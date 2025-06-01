"""
Simple calculation of vacancy formation energy with the following assumption

* Only one vacancy per cell.
* Cell size is sufficiently large.
* There is only one unique site per element.
* The system is in equilibrium with the elemental phase
"""

from functools import singledispatch

from aiida import orm
from aiida.engine import WorkChain, ProcessSpec, calcfunction
from aiida_vasp.workchains.v2.relax import VaspRelaxWorkChain
from aiida_grouppathx import GroupPathX
from matchest.aiida_utils.process.transformation import make_vac, make_supercell


class SimpleVacancyWorkChain(WorkChain):
    """Run simple vacancy calculation"""

    @classmethod
    def define(cls, spec: ProcessSpec):
        """Define the process"""

        super().define(spec)
        spec.expose_inputs(VaspRelaxWorkChain, namespace="relax")
        spec.input("elemental_group_path", valid_type=orm.Str, required=False)
        spec.input("supercell_workchain_updates", valid_type=orm.Dict, required=False)
        spec.input("supercell_dim", valid_type=orm.List)
        spec.input_namespace("elemental_structures", dynamic=True, required=False)

        spec.outline(
            cls.setup,
            cls.run_bulk_relax,
            cls.inspect_bulk_relax,
            cls.run_vac,
            cls.finalize,
        )

        spec.output_namespace("relaxed_structures", dynamic=True)
        spec.output_namespace("miscs", dynamic=True)
        spec.output_namespace("vacancy_formation_energies", dynamic=True)
        spec.exit_code(
            501, "ERROR_BULK_RELAX_FAILED", message="Bulk relaxation is failed"
        )
        spec.exit_code(
            500,
            "ERROR_ELEMENTAL_NOT_CALCULATED",
            message="Elemental reference calculations has not been finished yet.",
        )

    def setup(self):
        """
        Setup the inputs for the relaxation workchain
        """
        self.ctx.relax_inputs = self.exposed_inputs(
            VaspRelaxWorkChain, namespace="relax"
        )
        self.ctx.relaxed_bulk = None
        if "supercell_workchain_updates" in self.inputs:
            self.ctx.supercell_workchain_updates = (
                self.inputs.supercell_workchain_updates
            )
        else:
            self.ctx.supercell_workchain_updates = {}
        self.ctx.relax_settings_update = self.ctx.supercell_workchain_updates.get(
            "relax_settings", {}
        )
        self.ctx.options_update = self.ctx.supercell_workchain_updates.get(
            "options", {}
        )
        self.ctx.settings_update = self.ctx.supercell_workchain_updates.get(
            "settings", {}
        )
        self.ctx.incar_update = self.ctx.supercell_workchain_updates.get("incar", {})
        self.ctx.elems = list(
            set(self.inputs.relax.structure.get_ase().symbols)
        )  # Elements to make vacancy
        self.ctx.formula = (
            self.ctx.relax_inputs.structure.get_ase().get_chemical_formula()
        )
        self.ctx.supercell_dim = self.inputs.supercell_dim.get_list()

        # Check if the elemental energies has been calculated
        if "elemental_group_path" in self.inputs:
            workpath = GroupPathX(self.inputs.elemental_group_path.value)
            for elem in self.ctx.elems:
                if not workpath[f"{elem}_bulk"].node.is_finished_ok:
                    return self.exit_codes.ERROR_ELEMENTAL_NOT_CALCULATED
            self.ctx.calculate_elemental = False
        else:
            self.ctx.calculate_elemental = True

    def run_bulk_relax(self):
        """Run the bulk relaxation"""

        bulk_relax = self.submit(VaspRelaxWorkChain, **self.ctx.relax_inputs)
        return {"bulk_relax_workchain": bulk_relax}

    def inspect_bulk_relax(self):
        """Inspect the bulk relaxation"""

        if self.ctx.bulk_relax_workchain.is_finished_ok:
            self.ctx.relaxed_bulk = (
                self.ctx.bulk_relax_workchain.outputs.relax.structure
            )
            return
        return self.exit_codes.ERROR_BULK_RELAXATION_FAILED

    def run_vac(self):
        """Run vacancy structure calculations"""

        # Take the relaxed bulk structure as reference structure
        ref_structure = self.ctx.relaxed_bulk
        running = {}

        inputs = self.ctx.relax_inputs
        elems = self.ctx.elems
        # Run elemental calculations
        if self.ctx.calculate_elemental:
            for elem in elems:
                structure = self.inputs.elemental_structures[elem]
                inputs.structure = structure
                inputs.metadata.label = f"{elem} ELEMENTAL RELAX"
                running[f"workchain_{elem}"] = self.submit(VaspRelaxWorkChain, **inputs)

        # Update the inputs for supercell calculations
        relax_settings = inputs.relax_settings.get_dict()
        relax_settings["volume"] = False
        relax_settings["shape"] = False
        inputs.relax_settings = orm.Dict(dict=relax_settings)
        # Update other configuration ports
        inputs.vasp.settings = update_dict_node(
            inputs.vasp.settings, self.ctx.settings_update
        )
        inputs.vasp.options = update_dict_node(
            inputs.vasp.options, self.ctx.options_update
        )
        if self.ctx.incar_update:
            param = inputs.vasp.parameters.get_dict()
            param["incar"].update(self.ctx.incar_update)
            inputs.vasp.parameters = orm.Dict(dict=param)

        # Elements to make vacancy of
        superdim = "{}{}{}".format(*self.ctx.supercell_dim)
        for elem in elems:
            # Find the first occurance of the element
            # This assumes all atoms of the same element are equivalent by symmetry, which may not be the case
            # A better way is to spglib to find unique sites of each element and calculate for them all
            i_elem = ref_structure.get_ase().get_chemical_symbols().index(elem)
            #  This generate a vacancy cell
            vac_cell = make_vac(ref_structure, [i_elem], self.ctx.supercell_dim)
            inputs.structure = vac_cell
            inputs.metadata.label = f"{self.ctx.formula} {superdim} V_{elem} RELAX"
            running[f"workchain_V_{elem}"] = self.submit(VaspRelaxWorkChain, **inputs)

        # Supercell calculation
        supercell_ref = make_supercell(ref_structure, self.ctx.supercell_dim)[
            "structure"
        ]
        inputs.structure = supercell_ref
        inputs.metadata.label = f"{self.ctx.formula} {superdim} RELAX"
        running["workchain_supercell"] = self.submit(VaspRelaxWorkChain, **inputs)

        return running

    def finalize(self):
        """Compute the vacancy formation energy and set the outputs of the workchain"""

        # Collect the vacancy formation energy results
        for elem in self.ctx.elems:
            vac_workchain = self.ctx[f"workchain_V_{elem}"]
            if self.ctx.calculate_elemental:
                elem_misc = self.ctx[f"workchain_{elem}"].outputs.misc
            else:
                elem_misc = GroupPathX(self.inputs.elemental_group_path.value)[
                    f"{elem}_bulk"
                ].node.outputs.misc
            formation_energy = compute_formation_energy(
                self.ctx.workchain_supercell.outputs.misc,
                vac_workchain.outputs.misc,
                elem_misc,
            )
            self.out(f"miscs.V_{elem}", vac_workchain.outputs.misc)
            self.out(f"miscs.element_{elem}", elem_misc)
            self.out(f"vacancy_formation_energies.V_{elem}", formation_energy)
            self.out(
                f"relaxed_structures.V_{elem}", vac_workchain.outputs.relax.structure
            )

        self.out("miscs.supercell", self.ctx.workchain_supercell.outputs.misc)
        self.out(
            "relaxed_structures.supercell",
            self.ctx.workchain_supercell.outputs.relax.structure,
        )


def update_dict_node(dict_node, update_dict):
    if update_dict:
        pydict = dict_node.get_dict()
        pydict.update(update_dict)
        return orm.Dict(dict=pydict)
    return dict_node


@calcfunction
def compute_formation_energy(ref_work_misc, v_work_misc, elem_misc):
    """Show and return the vacancy formation energy given the supercell, vacancy-containing cell
    and the elemental reference calculation.
    The vacancy formation energy returns assumes the X-rich limit where the system is in
    equilibirum with the elemental phase.
    """
    evac = total_energy(v_work_misc)
    ebulk = total_energy(ref_work_misc)
    eelem = energy_per_atom(elem_misc)
    q = orm.QueryBuilder()
    q.append(orm.Node, filters={"pk": elem_misc.pk}, tag="dict")
    q.append(orm.CalcJobNode, with_outgoing="dict", tag="calc")
    q.append(orm.StructureData, with_outgoing="calc", project=["*"])
    elem_structure = q.first()[0]
    elem = elem_structure.sites[0].kind_name

    e_form = evac + eelem - ebulk
    return orm.Dict({
        f"V_{elem}": e_form,
        "E_supercell": ebulk,
        "E_vac_cell": evac,
        f"E_{elem}": eelem,
    })


@singledispatch
def total_energy(node):
    """Return the total energy"""
    raise NotImplementedError(f"total_energy is not implemented for {type(node)}")


@total_energy.register(orm.Dict)
def _(node):
    """Return the total energy"""
    return node["total_energies"]["energy_extrapolated"]


@total_energy.register(orm.ProcessNode)
def _(node):
    """Return the total energy"""
    return node.outputs.misc["total_energies"]["energy_extrapolated"]


@singledispatch
def energy_per_atom(node):
    """Return the energy per atom"""
    raise NotImplementedError(f"energy_per_atom is not implemented for {type(node)}")


@energy_per_atom.register(orm.ProcessNode)
def _(node):
    """Return the energy per atom"""
    eng = node.outputs.misc["total_energies"]["energy_extrapolated"]
    return eng / len(node.inputs.structure.sites)


@energy_per_atom.register(orm.Dict)
def _(node):
    """Return the energy per atom"""
    eng = node["total_energies"]["energy_extrapolated"]
    # Query to find the calculation node created this Dict node and its input structure
    q = orm.QueryBuilder()
    q.append(orm.Node, filters={"pk": node.pk}, tag="dict")
    q.append(orm.CalcJobNode, with_outgoing="dict", tag="calc")
    q.append(orm.StructureData, with_outgoing="calc", project=["*"])
    structure = q.first()[0]
    return eng / len(structure.sites)

"""
Pymatgen related tools
"""

from typing import Tuple

from aiida.plugins.factories import WorkflowFactory
from pymatgen.core.composition import (
    Composition,
    formula_double_format,
    gcd,
    get_el_sp,
)
from pymatgen.entries.computed_entries import ComputedStructureEntry

from .vasp import get_functional, get_u_map


def get_energy_from_misc(misc):
    """
    Get energy from misc output Dict/dictionary
    """
    if "energy_no_entropy" in misc["total_energies"]:
        return misc["total_energies"]["energy_no_entropy"]
    else:
        return misc["total_energies"]["energy_extrapolated"]


def load_mp_struct(mp_id, api_key=None):
    """
    Load Material Project structures using its api
    """
    # Query the database
    from aiida.orm import QueryBuilder, StructureData

    q = QueryBuilder()
    # For backward compatibility - also query the extras field
    q.append(
        StructureData,
        filters={"or": [{"extras.mp_id": mp_id}, {"attributes.mp_id": mp_id}]},
    )
    exist = q.first()

    if exist:
        return exist[0]
    # No data in the db yet - import from pymatgen
    from mp_api.client import MPRester

    if api_key is not None:
        rester = MPRester(api_key)
    else:
        # Use default api_key defined via environmental variable or pmgrc file
        rester = MPRester()
    struc = rester.get_structure_by_material_id(mp_id)
    magmom = struc.site_properties.get("magmom")
    strucd = StructureData(pymatgen=struc)
    strucd.label = strucd.get_formula()
    strucd.description = f"Imported from Materials Project ID={mp_id}"
    strucd.base.attributes.set("mp_id", mp_id)
    if magmom:
        strucd.base.attributes.set("mp_magmom", magmom)
    strucd.store()
    strucd.base.extras.set("mp_id", mp_id)
    if magmom:
        strucd.base.extras.set("mp_magmom", magmom)

    return strucd


def reduce_formula_no_polyanion(sym_amt, iupac_ordering=False) -> Tuple[str, int]:
    """
    Helper method to reduce a sym_amt dict to a reduced formula and factor.
    Unlike the original pymatgen version, this function does not do any polyanion reduction

    Args:
        sym_amt (dict): {symbol: amount}.
        iupac_ordering (bool, optional): Whether to order the
            formula by the iupac "electronegativity" series, defined in
            Table VI of "Nomenclature of Inorganic Chemistry (IUPAC
            Recommendations 2005)". This ordering effectively follows
            the groups and rows of the periodic table, except the
            Lanthanides, Actanides and hydrogen. Note that polyanions
            will still be determined based on the true electronegativity of
            the elements.

    Returns:
        (reduced_formula, factor).
    """
    syms = sorted(sym_amt.keys(), key=lambda x: [get_el_sp(x).X, x])

    syms = list(filter(lambda x: abs(sym_amt[x]) > Composition.amount_tolerance, syms))

    factor = 1
    # Enforce integers for doing gcd.
    if all(int(i) == i for i in sym_amt.values()):
        factor = abs(gcd(*(int(i) for i in sym_amt.values())))

    polyanion = []

    syms = syms[: len(syms) - 2 if polyanion else len(syms)]

    if iupac_ordering:
        syms = sorted(syms, key=lambda x: [get_el_sp(x).iupac_ordering, x])

    reduced_form = []
    for s in syms:
        normamt = sym_amt[s] * 1.0 / factor
        reduced_form.append(s)
        reduced_form.append(formula_double_format(normamt))

    reduced_form = "".join(reduced_form + polyanion)
    return reduced_form, factor


def get_entry_from_calc(calc):
    """Get a ComputedStructure entry from a given calculation/workchain"""
    misc = calc.outputs.misc
    energy = get_energy_from_misc(misc)
    in_structure = calc.inputs.structure

    # Check if there is any output structure - support for multiple interfaces
    if "structure" in calc.outputs:
        out_structure = calc.outputs.structure
    elif "relax" in calc.outputs:
        out_structure = calc.outputs.relax.structure
    elif "relax__structure" in calc.outputs:
        out_structure = calc.outputs.relax__structure
    else:
        out_structure = None

    if out_structure:
        entry_structure = out_structure.get_pymatgen()
    else:
        entry_structure = in_structure.get_pymatgen()

    if calc.process_label == "VaspCalculation":
        incar = calc.inputs.parameters.get_dict()
        pots = {pot.functional for pot in calc.inputs.potential.values()}
        if len(pots) != 1:
            raise RuntimeError(
                "Inconsistency in POTCAR functionals! Something is very wrong..."
            )
        pot = pots.pop()

    elif calc.process_label == "VaspWorkChain":
        incar = calc.inputs.parameters["incar"]
        pot = calc.inputs.potential_family.value
    elif calc.process_class == WorkflowFactory("vaspu.relax"):
        # For backword compatibility
        try:
            incar = calc.inputs.vasp.parameters["incar"]
            pot = calc.inputs.vasp.potential_family.value
        except AttributeError:
            incar = calc.inputs.parameters["vasp"]
            pot = calc.inputs.potential_family.value
    elif calc.process_class == WorkflowFactory("vasp.relax"):
        incar = calc.inputs.parameters["incar"]
        pot = calc.inputs.potential_family.value
    else:
        raise RuntimeError("Cannot determine calculation inputs")

    data = {
        "functional": get_functional(incar, pot),
        "umap": get_u_map(in_structure, incar.get("ldauu")),
        "calc_uuid": calc.uuid,
        "volume": entry_structure.volume,
    }

    out_entry = ComputedStructureEntry(entry_structure, energy, parameters=data)
    return out_entry

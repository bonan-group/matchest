"""
Elastic workflow for AiiDA-VASP.

This module implements a workflow to calculate elastic constants, including:
    - Strain generation for crystal structures
    - Stress calculation via VASP relaxation
    - Elastic tensor composition and validation
"""

from aiida_vasp.workchains.v2.relax import VaspRelaxWorkChain
import ase

# from ase.units import GPa
import numpy as np
from pymatgen.analysis.elasticity import (
    DeformedStructureSet,
    ElasticTensor,
    Strain,
    Stress,
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import spglib

from aiida import orm
from aiida.engine import ToContext, WorkChain, calcfunction


class VaspElasticWorkChain(WorkChain):
    """
    AiiDA WorkChain for calculating the elastic tensor of a material using VASP.
    This workchain performs a full relaxation of the input structure, standardizes it,
    generates deformed structures, and runs multiple relaxations on these deformed structures
    to compute the elastic tensor.
    """

    _base_workchain = VaspRelaxWorkChain

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "elastic_settings",
            valid_type=orm.Dict,
            required=False,
            help="Settings of elastic tensor calculation, valid options: use_symmetry, symprec",
        )
        # 基于VaspRelaxWorkChain的设置输入
        spec.expose_inputs(VaspRelaxWorkChain, "relax")
        spec.output("elastic_tensor", valid_type=orm.ArrayData)
        spec.output(
            "relaxed_structure",
            valid_type=orm.StructureData,
            help="The fully relaxed structure after the full relaxation step",
        )
        spec.output(
            "primitive_structure",
            valid_type=orm.StructureData,
            help="The standardized primitive structure or unit cell after the full relaxation step",
        )

        # Outline of the workchain,steps to be executed
        spec.outline(
            cls.setup,
            cls.full_relax,
            cls.standardize,
            cls.run_relax_multi,
            cls.compose_elastic_tensor,
        )
        spec.exit_code(
            500,
            "ERROR_SUB_PROCESS_FAILED",
            message="The subprocess has failed.",
        )
        spec.exit_code(
            501,
            "ERROR_ELASTIC_TENSOR_COMPOSITION",
            message="The elastic tensor could not be composed from the deformed structure relaxations.",
        )

    def setup(self):
        """
        Initialize context variables
        """

        self.ctx.relax_inputs = self.exposed_inputs(VaspRelaxWorkChain, "relax")

        pdict = self.ctx.relax_inputs.vasp.parameters.get_dict()
        self.ctx.elastic_settings = {} if "elastic_settings" not in self.inputs else self.inputs.elastic_settings
        pdict["incar"].pop("ibrion", None)
        pdict["incar"].pop("isif", None)
        pdict["incar"].pop("nsw", None)
        if pdict != self.ctx.relax_inputs.vasp.parameters.get_dict():
            self.logger.warn(
                """
                             The VASP parameters have been modified to remove `ibrion`,isif`,
                             and `nsw` settings for the elastic tensor calculation.
                             Please ensure that the modified parameters are suitable for your calculation.
                             """
            )
            self.ctx.relax_inputs.vasp.parameters = orm.Dict(dict=pdict)

        self.ctx.deformed_structures = []
        self.ctx.deformed_strains = []
        self.ctx.workchains = {}

    def full_relax(self):
        """
        Run a full relaxation of the input structure
        """
        self.report("Running full relaxation")
        relax_inputs = self.ctx.relax_inputs
        # relax_inputs.structure = self.inputs.structure
        # self.base_workchain的计算参数和输入结构由外部导入
        running = self.submit(self._base_workchain, **relax_inputs)
        return ToContext(full_relax=running)

    def standardize(self):
        """
        Standardize the structure after the full relaxation
        You can use spglib to standardize the structure or pymatgen's SpacegroupAnalyzer.
        """
        self.report("Standardizing the structure")
        relaxed_structure = self.ctx.full_relax.outputs.relax.structure
        self.out("relaxed_structure", relaxed_structure)
        primitive_type = self.ctx.elastic_settings.get("primitive_type", "conventional")
        if primitive_type == "conventional":
            # Use pymatgen's SpacegroupAnalyzer to get the conventional standard structure
            self.report("Using pymatgen SpacegroupAnalyzer to standardize the structure")
            conventional_structure = SpacegroupAnalyzer(
                relaxed_structure.get_pymatgen()
            ).get_conventional_standard_structure()
            self.ctx.reference_structure = orm.StructureData(pymatgen=conventional_structure)
        elif primitive_type == "primitive":
            atoms = relaxed_structure.get_ase()
            # standardize the structure using spglib
            # Use spglib to standardize the structure
            prim_atoms_data = spglib.find_primitive(
                (atoms.cell, atoms.get_scaled_positions(), atoms.numbers),
                symprec=self.ctx.elastic_settings.get("symprec", 1e-3),
            )
            primitive_lat, primitive_pos, primitive_numbers = prim_atoms_data
            primitive_atoms = ase.Atoms(
                cell=primitive_lat,
                scaled_positions=primitive_pos,
                numbers=primitive_numbers,
                pbc=True,
            )
            self.ctx.reference_structure = orm.StructureData(ase=primitive_atoms)
        else:
            raise ValueError(
                f'Unknown primitive type: {primitive_type}. Supported types are "conventional" and "primitive".'
            )
        self.out("primitive_structure", self.ctx.reference_structure)

    def run_relax_multi(self):
        """
        Run multiple relaxation calculations for the deformed structures
        """
        deformed = generate_deformed_structures(
            self.ctx.reference_structure,
            normal_strains=orm.List(self.ctx.elastic_settings.get("normal_strains", None)),
            shear_strains=orm.List(self.ctx.elastic_settings.get("shear_strains", None)),
            symmetry=self.ctx.elastic_settings.get("use_symmetry", True),
        )
        launched_calculations = {}
        inputs = self.ctx.relax_inputs
        base_relax_settings = inputs.relax_settings.get_dict()
        # Update the base relaxation settings for the deformed structures
        base_relax_settings["volume"] = False
        base_relax_settings["shape"] = False

        for key, value in deformed.items():
            if key.startswith("structure_"):
                d_structure = value
                inputs.structure = d_structure
                relax_settings = base_relax_settings.copy()
                relax_settings["label"] = f"relax_deformed_{key}"
                inputs.relax_settings = orm.Dict(dict=relax_settings)
                running = self.submit(self._base_workchain, **inputs)
                launched_calculations["workchain_deformed_" + key] = running
            if key.startswith("deformation_strains"):
                self.ctx.deformations = value  # 为orm.array数据类型
        return ToContext(**launched_calculations)

    def compose_elastic_tensor(self):
        """
        Get the elastic tensor from by compose_elastic_tensor method
        """
        self.report("Composing the elastic tensor from the deformed structure relaxations")
        miscs = {key: self.ctx[key].outputs.misc for key in self.ctx if key.startswith("workchain_deformed")}
        self.report("finished collecting workchain outputs for elastic tensor composition")
        # Check  the deformations data and miscs data
        self.report(f"deform_datas: {self.ctx.deformations.get_array('deformation_strains')}")
        self.report(f"miscs: {miscs}")
        self.out(
            "elastic_tensor",
            get_elastic_tensor(deform_datas=self.ctx.deformations, **miscs),
        )


@calcfunction
def generate_deformed_structures(
    structure: orm.StructureData,
    normal_strains: orm.List,
    shear_strains: orm.List,
    symmetry: bool = True,
):
    """
    generate deformed structures by pymatgen's DeformedStructureSet.
    Args:
        structure (orm.StructureData): The input structure to be deformed.
        normal_strains (orm.List, optional): List of normal strains to apply. Defaults to None.
        shear_strains (orm.List, optional): List of shear strains to apply. Defaults to None.
        symmetry (bool, optional): Whether to use symmetry in the deformation. Defaults to True.
    """
    if normal_strains is None:
        normal_strains = (-0.01, -0.005, 0.005, 0.01)
    else:
        normal_strains = normal_strains.get_list()
    if shear_strains is None:
        shear_strains = (-0.06, -0.03, 0.03, 0.06)
    else:
        shear_strains = shear_strains.get_list()
    deformed_structures = DeformedStructureSet(
        structure.get_pymatgen(),
        norm_strains=normal_strains,
        shear_strains=shear_strains,
        symmetry=bool(symmetry),
    )
    output = {}
    for i, d_structure in enumerate(deformed_structures.deformed_structures):
        # 转换为 AiiDA StructureData
        structure_data = orm.StructureData(pymatgen=d_structure)
        structure_data.label = f"deformed_{i}"
        structure_data.description = f"Deformed structure {i} for elastic tensor calculation"
        output[f"structure_{i}"] = structure_data
    strain_data = orm.ArrayData()
    strain_data.set_array("deformation_strains", np.array(deformed_structures.deformations))
    output["deformation_strains"] = strain_data
    return output


def get_elastic_tensor(deform_datas: orm.ArrayData, **kwargs):
    """
    get the elastic tensor from the results of the deformed structure relaxations.
    """
    strains = []
    stresses = []
    # 得到弛豫完后结构的应力，再得到对应的应变，然后
    # elastic_tensor = ElasticTensor.from_diff_fit(strains, stresses)得到弹性张量
    deform_datas = deform_datas.get_array("deformation_strains")

    miscs = []
    for key, value in kwargs.items():
        if key.startswith("workchain_deformed"):
            idx = int(key.split("_")[-1])
            miscs.append([idx, value])

    miscs = sorted(miscs, key=lambda x: x[0])

    for idx, misc in miscs:
        stress = -np.array(misc.get_dict()["stress"]) * 0.1  # Convert units to GPa from kBar
        stress_matrix = Stress(stress)
        stresses.append(stress_matrix)

        strain_matrix = Strain.from_deformation(deform_datas[idx])
        strains.append(strain_matrix)
    try:
        elastic_tensor = ElasticTensor.from_diff_fit(strains, stresses)
    except Exception as e:
        raise ValueError(f"Failed to compose the elastic tensor: {e}") from e

    elastic_array = orm.ArrayData()
    elastic_array.set_array("elastic_tensor", elastic_tensor.voigt)
    elastic_array.store()
    return elastic_array

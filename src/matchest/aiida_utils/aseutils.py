"""
Utilities for using MACE with aiida-vasp
"""

import io

import ase
import numpy as np
from aiida import orm
from aiida_vasp.parsers.content_parsers.vasprun import VasprunParser
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import GPa


def split_set(data, a, b, shuffle=True):
    """
    Split the dataset into three parts
    """
    if shuffle:
        idx = np.arange(len(data))
        idx = np.random.permutation(idx)
        data = [data[i] for i in idx]

    n = len(data)
    ia = int(np.floor(n * a))
    ib = int(np.floor(n * b))
    return [data[:ia], data[ia:ib], data[ib:]]


def compose_atoms(calcnode: orm.ProcessNode) -> ase.Atoms:
    """Compose an Atoms object with a calculation node"""

    if "vasp" in calcnode.process_label.lower():
        atoms = _compose_atoms_vasp(calcnode)
    if "abacus" in calcnode.process_label.lower():
        atoms = _compose_atoms_abacus(calcnode)
    atoms.info["aiida_uuid"] = calcnode.uuid
    atoms.info["aiida_label"] = calcnode.label
    return atoms


def _compose_atoms_abacus(node) -> ase.Atoms:
    misc = node.outputs.misc
    if "relax" in node.outputs:
        structure = node.outputs.structure
    else:
        structure = node.inputs.abacus.structure
    atoms = structure.get_ase()
    eng = misc["total_energy"]
    forces = np.array(misc["final_forces"])
    # Convert to eV / A^3, aiida-abacus uses GPa as the unit
    if "final_stress" in misc:
        stress = (np.array(misc["final_stress"]) * 0.1 * GPa * -1).tolist()
    else:
        stress = (np.array(misc["all_stress"][-1]) * 0.1 * GPa * -1).tolist()

    atoms = structure.get_ase()
    sp = SinglePointCalculator(atoms, energy=eng, forces=forces, stress=stress)
    atoms.calc = sp

    return atoms


def _compose_atoms_vasp(calcnode) -> ase.Atoms:
    """Compose an Atoms object with a calculation node, parsing from the retrieved files"""

    links = [x.link_label for x in calcnode.base.links.get_outgoing().all()]
    misc = calcnode.outputs.misc.get_dict()
    if "force" in links and "stress" in links and "energy_free" in misc["total_energies"]:
        eng = misc["total_energies"]["energy_free"]
        forces = calcnode.outputs.forces.get_array("final")
        stress = -calcnode.outputs.stress.get_array("final") * (GPa / 10)
    elif (
        "forces" in misc and "stress" in misc and "energy_free" in misc["total_energies"]
    ):  # Forces and stress has been parsed and made avaliable
        eng = misc["total_energies"]["energy_free"]
        if "final" in misc["forces"]:
            forces = misc["forces"]["final"]
        else:
            forces = misc["forces"]
        if "final" in misc["stress"]:
            stress = misc["stress"]["final"]
        else:
            stress = misc["stress"]
        forces = np.array(forces)
        stress = np.array(stress)
        stress = -stress * (GPa / 10)
    else:
        # Parse from retrieved files
        with calcnode.outputs.retrieved.open("vasprun.xml", "rb") as fh:
            buffer = io.BytesIO(fh.read())
            buffer.seek(0)
            parser = VasprunParser(handler=buffer)
            parser._settings["energy_type"] = ["energy_extrapolated", "energy_free"]

        eng = parser.total_energies["energy_free"]
        forces = parser.final_forces
        # VASP gives stress in kBar, we convert it to eV based to be consistent
        stress = -parser.final_stress * (GPa / 10)

    atoms = calcnode.inputs.structure.get_ase()
    sp = SinglePointCalculator(atoms, energy=eng, forces=forces, stress=stress)
    atoms.calc = sp
    return atoms


def compose_atoms_mace(
    calcnode, energy_key="REF_energy", forces_key="REF_forces", stress_key="REF_stress"
) -> ase.Atoms:
    """
    Compose a atoms which can be used as training data point for MACE
    The energy, forces and stress are saved as "REF_energy", "REF_forces", "REF_stress"
    """
    atoms = compose_atoms(calcnode)
    atoms.set_array(forces_key, atoms.get_forces())
    atoms.info[energy_key] = atoms.get_potential_energy()
    atoms.info[stress_key] = atoms.get_stress(voigt=False)
    # Clear other data
    atoms.info.pop("energy", None)
    atoms.info.pop("stress", None)
    atoms.arrays.pop("forces", None)
    atoms.calc = None
    return atoms

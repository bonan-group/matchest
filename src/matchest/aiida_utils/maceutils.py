from aiida.engine.processes import calcfunction
import aiida.orm as orm
from ase.build import sort, make_supercell
import ase
import numpy as np
from aiida_vasp.parsers.content_parsers.vasprun import VasprunParser
import io

def split_set(data, a, b, shuffle=True):
    """
    Split the dataset into three parts
    """
    if shuffle:
        idx = np.arange(len(data))
        idx = np.random.permutation(idx)
        data = [data[i] for i in idx]
        
    n = len(data)
    ia = int(np.floor(n * a ))
    ib = int(np.floor(n * b))
    return [data[:ia], data[ia:ib], data[ib:]]


def compose_atoms(calcnode) -> ase.Atoms:
    """Compose an Atoms object with a calculation node, parsing from the retrieved files"""
    from ase.units import GPa
    from ase.calculators.singlepoint import SinglePointCalculator
    links = [x.link_label for x in calcnode.base.links.get_outgoing().all()]
    misc = calcnode.outputs.misc.get_dict()
    if 'force' in links and 'stress' in links and 'energy_free' in  misc['total_energies']:
        eng = misc['total_energies']['energy_free']
        forces = calcnode.outputs.forces.get_array('final')
        stress = calcnode.outputs.stress.get_array('final') * (GPa / 10)
    elif 'forces' in misc and 'stress' in misc and 'energy_free' in  misc['total_energies']:  # Forces and stress has been parsed and made avaliable
        eng = misc['total_energies']['energy_free']
        if 'final' in misc['forces']:
            forces = misc['forces']['final']
        else:
            forces = misc['forces']
        if 'final' in misc['stress']:
            stress = misc['stress']['final']
        else:
            stress = misc['stress']
        forces = np.array(forces)
        stress = np.array(stress)
        stress = -stress * (GPa / 10)
    else:
        # Parse from retrieved files
        with calcnode.outputs.retrieved.open('vasprun.xml', 'rb') as fh:
            buffer = io.BytesIO(fh.read())
            buffer.seek(0)
            parser = VasprunParser(handler=buffer)
            parser._settings['energy_type'] = ['energy_extrapolated', 'energy_free']
    
        eng = parser.total_energies['energy_free']
        forces = parser.final_forces
        # VASP gives stress in kBar, we convert it to eV based to be consistent
        stress = -parser.final_stress * (GPa / 10)
        
    atoms = calcnode.inputs.structure.get_ase()
    sp = SinglePointCalculator(atoms, energy=eng, forces=forces, stress=stress)
    atoms.calc = sp
    atoms.info['uuid'] = calcnode.uuid
    atoms.info['label'] = calcnode.label
    return atoms

    
def compose_atoms_mace(calcnode):
    """
    Compose a atoms which can be used as training data point for MACE
    The energy, forces and stress are saved as "REF_energy", "REF_forces", "REF_stress"
    """
    atoms = compose_atoms(calcnode)
    atoms.set_array("REF_forces", atoms.get_forces())
    atoms.info("REF_energy", atoms.get_potential_energy())
    atoms.info("REF_stress", atoms.get_stress(voigt=False))
    # Clear other data
    atoms.info.pop('energy', None)
    atoms.info.pop('stress', None )
    atoms.arrays.pop('forces', None )
    atoms.calc = None
    return atoms


@calcfunction
def get_supercell(node, supercell):
    """Short-cut function to generate super cell and record the provenance"""
    atoms = node.get_ase()
    output = sort(make_supercell(atoms, supercell.get_list()))
    return orm.StructureData(ase = output)
    
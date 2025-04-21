"""
Routines to handle kpoints, mostly based on: https://github.com/WMD-group/kgrid

The reciprocal space of a given structure needs to be sampled with a mesh,
the routines in this module is for working out the grid needed with a given
spacing (in the reciprocal space) or a cut off length (in the real space).

The former has different units in different codes. In CASTEP, it has a unit
of 2pi A^-1 while in VASP/QE it does not have the 2pi factor.
In addition, for lattices with orthogonal vectors, the length of a* is simply
1/a, but it is not the case in monoclinic/triclinic systems. So the user may
choose to calculate the mesh based on the real space lattice vectors length 
or with reciprocal lattice vector lengths. 
"""
import numpy as np
from typing import Optional
import ase.io
from typing import List, Tuple


def calc_kpt_tuple_naive(atoms, cutoff_length=10, rounding='up'):
    """Calculate k-point grid using real-space lattice vectors"""

    abc = atoms.cell.cellpar()[:3]
    k_samples = np.divide(2*cutoff_length,abc)
    if rounding == 'up':
        k_samples = np.ceil(k_samples)
    else:
        k_samples = np.floor(k_samples + 0.5)
    return tuple((int(x) for x in k_samples))


def calc_kpt_tuple_recip(atoms, cutoff_length=10, rounding='up'):
    """Calculate reciprocal-space sampling with real-space parameter"""

    # Get reciprocal lattice vectors with ASE. Note that ASE does NOT include
    # the 2*pi factor used in many definitions of these vectors; the underlying
    # method is just a matrix inversion and transposition
    recip_cell = atoms.cell.reciprocal()

    # Get reciprocal cell vector magnitudes according to Pythagoras' theorem
    abc_recip = np.linalg.norm(recip_cell, axis=1)

    k_samples = abc_recip * 2 * cutoff_length

    # Rounding
    if rounding == 'up':
        k_samples = np.ceil(k_samples)
    else:
        k_samples = np.floor(k_samples + 0.5)
    return tuple((int(x) for x in k_samples))

def calc_kpt_tuple(atoms, cutoff_length=10, realspace=False, mode='default'):
    """
    Return the kpoint mesh in a tuple with given structure and real space cut off distance
    """

    if mode.lower() == 'default':
        rounding = 'up'
    elif mode.lower() == 'vasp_auto':
        cutoff_length = (cutoff_length) / 2.
        rounding = 'nearest'
    elif mode.lower() == 'kspacing':
        cutoff_length = np.pi / cutoff_length
        rounding = 'up'
    elif mode.lower() == 'castep_mp_spacing':
        cutoff_length = 1 / (2 * cutoff_length)
        rounding = 'up'

    if realspace:
        return calc_kpt_tuple_naive(atoms, cutoff_length=cutoff_length,
                                    rounding=rounding)
    else:
        return calc_kpt_tuple_recip(atoms, cutoff_length=cutoff_length,
                                    rounding=rounding)


def get_increments(lattice_lengths: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Calculate the vector l0 of increments between significant length cutoffs for each reciprocal lattice vector."""
    return tuple(1. / (2 * a) for a in lattice_lengths)

def cutoff_series(atoms: ase.Atoms, l_min: float, l_max: float, decimals: int = 4) -> List[float]:
    """Find multiples of l0 members within a range."""
    recip_cell = atoms.cell.reciprocal()
    lattice_lengths = np.linalg.norm(recip_cell, axis=1) 
    l0 = get_increments(lattice_lengths)
    members = set()
    for li in l0:
        n_min = np.ceil(l_min / li)
        members.update(set(np.around(np.arange(n_min * li, l_max, li), decimals=decimals)))
    return sorted(members)

def kspacing_series(atoms: ase.Atoms, l_min: float, l_max: float, decimals: int = 4) -> List[float]:
    """Find series of KSPACING values with different results."""
    return [np.pi / c for c in cutoff_series(atoms, l_min, l_max, decimals=decimals)]

def kpoints_main(filename: str, file_type: Optional[str], l_min: float, l_max: float, comma_sep: bool, 
                 vasp: bool, realspace: bool) -> None:
    atoms = ase.io.read(filename, format=file_type) if file_type else ase.io.read(filename)
    cutoffs = cutoff_series(atoms, l_min, l_max)
    kspacing = [0.5 / c for c in cutoffs] if not vasp else [np.pi / c for c in cutoffs]
    samples = [calc_kpt_tuple(atoms, cutoff_length=(cutoff - 1e-4), realspace=realspace) for cutoff in cutoffs]

    if comma_sep:
        def print_sample(sample: Tuple[int, int, int]) -> str:
            return ' '.join(str(x) for x in sample)
        print(','.join(print_sample(sample) for sample in samples))
    else:
        header = "Length cutoff (Å)  MP SPACING (2π/Å)    Samples" if not vasp else "Length cutoff (Å)  KSPACING (1/Å)     Samples"
        print(header)
        print("-----------------  -------------------  ------------" if not vasp else "-----------------  -----------------  ------------")
        fstring = "{0:16.3f}   {1:18.6f}   {2:3d} {3:3d} {4:3d}" if not vasp else "{0:16.3f}   {1:16.4f}   {2:3d} {3:3d} {4:3d}"
        for cutoff, s, sample in zip(cutoffs, kspacing, samples):
            print(fstring.format(cutoff, s, *sample))
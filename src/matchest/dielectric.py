"""
Module for computing the ionic dielectric constant from the second derivative matrix and Born effective charges.
"""

from ase.vibrations import Vibrations
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter
from shutil import rmtree
from tempfile import mkdtemp

from parsevasp.vasprun import Xml
from pathlib import Path
import numpy as np
from ase.units import Bohr, Ry, pi


def dof2idx(n):
    """Convert dof index to direction and ion index"""
    s = n // 3
    k = n % 3
    return k, s


def idx2dof(i, n):
    """Convert direction and ion index to dof idx"""
    return n * 3 + i


def _outcar_read_second_derivatives(lines):
    array = []
    for i, line in enumerate(lines):
        if "SECOND DERIVATIVES (NOT SYMMETRIZED)" in line:
            j = i + 3
            # Start from the third line below
            while j < (i + 999):
                subline = lines[j]
                tokens = subline.strip().split()
                # Stop when encoutering empty line
                if len(tokens) < 2:
                    break
                # Record the data
                array.append([float(x) for x in tokens[1:]])
                j += 1
        if array:
            break
    return np.array(array)


def _inv_second_deriv(cuni):
    """
    Return the filtered inverse of the second derivative matrix.
    The first three modes are translational modes, which are not included in the inverse.
    Soft modes close to 0 frequency are also not included.
    The matrix is symmetrized before inversion.
    """
    dof = cuni.shape[0]
    assert dof == cuni.shape[1]

    # Diagonalize the matrix using eigh for symmetric/hermitian matrices
    hfeig, ct = np.linalg.eigh(cuni)

    # Create a temporary matrix for filtered inverse calculation
    ct_filtered = np.zeros_like(ct)

    # Skip the lowest three eigenvalues (translational modes) and soft modes close to 0 frequency
    threshold = 1e-3
    for n2 in range(3, dof):
        if hfeig[n2] > threshold:
            ct_filtered[:, n2] = ct[:, n2] / hfeig[n2]

    # Reconstruct the matrix using filtered eigenvectors and eigenvalues
    ceidb = np.dot(ct_filtered, ct.T)
    return ceidb


def compute_ionic_dielectric(second_dev, born_charges, unitcell):
    """
    Compute the ionic dielectric constant from the second derivative matrix and Born effective charges.
    Based on 10.1103/PhysRevB.72.035105.
    """

    second_dev = 0.5 * (second_dev + second_dev.T)  # Symmetrize
    inv_second_dev = _inv_second_deriv(second_dev)
    epsilon_ion = np.zeros((3, 3))
    ndof = inv_second_dev.shape[0]
    for dof1 in range(ndof):
        k, s = dof2idx(dof1)  # noqa: E741
        for dof2 in range(ndof):
            l, sp = dof2idx(dof2)  # noqa: E741
            for i in range(3):
                for j in range(3):
                    epsilon_ion[i, j] += born_charges[s, i, k] * inv_second_dev[dof1, dof2] * born_charges[sp, j, l]
    omega = np.cross(unitcell[0], unitcell[1]) @ unitcell[2]  # Volume
    factor = 2 * Ry * Bohr * 4 * pi / omega
    epsilon_ion *= factor
    return epsilon_ion


def check_equivalence(folder):
    """
    Compute the ionic dielectric constant wih and compare with the one in vasprun.xml

    This validates the implementation of the ionic dielectric constant calculation
    compared with VASP's internal routine.
    """
    second_dev = _outcar_read_second_derivatives((folder / "OUTCAR").read_text().split("\n"))
    xml_parser = Xml(file_path=folder / "vasprun.xml")
    born = xml_parser.get_born()
    epsilon_ion = compute_ionic_dielectric(second_dev, born, xml_parser.get_unitcell("last"))
    epsilon_ion_ref = xml_parser.get_epsilon_ion()
    return epsilon_ion, epsilon_ion_ref


def get_ionic_dielectric(atoms, calc):
    """
    Compute the ionic dielectric constant from the atoms object and calculator.

    The atoms object should have the Born effective charges stored in the 'born_pred' array.

    :return: the Hessian matrix and the ionic dielectric constant.
    """
    atoms.calc = calc
    opt = LBFGS(UnitCellFilter(atoms), logfile=None)
    opt.run(steps=500, fmax=0.001)

    cache_name = mkdtemp(prefix="vib-cache-")
    if Path(cache_name).is_dir():
        rmtree(cache_name)
    vib = Vibrations(atoms, name=cache_name)
    vib.run()
    vib.get_frequencies()
    hessian = vib.H
    born = atoms.get_array("born_pred").reshape((len(atoms), 3, 3))

    # Remove the cache directory after use
    rmtree(cache_name)

    return -hessian, compute_ionic_dielectric(-hessian, born, atoms.cell)

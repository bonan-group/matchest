from pathlib import Path
import ase
import ase.io
import spglib
from typing import Optional

from .main import mc


def get_primitive_atoms(
    atoms: ase.Atoms,
    threshold: float = 1e-5,
    angle_tolerance: float = -1.0,
    print_spacegroup: bool = False,
) -> ase.Atoms:
    """Convert ASE Atoms to primitive cell using spglib"""
    atoms_spglib = (
        atoms.cell.array,
        atoms.get_scaled_positions(),
        atoms.numbers,
    )
    spacegroup = spglib.get_spacegroup(
        atoms_spglib, symprec=threshold, angle_tolerance=angle_tolerance)
    if print_spacegroup:
        print(f"Space group: {spacegroup}")
    cell, positions, atomic_numbers = spglib.find_primitive(
        atoms_spglib, symprec=threshold, angle_tolerance=angle_tolerance)
    primitive_atoms = ase.Atoms(
        scaled_positions=positions,
        cell=cell,
        numbers=atomic_numbers,
        pbc=True)
    return primitive_atoms

def get_primitive(
    input_file: Path = Path('POSCAR'),
    input_format: Optional[str] = None,
    output_file: Optional[Path] = None,
    output_format: Optional[str] = None,
    threshold: float = 1e-5,
    angle_tolerance: float = -1.,
    verbose: bool = False,
    precision: int = 6) -> None:
    if output_file is None:
        verbose = True
    float_format_str = f"{{:{precision+4}.{precision}f}}"
    def format_float(x: float) -> str:
        return float_format_str.format(x)
    atoms = ase.io.read(input_file, format=input_format)
    atoms = get_primitive_atoms(
        atoms, threshold=threshold, angle_tolerance=angle_tolerance, print_spacegroup=verbose
    )
    if verbose:
        print("Primitive cell vectors:")
        for row in atoms.cell:
            print(" ".join(map(format_float, row)))
        print("Atomic positions and proton numbers:")
        for position, number in zip(atoms.get_scaled_positions(), atoms.numbers):
            print(" ".join(map(format_float, position)) + f"\t{number}")
    if output_file is None:
        pass
    else:
        atoms.write(output_file, format=output_format)


def get_spacegroup(
    filename: Optional[Path] = None,
    filetype: Optional[str] = None):
    if filename is None:
        for candidate in ("geometry.in", "POSCAR", "castep.cell"):
            if (structure_file := Path.cwd() / candidate).is_file():
                filename = structure_file
                break
        else:
            raise ValueError("Input file not specified, no default found.")
    atoms = ase.io.read(str(filename), format=filetype)
    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    print("| Threshold / Å |    Space group    |")
    print("|---------------|-------------------|")
    for threshold in (1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1):
        print("|    {0:0.5f}    |  {1: <16} |".format(
            threshold, spglib.get_spacegroup(cell, symprec=threshold)))



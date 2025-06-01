"""
Utility module for analysing CASTEP data
"""

import numpy as np


def atoms_to_castep(atoms, index):
    """Convert ase atoms' index to castep like
    return (Specie, Ion) Depricatede, use ase_to_castep_index"""
    atom = atoms[index]
    symbol = atom.symbol
    # Start counter
    count = 0
    for atom in atoms:
        if atom.symbol == symbol:
            count += 1
        if atom.index == index:
            break
    return symbol, count


def ase_to_castep_index(atoms, indices):
    """Convert a list of indices to castep syle
    return list of (element, i in the same element)"""
    if isinstance(indices, int):
        indices = [indices]
    symbols = np.array(atoms.get_chemical_symbols())
    iatoms = np.arange(len(atoms))
    res = []
    # Iterate through given indices
    for i in indices:
        mask = symbols == symbols[i]  # Select the same species

        # Find the index via counting from first occurance
        for c, a in enumerate(iatoms[mask]):
            if a == i:
                res.append((symbols[i], c + 1))  # Castep start counting from 1
                break
    return res


def generate_ionic_fix_cons(atoms, indices, mask=None):
    """
    create ionic constraint section via indices and ase Atoms
    mask: a list of 3 integers, must be 0 (no fix) or 1 (fix this cartesian)
    """
    castep_indices = ase_to_castep_index(atoms, indices)
    count = 1
    lines = []
    if mask is None:
        mask = (1, 1, 1)
    for symbol, i in castep_indices:
        if mask[0]:
            lines.append(f"{count:<4d} {symbol:<2}    {i:<4d} 1 0 0")
        if mask[1]:
            lines.append(f"{count + 1:<4d} {symbol:<2}    {i:<4d} 0 1 0")
        if mask[2]:
            lines.append(f"{count + 2:<4d} {symbol:<2}    {i:<4d} 0 0 1")
        count += sum(mask)
    return lines


def castep_to_atoms(atoms, specie, ion):
    """Convert castep like index to ase Atoms index"""
    return [atom for atom in atoms if atom.symbol == specie][ion - 1].index


def sort_atoms_castep(atoms, copy=True, order=(0, 1, 2)):
    """
    Sort atoms to castep style
    :param copy: If True then return a copy of the atoms.
    :param order: orders of coordinates. (0, 1, 2) means the sorted atoms
    will be ascending by x, then y, then z if there are equal x or ys.
    """
    if copy:
        atoms = atoms.copy()

    # Sort castep style
    for i in reversed(order):
        isort = np.argsort(atoms.positions[:, i], kind="mergesort")
        atoms.positions = atoms.positions[isort]
        atoms.numbers = atoms.numbers[isort]

    isort = np.argsort(atoms.numbers, kind="mergesort")
    atoms.positions = atoms.positions[isort]
    atoms.numbers = atoms.numbers[isort]

    return atoms


def take_popn(seed):
    """
    Take section of population analysis from a seed.castep file
    Return a list of StringIO of the population analysis section
    """
    import io

    popns = []
    rec = False
    with open(seed + ".castep") as fh:
        for line in fh:
            if "Atomic Populations (Mulliken)" in line:
                record = io.StringIO()
                rec = True

            # record information
            if rec is True:
                if line.strip() == "":
                    rec = False
                    record.seek(0)
                    popns.append(record)
                else:
                    record.write(line)

    return popns


def read_popn(fn):
    """Read population file into pandas dataframe"""
    import pandas as pd

    table = pd.read_table(fn, sep=r"\s\s+", header=2, comment="=", engine="python")
    return table


def count_scf_lines(lines):
    """
    Extract the number of SCF cycles in the CASTEP files
    """
    counter = -1
    lengths = []
    for line in lines:
        if "Initial" in line:
            counter = 1
            continue
        if "----" in line and counter >= 1:
            lengths.append(counter)
            counter = -1
            continue
        # Counting the SCF lines
        if counter >= 1:
            counter += 1
    return lengths

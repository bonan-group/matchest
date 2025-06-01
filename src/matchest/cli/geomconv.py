"""
Getting geometry optimization convergence information

Analyze VASP geometry optimization convergence from OUTCAR file.

This module provides a function to extract and analyze convergence information
from a VASP OUTCAR file, including energy, forces, stress, computational time,
and convergence status.

Based on Fortran code written by Yida Yang

Functions
---------
analyze_vasp_convergence(outcar_file: str) -> None
    Analyze VASP convergence from OUTCAR file and print results.
"""

import re
from typing import List

import numpy as np


def print_vasp_conv(outcar_file: str) -> None:
    """
    Extract and analyze VASP geometry optimization convergence information.

    Parameters
    ----------
    outcar_file : str
        Path to the VASP OUTCAR file to analyze.

    Returns
    -------
    None
        Prints formatted convergence information to stdout.

    Notes
    -----
    The function extracts the following information for each ionic step:
    - Total energy (eV)
    - Energy difference between steps (eV)
    - Maximum atomic force (eV/Å)
    - Maximum stress component (GPa)
    - Computational time (s)
    - Average drift
    - Convergence status (YES/NO)

    Example
    -------
    >>> analyze_vasp_convergence("OUTCAR")
    Step |   E (eV)  |  dE (eV)   | Fmax (eV/Å) | Smax (Gpa) | Time (s) | Average drift  | Convergence |
    -----------------------------------------------------------------------------------------------------
      1    -42.123456     --------      0.123456     1.234567    120.0        0.000123        YES
      2    -42.123789      0.000333     0.098765     0.987654    240.0        0.000098        YES
    EDIFFG= 0.100
    """
    # Initialize variables
    natoms = 0
    volume = 0.0
    ediffg = 0.0
    energies: List[float] = []
    forces: List[np.ndarray] = []  # Each step has a (natoms, 3) array
    stresses: List[np.ndarray] = []  # Each step has a (3, 3) array
    cpu_times: List[float] = []
    total_drifts: List[np.ndarray] = []  # Each step has a (3,) array

    # Regular expressions for pattern matching
    patterns = {
        "nions": re.compile(r"NIONS\s*=\s*(\d+)"),
        "energy": re.compile(r"^  energy\s*without\s*entropy=\s*([-\d.]+)"),
        "positions": re.compile(r"POSITION\s*TOTAL-FORCE \(eV/Angst\)"),
        "volume": re.compile(r"volume of cell\s*:\s*([\d.]+)"),
        "stress": re.compile(r"^  external pressure =\s*([\d.-]+) kB"),
        "loop_time": re.compile(r"LOOP\+:\s*cpu time\s*([\d.]+)"),
        "ediffg": re.compile(r"EDIFFG\s*=\s*([-\d.]+)"),
    }

    try:
        with open(outcar_file) as f:
            lines = f.readlines()
    except OSError:
        print(f"Error: Cannot open input file {outcar_file}")
        return

    # First pass to get number of atoms
    for line in lines:
        if match := patterns["nions"].search(line):
            natoms = int(match.group(1))
            break

    if natoms == 0:
        print("Error: Could not determine number of atoms from OUTCAR")
        return

    # Second pass to get other information
    step = -1  # Will be incremented when we find a new step
    reading_forces = False
    atoms_read = 0

    for line in lines:
        # Get EDIFFG
        if not ediffg and (match := patterns["ediffg"].search(line)):
            ediffg = float(match.group(1))

        # Get volume
        if not volume and (match := patterns["volume"].search(line)):
            volume = float(match.group(1))

        # Get energy
        if match := patterns["energy"].search(line):
            energies.append(float(match.group(1)))
            reading_forces = False

        # Get forces
        if patterns["positions"].search(line):
            step += 1
            reading_forces = True
            atoms_read = 0
            forces.append(np.zeros((natoms, 3)))
            total_drifts.append(np.zeros(3))
            # Skip the next line (header)
            continue

        if reading_forces and atoms_read < natoms:
            parts = line.split()
            if len(parts) >= 6:
                forces[step][atoms_read] = np.abs([float(x) for x in parts[3:6]])
                atoms_read += 1
            if atoms_read == natoms:
                # Next line is blank, then total drift
                reading_forces = False

        # Get total drift (appears after forces)
        if line.strip().startswith("total drift:"):
            parts = line.split()
            if len(parts) >= 5:
                total_drifts[step] = np.array([float(x) for x in parts[2:5]])

        # Get stress
        if patterns["stress"].search(line):
            #   external pressure =     2144.70 kB  Pullay stress =        0.00 kB
            stresses.append(float(line.split()[3]))

        # Get CPU time
        if match := patterns["loop_time"].search(line):
            if step >= 0:  # Ensure we've started collecting steps
                cpu_times.append(float(match.group(1)))

    # Process collected data
    if not energies:
        print("Error: No energy data found in OUTCAR")
        return

    nsteps = len(energies)
    max_forces = [np.max(np.abs(force)) for force in forces]
    max_stresses = [stress / 10 for stress in stresses]  # Convert to GPa
    avg_drifts = [np.mean(np.abs(drift)) for drift in total_drifts]
    converged = ["YES" if fmax < abs(ediffg) else "NO" for fmax in max_forces]

    # Print results
    print(
        " Step |   E (eV)  |   dE (eV)    |  Fmax (eV/Å) | Smax (Gpa) | Time (s) | Average drift  | Convergence |"
    )
    print(
        "---------------------------------------------------------------------------------------------------------"
    )

    for i in range(nsteps):
        if i == 0:
            de = "--------"
        else:
            de = f"{energies[i] - energies[i-1]:.6f}"
        # In case of bad termination
        if len(cpu_times) == i:
            cpu_times.append(np.nan)
        print(
            f"  {i+1:2d}   {energies[i]:11.6f}   {de:>11}   {max_forces[i]:11.6f}   "
            f"{max_stresses[i]:11.6f}   {cpu_times[i]:8.1f}       {avg_drifts[i]:11.6f}       {converged[i]:>7}"
        )

    print(f"EDIFFG= {ediffg:6.3f}")

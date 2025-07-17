"""
VASP input file checker module.

This module provides tools to check VASP input files for optimal parallelization
settings and computational efficiency.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import click
from ..utils.kmesh import get_ir_kpoints_and_weights

CHEMICAL_SYMBOLS = [
    # 0
    "X",
    # 1
    "H",
    "He",
    # 2
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    # 3
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    # 4
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    # 5
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    # 6
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    # 7
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

ATOMIC_NUMBERS = {symbol: Z for Z, symbol in enumerate(CHEMICAL_SYMBOLS)}


class InputCheckError(ValueError):
    """Error indicating problem with the input file"""

    pass


@dataclass
class CalculationInfo:
    """Information about a VASP calculation."""

    path: Path
    n_atoms: int
    n_kpoints: int
    n_bands: int
    n_electrons: int
    ncore: Optional[int] = None
    kpar: Optional[int] = None
    npar: Optional[int] = None
    issues: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []

    @property
    def computational_cost(self) -> float:
        """Estimate computational cost proportional to n_kpoints * n_bands^2."""
        return self.n_kpoints * (self.n_bands**2)


class VASPInputChecker:
    """
    A class to check VASP input files for optimal settings and potential issues.

    This checker examines INCAR, POSCAR, POTCAR, and KPOINTS files to:
    1. Parse parallelization settings (NCORE, KPAR, NPAR)
    2. Estimate computational requirements
    3. Validate parallelization settings against calculation size
    """

    def __init__(self, root_dir, min_cost_threshold: float = 1000.0, max_cost_threshold: float = 1e6):
        """
        Initialize the VASP input checker.

        Args:
            min_cost_threshold: Minimum cost below which parallelization warnings are issued
            max_cost_threshold: Maximum cost above which efficiency warnings are issued
        """
        self.root_dir = Path(root_dir)
        self.min_cost_threshold = min_cost_threshold
        self.max_cost_threshold = max_cost_threshold

        # Define file paths as attributes
        self.incar_path = self.root_dir / "INCAR"
        self.kpoints_path = self.root_dir / "KPOINTS"
        self.poscar_path = self.root_dir / "POSCAR"
        self.potcar_path = self.root_dir / "POTCAR"

    def parse_incar(self) -> Dict[str, Union[str, int, float]]:
        """
        Parse INCAR file into a dictionary.

        Args:
            incar_path: Path to INCAR file

        Returns:
            Dictionary of INCAR parameters
        """
        incar_dict = {}

        if not self.incar_path.exists():
            raise InputCheckError("INCAR does not exist!")

        with open(self.incar_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("!"):
                    continue

                if "=" in line:
                    # Split on first = only
                    key, value = line.split("=", 1)
                    key = key.strip().upper()
                    value = value.strip()

                    # Remove inline comments
                    value = re.split(r"[#!]", value)[0].strip()

                    # Try to convert to appropriate type
                    if value.upper() in ["TRUE", ".TRUE.", "T"]:
                        incar_dict[key] = True
                    elif value.upper() in ["FALSE", ".FALSE.", "F"]:
                        incar_dict[key] = False
                    else:
                        try:
                            # Try integer first
                            if "." not in value and "E" not in value.upper():
                                incar_dict[key] = int(value)
                            else:
                                incar_dict[key] = float(value)
                        except ValueError:
                            incar_dict[key] = value

        return incar_dict

    def parse_poscar_elements(self) -> Tuple[List[str], List[int]]:
        """
        Parse POSCAR file to get element types and counts.

        Args:
            poscar_path: Path to POSCAR file

        Returns:
            Tuple of (element_types, element_counts)
        """
        with open(self.poscar_path, "r") as f:
            lines = f.readlines()
        elems = (
            lines[5].strip().split()
        )  # Read the element names (this line is optional but we enforce it for good practice)
        counts = [int(value) for value in lines[6].strip().split()]
        return elems, counts

    def parse_poscar_structure(
        self,
    ) -> Tuple[List[List[float]], List[str], List[int], List[List[float]]]:
        """
        Parse the POSCAR to get the lattice vectors and atomic positions

        Args:
            poscar_path: Path to POSCAR file
        Returns:
            A tuple containing:
            - lattice_vectors: 3x3 matrix of lattice vectors in Angstroms
            - element_types: List of element symbols
            - element_counts: List of atom counts for each element type
            - atomic_positions: List of atomic positions (fractional or cartesian)
        """
        if not self.poscar_path.exists():
            raise InputCheckError(f"POSCAR file not found: {self.poscar_path}")

        with open(self.poscar_path, "r") as f:
            lines = f.readlines()

        # Strip whitespace from all lines
        lines = [line.strip() for line in lines]

        # Remove empty lines
        lines = [line for line in lines if line]

        if len(lines) < 8:
            raise ValueError(f"POSCAR file {self.poscar_path} is too short (minimum 8 lines required)")

        # Line 0: Comment line (ignored for now)

        # Line 1: Scaling factor
        try:
            scaling_factor = float(lines[1])
        except ValueError as e:
            raise ValueError(f"Invalid scaling factor in POSCAR: {lines[1]}") from e

        # Lines 2-4: Lattice vectors
        lattice_vectors = []
        for i in range(2, 5):
            try:
                vector = [float(x) for x in lines[i].split()]
                if len(vector) != 3:
                    raise ValueError(f"Lattice vector {i - 1} must have 3 components")
                # Apply scaling factor
                vector = [x * scaling_factor for x in vector]
                lattice_vectors.append(vector)
            except ValueError as e:
                raise ValueError(f"Invalid lattice vector on line {i + 1}: {lines[i]}") from e

        # Line 5: Element types (optional in older VASP versions)
        # Check if line 5 contains element symbols or numbers
        line5_tokens = lines[5].split()
        try:
            # Try to parse as numbers - if successful, this is the atom counts line
            [int(x) for x in line5_tokens]
            # This means no element symbols are provided
            element_types = []
            element_counts_line_idx = 5
        except ValueError:
            # This line contains element symbols
            element_types = line5_tokens
            element_counts_line_idx = 6

        # Element counts line
        if element_counts_line_idx >= len(lines):
            raise ValueError("POSCAR file is missing element counts line")

        try:
            element_counts = [int(x) for x in lines[element_counts_line_idx].split()]
        except ValueError as e:
            raise ValueError(
                f"Invalid element counts on line {element_counts_line_idx + 1}: " f"{lines[element_counts_line_idx]}"
            ) from e

        # If no element types were provided, create generic names
        if not element_types:
            element_types = [f"X{i + 1}" for i in range(len(element_counts))]

        # Validate that element_types and element_counts have same length
        if len(element_types) != len(element_counts):
            raise ValueError(
                f"Number of element types ({len(element_types)}) does not match "
                f"number of counts ({len(element_counts)})"
            )

        total_atoms = sum(element_counts)

        # Line after element counts: Selective dynamics (optional)
        coord_line_idx = element_counts_line_idx + 1

        if coord_line_idx < len(lines) and lines[coord_line_idx].lower().startswith("s"):
            coord_line_idx += 1

        # Coordinate type line (Direct/Cartesian)
        if coord_line_idx >= len(lines):
            raise ValueError("POSCAR file is missing coordinate type specification")

        coord_type = lines[coord_line_idx].lower()
        is_direct = coord_type.startswith("d") or coord_type.startswith("f")  # Direct or Fractional
        is_cartesian = coord_type.startswith("c")  # Cartesian

        if not (is_direct or is_cartesian):
            raise ValueError(
                f"Invalid coordinate type: {lines[coord_line_idx]}. "
                "Must start with 'D'irect, 'F'ractional, or 'C'artesian"
            )

        # Atomic positions
        atomic_positions = []
        start_pos_line = coord_line_idx + 1

        if start_pos_line + total_atoms > len(lines):
            available_lines = len(lines) - start_pos_line
            raise ValueError(
                f"POSCAR file has insufficient position lines. "
                f"Expected {total_atoms} positions, but only {available_lines} lines available"
            )

        for i in range(start_pos_line, start_pos_line + total_atoms):
            try:
                position_line = lines[i].split()
                if len(position_line) < 3:
                    raise ValueError(f"Position line {i + 1} has fewer than 3 coordinates")

                # Extract x, y, z coordinates
                position = [float(position_line[j]) for j in range(3)]

                # If cartesian coordinates, apply scaling factor
                if is_cartesian:
                    position = [x * scaling_factor for x in position]

                atomic_positions.append(position)

            except ValueError as e:
                raise ValueError(f"Invalid atomic position on line {i + 1}: {lines[i]}") from e

        return lattice_vectors, element_types, element_counts, atomic_positions

    def parse_potcar(self) -> List[Tuple[str, str, float]]:
        """
        Parse POTCAR file to get valence electron information in the correct order.

        Args:
            potcar_path: Path to POTCAR file

        Returns:
            List of valence electrons for each element in the same order as elements_order
        """
        valence_list = []

        with open(self.potcar_path, "r") as f:
            content = f.read()

        # Split POTCAR into individual sections - each starts with PAW_XXX or US_XXX
        sections = re.split(r"^( *PAW_\w+| *US_\w+)", content, flags=re.MULTILINE)
        potcar_sections = []
        for i, sec in enumerate(sections):
            sec = sec.strip()
            if not sec:
                continue
            if sec.startswith("PAW_") or sec.startswith("US_"):
                potcar_sections.append(sections[i + 1])

        # Extract valence from each section in order
        for section in potcar_sections:
            lines = section.split("\n")
            # Find the header line and extract element e.g. 'Li_sv XXXXXX'
            header_element = re.match(r"^[A-Z][a-z]?", lines[0].strip()).group(0)
            header_symbol = lines[0].strip().split()[0]
            # The second line contains the number of valence electrons
            valence = float(lines[1])
            valence_list.append((header_element, header_symbol, valence))
        return valence_list

    def parse_kpoints(self):
        """
        Parse KPOINTS file to get the kpoints or mesh size.

        Args:
            kpoints_path: Path to KPOINTS file

        Returns:
            Total number of k-points
        """
        with open(self.kpoints_path, "r") as f:
            lines = f.readlines()

        if len(lines) < 3:
            raise ValueError(f"KPOINTS file {self.kpoints_path} is too short (minimum 3 lines required)")

        # Check the generation mode
        mode = lines[2].strip().lower()

        if mode.startswith("g") or mode.startswith("m"):  # Gamma or Monkhorst-Pack
            kpoint_line = lines[3].strip().split()
            shifts = [int(value) for value in lines[4].strip().split()] if len(lines) > 4 else None
            kx, ky, kz = map(int, kpoint_line[:3])
            return (kx, ky, kz), mode[0].lower(), shifts
        elif mode.startswith("a"):  # Automatic
            # Default automatic k-point generation
            return -1
        # Parse explicit kpoints
        is_cartesian = mode.startswith("c") or mode.startswith("k")
        coords = []
        weights = []
        for line in lines[3:]:
            tokens = line.strip().split()
            coords.append([float(value) for value in tokens[:3]])
            weights.append(float(tokens[3]))
        return is_cartesian, coords, weights

    def get_ir_kpoints_and_weights(
        self,
        is_time_reversal: bool = True,
        symprec: float = 1e-5,
        symmetry_reduce: bool = True,
    ):
        """
        Get the irreducible k-points and weights from a POSCAR file.

        """
        lattice_vectors, element_types, element_counts, atomic_positions = self.parse_poscar_structure()

        # Symbols
        chemical_symbols = []
        for name, count in zip(element_types, element_counts):
            chemical_symbols.extend([name] * count)

        kpoints_info = self.parse_kpoints()
        if not isinstance(kpoints_info[0], tuple):
            if kpoints_info[0] is False:
                return kpoints_info[1], kpoints_info[2]
            raise NotImplementedError("Only direct coordinate kpoints are supported for now!")
        # If kpoints_info is -1, it means automatic generation
        if kpoints_info == -1:
            return None
        mesh = kpoints_info[0]
        mode = kpoints_info[1]
        shift = kpoints_info[2]
        if mode == "g":
            assert shift[0] == 0 or shift is None
            shift = None
        elif mode == "m":
            assert shift[0] == 0 or shift is None
            shift = [1, 1, 1]
        ir_kpoints_weights = get_ir_kpoints_and_weights(
            lattice_vectors,
            atomic_positions,
            [ATOMIC_NUMBERS[value] for value in chemical_symbols],
            mesh,
            is_time_reversal=is_time_reversal,
            symprec=symprec,
            is_shift=shift,
            symmetry_reduce=symmetry_reduce,
        )
        return ir_kpoints_weights

    def estimate_nkpts(self, is_time_reversal: bool = True, symprec: float = 1e-5, symmetry_reduce: bool = True) -> int:
        """
        Estimate the number of k-points based on POSCAR and mesh.

        Args:
            poscar_path: Path to POSCAR file
            mesh: K-point mesh size or number of k-points
            is_time_reversal: Whether to consider time-reversal symmetry
            symprec: Symmetry precision for k-point generation
            symmetry_reduce: Whether to reduce k-points using symmetry

        Returns:
            Estimated number of k-points
        """
        output = self.get_ir_kpoints_and_weights(is_time_reversal, symprec, symmetry_reduce)
        if output is None:
            return None
        return len(output[0])

    def estimate_bands(
        self, elements: List[str], counts: List[int], valence_list: List[Tuple[str, str, int]]
    ) -> Tuple[int, int]:
        """
        Estimate number of electrons and bands.

        Args:
            elements: List of element symbols
            counts: List of atom counts for each element
            valence_list: List of valence electrons for each element in the same order

        Returns:
            Tuple of (n_electrons, n_bands)
        """
        n_electrons = 0

        for i, (element, count) in enumerate(zip(elements, counts)):
            potcar_elem, potcar_symbol, valence = valence_list[i]
            if potcar_elem != element:
                raise ValueError(
                    f"Element mismatch: {element} in POSCAR does not match {potcar_elem}->{potcar_symbol} in POTCAR"
                )
            n_electrons += count * valence

        # Estimate number of bands (typically NBANDS = 1.3 * n_electrons/2 for insulators/semiconductors)
        # For metals, might need more bands
        n_bands = max(int(1.3 * n_electrons / 2), n_electrons // 2 + 10)

        return n_electrons, n_bands

    def check_calculation(self) -> CalculationInfo:
        """
        Check a single VASP calculation directory.

        Returns:
            CalculationInfo object with analysis results
        """
        # Parse input files
        incar_dict = self.parse_incar()
        elements, counts = self.parse_poscar_elements()
        valence_list = self.parse_potcar()
        n_kpoints = self.estimate_nkpts()

        if not elements:
            return CalculationInfo(
                path=self.root_dir,
                n_atoms=0,
                n_kpoints=n_kpoints if n_kpoints else 0,
                n_bands=0,
                n_electrons=0,
                issues=["Could not read POSCAR file"],
            )

        n_atoms = sum(counts)
        n_electrons, n_bands = self.estimate_bands(elements, counts, valence_list)

        # Extract parallelization settings
        ncore = incar_dict.get("NCORE")
        kpar = incar_dict.get("KPAR")
        npar = incar_dict.get("NPAR")

        calc_info = CalculationInfo(
            path=self.root_dir,
            n_atoms=n_atoms,
            n_kpoints=n_kpoints,
            n_bands=n_bands,
            n_electrons=n_electrons,
            ncore=ncore,
            kpar=kpar,
            npar=npar,
        )

        # Check for issues
        self._check_parallelization_issues(calc_info)
        self._check_computational_efficiency(calc_info)

        return calc_info

    def _check_parallelization_issues(self, calc_info: CalculationInfo):
        """Check for parallelization-related issues."""

        # Check if parallelization tags are present for large calculations
        if calc_info.computational_cost > self.min_cost_threshold:
            if calc_info.ncore is None and calc_info.kpar is None and calc_info.npar is None:
                calc_info.issues.append("Large calculation without parallelization tags (NCORE/KPAR/NPAR)")

        # Check NCORE value
        if calc_info.ncore is not None:
            if calc_info.ncore > calc_info.n_bands:
                calc_info.issues.append(
                    f"NCORE ({calc_info.ncore}) is larger than number of bands ({calc_info.n_bands})"
                )
            elif calc_info.ncore < 1:
                calc_info.issues.append(f"Invalid NCORE value: {calc_info.ncore}")

        # Check KPAR value
        if calc_info.kpar is not None:
            if calc_info.kpar > calc_info.n_kpoints:
                calc_info.issues.append(
                    f"KPAR ({calc_info.kpar}) is larger than number of k-points ({calc_info.n_kpoints})"
                )
            elif calc_info.kpar < 1:
                calc_info.issues.append(f"Invalid KPAR value: {calc_info.kpar}")

        # Check for conflicting parallelization settings
        if calc_info.ncore is not None and calc_info.npar is not None:
            calc_info.issues.append("Both NCORE and NPAR are set (NCORE is preferred)")

    def _check_computational_efficiency(self, calc_info: CalculationInfo):
        """Check for computational efficiency issues."""

        # Warn about very large calculations
        if calc_info.computational_cost > self.max_cost_threshold:
            calc_info.issues.append(
                f"Very large calculation (cost: {calc_info.computational_cost:.2e}). "
                "Consider reducing k-point density or using hybrid functionals carefully."
            )

        # Check for very small calculations that might not parallelize well
        if calc_info.computational_cost < self.min_cost_threshold / 10:
            calc_info.issues.append("Very small calculation - parallelization may not be beneficial")

        # Check k-point to atom ratio
        if calc_info.n_atoms > 0:
            kpoint_per_atom = calc_info.n_kpoints / calc_info.n_atoms
            if kpoint_per_atom > 10:
                calc_info.issues.append(
                    f"High k-point density ({kpoint_per_atom:.1f} k-points/atom). " "Consider reducing k-point mesh."
                )


class VaspScanner:
    """A scanner for VASP calculations in a directory."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def scan_directory(self, recursive: bool = True) -> List[CalculationInfo]:
        """
        Scan directory for VASP calculations and check them.

        Args:
            root_dir: Root directory to scan
            recursive: Whether to scan subdirectories recursively

        Returns:
            List of CalculationInfo objects for found calculations
        """
        calculations = []
        root_dir = self.directory.resolve()

        def is_vasp_calculation(directory: Path) -> bool:
            """Check if directory contains VASP input files."""
            required_files = ["INCAR", "POSCAR"]
            return all((directory / f).exists() for f in required_files)

        if recursive:
            for dirpath in root_dir.rglob("*"):
                if dirpath.is_dir() and is_vasp_calculation(dirpath):
                    checker = VASPInputChecker(dirpath)
                    calc_info = checker.check_calculation()
                    calculations.append(calc_info)
        else:
            for item in root_dir.iterdir():
                if item.is_dir() and is_vasp_calculation(item):
                    checker = VASPInputChecker(item)
                    calc_info = checker.check_calculation()
                    calculations.append(calc_info)

        # Also check the root directory itself
        if is_vasp_calculation(root_dir):
            checker = VASPInputChecker(root_dir)
            calc_info = checker.check_calculation()
            calculations.append(calc_info)

        return calculations

    def generate_report(self, calculations: List[CalculationInfo] = None, output_file: Optional[Path] = None) -> str:
        """
        Generate a report of the calculation analysis.

        Args:
            calculations: List of CalculationInfo objects
            output_file: Optional file to write report to

        Returns:
            Report as a string
        """
        if calculations is None:
            calculations = self.scan_directory()
        report_lines = []
        report_lines.append("VASP Input File Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Total calculations checked: {len(calculations)}")
        report_lines.append("")

        # Summary statistics
        problematic = [calc for calc in calculations if calc.issues]
        report_lines.append(f"Calculations with issues: {len(problematic)}")
        report_lines.append("")

        # Detailed analysis
        for i, calc in enumerate(calculations, 1):
            report_lines.append(f"Calculation {i}: {calc.path}")
            report_lines.append(f"  Atoms: {calc.n_atoms}")
            report_lines.append(f"  K-points: {calc.n_kpoints}")
            report_lines.append(f"  Bands: {calc.n_bands}")
            report_lines.append(f"  Electrons: {calc.n_electrons}")
            report_lines.append(f"  Computational cost: {calc.computational_cost:.2e}")

            if calc.ncore is not None:
                report_lines.append(f"  NCORE: {calc.ncore}")
            if calc.kpar is not None:
                report_lines.append(f"  KPAR: {calc.kpar}")
            if calc.npar is not None:
                report_lines.append(f"  NPAR: {calc.npar}")

            if calc.issues:
                report_lines.append("  Issues:")
                for issue in calc.issues:
                    report_lines.append(f"    - {issue}")
            else:
                report_lines.append("  No issues found")

            report_lines.append("")

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report)

        return report


# CLI interface
@click.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--recursive/--no-recursive", default=True, help="Recursively scan subdirectories")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output report to file")
@click.option(
    "--min-cost", type=float, default=1000.0, help="Minimum computational cost threshold for parallelization warnings"
)
@click.option(
    "--max-cost", type=float, default=1e6, help="Maximum computational cost threshold for efficiency warnings"
)
def check_vasp_inputs(directory, recursive, output, min_cost, max_cost):
    """
    Check VASP input files for optimal parallelization and efficiency.

    DIRECTORY: Path to directory containing VASP calculations to check.
    """
    checker = VASPInputChecker(min_cost_threshold=min_cost, max_cost_threshold=max_cost)

    click.echo(f"Scanning {'recursively' if recursive else 'non-recursively'}: {directory}")
    calculations = checker.scan_directory(directory, recursive=recursive)

    if not calculations:
        click.echo("No VASP calculations found.")
        return

    report = checker.generate_report(calculations, output_file=output)

    if output:
        click.echo(f"Report written to: {output}")
    else:
        click.echo(report)


if __name__ == "__main__":
    check_vasp_inputs()

"""
Tests for VASPInputChecker and VaspScanner classes.

This module contains comprehensive tests for the VASP input file checking functionality
using pytest fixtures with example VASP calculation directories.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch

from matchest.cli.vaspcheck import VASPInputChecker, VaspScanner, CalculationInfo, InputCheckError


# Test fixtures
@pytest.fixture
def test_data_dir():
    """Path to test fixture data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def basic_calc_dir(test_data_dir):
    """Path to basic calculation test data."""
    return test_data_dir / "basic_calculation"


@pytest.fixture
def large_calc_dir(test_data_dir):
    """Path to large calculation test data."""
    return test_data_dir / "large_calculation"


@pytest.fixture
def small_calc_dir(test_data_dir):
    """Path to small calculation test data."""
    return test_data_dir / "small_calculation"


@pytest.fixture
def parallelization_issues_dir(test_data_dir):
    """Path to calculation with parallelization issues."""
    return test_data_dir / "parallelization_issues"


@pytest.fixture
def multi_calc_dir(test_data_dir):
    """Path to directory containing multiple calculations."""
    return test_data_dir / "multi_calc_directory"


@pytest.fixture
def temp_calc_dir():
    """Create a temporary calculation directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def basic_checker(basic_calc_dir):
    """VASPInputChecker instance for basic calculation."""
    return VASPInputChecker(basic_calc_dir)


@pytest.fixture
def large_checker(large_calc_dir):
    """VASPInputChecker instance for large calculation."""
    return VASPInputChecker(large_calc_dir)


@pytest.fixture
def small_checker(small_calc_dir):
    """VASPInputChecker instance for small calculation."""
    return VASPInputChecker(small_calc_dir)


@pytest.fixture
def issues_checker(parallelization_issues_dir):
    """VASPInputChecker instance for calculation with issues."""
    return VASPInputChecker(parallelization_issues_dir)


class TestVASPInputChecker:
    """Test cases for VASPInputChecker class."""

    def test_init(self, basic_calc_dir):
        """Test VASPInputChecker initialization."""
        checker = VASPInputChecker(basic_calc_dir)

        assert checker.root_dir == Path(basic_calc_dir)
        assert checker.min_cost_threshold == 1000.0
        assert checker.max_cost_threshold == 1e6
        assert checker.incar_path == basic_calc_dir / "INCAR"
        assert checker.kpoints_path == basic_calc_dir / "KPOINTS"
        assert checker.poscar_path == basic_calc_dir / "POSCAR"
        assert checker.potcar_path == basic_calc_dir / "POTCAR"

    def test_init_custom_thresholds(self, basic_calc_dir):
        """Test VASPInputChecker initialization with custom thresholds."""
        checker = VASPInputChecker(basic_calc_dir, min_cost_threshold=500.0, max_cost_threshold=5e5)

        assert checker.min_cost_threshold == 500.0
        assert checker.max_cost_threshold == 5e5

    def test_parse_incar_basic(self, basic_checker):
        """Test parsing basic INCAR file."""
        incar_dict = basic_checker.parse_incar()

        assert isinstance(incar_dict, dict)
        assert incar_dict["SYSTEM"] == "Si bulk calculation"
        assert incar_dict["ISTART"] == 0
        assert incar_dict["ENCUT"] == 520
        assert incar_dict["EDIFF"] == 1e-6
        assert incar_dict["LREAL"] is False

    def test_parse_incar_with_parallelization(self, large_checker):
        """Test parsing INCAR file with parallelization settings."""
        incar_dict = large_checker.parse_incar()

        assert incar_dict["NCORE"] == 4
        assert incar_dict["KPAR"] == 2

    def test_parse_incar_missing_file(self, temp_calc_dir):
        """Test parsing INCAR when file is missing."""
        checker = VASPInputChecker(temp_calc_dir)

        with pytest.raises(InputCheckError, match="INCAR does not exist"):
            checker.parse_incar()

    def test_parse_poscar_elements_basic(self, basic_checker):
        """Test parsing POSCAR elements."""
        elements, counts = basic_checker.parse_poscar_elements()

        assert elements == ["Si"]
        assert counts == [2]

    def test_parse_poscar_elements_multi(self, large_checker):
        """Test parsing POSCAR with multiple element types."""
        elements, counts = large_checker.parse_poscar_elements()

        assert elements == ["Si"]
        assert counts == [16]

    def test_parse_poscar_structure_basic(self, basic_checker):
        """Test parsing POSCAR structure."""
        lattice_vectors, element_types, element_counts, atomic_positions = basic_checker.parse_poscar_structure()

        assert len(lattice_vectors) == 3
        assert lattice_vectors[0] == [5.43, 0.0, 0.0]
        assert element_types == ["Si"]
        assert element_counts == [2]
        assert len(atomic_positions) == 2
        assert atomic_positions[0] == [0.0, 0.0, 0.0]
        assert atomic_positions[1] == [0.25, 0.25, 0.25]

    def test_parse_poscar_structure_missing_file(self, temp_calc_dir):
        """Test parsing POSCAR structure when file is missing."""
        checker = VASPInputChecker(temp_calc_dir)

        with pytest.raises(InputCheckError, match="POSCAR file not found"):
            checker.parse_poscar_structure()

    def test_parse_potcar_basic(self, basic_checker):
        """Test parsing POTCAR file."""
        valence_list = basic_checker.parse_potcar()

        assert len(valence_list) == 1
        element, symbol, valence = valence_list[0]
        assert element == "Si"
        assert symbol.startswith("Si")
        assert valence == 4.0

    def test_parse_kpoints_basic(self, basic_checker):
        """Test parsing KPOINTS file."""
        kpoints_info = basic_checker.parse_kpoints()

        mesh, mode, shifts = kpoints_info
        assert mesh == (4, 4, 4)
        assert mode == "m"  # Monkhorst-Pack
        assert shifts == [0, 0, 0]

    def test_parse_kpoints_large(self, large_checker):
        """Test parsing KPOINTS file for large calculation."""
        kpoints_info = large_checker.parse_kpoints()

        mesh, mode, shifts = kpoints_info
        assert mesh == (8, 8, 8)
        assert mode == "m"
        assert shifts == [0, 0, 0]

    def test_estimate_nkpts(self, large_checker):
        """Test k-point estimation."""
        # Mock the k-points function to return a known result
        n_kpts = large_checker.estimate_nkpts()
        assert n_kpts == 20

    def test_estimate_bands_basic(self, basic_checker):
        """Test band estimation."""
        elements = ["Si"]
        counts = [2]
        valence_list = [["Si", "Si", 4]]  # Silicon has 4 valence electrons

        n_electrons, n_bands = basic_checker.estimate_bands(elements, counts, valence_list)

        assert n_electrons == 8  # 2 atoms * 4 valence electrons
        assert n_bands >= 4  # At least half the electrons
        assert n_bands >= 14  # Should be at least n_electrons//2 + 10

    def test_estimate_bands_mismatch_error(self, basic_checker):
        """Test band estimation with element mismatch."""
        elements = ["Al"]  # Different from POTCAR
        counts = [2]
        valence_list = [("Si", "Si", 4.0)]  # POTCAR is for Si

        with pytest.raises(ValueError, match="Element mismatch"):
            basic_checker.estimate_bands(elements, counts, valence_list)

    @patch.object(VASPInputChecker, "estimate_nkpts")
    def test_check_calculation_basic(self, mock_estimate_nkpts, basic_checker):
        """Test checking a basic calculation."""
        mock_estimate_nkpts.return_value = 10

        calc_info = basic_checker.check_calculation()

        assert isinstance(calc_info, CalculationInfo)
        assert calc_info.path == basic_checker.root_dir
        assert calc_info.n_atoms == 2
        assert calc_info.n_kpoints == 10
        assert calc_info.n_bands > 0
        assert calc_info.n_electrons == 8
        assert calc_info.ncore is None
        assert calc_info.kpar is None
        assert calc_info.npar is None

    @patch.object(VASPInputChecker, "estimate_nkpts")
    def test_check_calculation_with_parallelization(self, mock_estimate_nkpts, large_checker):
        """Test checking calculation with parallelization settings."""
        mock_estimate_nkpts.return_value = 100

        calc_info = large_checker.check_calculation()

        assert calc_info.ncore == 4
        assert calc_info.kpar == 2
        assert calc_info.n_atoms == 16

    @patch.object(VASPInputChecker, "estimate_nkpts")
    def test_check_parallelization_issues(self, mock_estimate_nkpts, issues_checker):
        """Test detection of parallelization issues."""
        mock_estimate_nkpts.return_value = 5

        calc_info = issues_checker.check_calculation()

        # Should detect issues with NCORE and KPAR being too large
        issue_messages = " ".join(calc_info.issues)
        assert "NCORE" in issue_messages and "larger than number of bands" in issue_messages
        assert "KPAR" in issue_messages and "larger than number of k-points" in issue_messages
        assert "Both NCORE and NPAR are set" in issue_messages

    @patch.object(VASPInputChecker, "estimate_nkpts")
    def test_check_computational_efficiency_large(self, mock_estimate_nkpts, basic_checker):
        """Test detection of large calculation efficiency issues."""
        # Set very high computational cost
        mock_estimate_nkpts.return_value = 1000
        basic_checker.max_cost_threshold = 1000  # Lower threshold for testing

        calc_info = basic_checker.check_calculation()

        # Should detect high computational cost
        issue_messages = " ".join(calc_info.issues)
        assert "Very large calculation" in issue_messages

    @patch.object(VASPInputChecker, "estimate_nkpts")
    def test_check_computational_efficiency_small(self, mock_estimate_nkpts, basic_checker):
        """Test detection of very small calculation."""
        mock_estimate_nkpts.return_value = 1
        basic_checker.min_cost_threshold = 10000  # High threshold for testing

        calc_info = basic_checker.check_calculation()

        # Should detect very small calculation
        issue_messages = " ".join(calc_info.issues)
        assert "Very small calculation" in issue_messages

    @patch.object(VASPInputChecker, "estimate_nkpts")
    def test_check_high_kpoint_density(self, mock_estimate_nkpts, basic_checker):
        """Test detection of high k-point density."""
        mock_estimate_nkpts.return_value = 100  # High k-point count for 2 atoms

        calc_info = basic_checker.check_calculation()

        # Should detect high k-point density (100 k-points / 2 atoms = 50 > 10)
        issue_messages = " ".join(calc_info.issues)
        assert "High k-point density" in issue_messages

    def test_missing_poscar_elements(self, temp_calc_dir):
        """Test handling of missing POSCAR elements."""
        checker = VASPInputChecker(temp_calc_dir)

        # Create minimal files
        (temp_calc_dir / "INCAR").write_text("SYSTEM = Test")
        (temp_calc_dir / "POTCAR").write_text("PAW_PBE Si 05Jan2001\n4.0\ntest data")
        (temp_calc_dir / "KPOINTS").write_text("Test\n0\nM\n2 2 2\n0 0 0")

        # Mock estimate_nkpts to avoid complex setup
        with patch.object(checker, "estimate_nkpts", return_value=8):
            # Mock parse_poscar_elements to return empty
            with patch.object(checker, "parse_poscar_elements", return_value=([], [])):
                calc_info = checker.check_calculation()

        assert calc_info.n_atoms == 0
        assert "Could not read POSCAR file" in calc_info.issues

    @pytest.mark.parametrize("fixture_name", ["basic_calc_dir", "large_calc_dir", "small_calc_dir"])
    def test_vasp_input_checker(self, request, fixture_name):
        calc_dir = request.getfixturevalue(fixture_name)
        checker = VASPInputChecker(calc_dir)
        info = checker.check_calculation()
        assert info.n_atoms > 0
        assert info.n_kpoints > 0
        assert info.n_bands > 0
        assert info.n_electrons > 0
        assert isinstance(info.issues, list)
        # Should not raise for valid input
        assert info.path == calc_dir


class TestVaspScanner:
    """Test cases for VaspScanner class."""

    def test_init(self, multi_calc_dir):
        """Test VaspScanner initialization."""
        scanner = VaspScanner(multi_calc_dir)
        assert scanner.directory == Path(multi_calc_dir)

    @patch.object(VASPInputChecker, "estimate_nkpts")
    def test_scan_directory_recursive(self, mock_estimate_nkpts, multi_calc_dir):
        """Test recursive directory scanning."""
        mock_estimate_nkpts.return_value = 10

        scanner = VaspScanner(multi_calc_dir)
        calculations = scanner.scan_directory(recursive=True)

        # Should find 2 calculations (calc1 and calc2)
        assert len(calculations) == 2

        # Check that both calculations were found
        calc_paths = {calc.path.name for calc in calculations}
        assert "calc1" in calc_paths
        assert "calc2" in calc_paths

    @patch.object(VASPInputChecker, "estimate_nkpts")
    def test_scan_directory_non_recursive(self, mock_estimate_nkpts, multi_calc_dir):
        """Test non-recursive directory scanning."""
        mock_estimate_nkpts.return_value = 10

        scanner = VaspScanner(multi_calc_dir)
        calculations = scanner.scan_directory(recursive=False)

        # Should find 2 calculations at the immediate subdirectory level
        assert len(calculations) == 2

    @patch.object(VASPInputChecker, "estimate_nkpts")
    def test_scan_single_calculation(self, mock_estimate_nkpts, basic_calc_dir):
        """Test scanning a single calculation directory."""
        mock_estimate_nkpts.return_value = 10

        scanner = VaspScanner(basic_calc_dir)
        calculations = scanner.scan_directory()

        # Should find 1 calculation (the root directory itself)
        assert len(calculations) == 1
        assert calculations[0].path == Path(basic_calc_dir)

    def test_scan_empty_directory(self, temp_calc_dir):
        """Test scanning an empty directory."""
        scanner = VaspScanner(temp_calc_dir)
        calculations = scanner.scan_directory()

        # Should find no calculations
        assert len(calculations) == 0

    @patch.object(VASPInputChecker, "estimate_nkpts")
    def test_generate_report(self, mock_estimate_nkpts, multi_calc_dir):
        """Test report generation."""
        mock_estimate_nkpts.return_value = 10

        scanner = VaspScanner(multi_calc_dir)
        calculations = scanner.scan_directory()
        report = scanner.generate_report(calculations)

        assert "VASP Input File Analysis Report" in report
        assert "Total calculations checked: 2" in report
        assert "Calculation 1:" in report
        assert "Calculation 2:" in report

    @patch.object(VASPInputChecker, "estimate_nkpts")
    def test_generate_report_with_issues(self, mock_estimate_nkpts, parallelization_issues_dir):
        """Test report generation with problematic calculations."""
        mock_estimate_nkpts.return_value = 5

        scanner = VaspScanner(parallelization_issues_dir)
        calculations = scanner.scan_directory()
        report = scanner.generate_report(calculations)

        assert "Calculations with issues: 1" in report
        assert "Issues:" in report

    @patch.object(VASPInputChecker, "estimate_nkpts")
    def test_generate_report_to_file(self, mock_estimate_nkpts, multi_calc_dir, temp_calc_dir):
        """Test report generation to file."""
        mock_estimate_nkpts.return_value = 10

        scanner = VaspScanner(multi_calc_dir)
        calculations = scanner.scan_directory()

        output_file = temp_calc_dir / "report.txt"
        report = scanner.generate_report(calculations, output_file=output_file)

        assert output_file.exists()
        file_content = output_file.read_text()
        assert file_content == report
        assert "VASP Input File Analysis Report" in file_content


class TestCalculationInfo:
    """Test cases for CalculationInfo dataclass."""

    def test_calculation_info_init(self):
        """Test CalculationInfo initialization."""
        calc_info = CalculationInfo(path=Path("/test"), n_atoms=4, n_kpoints=10, n_bands=20, n_electrons=16)

        assert calc_info.path == Path("/test")
        assert calc_info.n_atoms == 4
        assert calc_info.n_kpoints == 10
        assert calc_info.n_bands == 20
        assert calc_info.n_electrons == 16
        assert calc_info.ncore is None
        assert calc_info.kpar is None
        assert calc_info.npar is None
        assert calc_info.issues == []

    def test_calculation_info_computational_cost(self):
        """Test computational cost calculation."""
        calc_info = CalculationInfo(path=Path("/test"), n_atoms=4, n_kpoints=10, n_bands=20, n_electrons=16)

        expected_cost = 10 * (20**2)  # n_kpoints * n_bands^2
        assert calc_info.computational_cost == expected_cost

    def test_calculation_info_with_parallelization(self):
        """Test CalculationInfo with parallelization settings."""
        calc_info = CalculationInfo(
            path=Path("/test"), n_atoms=4, n_kpoints=10, n_bands=20, n_electrons=16, ncore=4, kpar=2, npar=1
        )

        assert calc_info.ncore == 4
        assert calc_info.kpar == 2
        assert calc_info.npar == 1

    def test_calculation_info_with_issues(self):
        """Test CalculationInfo with custom issues."""
        issues = ["Issue 1", "Issue 2"]
        calc_info = CalculationInfo(
            path=Path("/test"), n_atoms=4, n_kpoints=10, n_bands=20, n_electrons=16, issues=issues
        )

        assert calc_info.issues == issues


class TestIntegration:
    """Integration tests combining multiple components."""

    @patch.object(VASPInputChecker, "estimate_nkpts")
    def test_end_to_end_workflow(self, mock_estimate_nkpts, multi_calc_dir, temp_calc_dir):
        """Test complete workflow from scanning to report generation."""
        mock_estimate_nkpts.return_value = 15

        # Create scanner and scan directory
        scanner = VaspScanner(multi_calc_dir)
        calculations = scanner.scan_directory(recursive=True)

        # Verify calculations were found
        assert len(calculations) == 2

        # Generate report to file
        output_file = temp_calc_dir / "integration_report.txt"
        report = scanner.generate_report(calculations, output_file=output_file)

        # Verify report content
        assert "Total calculations checked: 2" in report
        assert output_file.exists()

        # Verify individual calculation details
        for calc in calculations:
            assert calc.n_atoms > 0
            assert calc.n_kpoints == 15
            assert calc.n_bands > 0
            assert calc.n_electrons > 0

    @patch("matchest.cli.vaspcheck.get_ir_kpoints_and_weights")
    def test_real_kpoints_calculation(self, mock_get_ir_kpoints, basic_checker):
        """Test with actual k-point calculation (mocked external dependency)."""
        # Mock the external k-points library
        mock_get_ir_kpoints.return_value = [
            (
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.25, 0.25, 0.0],
                [0.0, 0.0, 0.25],
                [0.25, 0.0, 0.25],
                [0.0, 0.25, 0.25],
                [0.25, 0.25, 0.25],
            ),
            [0.125] * 8,
        ]

        n_kpts = basic_checker.estimate_nkpts()
        assert n_kpts == 8

        calc_info = basic_checker.check_calculation()
        assert calc_info.n_kpoints == 8

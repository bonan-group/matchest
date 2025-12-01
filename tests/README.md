# VASP Input Checker Tests

This directory contains comprehensive tests for the `VASPInputChecker` and `VaspScanner` classes.

## Test Structure

### Test Files

- `test_vaspcheck.py` - Main unit tests for VASPInputChecker class
- `conftest.py` - Shared pytest configuration and fixtures
- `run_tests.py` - Test runner script with various options

### Test Fixtures

The `fixtures/` directory contains example VASP calculation directories:

- `basic_calculation/` - Standard Si calculation with basic settings
- `large_calculation/` - Large supercell calculation with parallelization settings
- `small_calculation/` - Small calculation for testing efficiency warnings
- `parallelization_issues/` - Calculation with problematic parallelization settings
- `multi_calc_directory/` - Directory containing multiple calculations for scanner tests

Each fixture directory contains the four required VASP input files:
- `INCAR` - Input parameters
- `POSCAR` - Atomic structure
- `KPOINTS` - k-point mesh
- `POTCAR` - Pseudopotential information

## Running Tests

### Using pytest directly

```bash
# Run all tests
pytest tests/

# Run only unit tests
pytest tests/test_vaspcheck.py::TestVASPInputChecker

# Run only integration tests  
pytest tests/test_vaspcheck_integration.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=matchest.cli.vaspcheck --cov-report=html
```

### Using the test runner script

```bash
# Run all tests
python tests/run_tests.py

# Run only unit tests
python tests/run_tests.py --type unit

# Run only integration tests
python tests/run_tests.py --type integration

# Run with verbose output and coverage
python tests/run_tests.py --verbose --coverage
```

## Test Categories

### Unit Tests (`TestVASPInputChecker`)

- Initialization and attribute setting
- INCAR file parsing (basic and with parallelization)
- POSCAR structure and element parsing
- POTCAR valence electron extraction
- KPOINTS mesh parsing
- Band and electron count estimation
- Individual calculation checking
- Issue detection (parallelization problems, efficiency issues)

### Scanner Tests (`TestVaspScanner`)

- Directory scanning (recursive and non-recursive)
- Multiple calculation detection
- Report generation
- File output handling

### Integration Tests

- Complete workflow testing (scan → check → report)
- Performance testing with many calculations
- Error handling for corrupted/incomplete files
- Edge cases and boundary conditions
- Data validation and consistency checks

## Mocking Strategy

The tests use `unittest.mock.patch` to mock external dependencies:

- `estimate_nkpts()` method is mocked to avoid complex k-point calculations
- `get_ir_kpoints_and_weights()` function is mocked to avoid dependency on external k-point libraries

This allows tests to focus on the logic of the VASP checker without requiring:
- Actual k-point symmetry calculations
- Large computational resources
- External scientific libraries

## Test Fixtures Design

Each test fixture represents a different scenario:

1. **Basic Calculation**: Standard case with no issues
2. **Large Calculation**: Tests parallelization settings and large system warnings  
3. **Small Calculation**: Tests efficiency warnings for small systems
4. **Parallelization Issues**: Tests detection of problematic settings
5. **Multi-Calculation**: Tests scanner functionality with multiple directories

## Adding New Tests

To add new tests:

1. Create new test methods in the appropriate test class
2. Use existing fixtures or create new ones in `fixtures/`
3. Mock external dependencies as needed
4. Follow the naming convention `test_description_of_what_is_tested`
5. Use descriptive docstrings

Example:

```python
def test_new_functionality(self, basic_checker):
    """Test description of the new functionality."""
    # Test implementation
    result = basic_checker.some_method()
    assert result == expected_value
```

## Coverage

The tests aim for high coverage of:

- All public methods in `VASPInputChecker` and `VaspScanner`
- Error handling paths
- Edge cases and boundary conditions
- Integration between components

Run with coverage to see current coverage levels:

```bash
pytest tests/ --cov=matchest.cli.vaspcheck --cov-report=html
```

This will generate an HTML coverage report in `htmlcov/index.html`.

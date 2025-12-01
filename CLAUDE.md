# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**matchest** is a Python package providing scripts and tools for computational materials science, with three main components:
1. **CLI tools** for crystal structure analysis and DFT code utilities (VASP, CASTEP)
2. **AiiDA integration** for automated computational workflows
3. **Pymatgen utilities** for materials structure manipulation

Main dependencies: `spglib`, `ase`, `pymatgen`

Optional dependencies:
- **AiiDA features** (`[aiida]`): `aiida-core`, `aiida-vasp`, `aiida-workgraph`, `numpy` - Required for workflow automation and workgraph functionality

## Development Commands

### Environment Setup
The project uses a virtual environment located at `.venv/`:

```bash
# Activate virtual environment
source .venv/bin/activate
```

### Package Management
This project uses **hatch** as the build system and **uv** for dependency management.

**uv is the preferred tool** for managing and updating dependencies:

```bash
# Install dependencies with uv
uv pip install -e .

# Install with AiiDA support (for workflows and workgraphs)
uv pip install -e ".[aiida]"

# Update dependencies
uv pip install --upgrade <package>

# Sync dependencies from pyproject.toml
uv pip sync

# Install in development mode (alternative)
pip install -e .

# Install with AiiDA support (alternative)
pip install -e ".[aiida]"

# Build the package
hatch build
```

### Code Quality
```bash
# Format code (runs ruff format)
hatch fmt -f

# Lint code (runs ruff check)
hatch fmt -l

# Both format and lint
hatch fmt

# Type checking
hatch run types:check
```

### Pre-commit Hooks
The project uses pre-commit hooks for automatic formatting and linting:
```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_specific.py

# Run with coverage
pytest --cov=matchest --cov-report=html
```

## Architecture

### CLI Structure (`src/matchest/cli/`)
All CLI commands are registered via Click in [main.py](src/matchest/cli/main.py) under the `mc` command group (accessible as `matc` command):

- **Structure tools** (`structure.py`): Primitive cell conversion, spacegroup analysis
- **K-points** (`kpoints.py`): K-point grid calculation for DFT codes
- **Geometry convergence** (`geomconv.py`): VASP convergence analysis
- **CASTEP utilities** ([casteputils.py](src/matchest/casteputils.py), [dotcastep.py](src/matchest/dotcastep.py)): SCF analysis, timing extraction

Each CLI command follows the pattern:
```python
@mc.command()
@click.option(...)
def command_name(...):
    from .module import function
    function(...)
```

### AiiDA Integration (`src/matchest/aiida_utils/`)
Provides higher-level workflow abstractions for AiiDA-based calculations.

**Note**: AiiDA features require installation with the `[aiida]` extra: `pip install matchest[aiida]`

- **Workflows** (`workflows/`): Complete computational protocols
  - [simple_vac.py](src/matchest/aiida_utils/workflows/simple_vac.py): Vacancy formation energy calculations using `VaspRelaxWorkChain`
  - [elastic.py](src/matchest/aiida_utils/workflows/elastic.py): Elastic tensor calculations via strain-stress relationships
  - Pattern: Extend `WorkChain`, expose inputs from underlying workchains, implement `setup()` → `run_*()` → `inspect_*()` → `finalize()` outline
  - Entry point: `matchest.vasp.elastic` for VaspElasticWorkChain

- **Workgraphs** (`workgraphs/`): AiiDA WorkGraph-based workflows (requires `aiida-workgraph`)
  - [phono3py.py](src/matchest/aiida_utils/workgraphs/phono3py.py): Phono3py calculations for 2nd and 3rd order force constants

- **Process utilities** (`process/`): Structure manipulation for workflows
  - [transformation.py](src/matchest/aiida_utils/process/transformation.py): Supercell creation, vacancy generation
  - [battery.py](src/matchest/aiida_utils/process/battery.py): Battery materials specific tools

- **Integration modules**:
  - [pmg.py](src/matchest/aiida_utils/pmg.py): Bridge between AiiDA and pymatgen (structure conversion, Materials Project integration)
  - [vasp.py](src/matchest/aiida_utils/vasp.py): VASP-specific utilities (functional parsing, U value handling)
  - [maceutils.py](src/matchest/aiida_utils/maceutils.py): MACE ML potential integration

### Key Patterns

**Structure handling**: The codebase uses ASE `Atoms` objects as the primary structure representation, with conversions to/from:
- Pymatgen `Structure` via `strucd.get_pymatgen()/StructureData(pymatgen=...)`
- AiiDA `StructureData` via `strucd.get_ase()/StructureData(ase=...)`
- File I/O via `ase.io.read()/ase.io.write()`

**AiiDA WorkChain pattern**: Workflows in `aiida_utils/workflows/` follow AiiDA conventions:
- Use `expose_inputs()` to inherit parameters from base workchains
- Store intermediate results in `self.ctx`
- Submit sub-workchains via `self.submit()`
- Return exit codes for error handling
- Use `@calcfunction` decorator for provenance tracking of pure functions. The function's argument should be aiida.orm classes.

**AiiDA WorkGraph pattern**: Workgraphs in `aiida_utils/workgraphs/` use the aiida-workgraph framework:
- Define tasks using `@task` or `@task.graph` decorators
- Use dynamic namespaces for flexible outputs
- Chain tasks by connecting outputs to inputs
- Submit workgraphs with `.submit()` for execution 

**CLI argument handling**: All CLI commands accept structure files with auto-format detection (ASE), with optional `--input-format`/`--output-format` overrides when needed.

## Important Notes

- **Ruff configuration**: Project uses `ruff_defaults.toml` for consistent formatting (120 char line length, double quotes)
- **Version management**: Version is stored in `src/matchest/__about__.py` and managed by hatch
- **AiiDA dependencies**: The `aiida_utils` module requires AiiDA to be installed via the `[aiida]` extra. Importing `matchest.aiida_utils` without AiiDA will raise a helpful error message
- **Symmetry operations**: All symmetry detection uses `spglib` with configurable tolerance parameters
- **Entry points**: The package registers AiiDA workflow entry points (e.g., `matchest.vasp.elastic` for VaspElasticWorkChain)

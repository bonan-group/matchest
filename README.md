# matchest

**matchest** is a Python package providing scripts and tools for computational materials science, developed and maintained by the Bonan research group.

> **Note:** This package contains common code shared within our research group. While we strive for quality, the code is provided as-is without guarantees of correctness or completeness. Feedback, bug reports, and contributions are welcome!

## Overview

matchest provides three main components for materials science research:

### 1. CLI Tools
Command-line utilities for everyday computational materials science tasks:
- **Structure analysis**: Primitive cell conversion, space group determination
- **K-point grids**: Systematic k-point mesh generation for DFT convergence testing
- **VASP utilities**: Input validation, parallelization checking, convergence analysis
- **CASTEP utilities**: SCF analysis, timing extraction

### 2. AiiDA Integration
High-level workflow abstractions for automated calculations:
- **Workflows**: Elastic tensor calculations, vacancy formation energies
- **Workgraphs**: Calculating force constants for Phono3py. Current implemented designed for zincblende and rocksalt structures, but can be adapted for other systems easily.
- **Utilities**: Structure transformation, pymatgen integration, MACE ML potentials

### 3. Python Utilities
Reusable modules for materials structure manipulation and analysis using ASE, pymatgen, and spglib.

## Documentation

Full documentation is available at [matchest.readthedocs.io](https://matchest.readthedocs.io).

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

**Basic installation** (CLI tools and utilities):
```bash
pip install matchest
```

**With AiiDA support** (for workflows):
```bash
pip install matchest[aiida]
```

**Development installation**:
```bash
git clone https://github.com/bonan-group/matchest.git
cd matchest
pip install -e .              # Basic
pip install -e ".[aiida]"     # With AiiDA
pip install -e ".[docs]"      # With documentation dependencies
pip install -e ".[pre-commit]"  # With pre-commit hooks
```

## Features

### CLI Commands

- `mat prim` - Convert structures to primitive cells
- `mat spg` - Space group analysis
- `mat kpoints` - K-point grid generation
- `mat vasp-check` - VASP input validation and parallelization checker
- `mat vasp-conv` - Geometry optimization convergence analysis
- `mat castep-scf-info` - CASTEP SCF iteration analysis
- `mat castep-timing` - CASTEP timing analysis

See `mat --help` or the [documentation](https://matchest.readthedocs.io) for all commands.

### AiiDA Workflows

- **VaspElasticWorkChain**: Automated elastic tensor calculations
- **SimpleVacancyWorkChain**: Vacancy formation energy calculations
- **Phono3py workgraph**: 2nd/3rd order force constants for thermal transport

## Contributing

Contributions, bug reports, and feedback are welcome! Please open an issue or pull request on [GitHub](https://github.com/bonan-group/matchest).

## Dependencies

**Core dependencies:**
- spglib
- ase
- pymatgen
- click
- tqdm
- tabulate

**Optional dependencies:**
- AiiDA ecosystem (`[aiida]` extra): aiida-core, aiida-vasp, aiida-workgraph
- Documentation (`[docs]` extra): Sphinx, MyST, pydata-sphinx-theme
- Pre-commit hooks (`[pre-commit]` extra)

## Disclaimer

This package is developed for internal use within our research group and shared publicly in the spirit of open science. While we use these tools in our research, we cannot guarantee that all code is bug-free or suitable for all use cases. Use at your own discretion and always validate results independently.

**Feedback and contributions to improve the code are highly encouraged!**

## License

`matchest` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

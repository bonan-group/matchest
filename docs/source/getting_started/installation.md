# Installation

## Requirements

- Python 3.8 or later
- `spglib`, `ase`, `pymatgen` (automatically installed)

## Basic Installation

Install matchest from PyPI:

```bash
pip install matchest
```

This installs the core CLI tools and utilities.

## Installation with AiiDA Support

For AiiDA workflows and workgraphs, install with the `[aiida]` extra:

```bash
pip install matchest[aiida]
```

This includes:
- `aiida-core>=2.0`
- `aiida-vasp`
- `aiida-workgraph`
- `numpy`

## Development Installation

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/bonan-group/matchest.git
cd matchest
pip install -e .
```

With AiiDA support:

```bash
pip install -e ".[aiida]"
```

## Verifying Installation

Check that the CLI is available:

```bash
matchest --help
# or
mat --help
```

## Documentation Build (Optional)

To build this documentation locally:

```bash
pip install matchest[docs]
cd docs
make html
```

The built documentation will be in `docs/build/html/`.

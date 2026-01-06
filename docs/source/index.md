# matchest Documentation

**matchest** is a Python package providing scripts and tools for computational materials science, developed and maintained by the Bonan research group.

:::{admonition} Disclaimer
:class: warning

This package contains common code shared within our research group. While we strive for quality, the code is provided as-is without guarantees of correctness or completeness.

**Use at your own discretion and always validate results independently.**

Feedback, bug reports, and contributions are highly encouraged!
:::

## Features

- **CLI Tools**: Command-line utilities for crystal structure analysis and DFT code utilities (VASP, CASTEP)
- **AiiDA Integration**: Automated computational workflows for materials science calculations
- **Pymatgen Utilities**: Advanced structure manipulation and analysis tools

## Quick Links

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Getting Started
:link: getting_started/index
:link-type: doc

New to matchest? Start here with installation and quickstart guides.
:::

:::{grid-item-card} CLI Tools
:link: cli/index
:link-type: doc

Command-line tools for structure analysis, k-points, VASP/CASTEP utilities.
:::

:::{grid-item-card} AiiDA Workflows
:link: aiida/index
:link-type: doc

Automated workflows for elastic constants, vacancy formation, phonon calculations.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

Detailed API documentation for Python developers.
:::

::::

## Table of Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

getting_started/index
cli/index
aiida/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

cli/reference
api/index
```

## Installation

Install matchest from PyPI:

```bash
pip install matchest
```

For AiiDA workflows, install with the `aiida` extra:

```bash
pip install matchest[aiida]
```

## License

matchest is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

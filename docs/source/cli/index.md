# CLI Tools

matchest provides a comprehensive set of command-line tools accessible via the `matchest` (or `mat`) command.

## Overview

The CLI is organized into several categories:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Structure Tools
:link: structure_tools
:link-type: doc

Primitive cell conversion, space group analysis
:::

:::{grid-item-card} K-points
:link: kpoints
:link-type: doc

K-point grid calculation for DFT codes
:::

:::{grid-item-card} VASP Tools
:link: vasp_tools
:link-type: doc

Convergence analysis, input validation, force tracking
:::

:::{grid-item-card} CASTEP Tools
:link: castep_tools
:link-type: doc

SCF analysis, timing extraction
:::

::::

## Command List

All commands are accessible via `matchest <command>` or `mat <command>`:

**Structure Analysis:**
- `prim` - Convert to primitive cell
- `spg` - Get space group information
- `pmg-convert-cell` - Convert cell using pymatgen

**K-points:**
- `kpoints` - Calculate systematic k-point series

**VASP Utilities:**
- `vasp-conv` - Convergence analysis
- `vasp-check` - Input file validation and parallelization check
- `vasp-max-force` - Track maximum forces per cycle
- `trim-vasprun` - Reduce vasprun.xml size

**CASTEP Utilities:**
- `castep-scf-info` - SCF iteration analysis
- `castep-timing` - Timing information extraction

**Visualization:**
- `view-files-ovito` - View structures in OVITO

**Utilities:**
- `charge-neutral-combinations` - Compute charge-neutral formulas

## Detailed Documentation

```{toctree}
:maxdepth: 2

structure_tools
kpoints
vasp_tools
castep_tools
reference
```

## General Usage

```bash
# Get help for any command
matchest <command> --help

# Examples
matchest prim --help
matchest vasp-check --help
```

Most structure-related commands accept:
- `--input-format` - Specify input file format (auto-detected by default)
- `--output-format` - Specify output file format
- Common formats: vasp, cif, xsf, aims, etc. (via ASE)

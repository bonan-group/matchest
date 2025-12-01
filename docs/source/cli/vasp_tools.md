# VASP Tools

Utilities for analyzing and validating VASP calculations.

## vasp-conv - Convergence Analysis

Analyze geometry optimization convergence from VASP OUTCAR files.

### Usage

```bash
matchest vasp-conv [FILENAME]
```

Shows for each ionic step: total energy, energy difference, maximum force, maximum stress, time, and convergence status.

## vasp-check - Input Validation and Parallelization

Comprehensive VASP input file checker for optimal parallelization and efficiency.

### Usage

```bash
matchest vasp-check [DIRECTORY] [OPTIONS]
```

### Options

- `--recursive/--no-recursive` - Recursively scan subdirectories (default: True)
- `-o, --output PATH` - Write report to file
- `--table/--no-table` - Display as table (default: True)
- `--queue` - Scan running/queued SLURM jobs instead of directory

### Examples

```bash
# Check single calculation
matchest vasp-check .

# Scan directory recursively
matchest vasp-check /calculations --recursive

# Check running SLURM jobs
matchest vasp-check --queue

# Generate report file
matchest vasp-check /calculations --output report.txt
```

### What It Checks

The checker analyzes:

**Parallelization:**
- NCORE, KPAR, NPAR settings
- Number of MPI tasks
- Band and k-point parallelization efficiency

**Computational Efficiency:**
- Computational cost estimation (n_kpoints × n_bands²)
- K-point spacing validation
- Long electronic/ionic loop detection

**Input Validation:**
- File consistency (INCAR, POSCAR, POTCAR, KPOINTS)
- Hybrid DFT detection

### Common Issues Detected

- Missing parallelization tags for large calculations
- NCORE too small resulting in inefficient band groups
- KPAR larger than k-points
- Unusual k-point spacing
- Very long LOOP/LOOP+ times

## vasp-max-force - Track Maximum Forces

Extract maximum forces from each VASP ionic step.

### Usage

```bash
matchest vasp-max-force OUTCAR
```

Shows the maximum force magnitude and atom index for each ionic step.

## trim-vasprun - Reduce vasprun.xml Size

Remove specific XML tags from vasprun.xml to reduce file size.

### Usage

```bash
matchest trim-vasprun INPUT_FILE TAG_TO_REMOVE
```

### Examples

```bash
# Remove projected DOS
matchest trim-vasprun vasprun.xml projected

# Remove partial charge data
matchest trim-vasprun vasprun.xml partial
```

**Warning:** This tool permanently modifies the vasprun.xml file. Make backups before use!

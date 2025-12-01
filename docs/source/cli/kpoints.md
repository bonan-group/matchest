# K-points Tools

Calculate systematic k-point grids for DFT calculations.

## kpoints - Calculate K-point Series

Generate a series of k-point grids for convergence testing.

### Usage

```bash
matchest kpoints [FILENAME] [OPTIONS]
```

### Options

- `--min FLOAT` - Minimum real-space cutoff in Å (default: 10)
- `--max FLOAT` - Maximum real-space cutoff in Å (default: 30)
- `--comma-sep` - Output as comma-separated list
- `--vasp` - Use VASP KSPACING format (1/Å) instead of CASTEP MP_SPACING (2π/Å)
- `--realspace` - Use real-space lattice lengths instead of reciprocal

### Examples

```bash
# CASTEP format (MP SPACING in 2π/Å)
matchest kpoints POSCAR --min 10 --max 30

# VASP format (KSPACING in 1/Å)
matchest kpoints POSCAR --vasp --min 10 --max 30

# Comma-separated for scripting
matchest kpoints POSCAR --comma-sep
```

### Output

**Standard format (CASTEP):**
```
Length cutoff (Å)  MP SPACING (2π/Å)    Samples
-----------------  -------------------  ------------
          10.000            0.050000     2   2   2
          15.000            0.033333     3   3   3
...
```

**Comma-separated:**
```
2 2 2,3 3 2,3 3 3,4 4 3,...
```

### Background

K-point grids are critical for DFT convergence. This tool:
- Generates systematic series based on real-space cutoffs
- Accounts for crystal symmetry to find unique grids
- Supports both CASTEP (2π/Å) and VASP (1/Å) conventions

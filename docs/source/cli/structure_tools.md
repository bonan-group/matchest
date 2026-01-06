# Structure Tools

Commands for crystal structure analysis and manipulation.

## prim - Convert to Primitive Cell

Convert crystal structures to their primitive cell using spglib.

### Usage

```bash
matchest prim [INPUT_FILE] [OPTIONS]
```

### Key Options

- `-t, --threshold FLOAT` - Distance threshold in Å for symmetry reduction (default: 1e-5)
- `-o, --output-file PATH` - Output file path
- `-v, --verbose` - Print output to screen even when writing to file

### Examples

```bash
# Convert POSCAR to primitive cell
matchest prim POSCAR -o POSCAR.prim

# Convert CIF with custom tolerance
matchest prim structure.cif -t 1e-4 -o primitive.cif
```

## spg - Space Group Analysis

Display space group information at different symmetry tolerances.

### Usage

```bash
matchest spg [FILENAME] [OPTIONS]
```

### Examples

```bash
# Auto-detect structure file
matchest spg

# Specify file
matchest spg structure.cif
```

Shows space group at different tolerance thresholds from 1e-5 to 1e-2 Å.

## pmg-convert-cell - Pymatgen Cell Conversion

Convert structures to standard/primitive cells using pymatgen's SpaceGroupAnalyzer.

### Usage

```bash
matchest pmg-convert-cell INPUT OUTPUT [OPTIONS]
```

### Options

- `-c, --cell-type` - Cell type: standard, primitive, primitive-standard (default: standard)
- `-o, --output-type` - Output format: cif, poscar (default: cif)
- `--symprec FLOAT` - Symmetry tolerance (default: 1e-5)

### Examples

```bash
# Convert to conventional standard structure
matchest pmg-convert-cell input.cif output.cif

# Get primitive cell as POSCAR
matchest pmg-convert-cell input.cif output.vasp -c primitive -o poscar
```

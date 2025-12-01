# Quick Start

This guide demonstrates common matchest use cases.

## CLI Tools Examples

### Convert to Primitive Cell

```bash
# Convert POSCAR to primitive cell
matchest prim POSCAR -o POSCAR.prim

# With symmetry tolerance
matchest prim POSCAR -t 1e-4 -o POSCAR.prim
```

### Check Space Group

```bash
# Auto-detect input file (POSCAR, geometry.in, or castep.cell)
matchest spg

# Specify file
matchest spg structure.cif
```

### Calculate K-points Grid

```bash
# CASTEP MP spacing format
matchest kpoints POSCAR --min 10 --max 30

# VASP KSPACING format
matchest kpoints POSCAR --vasp --min 10 --max 30

# Comma-separated output
matchest kpoints POSCAR --comma-sep
```

### Check VASP Inputs

```bash
# Check single calculation
matchest vasp-check /path/to/calculation

# Scan directory recursively
matchest vasp-check /calculations --recursive

# Check running SLURM jobs
matchest vasp-check --queue

# Generate report file
matchest vasp-check ./my-calc --output report.txt --table
```

### Analyze VASP Convergence

```bash
matchest vasp-conv OUTCAR
```

## AiiDA Workflows (Requires `[aiida]` extra)

### Elastic Tensor Calculation

```python
from aiida import load_profile, orm
from aiida.engine import submit
from matchest.aiida_utils.workflows.elastic import VaspElasticWorkChain

load_profile()

# Setup inputs
builder = VaspElasticWorkChain.get_builder()
builder.relax.structure = orm.StructureData(...)
builder.relax.vasp.code = orm.load_code('vasp@localhost')
builder.relax.vasp.parameters = orm.Dict(dict={...})
# ... configure other inputs

# Submit workflow
node = submit(builder)
print(f"Submitted WorkChain with PK={node.pk}")
```

## Python API Examples

### Structure Conversion

```python
from matchest.cli.structure import get_primitive_atoms
import ase.io

atoms = ase.io.read("POSCAR")
primitive = get_primitive_atoms(atoms, threshold=1e-5)
ase.io.write("POSCAR.prim", primitive, format="vasp")
```

### K-points Calculation

```python
from matchest.cli.kpoints import calc_kpt_tuple
import ase.io

atoms = ase.io.read("POSCAR")
kpoints = calc_kpt_tuple(atoms, cutoff_length=15.0, realspace=False)
print(f"K-point grid: {kpoints}")
```

## Next Steps

- Explore [CLI Tools](../cli/index) for detailed command documentation
- Check [AiiDA Workflows](../aiida/index) for automated calculation workflows
- See [API Reference](../api/index) for Python development

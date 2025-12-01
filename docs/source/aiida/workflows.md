# AiiDA Workflows

WorkChains for automated multi-step calculations.

## VaspElasticWorkChain

Calculate elastic constants using VASP.

### Overview

This workflow:
1. Performs full structural relaxation
2. Standardizes the structure to primitive cell
3. Generates deformed structures using pymatgen
4. Runs VASP relaxations on deformed structures
5. Computes elastic tensor from stress-strain relationship

### Usage

```python
from aiida import orm
from aiida.engine import submit
from matchest.aiida_utils.workflows.elastic import VaspElasticWorkChain

builder = VaspElasticWorkChain.get_builder()

# Structure
builder.relax.structure = orm.StructureData(...)

# VASP settings
builder.relax.vasp.code = orm.load_code('vasp@localhost')
builder.relax.vasp.parameters = orm.Dict(dict={
    'incar': {
        'encut': 520,
        'ismear': 0,
        'sigma': 0.05,
        'ediff': 1e-6,
        'prec': 'Accurate',
    }
})
builder.relax.vasp.kpoints = orm.KpointsData(...)
builder.relax.vasp.potential_family = 'PBE.54'
builder.relax.vasp.potential_mapping = orm.Dict(...)
builder.relax.vasp.options = orm.Dict(dict={
    'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 16},
    'max_wallclock_seconds': 3600,
})

# Submit
node = submit(builder)
```

### Inputs

The workflow exposes all inputs from `VaspRelaxWorkChain` under the `relax` namespace plus optional `elastic_settings` for symmetry parameters.

### Outputs

- `elastic_tensor` (ArrayData) - Elastic tensor in Voigt notation
- `relaxed_structure` (StructureData) - Fully relaxed structure
- `primitive_structure` (StructureData) - Standardized primitive cell

### Entry Point

Registered as: `matchest.vasp.elastic`

## Simple Vacancy Formation Workflow

Calculate vacancy formation energies.

### Overview

This workflow:
1. Relaxes the pristine supercell
2. Creates vacancy structures
3. Relaxes vacancy structures
4. Computes formation energies

Usage is similar to the elastic workflow but with vacancy-specific configuration parameters.

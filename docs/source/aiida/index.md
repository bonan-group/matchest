# AiiDA Integration

matchest provides AiiDA workflows and utilities for automated computational materials science calculations.

:::{note}
AiiDA features require installation with the `[aiida]` extra:
```bash
pip install matchest[aiida]
```
:::

## Overview

The AiiDA integration includes:

**Workflows** (`matchest.aiida_utils.workflows`):
- VaspElasticWorkChain - Elastic tensor calculations
- Simple vacancy formation - Vacancy formation energy calculations

**Workgraphs** (`matchest.aiida_utils.workgraphs`):
- Phono3py - Phonon and thermal transport calculations

**Utilities** (`matchest.aiida_utils`):
- Structure transformation and manipulation
- Pymatgen-AiiDA integration
- VASP-specific utilities
- MACE ML potential integration
- Battery materials tools

## Contents

```{toctree}
:maxdepth: 2

workflows
workgraphs
utilities
```

## Quick Example

```python
from aiida import load_profile, orm
from aiida.engine import submit
from matchest.aiida_utils.workflows.elastic import VaspElasticWorkChain

# Load AiiDA profile
load_profile()

# Setup workflow
builder = VaspElasticWorkChain.get_builder()
builder.relax.structure = structure_node
builder.relax.vasp.code = vasp_code
builder.relax.vasp.parameters = parameters_dict
# ... configure remaining inputs

# Submit
node = submit(builder)
print(f"Submitted elastic workflow: {node.pk}")
```

## Entry Points

The package registers AiiDA workflow entry points:

- `matchest.vasp.elastic` - VaspElasticWorkChain

Access via:

```python
from aiida.plugins import WorkflowFactory

ElasticWorkChain = WorkflowFactory('matchest.vasp.elastic')
```

## Dependencies

AiiDA features require:
- `aiida-core>=2.0,<3`
- `aiida-vasp`
- `aiida-workgraph`
- `numpy`

These are automatically installed with `pip install matchest[aiida]`.

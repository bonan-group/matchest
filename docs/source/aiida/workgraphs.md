# AiiDA WorkGraphs

Modern graph-based workflows using aiida-workgraph.

## Phono3py Workgraph

Calculate 2nd and 3rd order force constants using phono3py.

### Overview

This workgraph automates:
- Supercell generation for phonon calculations
- Displacement creation (2nd and 3rd order)
- Force calculations on displaced structures
- Force constant extraction with phono3py

### Requirements

```bash
pip install matchest[aiida]  # includes aiida-workgraph
```

### Usage

```python
from aiida import orm
from matchest.aiida_utils.workgraphs.phono3py import phono3py_workflow

# Define inputs
inputs = {
    'structure': structure_node,
    'code': vasp_code,
    'supercell_matrix': [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    'parameters': vasp_parameters,
    # ... other inputs
}

# Create and submit workgraph
wg = phono3py_workflow.build(**inputs)
wg.submit()
```

### Features

- Dynamic namespace outputs for flexible data handling
- Automatic displacement generation
- Parallel force calculations
- Integration with phono3py for force constant extraction

### Outputs

- 2nd order force constants
- 3rd order force constants
- Phonon band structures
- Thermal conductivity (if configured)

## WorkGraph Benefits

WorkGraphs provide a more flexible alternative to traditional WorkChains:
- Dynamic task graphs
- Conditional execution
- Better debugging and visualization

See the [aiida-workgraph documentation](https://aiida-workgraph.readthedocs.io/) for more details.

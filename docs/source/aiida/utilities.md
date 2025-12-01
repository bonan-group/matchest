# AiiDA Utilities

Supporting modules for AiiDA workflows.

## Structure Transformation (`transformation.py`)

Tools for creating supercells and defect structures.

### Key Functions

- `create_supercell()` - Generate supercells from structures
- `create_vacancy_structure()` - Create vacancy defect structures

### Example

```python
from matchest.aiida_utils.process.transformation import (
    create_supercell,
    create_vacancy_structure,
)

# Create supercell
supercell = create_supercell(structure, scaling_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])

# Create vacancy
vacancy_struct = create_vacancy_structure(structure, site_index=0)
```

## Pymatgen Integration (`pmg.py`)

Bridge between AiiDA and pymatgen.

### Key Functions

- `aiida_to_pymatgen()` - Convert AiiDA StructureData to pymatgen Structure
- `pymatgen_to_aiida()` - Convert pymatgen Structure to AiiDA StructureData
- `get_conventional_structure()` - Get conventional cell
- Materials Project integration for structure queries

### Example

```python
from matchest.aiida_utils.pmg import aiida_to_pymatgen, pymatgen_to_aiida

# Convert AiiDA to pymatgen
pmg_struct = aiida_to_pymatgen(aiida_structure)

# Convert back
aiida_struct = pymatgen_to_aiida(pmg_struct)
```

## VASP Utilities (`vasp.py`)

VASP-specific helper functions.

- `parse_functional()` - Determine exchange-correlation functional
- `apply_hubbard_u()` - Apply DFT+U corrections

## MACE Utilities (`maceutils.py`)

Integration with MACE machine learning potentials for pre-relaxation and high-throughput screening.

## Battery Tools (`battery.py`)

Specialized utilities for battery materials:
- Voltage profile calculations
- Ion migration analysis
- Intercalation site analysis

## Common Patterns

### @calcfunction Decorator

Many utilities use `@calcfunction` for provenance tracking:

```python
from aiida.engine import calcfunction

@calcfunction
def my_transformation(structure):
    # Transformations here
    return transformed_structure
```

This ensures all operations are tracked in AiiDA's provenance graph.

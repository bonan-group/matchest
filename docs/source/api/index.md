# API Reference

Python API documentation for matchest modules.

## Module Overview

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} CLI Modules
Python functions underlying CLI commands
:::

:::{grid-item-card} AiiDA Workflows
WorkChains and workgraphs for automated calculations
:::

:::{grid-item-card} Utilities
Structure manipulation, k-points, analysis tools
:::

:::{grid-item-card} CASTEP/VASP
Code-specific parsing and utilities
:::

::::

## Auto-generated Documentation

```{eval-rst}
.. automodule:: matchest
   :members:
   :undoc-members:
   :show-inheritance:

CLI Modules
-----------

.. automodule:: matchest.cli.structure
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: matchest.cli.kpoints
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: matchest.cli.geomconv
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: matchest.cli.vaspcheck
   :members:
   :undoc-members:
   :show-inheritance:

AiiDA Workflows
---------------

.. automodule:: matchest.aiida_utils.workflows.elastic
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: matchest.aiida_utils.workflows.simple_vac
   :members:
   :undoc-members:
   :show-inheritance:

AiiDA Workgraphs
----------------

.. automodule:: matchest.aiida_utils.workgraphs.phono3py
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: matchest.utils.kmesh
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: matchest.aiida_utils.pmg
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: matchest.aiida_utils.vasp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: matchest.aiida_utils.process.transformation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: matchest.aiida_utils.process.battery
   :members:
   :undoc-members:
   :show-inheritance:

CASTEP Utilities
----------------

.. automodule:: matchest.casteputils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: matchest.dotcastep
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Examples

### Importing Modules

```python
# CLI functions
from matchest.cli.structure import get_primitive_atoms
from matchest.cli.kpoints import calc_kpt_tuple

# AiiDA workflows
from matchest.aiida_utils.workflows.elastic import VaspElasticWorkChain

# Utilities
from matchest.aiida_utils.pmg import aiida_to_pymatgen
```

### Structure Analysis

```python
import ase.io
from matchest.cli.structure import get_primitive_atoms

atoms = ase.io.read("POSCAR")
primitive = get_primitive_atoms(atoms, threshold=1e-5)
print(f"Reduced from {len(atoms)} to {len(primitive)} atoms")
```

### K-points Calculation

```python
from matchest.cli.kpoints import calc_kpt_tuple, cutoff_series
import ase.io

atoms = ase.io.read("POSCAR")

# Single k-point grid
kpts = calc_kpt_tuple(atoms, cutoff_length=15.0)
print(f"K-points: {kpts}")

# Series for convergence
cutoffs = cutoff_series(atoms, l_min=10, l_max=30)
for cutoff in cutoffs:
    kpts = calc_kpt_tuple(atoms, cutoff_length=cutoff)
    print(f"{cutoff:.2f} Å → {kpts}")
```

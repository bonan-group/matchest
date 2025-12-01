# CASTEP Tools

Utilities for analyzing CASTEP calculations.

## castep-scf-info - SCF Analysis

Extract and display the number of SCF iterations for each geometry step.

### Usage

```bash
matchest castep-scf-info DOT_CASTEP
```

### Output

```
15
12
10
8
7
...
```

Each line shows the number of SCF iterations for that geometry step.

## castep-timing - Timing Analysis

Analyze timing information from CASTEP calculations.

### Usage

```bash
matchest castep-timing DOT_CASTEP
```

### Output

```
Average time on each electronic loop : 12.345 s
Average time on each ionic loop      : 185.234 s
K-point parallelisation              : 4
G-vector parallelisation             : 8
Total number of processsors          : 32
```

Provides:
- Average electronic SCF loop time
- Average ionic (geometry) loop time
- Parallelization settings (k-point, G-vector)
- Total MPI processes used

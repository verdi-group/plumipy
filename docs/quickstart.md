# Quick start

!!! warning

    These instructions are very preliminary, expect large changes in the API soon.

## Required inputs

You need to have calculated:

1. `CONTCAR` (structure) & `OUTCAR` (forces) for the ground state
1. `CONTCAR` (structure) & `OUTCAR` (forces) for the excited state
1. `band.yaml` from Phonopy (if phonopy) or DFPT/FD `OUTCAR` from VASP

## Calculate the spectrum

```
from plumipy import calculate_spectrum

calculate_spectrum(
    path_structure_gs="CONTCAR_GS",
    path_structure_es="CONTCAR_ES",
    phonons_source="Phonopy",
    path_phonon_band="./band.yaml",
    temperature=0,
    zpl=3339,
    tmax=2000,
    gamma=10,
    forces=None, 
)
```

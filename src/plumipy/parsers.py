"""Custom parsers for the `plumipy` package."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import yaml


class ReadFiles:
    def __init__(self):
        pass

    def read_structure(self, path):
        """
        Input:   1. path - Location of POSCAR or CONTCAR file as a string.

        Outputs: 1. Position vectors of all the atoms as numpy array of shape (total number of atoms, 3), where 3 is
                    the x,y and z space coordinates.
                 2. Dictionary of atomic species and there corresponding number of atoms.
        """
        with open(path) as file:
            lines = file.readlines()

            scaling_factor = float(lines[1].strip())
            lattice_vectors = [lines[i].strip().split() for i in range(2, 5)]
            lattice_vectors = scaling_factor * (np.array(lattice_vectors).astype(float))

            atomic_species = lines[5].strip().split()
            number_of_atoms = np.array(lines[6].strip().split()).astype(int)
            tot = sum(number_of_atoms)

            lattice_type = lines[7].strip()

            atomic_positions = [lines[i].strip().split() for i in range(8, 8 + tot)]
            atomic_positions = np.array(atomic_positions).astype(float)
            atoms = dict(zip(atomic_species, number_of_atoms))

            if lattice_type != "Direct":
                lattice_inv = np.linalg.inv(lattice_vectors.T)
                atomic_positions = np.array(
                    [np.dot(lattice_inv, vec) for vec in atomic_positions]
                )

            atomic_positions[atomic_positions > 0.99] -= 1
            atomic_positions = np.dot(atomic_positions, lattice_vectors)
            return (atomic_positions, atoms)

    def read_phonons_phonopy(self, path: str | Path) -> tuple:
        """
        Input:   1. path: Location of band.yaml file as a string.

        Outputs: 1. Atomic_masses is a 1D array of masses (AMU) of each atom in the same sequence as
                    that of Atomic positions in previous function.
                 2. Phonon frequencies (THz) as a 1D at Gamma point. Length of the array = number of normal modes.
                 3. Eigenvectors corresponding to the phonon frequencies as a 3D array of
                    shape (number of normal mode, number of atoms, 3), where 3 is the x,y and z coordinates.
        """
        file_path = Path(path)

        if file_path.suffix in (".yml", ".yaml"):
            with file_path.open("r") as handle:
                data = yaml.safe_load(handle)

        yaml_freqs = np.array(
            [
                [band_dict["frequency"] for band_dict in qpoint_data["band"]]
                for qpoint_data in data["phonon"]
            ]
        )
        yaml_freqs = np.concatenate([yaml_freqs[0, :], yaml_freqs[1, :]], axis=0)
        yaml_freqs[yaml_freqs < 0] = 0  # TODO: remove this rather hacky solution 😅

        eigenvectors = np.array(
            [
                [band_dict["eigenvector"] for band_dict in phonon_data["band"]]
                for phonon_data in data["phonon"]
            ]
        )
        complex_eigenvectors = (
            eigenvectors[:, :, :, :, 0] + eigenvectors[:, :, :, :, 1] * 1j
        )
        final_eigenvectors = np.concatenate(
            [complex_eigenvectors[0, :, :, :], complex_eigenvectors[1, :, :, :]], axis=0
        )
        atomic_masses = [point["mass"] for point in data["points"]]

        return atomic_masses, yaml_freqs, final_eigenvectors.real

    def read_phonons_vasp(self, path, atoms):
        """
        From VASP OUTCAR.
        Input:   1. path: Location of band.yaml file as a string.
                 2. atoms: Dictionary of atomic species and there corresponding number of atoms.

        Outputs:  1.Atomic_masses is a 1D array of masses (AMU) of each atom in the same sequence as
                    that of Atomic positions in previous function.
                  2. Phonon frequencies (THz) as a 1D at Gamma point. Length of the array = number of normal modes.
                  3. Eigenvectors corresponding to the phonon frequencies as a 3D array of
                    shape (number of normal mode, number of atoms, 3), where 3 is the x,y and z coordinates.
        """

        freqs = []
        normal_modes = []
        number_of_atoms = np.array([i[1] for i in atoms.items()])
        total_atoms = np.sum(number_of_atoms)

        with open(path) as file:
            lines = [line.strip() for line in file]

            index = lines.index("Mass of Ions in am")
            atomic_masses = lines[index + 1].split()[2:]
            atomic_masses = np.array(atomic_masses).astype(float)

            index_init = lines.index(
                "Eigenvectors and eigenvalues of the dynamical matrix"
            )
            index_final = lines.index(
                "ELASTIC MODULI CONTR FROM IONIC RELAXATION (kBar)"
            )

            for i in range(index_init, index_final + 1):
                internal_modes = []
                if "THz" in lines[i]:
                    freqs.append(lines[i].split()[lines[i].split().index("THz") - 1])
                    internal_modes = [
                        lines[j].split() for j in range(i + 2, i + 2 + total_atoms)
                    ]
                    normal_modes.append(internal_modes)

        atomic_masses = np.repeat(atomic_masses, number_of_atoms)
        freqs = np.array(freqs).astype(float)
        sort = np.argsort(freqs)
        freqs = freqs[sort]
        normal_modes = np.array(normal_modes).astype(float)[..., 3:]
        normal_modes = normal_modes[sort]
        return atomic_masses, freqs, normal_modes

    def read_forces(self, path):
        """
        Reads and stores the Forces (eV/Angstrom) on each atom from the OUTCAR file and returns a 2D array.
        """
        forces = []
        start_collecting = False
        lines_buffer = []

        with open(path) as file:
            for line in file:
                if "TOTAL-FORCE" in line:
                    start_collecting = True
                    continue
                if start_collecting:
                    if "total drift:" in line:
                        break
                    lines_buffer.append(line.strip())
        for line in lines_buffer[1:-1]:
            numbers = [float(num) for num in line.split()]
            forces.append(numbers)
        forces = np.array(forces)
        return forces[:, 3:]

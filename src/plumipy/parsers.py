"""Custom parsers for the `plumipy` package."""

import numpy as np


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

    def read_phonons_phonopy(self, path):
        """
        Input:   1. path: Location of band.yaml file as a string.

        Outputs: 1. Atomic_masses is a 1D array of masses (AMU) of each atom in the same sequence as
                    that of Atomic positions in previous function.
                 2. Phonon frequencies (THz) as a 1D at Gamma point. Length of the array = number of normal modes.
                 3. Eigenvectors corresponding to the phonon frequencies as a 3D array of
                    shape (number of normal mode, number of atoms, 3), where 3 is the x,y and z coordinates.
        """
        with open(path) as file:
            lines = [ts.strip() for ts in file]

        freqs = []
        normal_modes = []

        with open(path) as file:
            atomic_masses = [line.split()[1] for line in file if "mass:" in line]

        atomic_masses = np.array(atomic_masses).astype(float)
        total_atoms = len(atomic_masses)

        with open(path) as file:
            line_number = -1
            for line in file:
                line_number += 1
                if "frequency:" in line:
                    freqs.append(float(line.split()[1]))
                    ev_internal = []
                    for i in range(
                        line_number + 3, line_number + 4 * total_atoms + 2, 4
                    ):
                        xyz = [lines[i + j].split()[2] for j in range(3)]
                        ev_internal.append(xyz)
                    normal_modes.append(ev_internal)
        freqs = np.array(freqs).astype(float)
        freqs[freqs < 0] = 0
        normal_modes = np.array(
            [
                [[float(x.strip(",")) for x in sublist] for sublist in outer]
                for outer in normal_modes
            ]
        )
        return atomic_masses, freqs, normal_modes

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

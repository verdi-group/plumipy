# ruff: noqa: B008
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plumipy import constants


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


class Photoluminescence(ReadFiles):
    def __init__(self):
        """
        Define all the variables by reading the input files like POSCAR_GS/CONTCAR_GS, POSCAR_ES/CONTCAR_ES, and band.yaml.
        """
        super().__init__()

    def get_direct_mesh(self, direct_min, direct_max, reciprocal_max):
        """Obtain a 1D array with equal intervals for the direct variable.

        The reciprocal variable is typically energy, whereas the direct variable is time.

        Args:
            direct_min (float): Lower bound of the direct variable.
            direct_max (float): Upper bound of the direct variable.
            reciprocal_max (float): Maximum allowed value of the reciprocal variable.

        Returns:
            numpy.ndarray: 1D array of direct variable values values.
        """
        direct_spacing = (2 * np.pi) / (2 * reciprocal_max)
        return np.arange(direct_min, direct_max, direct_spacing)

    def fourier(self, iv_array, function):
        """
        Discrete Fourier transform (DFT) using FFT algorithm.
        Here, DFT is approximated as Continuous Fourier Transform.

        Inputs: Independent variable and the function to be Fourier Transformed.

        Outputs: 1D array of reciprocal variable (generally Energy or frequency in this case) on which
        FFT has been performed and 1D array of the DFT result.
        """
        iv_spacing = iv_array[1] - iv_array[0]
        rv_array = 2 * np.pi * np.fft.fftfreq(len(iv_array), iv_spacing)
        sort = np.argsort(rv_array)
        rv_array = rv_array[sort]
        discrete_fourier = np.fft.fft(function)[sort]
        discrete_fourier = (
            iv_spacing * discrete_fourier * np.exp(-1j * rv_array * iv_array[0])
        )

        return rv_array, discrete_fourier

    def inverse_fourier(self, iv, function):
        """
        Inverse Discrete Fourier transform (IDFT) using FFT algorithm.
        Here, DFT is approximated as Continuous Fourier Transform.
        """
        div = iv[1] - iv[0]
        rv = 2 * np.pi * np.fft.fftfreq(len(iv), div)
        sort = np.argsort(rv)
        rv = rv[sort]
        idft = np.fft.ifft(function)[sort]
        idft = div * idft * np.exp(1j * rv * iv[0])
        return rv, idft

    def trapezoidal(self, integrand, iv, equally_spaced=True):
        """
        Calculates the integral using Trapezoidal Rule.

        Inputs: integrand and iv are arrays of same dimension. equally_spaced: determines whether the method should integrate using
        equally spaced or unequally spaced intervals.

        Output: Integration result.
        """
        div = iv[1] - iv[0]
        return (
            (div / 2) * (np.sum(integrand[1:-1]) + integrand[0] + integrand[-1])
            if equally_spaced
            else np.sum(
                np.array(
                    [
                        ((iv[i + 1] - iv[i]) / 2) * (integrand[i + 1] + integrand[i])
                        for i in range(len(iv) - 1)
                    ]
                )
            )
        )

    @staticmethod
    def time_scaling(t, reverse=False):
        """
        Changes time array t from femtoseconds to meV^-1. This is a necessary step after initializing time through IV
        function in order to maintain consistency in units while performing Fourier Transform.
        """
        conversion_factor = (constants.h.to("meV * fs") / (2 * np.pi)).magnitude
        return t * conversion_factor if reverse else t / conversion_factor

    def lorentzian(self, x, x0, sigma):
        """
        Used to fit Dirac-Delta as Lorentzian function, where sigma = 6 has units of meV.
        The factor of 0.8 multiplying sigma is to make this function have similarities to
        Gaussian for same standard deviation, sigma.
        """
        return ((1 / np.pi) * (sigma * 0.8)) / (((sigma * 0.8) ** 2) + ((x - x0) ** 2))

    def gaussian(self, x, x0, sigma):
        """
        Gaussian fit for Dirac-Delta with sigma = 6 (meV) as standard deviation.
        """
        return (1 / np.sqrt(2 * np.pi * (sigma**2))) * np.exp(
            -((x - x0) ** 2) / (2 * (sigma**2))
        )

    def config_coordinates(self, masses, positions_es, positions_gs, modes):
        """
        Calculates the qk factor (AMU^0.5-Angstrom) for different normal modes as a 1D array of
        length = total number of normal modes.
        """
        masses = np.sqrt(masses)
        r_diff = positions_es - positions_gs
        mr_diff = np.array([masses[i] * r_diff[i, :] for i in range(len(masses))])
        return np.array(
            [np.sum(mr_diff * modes[i, :, :]) for i in range(modes.shape[0])]
        )

    def config_coordinates_f(self, masses, forces_es, forces_gs, modes, energy_k):
        """
        Calculates the qk factor (AMU^0.5-Angstrom) for different normal modes as a 1D array of
        length = total number of normal modes. This function uses forces on atoms rather than their position vectors
        as used in previous function.
        """
        masses = np.sqrt(masses)
        forces_diff = forces_es - forces_gs
        mf_diff = np.array(
            [(1 / masses[i]) * forces_diff[i, :] for i in range(len(masses))]
        )
        qk = np.array([np.sum(mf_diff * modes[i, :, :]) for i in range(modes.shape[0])])
        return (1 / energy_k**2) * qk * 4180.069

    def calc_partial_hr(self, freqs, qk):
        """
        Calculates the Sk (unit less) as a 1D array of length equal to total number of normal modes.
        """
        return 2 * np.pi * freqs * (qk**2) * 0.166 / (2 * 1.05457)

    def spectral_function(
        self, partial_hr_factor, energy_k, energy_mev_positive, sigma=6, lorentz=False
    ):
        """
        Calculates S(hbar_omega) or S(E) (unit less) by using Gaussian or Lorentzian fit
        for Direc-Delta with sigma = 6 meV by default.

        Ek: Normal mode phonon energies.
        """
        self.sigma = sigma
        if not lorentz:
            specfun_energy = np.array(
                [
                    np.dot(partial_hr_factor, self.gaussian(i, energy_k, sigma))
                    for i in energy_mev_positive
                ]
            )
        else:
            specfun_energy = np.array(
                [
                    np.dot(partial_hr_factor, self.lorentzian(i, energy_k, sigma))
                    for i in energy_mev_positive
                ]
            )
        return specfun_energy

    def fourier_spectral_function(
        self, s_k, energy_k, specfun_energy, energy_mev_positive
    ):
        """
        Calculates the Fourier transform of S(E) which is equal to S(t).
        """
        t_mev, s_t = self.fourier(energy_mev_positive, specfun_energy)
        s_t_exact = np.array([np.dot(s_k, np.exp(-1j * energy_k * i)) for i in t_mev])
        return t_mev, s_t, s_t_exact

    def calc_generating_function(
        self, s_k, s_t, t_mev, energy_k, energy_mev_positive, temperature
    ):  # noqa: ARG002
        """
        Calculates the generating function G(t).
        """
        if temperature == 0.0:
            generating_function = np.exp((s_t) - (np.sum(s_k)))
        else:
            boltzmann_constant = 8.61733326e-2  # Boltzmann constant in meV/k
            nk = 1 / ((np.exp(energy_k / (boltzmann_constant * temperature))) - 1)
            c_e = np.array(
                [
                    np.dot(nk * s_k, self.gaussian(i, energy_k, self.sigma))
                    for i in energy_mev_positive
                ]
            )
            c_t = self.fourier(energy_mev_positive, c_e)[1]
            c_t_inv = self.inverse_fourier(energy_mev_positive, c_e)[1]
            generating_function = np.exp(
                (s_t) - (np.sum(s_k)) + c_t + c_t_inv - 2 * np.sum(nk * s_k)
            )

        return generating_function

    def calc_optical_spectral_function(self, g_t, t_mev, zpl, gamma):
        """
        Calculates the optical spectra A(E).
        """
        energy_mev, optical_spectral_function = self.fourier(
            t_mev, (g_t * np.exp(1j * zpl * t_mev)) * np.exp(-(gamma * np.abs(t_mev)))
        )
        optical_spectral_function = (1 / len(t_mev)) * optical_spectral_function
        return energy_mev, optical_spectral_function

    def luminescence_intensity(self, energy_mev, a_e, zpl):
        """
        Calculates the normalized photoluminescence (PL), L(E)
        """
        a_e = a_e[(energy_mev >= (zpl - 500)) & (energy_mev <= (zpl + 100))]
        energy_mev = energy_mev[
            (energy_mev >= (zpl - 500)) & (energy_mev <= (zpl + 100))
        ]
        l_e = ((energy_mev**3) * a_e) / (
            self.trapezoidal(((energy_mev**3) * a_e), energy_mev)
        )
        return energy_mev, l_e

    def inverse_participation_ratio(self, modes):
        """
        Calculates the IPR (1D array) for each mode.
        """
        p = np.einsum("ijk -> ij", modes**2)
        return 1 / np.einsum("ij -> i", p**2)


def calculate_spectrum(
    gs_structure_path: str | Path,  #
    es_structure_path: str | Path,  # Path to excited state structure
    phonon_band_path: str | Path,  # Path to phonon band data
    phonons_source="Phonopy",  # Options: "VASP" or "Phonopy"
    temperature=0,  # Temperature
    zpl=3339,  # Zero Phonon Line (meV)           3405, algo-3395
    tmax=2000,  # Upper time limit (fs)
    gamma=10,  # Gamma value (meV)
    forces=None,  # (os.path.expanduser("./OUTCAR_T"), os.path.expanduser("./OUTCAR_GS")),  # Options: None or tuple (ES file path, GS file path)
):
    """Compute the luminescence spectrum and related factors.

    Args:
        path_structure_gs (str | os.PathLike): Path to the ground-state structure.
        path_structure_es (str | os.PathLike): Path to the excited-state structure.
        path_phonon_band (str | os.PathLike): Path to phonon band data.
        phonons_source (str, optional): Source/format of the phonon data.
            Supported values are ``"Phonopy"`` and ``"VASP"``. Defaults to
            ``"Phonopy"``.
        temperature (float, optional): Temperature in K used for thermal
            population/broadening. Defaults to ``0`` (no thermal effects).
        zpl (float, optional): Zero-phonon line energy in meV. Defaults to ``3339``.
        tmax (float, optional): Upper time limit in femtoseconds for the time-domain
            evaluation/integration. Defaults to ``2000``.
        gamma (float, optional): Homogeneous broadening (Lorentzian half-width)
            in meV. Defaults to ``10``.
        forces (tuple[str | os.PathLike, str | os.PathLike] | None, optional):
            Paths to force files ``(ES_path, GS_path)`` when forces should be
            read directly (e.g., OUTCARs). If ``None``, forces are obtained from
            the structures/phonons input. Defaults to ``None``.
    """
    pl = Photoluminescence()

    positions_gs, _ = pl.read_structure(Path(gs_structure_path).absolute().as_posix())
    positions_es, elements_es = pl.read_structure(
        Path(es_structure_path).absolute().as_posix()
    )

    if phonons_source == "Phonopy":
        masses, freqs, modes = pl.read_phonons_phonopy(
            Path(phonon_band_path).absolute().as_posix()
        )
        freqs = freqs[: int(freqs.shape[0] / 2)]
        modes = modes[: int(modes.shape[0] / 2), ...]
    else:
        masses, freqs, modes = pl.read_phonons_vasp(
            os.path.expanduser(phonon_band_path), elements_es
        )

    freqs[freqs < 0.1] = 0.0
    energy_k = constants.h.to("meV / THz").magnitude * freqs
    energy_k[energy_k == 0] = 0.00001

    if forces is not None:
        forces_es = np.loadtxt("./forces/FORCES1")  # pl.ReadForces(forces[0])
        forces_gs = pl.read_forces(forces[1])
        qk = pl.config_coordinates_f(masses, forces_es, forces_gs, modes, energy_k)
    else:
        qk = pl.config_coordinates(masses, positions_es, positions_gs, modes)

    partial_hr = pl.calc_partial_hr(freqs, qk)

    max_energy = 2.5 * zpl if zpl != 0 else 5000

    tmax_mev = pl.time_scaling(tmax)
    energy_mev_positive = pl.get_direct_mesh(0, max_energy, tmax_mev)
    specfun_energy = pl.spectral_function(partial_hr, energy_k, energy_mev_positive)

    t_mev, s_t, s_t_exact = pl.fourier_spectral_function(
        partial_hr, energy_k, specfun_energy, energy_mev_positive
    )

    g_t = pl.calc_generating_function(
        partial_hr, s_t, t_mev, energy_k, energy_mev_positive, temperature
    )

    energy_mev, a_e = pl.calc_optical_spectral_function(g_t, t_mev, zpl, gamma)

    energy_mev, l_e = pl.luminescence_intensity(energy_mev, a_e, zpl)

    t_fs = pl.time_scaling(t_mev, reverse=True)

    ipr = pl.inverse_participation_ratio(modes)

    """
    Analyses of results.
    """
    plt.scatter(energy_k, partial_hr, s=5, marker="s")
    plt.title(f"Total HR factor = {np.sum(partial_hr)}")
    plt.xlabel("Phonon Energy (meV)")
    plt.ylabel("$S_k$")
    plt.show()

    specfun_energy = specfun_energy[energy_mev_positive <= (max(energy_k) + 36)]
    energy_mev_positive = energy_mev_positive[
        energy_mev_positive <= (max(energy_k) + 36)
    ]
    s_t = s_t[(t_fs >= 0) & (t_fs <= 550)]
    s_t_exact = s_t_exact[(t_fs >= 0) & (t_fs <= 550)]
    g_t = g_t[(t_fs >= 0) & (t_fs <= 550)]
    t_fs = t_fs[(t_fs >= 0) & (t_fs <= 550)]

    plt.plot(energy_mev_positive, specfun_energy)
    plt.xlabel("Phonon Energy (meV)")
    plt.ylabel("S(E)")
    plt.show()

    plt.plot(t_fs, np.real(s_t), label="Real", color="red")
    plt.plot(t_fs, np.imag(s_t), label="Imaginary", color="blue")
    plt.legend()
    plt.xlabel("Time (fs)")
    plt.ylabel("S(t)")
    plt.show()

    plt.plot(t_fs, np.real(g_t), label="Real", color="red")
    plt.plot(t_fs, np.imag(g_t), label="Imaginary", color="blue")
    plt.legend()
    plt.xlabel("Time (fs)")
    plt.ylabel("G(t)")
    plt.show()

    plt.plot(energy_mev, np.log(np.abs(l_e)))
    plt.xlabel("Photon Energy (meV)")
    plt.ylabel("PL")
    # plt.xlim(1700, 2000)
    plt.show()

    plt.scatter(energy_k, ipr, s=5, marker="s")
    plt.title("Inverse Participation Ratio")
    plt.xlabel("Phonon Energy (meV)")
    plt.ylabel("IPR")
    plt.show()

    return (
        positions_gs,
        positions_es,
        qk,
        (energy_k, partial_hr),
        (energy_mev_positive, specfun_energy),
        (t_fs, s_t, s_t_exact),
        (g_t),
        (energy_mev, a_e),
        (l_e),
        ipr,
    )

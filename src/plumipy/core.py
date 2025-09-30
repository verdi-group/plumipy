import numpy as np
import matplotlib.pyplot as plt
import os

class ReadFiles:

  def __init__(self):
    pass

  def ReadStructure(self, path):

    """
    Input:   1. path - Location of POSCAR or CONTCAR file as a string.

    Outputs: 1. Position vectors of all the atoms as numpy array of shape (total number of atoms, 3), where 3 is
                the x,y and z space coordinates.
             2. Dictionary of atomic species and there corresponding number of atoms.
    """
    with open(path,'r') as file:

      lines = file.readlines()

      scaling_factor = float(lines[1].strip())
      lattice_vectors = [lines[i].strip().split() for i in range(2,5)]
      lattice_vectors = scaling_factor*(np.array(lattice_vectors).astype(float))

      atomic_species = lines[5].strip().split()
      number_of_atoms = np.array(lines[6].strip().split()).astype(int)
      tot = sum(number_of_atoms)

      lattice_type = lines[7].strip()

      atomic_positions = [lines[i].strip().split() for i in range(8,8+tot)]
      atomic_positions = np.array(atomic_positions).astype(float)
      atoms = dict(zip(atomic_species, number_of_atoms))
      
      if lattice_type != "Direct":
        latticeInv = np.linalg.inv(lattice_vectors.T)
        Rd = np.array([np.dot(latticeInv,vec) for vec in atomic_positions])
        atomic_positions = Rd

      atomic_positions[atomic_positions > 0.99] -= 1
      atomic_positions = np.dot(atomic_positions, lattice_vectors)
      return (atomic_positions, atoms)
      
      

  def ReadPhononsPhonopy(self, path):

    """
    Input:   1. path: Location of band.yaml file as a string.

    Outputs: 1. Atomic_masses is a 1D array of masses (AMU) of each atom in the same sequence as
                that of Atomic positions in previous function.
             2. Phonon frequencies (THz) as a 1D at Gamma point. Length of the array = number of normal modes.
             3. Eigenvectors corresponding to the phonon frequencies as a 3D array of
                shape (number of normal mode, number of atoms, 3), where 3 is the x,y and z coordinates.
    """
    with open(path,'r') as file:
      lines = [ts.strip() for ts in file]

    atomic_masses = []
    freqs = []
    normal_modes = []
    with open(path,'r') as file:
      for line in file:
        if "mass:" in line:
          atomic_masses.append(line.split()[1])
    atomic_masses = np.array(atomic_masses).astype(float)
    total_atoms = len(atomic_masses)
    with open(path,'r') as file:
      line_number = -1
      for line in file:
        line_number += 1
        if "frequency:" in line:
          freqs.append(float(line.split()[1]))
          ev_internal = []
          for i in range(line_number+3,line_number + 4*total_atoms + 2,4):
            xyz = [lines[i+j].split()[2] for j in range(3)]
            ev_internal.append(xyz)
          normal_modes.append(ev_internal)
    freqs = np.array(freqs).astype(float)
    freqs[freqs<0] = 0
    normal_modes = np.array([[[float(x.strip(',')) for x in sublist] for sublist in outer] for outer in normal_modes])
    return atomic_masses, freqs, normal_modes

  def ReadPhononsVasp(self, path, atoms):

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


    with open(path, 'r') as file:
          lines = [line.strip() for line in file]

          index = lines.index("Mass of Ions in am")
          atomic_masses = lines[index + 1].split()[2:]
          atomic_masses = np.array(atomic_masses).astype(float)

          index_init = lines.index("Eigenvectors and eigenvalues of the dynamical matrix")
          index_final = lines.index("ELASTIC MODULI CONTR FROM IONIC RELAXATION (kBar)")


          for i in range(index_init, index_final + 1):
            internal_modes = []
            if "THz" in lines[i]:
              freqs.append(lines[i].split()[lines[i].split().index("THz") - 1])
              internal_modes = [lines[j].split() for j in range(i + 2, i + 2 + total_atoms)]
              normal_modes.append(internal_modes)

    atomic_masses = np.repeat(atomic_masses, number_of_atoms)
    freqs = np.array(freqs).astype(float)
    sort = np.argsort(freqs)
    freqs = freqs[sort]
    normal_modes = np.array(normal_modes).astype(float)[...,3:]
    normal_modes = normal_modes[sort]
    return atomic_masses, freqs, normal_modes

  def ReadForces(self, path):

    """
    Reads and stores the Forces (eV/Angstrom) on each atom from the OUTCAR file and returns a 2D array.
    """
    forces = []
    start_collecting = False
    lines_buffer = []

    with open(path, 'r') as file:
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
    forces = forces[:,3:]
    return forces



class Photoluminescence(ReadFiles):

  def __init__(self):

    """
    Define all the variables by reading the input files like POSCAR_GS/CONTCAR_GS, POSCAR_ES/CONTCAR_ES, and band.yaml.
    """
    super().__init__()

  def IV(self, iv_low, iv_high, rv_high):

    """
    This function can be used to obtain a 1D time array with equal intervals.

    iv: Independent Variable;
    rv: Reciprocal Variable.

    Inputs: Min max values of independent variable, and
    max value of rv required by the user.

    div: Minimum resolution of iv.

    Output: 1D array of independent variable (usually time in this case).
    """

    div = (2*np.pi)/(2*rv_high)
    return np.arange(iv_low, iv_high, div)

  def Fourier(self, iv, function):

    """
    Discrete Fourier transform (DFT) using FFT algorithm.
    Here, DFT is approximated as Continuous Fourier Transform.

    Inputs: Independent variable and the function to be Fourier Transformed.

    Outputs: 1D array of reciprocal variable (generally Energy or frequency in this case) on which
    FFT has been performed and 1D array of the DFT result.
    """
    div = iv[1] - iv[0]
    rv = 2*np.pi*np.fft.fftfreq(len(iv),div)
    sort = np.argsort(rv)
    rv = rv[sort]
    DFT = np.fft.fft(function)[sort]
    DFT = div*DFT*np.exp(-1j*rv*iv[0])
    return rv, DFT

  def InverseFourier(self, iv, function):

    """
    Inverse Discrete Fourier transform (IDFT) using FFT algorithm.
    Here, DFT is approximated as Continuous Fourier Transform.
    """
    div = iv[1] - iv[0]
    rv = 2*np.pi*np.fft.fftfreq(len(iv),div)
    sort = np.argsort(rv)
    rv = rv[sort]
    IDFT = np.fft.ifft(function)[sort]
    IDFT = div*IDFT*np.exp(1j*rv*iv[0])
    return rv, IDFT

  def Trapezoidal(self, integrand, iv, equally_spaced = True):

    """
    Calculates the integral using Trapezoidal Rule.

    Inputs: integrand and iv are arrays of same dimension. equally_spaced: determines whether the method should integrate using
    equally spaced or unequally spaced intervals.

    Output: Integration result.
    """
    div = iv[1] - iv[0]
    return (div/2)*(np.sum(integrand[1:-1]) + integrand[0] + integrand[-1]) if equally_spaced \
    else np.sum(np.array([((iv[i+1] - iv[i])/2)*(integrand[i+1] + integrand[i]) for i in range(len(iv)-1)]))

  def FreqToEnergy(self, freqs):

    """Coversion of frequencies (THz) to Energy (meV)."""

    return 4.13566*freqs


  def TimeScaling(self, t, reverse = False):

    """
    Changes time array t from femtoseconds to meV^-1. This is a necesaary step after initializing time through IV
    function in order to maintain consistency in units while performing Fourier Transform.
    """
    return t/658.2119 if reverse == False else t*658.2119

  def Lorentzian(self, x, x0, sigma):

    """
    Used to fit Dirac-Delta as Lorentzian function, where sigma = 6 has units of meV.
    The factor of 0.8 multiplying sigma is to make this function have similarities to
    Gaussian for same standard deviation, sigma.
    """
    return ((1/np.pi)*(sigma*0.8))/(((sigma*0.8)**2) + ((x - x0)**2))

  def Gaussian(self, x, x0, sigma):

    """
    Gaussian fit for Dirac-Delta with sigma = 6 (meV) as standard deviation.
    """
    return (1/np.sqrt(2*np.pi*(sigma**2)))*np.exp(-((x-x0)**2)/(2*(sigma**2)))

  def ConfigCoordinates(self, masses, R_es, R_gs, modes):

    """
    Calculates the qk factor (AMU^0.5-Angstrom) for different normal modes as a 1D array of
    length = total number of normal modes.
    """
    masses = np.sqrt(masses)
    R_diff = R_es - R_gs
    mR_diff = np.array([masses[i]*R_diff[i,:] for i in range(len(masses))])
    qk = np.array([np.sum(mR_diff*modes[i,:,:]) for i in range(modes.shape[0])])
    return qk

  def ConfigCoordinatesF(self, masses, F_es, F_gs, modes, Ek):

    """
    Calculates the qk factor (AMU^0.5-Angstrom) for different normal modes as a 1D array of
    length = total number of normal modes. This function uses forces on atoms rather than their position vectors
    as used in previous function.
    """
    masses = np.sqrt(masses)
    F_diff = F_es - F_gs
    mF_diff = np.array([(1/masses[i])*F_diff[i,:] for i in range(len(masses))])
    qk = np.array([np.sum(mF_diff*modes[i,:,:]) for i in range(modes.shape[0])])
    qk = (1/Ek**2)*qk*4180.069
    return qk

  def PartialHR(self, freqs, qk):

    """
    Calculates the Sk (unit less) as a 1D array of length equal to total number of normal modes.
    """
    return 2*np.pi*freqs*(qk**2)*0.166/(2*1.05457)

  def SpectralFunction(self, Sk, Ek, E_meV_positive, sigma = 6, Lorentz = False):

    """
    Calculates S(hbar_omega) or S(E) (unit less) by using Gaussian or Lorentzian fit
    for Direc-Delta with sigma = 6 meV by default.

    Ek: Normal mode phonon energies.
    """
    self.sigma = sigma
    if Lorentz == False:
      S_E = np.array([np.dot(Sk,self.Gaussian(i,Ek,sigma)) for i in E_meV_positive])
    else:
      S_E = np.array([np.dot(Sk,self.Lorentzian(i,Ek,sigma)) for i in E_meV_positive])
    return S_E

  def FourierSpectralFunction(self, Sk, Ek, S_E, E_meV_positive):

    """
    Calculates the Fourier transform of S(E) which is equal to S(t).
    """
    t_meV, S_t = self.Fourier(E_meV_positive, S_E)
    S_t_exact = np.array([np.dot(Sk,np.exp(-1j*Ek*i)) for i in t_meV])
    return t_meV, S_t, S_t_exact

  def GeneratingFunction(self, Sk, S_t, t_meV, Ek, E_meV_positive, T):

    """
    Calculates the generating function G(t).
    """
    if T == 0.0:
      G_t = np.exp((S_t) - (np.sum(Sk)))
    else:
      Kb = 8.61733326e-2 # Boltzmann constant in meV/k
      nk = 1/((np.exp(Ek/(Kb*T))) - 1)
      C_E = np.array([np.dot(nk*Sk,self.Gaussian(i,Ek,self.sigma)) for i in E_meV_positive])
      C_t = self.Fourier(E_meV_positive, C_E)[1]
      C_t_inv = self.InverseFourier(E_meV_positive, C_E)[1]
      G_t = np.exp((S_t) - (np.sum(Sk)) + C_t + C_t_inv - 2*np.sum(nk*Sk))
    return G_t

  def OpticalSpectralFunction(self, G_t, t_meV, zpl, gamma):

    """
    Calculates the optical spectra A(E).
    """
    E_meV, A_E =  self.Fourier(t_meV, (G_t*np.exp(1j*zpl*t_meV))*np.exp(-(gamma*np.abs(t_meV))))
    A_E = (1/len(t_meV))*A_E
    return E_meV, A_E

  def LuminescenceIntensity(self, E_meV, A_E, zpl):

    """
    Calculates the normalized photoluminescence (PL), L(E)
    """
    A_E = A_E[(E_meV >= (zpl - 500)) & (E_meV <= (zpl + 100))]
    E_meV = E_meV[(E_meV >= (zpl - 500)) & (E_meV <= (zpl + 100))]
    L_E = ((E_meV**3)*A_E)/(self.Trapezoidal(((E_meV**3)*A_E), E_meV))
    return E_meV, L_E

  def InverseParticipationRatio(self, modes):

    """
    Calculates the IPR (1D array) for each mode.
    """
    p = np.einsum("ijk -> ij", modes**2)
    IPR = 1/np.einsum("ij -> i", p**2)
    return IPR
  


def CalculateSpectrum(
  path_structure_gs = os.path.expanduser("./CONTCAR_GS"),  # Path to ground state structure
  path_structure_es = os.path.expanduser("./POSCAR100"),  # Path to excited state structure
  phonons_source = "Phonopy",  # Options: "VASP" or "Phonopy"
  path_phonon_band = os.path.expanduser("./band.yaml"),  # Path to phonon band data
  temperature = 0, # Temperature
  zpl = 3339,  # Zero Phonon Line (meV)           3405, algo-3395
  tmax = 2000,  # Upper time limit (fs)
  gamma = 10,  # Gamma value (meV)
  forces = None #(os.path.expanduser("./OUTCAR_T"), os.path.expanduser("./OUTCAR_GS")),  # Options: None or tuple (ES file path, GS file path)
):

    """
    Calculates all factors step by step.
    """
    pl = Photoluminescence()

    R_gs, atoms_gs = pl.ReadStructure(path_structure_gs)
    R_es, atoms_es = pl.ReadStructure(path_structure_es)

    if phonons_source == "Phonopy":
      masses, freqs, modes = pl.ReadPhononsPhonopy(path_phonon_band)
      freqs = freqs[:int(freqs.shape[0]/2)]
      modes = modes[:int(modes.shape[0]/2),...]
    else:
      masses, freqs, modes = pl.ReadPhononsVasp(path_phonon_band, atoms_es)

    freqs[freqs < 0.1] = 0.0
    Ek = pl.FreqToEnergy(freqs)
    Ek[Ek == 0] = 0.00001

    if forces != None:
      F_es = np.loadtxt("./forces/FORCES1") #pl.ReadForces(forces[0])
      F_gs = pl.ReadForces(forces[1])
      qk = pl.ConfigCoordinatesF(masses, F_es, F_gs, modes, Ek)
    else:
      qk = pl.ConfigCoordinates(masses, R_es, R_gs, modes)

    Sk = pl.PartialHR(freqs, qk)

    if zpl != 0:
      Emax = 2.5*zpl
    else:
      Emax = 5000
    tmax_meV = pl.TimeScaling(tmax)
    E_meV_positive = pl.IV(0, Emax, tmax_meV)
    S_E = pl.SpectralFunction(Sk, Ek, E_meV_positive)

    t_meV, S_t, S_t_exact = pl.FourierSpectralFunction(Sk, Ek, S_E, E_meV_positive)

    G_t = pl.GeneratingFunction(Sk, S_t, t_meV, Ek, E_meV_positive, temperature)


    E_meV, A_E = pl.OpticalSpectralFunction(G_t, t_meV, zpl, gamma)



    E_meV, L_E = pl.LuminescenceIntensity(E_meV, A_E, zpl)



    t_fs = pl.TimeScaling(t_meV, reverse = True)

    IPR = pl.InverseParticipationRatio(modes)



    """
    Analyses of results.
    """
    plt.scatter(Ek, Sk, s = 5, marker = "s")
    plt.title(f"Total HR factor = {np.sum(Sk)}")
    plt.xlabel("Phonon Energy (meV)")
    plt.ylabel("$S_k$")
    plt.show()

    S_E = S_E[E_meV_positive <= (max(Ek) + 36)]
    E_meV_positive = E_meV_positive[E_meV_positive <= (max(Ek) + 36)]
    S_t = S_t[(t_fs >= 0) & (t_fs <= 550)]
    S_t_exact = S_t_exact[(t_fs >= 0) & (t_fs <= 550)]
    G_t = G_t[(t_fs >= 0) & (t_fs <= 550)]
    t_fs = t_fs[(t_fs >= 0) & (t_fs <= 550)]


    plt.plot(E_meV_positive, S_E)
    plt.xlabel("Phonon Energy (meV)")
    plt.ylabel("S(E)")
    plt.show()

    plt.plot(t_fs, np.real(S_t), label = "Real", color = "red")
    plt.plot(t_fs, np.imag(S_t), label = "Imaginary", color = "blue")
    plt.legend()
    plt.xlabel("Time (fs)")
    plt.ylabel("S(t)")
    plt.show()


    plt.plot(t_fs, np.real(G_t), label = "Real", color = "red")
    plt.plot(t_fs, np.imag(G_t), label = "Imaginary", color = "blue")
    plt.legend()
    plt.xlabel("Time (fs)")
    plt.ylabel("G(t)")
    plt.show()

    plt.plot(E_meV, np.log(np.abs(L_E)))
    plt.xlabel("Photon Energy (meV)")
    plt.ylabel("PL")
    # plt.xlim(1700, 2000)
    plt.show()


    plt.scatter(Ek, IPR, s = 5, marker = "s")
    plt.title(f"Inverse Participation Ratio")
    plt.xlabel("Phonon Energy (meV)")
    plt.ylabel("IPR")
    plt.show()

    return (R_gs, R_es, qk, (Ek, Sk), (E_meV_positive, S_E), (t_fs, S_t, S_t_exact), (G_t), (E_meV, A_E), (L_E), IPR)


(R_gs, R_es, qk, (Ek, Sk), (E_meV_positive, S_E), (t_fs, S_t, S_t_exact), (G_t), (E_meV, A_E), (L_E), IPR) = CalculateSpectrum(
  path_structure_gs = os.path.expanduser("./CONTCAR_GS"),  # Path to ground state structure
  path_structure_es = os.path.expanduser("./POSCAR100"),  # Path to excited state structure
  phonons_source = "Phonopy",  # Options: "VASP" or "Phonopy"
  path_phonon_band = os.path.expanduser("./band.yaml"),  # Path to phonon band data
  temperature = 0, # Temperature
  zpl = 2000,  # Zero Phonon Line (meV)           3405, algo-3395
  tmax = 2000,  # Upper time limit (fs)
  gamma = 10,  # Gamma value (meV)
  forces = None #(os.path.expanduser("./OUTCAR_T"), os.path.expanduser("./OUTCAR_GS")),  # Options: None or tuple (ES file path, GS file path)
)





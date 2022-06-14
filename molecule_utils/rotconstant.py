import numpy as np
import scipy as sp
import mdtraj as md
## Define some constants
AU2S = sp.constants.physical_constants['reduced Planck constant'][0]/sp.constants.physical_constants['Hartree energy'][0] # s = \hbar/Hartree
AMU2UMA = sp.constants.physical_constants['unified atomic mass unit'][0]/sp.constants.electron_mass
B2A = sp.constants.physical_constants['Bohr radius'][0]/sp.constants.angstrom
HZ2MHZ = 1e-6


def centre_of_mass(crd, atmass):
    """Returns the centre of mass of the coordinates
    
    Arguments:
        crd {np.array(N,3)} -- Atomic coordinates
        atmass {np.array(N)} -- Atomic masses

    Returns:
        com {np.array(3) -- centre of mass coordinate
    """
    # center of mass
    return np.average(crd, axis=0, weights=atmass)

def inertia(crd, atmass):
    """
    Computes and returns the inertia axis 
    TODO add checks on arguments
    Arguments:
        crd {np.array(N,3)} -- Atomic coordinates
        atmass {np.array(N)} -- Atomic masses

    Returns:
        (eigval, eigvec) tuple of (np.array(3), np.array(3,3))
    """
    assert crd.shape[0] == atmass.shape[0]
    assert crd.shape[1] == 3
    ine_tensor = np.zeros((3, 3))
    # inertia tensor
    ine_tensor[0, 0] = (atmass * (crd[:, 1]**2 + crd[:, 2]**2)).sum()
    ine_tensor[1, 1] = (atmass * (crd[:, 0]**2 + crd[:, 2]**2)).sum()
    ine_tensor[2, 2] = (atmass * (crd[:, 0]**2 + crd[:, 1]**2)).sum()
    ine_tensor[0, 1] = ine_tensor[1, 0] = (-atmass * (crd[:, 0] * crd[:, 1])).sum() 
    ine_tensor[0, 2] = ine_tensor[2, 0] = (-atmass * (crd[:, 0] * crd[:, 2])).sum()
    ine_tensor[1, 2] = ine_tensor[2, 1] = (-atmass * (crd[:, 1] * crd[:, 2])).sum()
    # Diagonalization
    eigval, eigvec = np.linalg.eigh(ine_tensor)
    return (eigval, eigvec)

def get_B(crd, atmass):
    """
    Computes and returns the rotational constants in MHz
    Arguments:
        crd {np.array(N,3)} -- Atomic coordinates in bohr
        atmass {np.array(N)} -- Atomic masses
    Returns:
        rotational constants in MHz
    """    
    cntrmass = centre_of_mass(crd, atmass)
    new_crd = (crd - cntrmass)
    iner, _ = inertia(new_crd, atmass*AMU2UMA)
    return 1/(4*np.pi*iner)/AU2S*HZ2MHZ

def gettrajB(traj, mask=None):
    """
    Computes the rotational contants of a given trajectory
    Arguments:
        traj {mdtraj.Trajectory} -- mdtraj trajectory
        mask {array} -- list of selected frames
    Returns:
        {array(nframe,3)} -- rotational constants in MHz of each frames
    """
    if mask == None:
        mask = np.ones(traj.n_frames, dtype='bool')
    tmp_itens = md.compute_inertia_tensor(traj[mask])*AMU2UMA/(B2A**2)*100
    bvec = []
    for itns in tmp_itens:
        eigval,_ = np.linalg.eigh(itns, UPLO='L')
        bvec.append(1/(4*np.pi*eigval)/AU2S*HZ2MHZ)
    return np.array(bvec)

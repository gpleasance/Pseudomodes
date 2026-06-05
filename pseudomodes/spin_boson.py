"""
Module for simulating the spin-boson model using the pseudomode method.
"""
import numpy as np
import qutip as qt
from numpy import ndarray
from qutip import Qobj
from scipy.integrate import quad 
from mpmath import zeta
# from numpy.typing import ArrayLike

def coth(w: float):
    return 1./np.tanh(w)


def sd_power(
    w: float | list | ndarray, 
    coup: float, 
    w_cut: float, 
    s: float
) -> float:
    r"""
    Power-law spectral density w/ exponential cutoff

    .. math::

        J(\omega) = \alpha\frac{\omega^s}{\omega^{s-1}_c} e^{-\omega / \omega_c}

    where $\alpha$ is the coupling strength, $\omega_c$ is the cutoff frequency, and $s$ the Ohmicity parameter.

    Parameters
    ----------
    w : float | array_like
        Bath frequencies.
    coup : float
        Dimensionless coupling strength.
    w_cut : float
        Cutoff frequency.
    s : int
        Ohmicity parameter.

    Returns
    -------
    Spectral density at frequencies w. 
    """
    if type(w) == list:
        w = np.array(w)

    return coup * (w**(s) / w_cut**(s-1)) * np.exp(-w/w_cut)


def sd_ud(
    w: float | list | ndarray, 
    coup: float,
    width: float, 
    w_res: float
) -> float:
    r"""
    Underdamped Brownian spectral density. 

    .. math::

        J(\omega) = \frac{2\lambda^2\omega_0\Gamma\omega}{(\omega^2-\omega^2_0)^2 + \Gamma^2\omega^2}
    
    where $\lambda$ is the coupling strength, $\Gamma$ the bath width, and $\omega_0$ the bath resonance frequency.

    Parameters
    ----------
    w : float | array_like
        Bath frequencies.
    coup : float
        Coupling strength.
    width : float
        Bath width.
    w_res : float
        Bath resonance frequency.
    
    Returns
    -------
    Spectral density at frequencies w. 
    """
    if type(w) == list:
        w = np.array(w)

    return (2 * coup ** 2 * w_res * width * w) / ((w**2 - w_res**2) ** 2 + width**2 * w**2)


def bcf_power(
    t: float | list | ndarray,
    T: float,
    coup: float,
    w_cut: float,
    s: float
) -> complex:
    """
    Bath correlation function for power-law spectral density. 
    
    Parameters
    ----------
    t : float | array_like
        Time or array of times t.
    T : float
        Bath temperature.
    coup : float
        Dimensionless coupling strength.
    w_cut : float
        Cutoff frequency.
    s : int
        Ohmicity parameter.
    
    Returns
    -------
    Bath correlation function at time(s) t.
    """
    beta = 1./T
    if type(t) == list:
        t = np.array(t)

    return (1/np.pi) * coup * w_cut**(1-s) * beta**(-(s+1)) * (zeta(s+1,(1+beta*w_cut-1.0j*w_cut*t)/(beta*w_cut)) + 
            zeta(s+1,(1+1.0j*w_cut*t)/(beta*w_cut)))


def bcf_ud(
    t: float | list | ndarray,
    T: float,
    coup: float,
    width: float, 
    w_res: float,
    Nk: int,
) -> complex: 
    """
    Bath correlation function for underdamped Brownian spectral density. 

    Parameters
    ----------
    t : float | array_like
        Time or array of times t.
    T : float
        Bath temperature.
    coup : float
        Coupling strength.
    width : float
        Bath width.
    w_res : float
        Resonance frequency,
    Nk : int
        Number of Matsubara terms.

    Returns
    -------
    Bath correlation function at time(s) t.
    """
    beta = 1./T
    Om = np.sqrt(w_res ** 2 - width **2 / 4)
    if type(t) == list:
        t = np.array(t)

    # Check w_res > width / 2
    if not w_res > 0.5 * width:
        raise ValueError(f'Require w_res > (width / 2) for underdamped spectral density.')

    # Matsubara terms 
    def M(t, k):
        vk = 2 * np.pi * k / beta
        return np.real( ((-1j * 2) / beta) * sd_ud(-1j*vk, coup=coup, width=width, w_res=w_res) * np.exp(-vk * np.abs(t)) )

    return ((coup **2 * w_res) / (2 * Om)) * ((coth(0.5 * beta * (Om - 0.5j * width)) + 1) * np.exp(-1j*Om*t - 0.5 * width * np.abs(t)) \
                                              +  (coth(0.5 * beta * (Om + 0.5j * width)) - 1) * np.exp(1j*Om*t - 0.5 * width * np.abs(t))) \
                                              + np.sum([M(t, k) for k in range(1, Nk+1)], axis=0)


def dephase_integrand(
    w: float | list | ndarray, 
    t: float | list | ndarray,
    T: float,
    sd_type: str,
    **kwargs
) -> float:
    """
    Returns integrand for computing dephasing integral.  
    
    Parameters
    ----------
    w : array_like
        Frequency or array of frequencies w.
    t : float | array_like
        Time or array of times t. 
    T : float
        Bath temperature. 
    sd_type : str
        Spectral density type - use either 'PowerLaw' or 'Underdamped'.  

    Returns
    -------
    Integrand evaluated at frequencies w and times t.
    """
    beta = 1./T
    if t == type(list):
        t = np.array(t)
    if w == type(list):
        w = np.array(w)

    # Error checks
    expected_keys = {'PowerLaw' : {'coup', 'w_cut', 's'}, 'Underdamped' : {'coup', 'width', 'w_res'}}
    if sd_type not in expected_keys:
        raise ValueError(f"Invalid spectral density type '{sd_type}'. Use either 'PowerLaw' or 'Underdamped'.")

    missing = expected_keys[sd_type] - kwargs.keys()
    if missing:
        raise KeyError(f"Missing arguments for spectral density type '{sd_type}': {missing}")

    # include w = 0 case
    if sd_type == 'PowerLaw':
        return - (4/np.pi) * (sd_power(w, kwargs['coup'], kwargs['w_cut'], kwargs['s'])/w**2) * coth(0.5*beta*w) * (1 - np.cos(w*t))
    elif sd_type == 'Underdamped':
        return - (4/np.pi) * (sd_ud(w, kwargs['coup'], kwargs['width'], kwargs['w_res'])/w**2) * coth(0.5*beta*w) * (1 - np.cos(w*t))


def dephase_exp(
    tlist: list | ndarray, 
    init_bloch: list | ndarray,
    eps: float,
    T: float,
    sd_type: str,
    **kwargs
) -> tuple[ndarray, ...]:
    """
    Returns bloch vector for pure dephasing evolution.
    
    Parameters
    ----------
    tlist: array_like
        Array of times t.
    init_bloch: array_like, shape (3,)
        Initial bloch vector.  
    eps: float
        TLS energy gap. 
    T: float
        Bath temperature.
    sd_type: str
        Spectral density type - use either 'PowerLaw' or 'Underdamped'.
    
    Returns
    -------
    Time-evolved bloch vector for each t in tlist. 
    """
    # Error checks
    expected_keys = {'PowerLaw' : {'coup', 'w_cut', 's'}, 'Underdamped' : {'coup', 'width', 'w_res'}}
    if sd_type not in expected_keys:
        raise ValueError(f"Invalid spectral density type '{sd_type}'. Use either 'PowerLaw' or 'Underdamped'.")

    missing = expected_keys[sd_type] - kwargs.keys()
    if missing:
        raise KeyError(f"Missing arguments for spectral density type '{sd_type}': {missing}")

    if sd_type == 'PowerLaw':
        integral = lambda t: quad(lambda w: dephase_integrand(w, t, T, sd_type=sd_type, coup=kwargs['coup'], w_cut=kwargs['w_cut'], s=kwargs['s']), 0, np.inf)[0]
    elif sd_type == 'Underdamped':
        integral = lambda t: quad(lambda w: dephase_integrand(w, t, T, sd_type=sd_type, coup=kwargs['coup'], width=kwargs['width'], w_res=kwargs['w_res']), 0, np.inf)[0]

    # Initial coherence in TLS energy eigenbasis
    rho_10 = 0.5 * (init_bloch[0] + 1j * init_bloch[1])
    return (np.array([2 * np.real(np.exp(1j*eps*t)*np.exp(integral(t))*rho_10) for t in tlist]), 
            np.array([2 * np.imag(np.exp(1j*eps*t)*np.exp(integral(t))*rho_10) for t in tlist]),
            np.array([init_bloch[2] for t in tlist]))


def pm_liouvillian(
    H_sys: Qobj,
    coup_op: Qobj,
    coup_sys_pm: list | ndarray,
    pm_params: ndarray[complex],
    pm_dims: list[int]
) -> Qobj:
    """
    Constructs Liouvillian for PMME parameterized by {gp, Z}, where 

    Z = xi - 0.5i * gm

    Parameters
    ----------
    sys : Qobj, 
        System Hamiltonian. 
    coup_op : Obj,
        System coupling operator.
    sys_pm_coup : array_like, shape (n,)
        1D array of system-pm couplings. 
    pm_params : array_like, shape (n,n)
        Complex 2D array containing internal pm couplings xi, and decay rate gm.  
    pm_dims : list
        List of local pm dimensions. 

    Returns
    -------
    liouvillian : Qobj
    """
    # Error checks
    if not len(coup_sys_pm) == len(pm_params):
        raise ValueError(f"Inner dimensions of coup_sys_pm and pm_params must match.")
    if not np.allclose(pm_params.T, pm_params, atol=1e-12):
        raise ValueError(f"pm_params must be a complex symmmetric matrix.")

    sys_dims = H_sys.dims[0][0]
    H_sys = qt.tensor([H_sys] + [qt.identity(n) for n in pm_dims])
    coup_sys = qt.tensor([coup_op] + [qt.identity(n) for n in pm_dims])

    b_ops = []
    for n in range(len(pm_dims)): 
        b_ops.append(qt.tensor([qt.identity(sys_dims)] + [qt.destroy(N) if n == idx else qt.identity(N) for idx, N in enumerate(pm_dims)]))

    H_pm, H_int = 0, 0

    xi = np.real(0.5 * (pm_params + np.conj(pm_params)))
    gm = np.real(1j * (pm_params - np.conj(pm_params)))

    for i in range(len(pm_dims)):
        H_int += coup_sys_pm[i] * coup_sys * (b_ops[i] + b_ops[i].dag())
        for j in range(len(b_ops)):
            H_pm += xi[i][j] * b_ops[i].dag() * b_ops[j] 
    
    H = H_sys + H_pm + H_int

    # Construct GKSL part of dissipator
    result = qt.liouvillian(H=H, c_ops=[np.sqrt(decay) * b_ops[i] if decay > 0 else 0 * b_ops[i] for i, decay in enumerate(np.diag(gm))])

    # Add non-GKSL part (negative decay rates)
    for i, decay in enumerate(np.diag(gm)):
        if decay < 0:
            result -= qt.lindblad_dissipator(a=np.sqrt(np.abs(decay)) * b_ops[i])
    
    return result
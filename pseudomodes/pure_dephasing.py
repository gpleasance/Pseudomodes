import numpy as np
from scipy.integrate import quad 
from mpmath import zeta


def coth(w: float):
    return 1./np.tanh(w)


def sd_power(
    w: float, 
    alph: float, 
    w_cut: float, 
    s: float
) -> float:
    """
    Power-law spectral density w/ exponential cutoff.

    Parameters
    ----------
    w: float
        Bath frequencies.
    alph: float
        Dimensionless coupling strength.
    w_cut: float
        Cutoff frequency.
    s: int
        Ohmicity parameter.
    """
    return alph * (w**(s) / w_cut**(s-1)) * np.exp(-w/w_cut)


def bcf_power(
    t: float | list | np.ndarray,
    T: float,
    alph: float,
    w_cut: float,
    s: float
) -> complex:
    """
    Bath correlation function for power-law spectral density. 
    
    Parameters
    ----------
    t: float | ArrayLike
        Time or array of times t.
    T: float
        Bath temperature.
    alph: float
        Dimensionless coupling strength.
    w_cut: float
        Cutoff frequency.
    s: int
        Ohmicity parameter.
    
    Returns
    ----------
    Bath correlation functions at time(s) t.
    """
    beta = 1./T
    return complex((1/np.pi) * alph * w_cut**(1-s) * beta**(-(s+1)) * (zeta(s+1,(1+beta*w_cut-1.0j*w_cut*t)/(beta*w_cut)) + 
            zeta(s+1,(1+1.0j*w_cut*t)/(beta*w_cut))))


def dephase_integrand(
    w: float | list | np.ndarray, 
    t: float | list | np.ndarray,
    T: float,
    alph: float,
    w_cut: float,
    s: float
) -> float:
    """
    Returns integrand for computing dephasing integral.  
    
    Parameters
    ----------
    w: array
        Frequency or array of frequencies w.
    t: float | ArrayLike
        Time or array of times t. 

    Returns
    ----------
    Integrand evaluated at frequency(ies) w and time(s) t.
    """
    beta = 1./T 
    return - (4/np.pi) * (sd_power(w, alph, w_cut, s)/w**2) * coth(0.5*beta*w) * (1 - np.cos(w*t))


def dephase_exp(
    tlist: list | np.ndarray, 
    init_bloch: list | np.ndarray,
    eps: float,
    T: float,
    alph: float,
    w_cut: float,
    s: float
) -> tuple[np.ndarray, ...]:
    """
    Returns bloch vector for pure dephasing evolution.
    
    Parameters
    ----------
    tlist: array
        Array of times t.
    init_bloch: array of shape (3, 1)
        Initial bloch vector.  
    eps: float
        TLS energy gap. 
    
    Returns
    ----------
    Time-evolved bloch vector for each t in tlist
    """
    integral = lambda t: quad(dephase_integrand, 0, np.inf, args=(t, T, alph, w_cut, s))[0]

    # Initial coherence in TLS energy eigenbasis
    rho_10 = 0.5 * (init_bloch[0] + 1j * init_bloch[1])
    return (np.array([2 * np.real(np.exp(1j*eps*t)*np.exp(integral(t))*rho_10) for t in tlist]), 
            np.array([2 * np.imag(np.exp(1j*eps*t)*np.exp(integral(t))*rho_10) for t in tlist]),
            np.array([init_bloch[2] for t in tlist]))
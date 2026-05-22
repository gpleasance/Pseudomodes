import numpy as np
from numpy import linalg
from scipy.integrate import quad 
from scipy.linalg import expm
from mpmath import zeta
# from numpy.typing import ArrayLike


def coth(w: float):
    return 1./np.tanh(w)


def sd_power(
    w: float | list | np.ndarray, 
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
    w: float | list | np.ndarray, 
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
    t: float | list | np.ndarray,
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
    t: float | list | np.ndarray,
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
    w: float | list | np.ndarray, 
    t: float | list | np.ndarray,
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
    tlist: list | np.ndarray, 
    init_bloch: list | np.ndarray,
    eps: float,
    T: float,
    sd_type: str,
    **kwargs
) -> tuple[np.ndarray, ...]:
    """
    Returns bloch vector for pure dephasing evolution.
    
    Parameters
    ----------
    tlist: array_like
        Array of times t.
    init_bloch: array of shape (3, 1)
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


def pm_parameters(
    rk: list | np.ndarray, 
    zk: list | np.ndarray,
    info = False
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Extracts the pseudomode parameters from a fitted BCF of the form 

    .. math:: 
        C^E(t) = -i\sum^N_{k=1}r_ke^{-iz_kt}
    
    where 'r_k' are the fitted coefficients, and 'z_k' the fitted exponents. 

    Parameters
    ----------
    rk : array_like, shape (n,)
        Fitted complex coefficients r_k.
    zk : array_like, shape (n,) or shape (n,n)
        Fitted complex exponents z_k.
    info : bool, optional
        Provides information on PM parameters if set to True. Default is False.

    Returns
    -------
    g : ndarray

    xi : ndarray 

    gm : ndarray

    info : dict
    """
    # Error checks
    if not len(rk) == len(zk):
        raise ValueError(f"Dimensions of rk and zk must match.")
    
    # Check if list or array - if list, convert to array
    if type(rk) == list:
        rk = np.array(rk) 
    elif not rk.shape == (len(rk), ):
        raise ValueError(f"The coefficients rk must be input as a list or 1D array.")

    if type(zk) == list:
        zk = np.array(zk)
    else:
        if zk.shape == (len(zk), len(zk)):
            if np.allclose(zk, np.diag(np.diag(zk)), atol=1e-12):
                Lmb = zk
            else:
                raise ValueError(f"If zk is input as a square matrix, it must be diagonal.")
        elif zk.shape == (len(zk), ):
            Lmb = np.diag(zk)
        else:
            raise ValueError(f"The exponents zk must be input as a list, 1D array or a diagonal matrix.")

    r = np.sqrt(-1j*rk)
    c0 = np.sqrt(r @ r)
    r = r / c0  # convert to normalized 1D array 

    mu = r @ np.conj(r)  # r.dot(np.conj(r))
    alpha = (1/(np.sqrt(mu*mu-1))) * np.arctanh(np.sqrt((mu-1)/(mu+1))) 

    theta = alpha * np.sqrt(mu*mu-1)

    # exp of R matrix
    exp_R = np.identity(len(r)) + (1/(mu*mu-1)) * ((np.cosh(theta)*(2*mu-1)-mu) * np.outer(np.conj(r), r)
                                            - (mu - np.cosh(theta)) * np.outer(r, np.conj(r))
                                            - (np.cosh(theta)-1) * np.outer(r, r)
                                            - (np.cosh(theta)-1) * np.outer(np.conj(r), np.conj(r)))

    gm0 = -2 * np.imag(exp_R.dot(Lmb.dot(exp_R.T)))
    xi0 = np.real(exp_R.dot(Lmb.dot(exp_R.T)))

    S = linalg.eig(gm0)[1].T  # obtain S from eigenvectors of gm0

    # PM parameters 
    g = np.real(c0 * S.dot((1/(np.sqrt(2*(mu+1))))*(r + np.conj(r))))  # System-PM couplings
    xi = S.dot(xi0.dot(S.T))  # internal couplings
    gm = S.dot(gm0.dot(S.T))  # decay rates

    # Information 
    info_dict = {
    'sys couplings': 'g = \n {0}'.format(np.round(g, 5)),
    'int couplings': 'xi = \n {0}'.format(np.round(xi, 5)),
    'decay rates': 'gm = \n {0}'.format(np.round(np.diag(gm), 5)),
    'all': 'g = \n {0}\n\n'.format(np.round(np.real(g), 5))+'xi = \n {0}\n\n'.format(np.round(xi, 5)) + 'gm = \n {0}'.format(np.round(np.diag(gm), 5)) 
    }

    return (g, xi, gm, info_dict) if info==True else (g, xi, gm)


def bcf_pm(
    t: float | list | np.ndarray,
    g: list | np.ndarray, 
    xi: np.ndarray[np.float64],
    gm: list | np.ndarray[np.float64]
) -> complex:
    r"""
    Free PM correlation function 

    .. math::
        C^E(t) = \boldsymbol{g}'^Te^{-i\boldsymbol{Z}t}\boldsymbol{g}' , 

    where '\boldsymbol{Z} = \boldsymbol{\xi} - \frac{i}{2}\boldsymbol{\gamma}', and
        - '\boldsymbol{g}' are real system-PM couplings,
        - '\boldsymbol{\xi}' are internal couplings,
        - '\boldsymbol{\gamma}' are decay rates.

    Parameters
    ----------
    t : float, array_like
        Time or array of times t.
    g : array_like, shape (n,) 
        System-PM couplings (real).
    xi : ndarray, shape (n,n)
        Internal PM couplings (real, symmetric matrix). 
    gm : array_like, shape (n,n) or (n,)
        PM decay rates (real, diagonal matrix). 

    Returns
    -------
    PM correlation function at time(s) t. 
    """
    # Error checks
    if not len(g) == len(xi) == len(gm):
        raise ValueError(f"Dimensions of g, xi, and gm must match.")
    if type(g) == list:
        g = np.array(g)
    if g.dtype == complex:
        raise ValueError(f"g must be a real 1D array.")
    if not np.allclose(xi.T, xi, atol=1e-8) or xi.dtype == complex:
        raise ValueError(f"xi must be a real symmmetric matrix.")
    if gm.shape == (len(gm), len(gm)) and not np.allclose(gm.T, gm, atol=1e-12) or gm.dtype == complex:
        raise ValueError(f"If gm is input as a matrix, it must be real symmetric.")
    
    if gm.shape == (len(gm), ):
        gm = np.diag(gm)

    # Other error checks?
    Z = xi - 0.5j*gm

    if type(t) == list:
        tlist = np.array(t)
    elif type(t) == np.ndarray:
        tlist = t
    
    return g.T.dot(expm(-1j*Z*t).dot(g)) if np.isscalar(t) else np.array([g.T.dot(expm(-1j*Z*t).dot(g)) for t in tlist])
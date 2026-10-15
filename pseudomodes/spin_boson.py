"""
Module for simulating the spin-boson model with the pseudomode method.
"""
import numpy as np
import qutip as qt
from numpy import ndarray
from qutip import Qobj
from scipy.integrate import quad 
from mpmath import zeta, gamma
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
    w = np.asarray(w)

    return coup * (w**(s) / w_cut**(s-1)) * np.exp(-w/w_cut)


def sd_ud(
    w: float | list | ndarray, 
    coup: float,
    width: float, 
    w_res: float
) -> float:
    r"""
    Underdamped Brownian spectral density

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
    w = np.asarray(w)

    return (2 * coup ** 2 * w_res * width * w) / ((w**2 - w_res**2) ** 2 + width**2 * w**2)


def bcf_power(
    tlist: float | list | ndarray,
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
    tlist = np.atleast_1d(tlist)

    if T > 0:
        beta = 1./T
        return np.array([complex( (1/np.pi) * coup * w_cut**(1-s) * beta**(-(s+1)) * gamma(s+1) \
                          * (zeta(s+1,(1+beta*w_cut-1.0j*w_cut*t)/(beta*w_cut)) \
                          + zeta(s+1,(1+1.0j*w_cut*t)/(beta*w_cut)))
                          ) for t in tlist]
                        )
    elif T==0:
        return np.array([complex( (1/np.pi) * coup * w_cut**2 * gamma(s+1) * (1+1j*w_cut*t)**(-(s+1)) ) for t in tlist])


def bcf_ud(
    t: float | list | ndarray,
    T: float,
    coup: float,
    width: float, 
    w_res: float,
    Nk: int
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
        Resonance frequency.
    Nk : int
        Number of Matsubara terms.

    Returns
    -------
    Bath correlation function at time(s) t.
    """
    beta = 1./T
    Om = np.sqrt(w_res ** 2 - width **2 / 4)
    t = np.asarray(t)

    # Check w_res > width / 2
    if not w_res > 0.5 * width:
        raise ValueError(f'Require w_res > (width / 2) for underdamped spectral density.')

    # Matsubara terms 
    def M(t, k):
        vk = 2 * np.pi * k / beta
        return np.real( ((-1j * 2) / beta) * sd_ud(-1j*vk, coup=coup, width=width, w_res=w_res) * np.exp(-vk * np.abs(t)) )

    return ((coup **2 * w_res) / (2 * Om)) \
         * ((coth(0.5 * beta * (Om - 0.5j * width)) + 1) * np.exp(-1j*Om*t - 0.5 * width * np.abs(t)) \
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
    return (np.array([2 * np.real(np.exp(1j*eps*t) * np.exp(integral(t)) * rho_10) for t in tlist]), 
            np.array([2 * np.imag(np.exp(1j*eps*t) * np.exp(integral(t)) * rho_10) for t in tlist]),
            np.array([init_bloch[2] for t in tlist]))


# previously pm_liouvillian
def pmme(
    H_sys: Qobj,
    c_ops: Qobj | list[Qobj],
    pm_params: tuple[ndarray],
    pm_dims: tuple[int]
) -> Qobj:
    """
    Constructs PMME for PM environment parameterized by (gp, xi, gm).

    PM parameters must be input as ordered list (tuple?) [gp, xi, gm]. 

    Parameters (EDIT)
    ----------
    sys : Qobj
        System Hamiltonian. 
    c_ops : Obj
        System coupling operator.
    pm_params : tuple[ndarray]
        Tuple of PM parameters in format [gp, xi, gm]. \n
            - gp must be a 1d array of shape (n,) or 2d array of shape (n,len(c_ops)).
            - xi must be a symmetric 2d array of shape (n,n).
            - gm must be a 1d array of shape (n,) or diagonal 2d array of shape (n,n).
    pm_dims : tuple[int]
        Tuple of local PM dimensions. 

    Returns
    -------
    liouv : Qobj
    """
    # Error checks
    gp, xi, gm = pm_params

    c_ops = [c_ops] if not isinstance(c_ops, list) else c_ops

    # Check type and dimensions of PM parameter inputs
    if not len(gp) == len(xi) == len(gm):
        raise ValueError(f"Inner dimensions of PM arguments must match.")
    
    if not len(pm_dims) == len(gp):
        raise ValueError(f"Length of pm_dims must match length of each PM argument.")
    
    # Convert gp to 2d array of shape n * len(c_ops) 
    if gp.ndim == 1:
        gp = np.atleast_2d(gp).T

    # Check length of c_ops and number of columns of gpL, gpR match
    if not gp.shape[1] == len(c_ops): 
        raise ValueError(f"Number of columns of coupling array gp must match number of coupling operators.")

    # Check xi is real symmetric
    if np.iscomplexobj(xi):
        if not np.allclose(xi.imag, 0, atol=1e-12):
            raise ValueError(f"xi must be real.")
        xi = np.real(xi)
    
    if not np.allclose(xi.T, xi, atol=1e-12):
        raise ValueError(f"xi must be symmetric.")

    # Check gm is real diagonal 
    if np.iscomplexobj(gm):
        if not np.allclose(gm.imag, 0, atol=1e-12):
            raise ValueError(f"gm must be real.")
        gm = np.real(gm)

    if not gm.ndim == 1: 
        gm_off_diag = gm.copy()
        np.fill_diagonal(gm_off_diag, 0)

        if not np.allclose(gm_off_diag, 0, atol=1e-12):
            raise ValueError(f"gm must be diagonal.")
        
        # Convert gm to 1d array
        gm = np.diag(gm)

    sys_dims = H_sys.dims[0][0]

    H_sys = qt.tensor([H_sys] + [qt.identity(n) for n in pm_dims])

    c_ops = [qt.tensor([c] + [qt.identity(n) for n in pm_dims]) for c in c_ops]    

    b_ops = []
    for n in range(len(pm_dims)): 
        b_ops.append(qt.tensor([qt.identity(sys_dims)] + [qt.destroy(N) if n == idx else qt.identity(N) for idx, N in enumerate(pm_dims)]))

    H_pm, H_int = 0, 0

    for i in range(len(pm_dims)):
        # sum([coup[i][k] * ( c_ops[k].dag() * f_ops[env][i] + f_ops[env][i].dag() * c_ops[k]) for k in range(len(c_ops))])
        H_int += sum([gp[i][k] * c_ops[k] * (b_ops[i] + b_ops[i].dag()) for k in range(len(c_ops))])
        for j in range(len(b_ops)):
            H_pm += xi[i][j] * b_ops[i].dag() * b_ops[j] 
    
    H = H_sys + H_pm + H_int

    # Dissipator
    liouv = 0
    for i, decay_rate in enumerate(gm):
        if decay_rate >= 0:
            liouv += qt.liouvillian(c_ops=[np.sqrt(decay_rate) * b_ops[i]])
        elif decay_rate < 0:
            liouv -= qt.lindblad_dissipator(a=np.sqrt(np.abs(decay_rate)) * b_ops[i])
    
    return qt.liouvillian(H=H) + liouv
"""
Module for simulating the single-impurity Anderson model (SIAM) with the pseudomode method.
"""
import numpy as np
import qutip as qt
from numpy import ndarray
from qutip import Qobj
from scipy.integrate import quad
from scipy.special import j1
from scipy.fft import fft, fftshift, ifftshift
# from numpy.typing import ArrayLike


def fd(x):
    return 0.5 * (1 - np.tanh(0.5 * x)) 


def sd_lorentzian(
    w: float | list | ndarray,
    coup: float,
    width: float, 
    w_res: float
) -> float:
    r"""
    Lorentzian spectral density 

    .. math::

        J(\omega) = \frac{W^2\Gamma}{(\omega - \mu)^2 + \Gamma^2}
    
    where $W$ is the is the coupling strength, $\Gamma$ is the bath width, and $\mu$ is the chemical potential (resonance frequency). 
    
    Parameters
    ----------
    w : float | array_like 
        Frequency or array of frequencies w. 
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
    
    return coup ** 2 * width / ((w - w_res)**2 + width**2)


def bcf_lorentzian(
    t: float | list | ndarray,
    T: float,
    coup: float,
    width: float,
    w_res: float,
    Nk: int,
    bcf_type = 'gtr'
) -> float: 
    r"""
    Bath correlation function for Lorentzian spectral density. 

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
        Resonance frequency (chemical potential). 
    Nk : int
        Number of Matsubara terms.
    bcf_type : str, optional
        BCF type - use either 'gtr' or 'less'.
    Returns
    -------
    Bath correlation function at time(s) t.
    """
    if bcf_type not in ['gtr', 'less']:
        raise ValueError(f"Invalid BCF type '{bcf_type}'. Use either 'gtr' or 'less'.")

    t = np.asarray(t)

    # Matsubara terms
    def M(t, k):
        vk = np.pi * (k+1) * T
        return (-1j * T) * sd_lorentzian(-1j * vk, coup=coup, width=width, w_res=0) * np.exp(-1j * w_res * t - vk * np.abs(t))

    # Greater
    def _bcf_gtr(t):
        if T > 0:
            return (coup**2 / 4) * (1 - 1j*np.tan((0.5 * width) / T)) * np.exp(-1j * w_res * t - width * np.abs(t)) \
            + np.sum([M(t, k) for k in range(1, Nk+1)], axis=0) 
        else:
            return (coup**2 / 2) * np.exp(-1j * w_res * t - width * np.abs(t))

    # Lesser
    def _bcf_less(t): 
        if T > 0:
            return (coup**2 / 2) * fd((1j * width) / T) * np.exp(1j * w_res * t - width * np.abs(t)) \
            - np.conj(np.sum([M(t, k) for k in range(1, Nk+1)], axis=0))
        else:
            return 0

    if bcf_type == 'gtr':
        return np.where(t>=0, _bcf_gtr(t), np.conj(_bcf_gtr(-t)))
    elif bcf_type == 'less':
        return np.where(t>=0, _bcf_less(t), np.conj(_bcf_less(-t)))


def sd_semicircular(
    w: float | list | ndarray,
    coup: float,
    w_cut: float
) -> float:
    r"""
    Semicircular spectral density 

    .. math::

        J(\omega) = \frac{2\alpha}{\omega_c} * \sqrt{1 - \left(\frac{\omega}{\omega_c}\right)^2},  \quad \omega \in [-\omega_c, \omega_c],
    
    where $\alpha$ is the is the coupling strength, and $\omega_c$ is the bath cutoff frequency. 
    
    Parameters
    ----------
    w : float | array_like 
        Frequency or array of frequencies w. 
    coup : float 
        Coupling strength. 
    w_cut : float 
        Bath cutoff frequency. 
    
    Returns 
    -------
    Spectral density at frequencies w. 
    """
    w = np.asarray(w)
    
    return np.where(np.abs(w)<=w_cut, 2 * (coup / w_cut) * np.sqrt(1 - (w / w_cut)**2), 0) 


def bcf_semicircular(
    t: float | list | ndarray,
    T: float,
    mu: float,
    coup: float,
    w_cut: float
) -> float:
    r"""
    Bath correlation function for semicircular spectral density. 
    
    NB: limits ... 

    Parameters
    ----------
    t : float | array_like
        Time or array of times t. 
    T : float
        Bath temperature.
    mu : float
        Bath chemical potential.
    coup : float
        Coupling strength.
    w_cut : float
        Bath cutoff frequency.
  
    Returns
    -------
    Bath correlation function at time(s) t.
    """
    t = np.asarray(t)

    w, dw = np.linspace(-w_cut, w_cut, num=int(1e4), retstep=True)

    if T==0:
        # Fully occupied bath: mu >= w_cut
        if mu >= w_cut:
            bcf_gtr = 0 * t
            bcf_less = np.where(t==0, 0.5 * coup, coup * j1(w_cut * t)/(w_cut * t))

        # Fully empty bath: mu <= -w_cut
        if mu <= -w_cut:
            bcf_gtr = np.where(t==0, 0.5 * coup, coup * j1(w_cut * t)/(w_cut * t))
            bcf_less = 0 * t

        if w_cut >= np.abs(mu):
            # J(w) * (1 - f(w))
            G_gtr = np.where(w>=mu, sd_semicircular(w, coup=coup, w_cut=w_cut), 0)
            # J(w) * f(w)
            G_less = np.where(w<=mu, sd_semicircular(w, coup=coup, w_cut=w_cut), 0)
    
            # Riemann sum via outer product matrix multiplication:
            # (N_t, N_w) @ (N_w,) -> (N_t,)
            bcf_gtr = (dw / (2 * np.pi)) * (np.exp(-1j * np.outer(t, w)) @ G_gtr)
            bcf_less = (dw / (2 * np.pi)) * (np.exp(1j * np.outer(t, w)) @ G_less)

    elif T==np.infty:
        # Infinite temperature limit
        bcf_gtr = 0.5 * np.where(t==0, 0.5 * coup, coup * j1(w_cut * t)/(w_cut * t))
        bcf_less = 0.5 * np.where(t==0, 0.5 * coup, coup * j1(w_cut * t)/(w_cut * t))

    else:
        # General case for arbitrary T, mu
        # J(w) * (1 - f(w))
        G_gtr = sd_semicircular(w, coup=coup, w_cut=w_cut) * (1 - fd((w - mu) / T))
         # J(w) * f(w)
        G_less = sd_semicircular(w, coup=coup, w_cut=w_cut) * fd((w - mu) / T)

        # Riemann sum via outer product matrix multiplication:
        # (N_t, N_w) @ (N_w,) -> (N_t,)
        bcf_gtr = (dw / (2 * np.pi)) * (np.exp(-1j * np.outer(t, w)) @ G_gtr)
        bcf_less = (dw / (2 * np.pi)) * (np.exp(1j * np.outer(t, w)) @ G_less)

    return bcf_gtr, bcf_less


# GENERALIZE TO ARBITRARY SPECTRAL DENSITY???
def impurity_site_occ(
    tlist: float | list | ndarray,
    e0: float,
    T: float,
    coup: float,
    width: float, 
    w_res: float,
    init_occ: float
) -> float:
    r"""
    Impurity site occupation n(t) = \langle c^{\dagger}(t)c(t)\rangle for Lorentzian hybridization function.  

    Parameters
    ----------
    tlist : float | array_like
        Time or array of times t.
    e0 : float
        Impurity energy level gap. 
    T : float
        Bath temperature.
    coup : float
        Coupling strength.
    width : float
        Bath width. 
    w_res : float
        Resonance frequency (chemical potential). 
    init_occ : float
        Initial impurity site occuptation. 
        
    Returns
    -------
    nt : Impurity site occupation at time(s) t.
    """ 
    tlist = np.asarray(tlist)

    # Evaluate retarded Green's functions of impurity and bath
    def P(z):
        return z + 1j*w_res + width 

    # Analytical roots of Q and B: 
    # Q = (z + 1j*e0)*(z + 1j*w_res + W) + 0.5 * coup**2
    # B = [(z + 1j*e0)*(z + 1j*w_res + W) + 0.5 * coup**2] * (z + 1j*om)
    r1 = 0.5 * (-1j*e0 - 1j*w_res - width \
                - np.sqrt(-e0**2 + 2*e0*w_res - w_res**2 - 2j*e0*width + 2j*w_res*width + width**2 - 2*coup**2)
                )
    r2 = 0.5 * (-1j*e0 - 1j*w_res - width \
                + np.sqrt(-e0**2 + 2*e0*w_res - w_res**2 - 2j*e0*width + 2j*w_res*width + width**2 - 2*coup**2)
                )
  
    def rb(om):
        return -1j * om

    # Derivative of Q and B 
    def Q_deriv(z): 
        return 2*z + 1j*(w_res + e0) + width

    def B_deriv(z, om):
        return ((z + 1j*e0) * (z + 1j*w_res + width) + 0.5*coup**2) + (z + 1j*om)*(2*z + 1j*w_res + width + 1j*e0)

    # Laplace inversion formula 
    Gr_impurity = sum([(P(r) / Q_deriv(r)) * np.exp(r * tlist) for r in [r1, r2]])
    def Gr_bath(om, t): 
        om_val = float(om)
        roots = [rb(om_val), r1, r2]
        return sum((P(r) / B_deriv(r, om_val)) * np.exp(r * t) for r in roots)

    # Site occupation <c^{\dagger}(t)c(t)> 
    def integrand(om, t):
        om_val = float(om)
        if T > 0: 
            return np.abs(Gr_bath(om_val, t)) ** 2 * (1/(2 * np.pi)) * sd_lorentzian(om_val, coup, width, w_res) * fd((om_val - w_res) / T)
        else:
            return 0 
    
    def integral(t):
        return quad(integrand, -np.inf, np.inf, args=t)[0]

    nt = np.abs(Gr_impurity) ** 2 * init_occ + np.array([integral(t) for t in tlist])

    return nt


def pmme_fermion( 
    H_sys: Qobj,
    c_ops: Qobj | list[Qobj],
    pm_sys_coup: tuple[ndarray],
    pm_self: tuple[ndarray],
    pm_decay: tuple[ndarray]
) -> Qobj: 
    """
    Constructs PMME for greater (L) and lesser (R) PM environments parameterized 
    by (gpL, xiL, gmL), (gpR, xiR, gmR). 

    PM parameters must be input as tuples e.g., (gpL, gpR), where the first (left) 
    element corresponds to L, and the second (right) element to R.

    Parameters
    ----------
    H_sys : Qobj
        System Hamiltonian. 
    c_ops : Qobj, list[Oobj]
        Tuple of system coupling operators.
    pm_sys_coup : tuple[ndarray]
        Tuple of system-PM couplings in format (gpL, gpR). 
        Each element must be 1d array of shape (n,), or 2d array of shape (n,len(c_ops)). 
    pm_self : tuple[ndarray]
        Tuple of internal PM parameters (site energies, couplings) in format (xiL, xiR). 
        Each element must be 2d array of shape (n,n). 
    pm_decay : tuple[ndarray]
        Tuple of PM decay rates in format (gmL, gmR).
        Each element must be 1d array of shape (n,) or diagonal 2d array of shape (n,n). 
        
    Returns
    -------
    liouv : Qobj
    """
    # Re-assign PM parameters locally to lists
    gp = list(pm_sys_coup)
    xi = list(pm_self)
    gm = list(pm_decay)

    c_ops = [c_ops] if not isinstance(c_ops, list) else c_ops

    # Error checks
    if not len(gp[0]) == len(xi[0]) == len(gm[0]) and \
           len(gp[1]) == len(xi[1]) == len(gm[1]):
        raise ValueError(f"Inner dimensions of PM arguments must match.")

    # Convert gpL, gpR to 2d arrays of shape n * len(c_ops) 
    for env, coup in enumerate(gp): 
        if coup.ndim == 1:
            gp[env] = np.atleast_2d(coup).T

    # Check length of c_ops and number of columns of gpL, gpR match
    for coup in gp:
        if not coup.shape[1] == len(c_ops): 
            raise ValueError(f"Number of columns of coupling arrays gpL,gpR must match number of coupling operators.")

    # Check xiL, xiR is real symmetric
    def _check_real_symmetric(xi):
        if np.iscomplexobj(xi):
            if not np.allclose(xi.imag, 0, atol=1e-12):
                raise ValueError(f"The matrices (xiL, xiR) must be real.")
            xi = np.real(xi)

        if not np.allclose(xi.T, xi, atol=1e-12):
                raise ValueError(f"The matrices (xiL, xiR) must be symmetric.")
        return xi

    # Check gmL, gmR is real diagonal
    def _check_real_diag(gm):
        if np.iscomplexobj(gm):
            if not np.allclose(gm.imag, 0, atol=1e-12):
                raise ValueError(f"The matrices (gmL, gmR) must be real.")
            gm = np.real(gm)

        if not np.allclose(gm.T, gm, atol=1e-12):
            raise ValueError(f"The matrices (gmL, gmR) must be symmetric.")

        if not gm.ndim == 1: 
            gm_off_diag = gm.copy()
            np.fill_diagonal(gm_off_diag, 0)

            if not np.allclose(gm_off_diag, 0, atol=1e-12):
                raise ValueError(f"The matrices (gmL, gmR) must be diagonal.")
        
            # Convert gm to 1d array
            gm = np.diag(gm)
        return gm

    for env, _ in enumerate(xi): 
        xi[env] = _check_real_symmetric(xi[env])
        gm[env] = _check_real_diag(gm[env])


    #################################


    # No. of PMs for greater (L) and lesser (R) envs
    N1 = len(xi[0])
    N2 = len(xi[1])
    
    c_ops = [qt.tensor([c] + [qt.identity(2)] * (N1 + N2)) for c in c_ops]

    sys_dim = H_sys.dims[0][0]

    # Hamiltonian H_S 
    H_sys = qt.tensor([H_sys] + [qt.identity(2)] * (N1 + N2))

    # Environment + interaction 
    f_ops = [[], []] 

    # Greater (L)
    for n in range(N1):
        f_ops[0].append(
                       qt.tensor([qt.identity(sys_dim)] 
                       + [qt.destroy(2) if n == idx else qt.identity(2) for idx in range(N1)]
                       + [qt.identity(2)] * N2)
                       )
    
    # Lesser (R)
    for n in range(N2):
        f_ops[1].append(
                       qt.tensor([qt.identity(sys_dim)] 
                       + [qt.identity(2)] * N1
                       + [qt.destroy(2) if n == idx else qt.identity(2) for idx in range(N2)])
                       )
    
    # Hamiltonians H_E, H_I    
    H_pm, H_int = 0, 0

    for env, coup in enumerate(gp):
        for i in range(len(f_ops[env])):
            H_int += sum([coup[i][k] * ( c_ops[k].dag() * f_ops[env][i] + f_ops[env][i].dag() * c_ops[k]) for k in range(len(c_ops))])
            for j in range(len(f_ops[env])):
                H_pm += (xi[env])[i][j] * f_ops[env][i].dag() * f_ops[env][j] 

    Hp = H_sys + H_pm + H_int

    # Dissipators
    liouvL, liouvR = 0, 0

    # greater (L)
    for i, decay_rate in enumerate(gm[0]):
        if decay_rate >= 0:
            liouvL += qt.liouvillian(c_ops=[np.sqrt(decay_rate) * f_ops[0][i]])
        elif decay_rate < 0:
            liouvL -= qt.lindblad_dissipator(a=np.sqrt(np.abs(decay_rate)) * f_ops[0][i])

    # lesser (R)
    for i, decay_rate in enumerate(gm[1]):
        if decay_rate >= 0:
            liouvR += qt.liouvillian(c_ops=[np.sqrt(decay_rate) * f_ops[1][i].dag()])
        elif decay_rate < 0:
            liouvR -= qt.lindblad_dissipator(a=np.sqrt(np.abs(decay_rate)) * f_ops[1][i].dag())

    # Combine liouvillians 
    liouv = qt.liouvillian(H=Hp) + liouvR + liouvL

    return liouv  


# Function to compute spectrum A(w)
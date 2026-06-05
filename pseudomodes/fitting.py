"""
Module for exponentially fitting bath correlation functions and extracting pseudomode information.
"""
import numpy as np
from numpy import ndarray
from numpy import linalg
from lmfit import Parameters
# from numpy.typing import ArrayLike

def pm_parameters(
    rk: list | ndarray, 
    zk: list | ndarray,
    info = False
) -> tuple[ndarray, ndarray, ndarray, dict]:
    r"""
    Extracts the pseudomode parameters from a fitted BCF of the form 

    .. math:: 
        C^E(t) = -i\sum^N_{k=1} rk * e^{-i*zk*t}
    
    where rk are the fitted coefficients, and zk the fitted exponents. 

    Parameters
    ----------
    rk : array_like, shape (n,)
        Fitted complex coefficients.
    zk : array_like, shape (n,) or shape (n,n)
        Fitted complex exponents.
    info : bool, optional
        Provides information on PM parameters if set to True. Default is False.

    Returns
    -------
    Tuple of ndarrays containing PM parameters and dictionary.  

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
    r = r / c0  # convert to normalized array 

    mu = r @ np.conj(r) 
    alpha = (1/(np.sqrt(mu*mu-1))) * np.arctanh(np.sqrt((mu-1)/(mu+1))) 

    theta = alpha * np.sqrt(mu*mu-1)

    exp_R = np.identity(len(r)) + (1/(mu*mu-1)) * ((np.cosh(theta)*(2*mu-1)-mu) * np.outer(np.conj(r), r)
                                            - (mu - np.cosh(theta)) * np.outer(r, np.conj(r))
                                            - (np.cosh(theta)-1) * np.outer(r, r)
                                            - (np.cosh(theta)-1) * np.outer(np.conj(r), np.conj(r)))

    gm0 = -2 * np.imag(exp_R.dot(Lmb.dot(exp_R.T)))
    xi0 = np.real(exp_R.dot(Lmb.dot(exp_R.T)))

    S = linalg.eig(gm0)[1].T

    # PM parameters : sys-pm couplings, internal couplings, decay rates
    g = np.real(c0 * S.dot((1/(np.sqrt(2*(mu+1))))*(r + np.conj(r))))
    xi = S.dot(xi0.dot(S.T))
    gm = S.dot(gm0.dot(S.T))

    # Information on pm parameters
    info_dict = {
    'sys couplings': 'g = \n {0}'.format(np.round(g, 5)),
    'int couplings': 'xi = \n {0}'.format(np.round(xi, 5)),
    'decay rates': 'gm = \n {0}'.format(np.round(np.diag(gm), 5)),
    'all': 'g = \n {0}\n\n'.format(np.round(np.real(g), 5))+'xi = \n {0}\n\n'.format(np.round(xi, 5)) + 'gm = \n {0}'.format(np.round(np.diag(gm), 5)) 
    }

    return (g, xi, gm, info_dict) if info==True else (g, xi, gm)


def bcf_fit(
    t: float | list | ndarray,
    c_vector: list | ndarray,
    V_matrix: ndarray[complex]
) -> ndarray[complex]:
    r"""
    Fitted exponential BCF of the form 

    .. math::
        C^E(t) = c_vector^T * exp(-i * V_matrix * t) * c_vector

    where 'c_vector' is a complex array, and 'V_matrix' is a complex symmetric matrix. 

    Parameters
    ----------
    t : float, array_like
        Time or array of times t. 
    c_vector : array_like
        1D array of complex coefficients, shape (n,).
    V_matrix : array_like
        2D array of complex exponents, shape (n,n).
 
    Returns
    -------
    Fitted BCF at time(s) t.
    """
    # Error checks 
    if not len(c_vector) == len(V_matrix):
        raise ValueError(f"Inner dimensions of c_vector and V_matrix must match.")
    if not np.allclose(V_matrix.T, V_matrix, atol=1e-12):
        raise ValueError(f"V_matrix must be a complex symmmetric matrix.")

    tlist = np.asarray(t)
    c_vector = np.asarray(c_vector)

    # Diagonalize to find rk, zk - faster than using expm in fitting
    zk, S = linalg.eig(V_matrix)
    S = S / np.sqrt(np.sum(S**2, axis=0))  # re-scale to ensure orthonormal colums v^T[i]*v[j] = delta_{ij}
    rk = S.T @ c_vector

    result = (rk.T * rk * np.exp(-1j * np.outer(tlist, zk))).sum(axis=1)

    return result[0] if tlist.ndim==0 else result


def ps_fit(
    w: float | list | ndarray,
    c_vector: list | ndarray,
    V_matrix: ndarray
) -> complex | ndarray:
    """
    Power spectrum (Fourier transform) of exponentially fitted BCF 

    .. math::
        \gamma'(w) = 2 * Im[c_vector^T * (V_matrix - w)^{-1} * c_vector]

    where 'c_vector' is a complex array, and 'V_matrix' is a complex symmetric matrix.

    Parameters
    ----------
    w : float, array_like
        Frequency or array of frequencies w.
    c_vector : array_like
        1D array of complex coefficients, shape (n,).
    V_matrix : array_like
        2D array of complex exponents, shape (n,n).

    Returns
    ------- 
    Fitted power spectrum at frequency(ies) w.
    """
    # Error checks 
    if not len(c_vector) == len(V_matrix):
        raise ValueError(f"Inner dimensions of c_vector and V_matrix must match.")
    if not np.allclose(V_matrix.T, V_matrix, atol=1e-12):
        raise ValueError(f"V_matrix must be a complex symmmetric matrix.")
    
    wlist = np.asarray(w)
    c_vector = np.asarray(c_vector)

    # Diagonalize to find rk, zk - faster than using expm in fitting
    zk, S = linalg.eig(V_matrix)
    S = S / np.sqrt(np.sum(S**2, axis=0))  # re-scale to ensure orthonormal colums v^T[i]*v[j] = delta_{ij}
    rk = S.T @ c_vector
    
    result = 2 * np.imag(rk.T * rk * (zk - np.outer(wlist, np.ones(len(zk)))) ** -1).sum(axis=1)

    return result[0] if wlist.ndim==0 else result


def make_params(
    rkr_init: list | ndarray,
    rki_init: list | ndarray,
    zkr_init: list | ndarray,
    zki_init: list | ndarray,
    C0 : float
):
    """
    Create Parameters object initializing fitted parameters for arbitrary number of exponential terms. 

    NB: The fitted parameters rkr[-1] and rki[-1] in bcf_fit are constrained such that 

    rkr[-1] = -sum([rkr[0], ..., rkr[-2]])

    rki[-1] =  C0 - sum([rki[0], ..., rki[-2]])

    where C0 = bcf(0). 

    Parameters
    ----------
    rkr_init : array_like
        Array of initial values for rkr. 
    rki_init : array_like
        Array of initial values for rki.
    zkr_init : array_like
        Array of initial values of zkr.
    zki_init : array_like  
        Array of initial values of zki.

    Returns
    -------
    params : Parameters
    """
    # Error checks 
    if not len(rkr_init) == len(rki_init):
        raise ValueError(f"rkr_init and rki_init must have the same length.") 
    if not len(zkr_init) == len(zki_init):
        raise ValueError(f"zkr_init and zki_init must have the same length.")
    if not len(rkr_init) == len(zkr_init)-1 & len(rki_init) == len(zki_init)-1:
        raise ValueError(f"Missing parameter values from rkr_init, rki_init.")

    # Intialize fitting parameters rkr, rki, zkr, zki used in bcf_fit 
    params = Parameters()
    params.add('C0', value=C0, vary=False)

    for k in np.arange(1,len(zkr_init)+1):
        if k < max(np.arange(1,len(zkr_init)+1)):
            params.add(f"r{k}r", value=rkr_init[k-1])
            params.add(f"r{k}i", value=rki_init[k-1])
        elif k == len(zkr_init):
            rkr_other = " - ".join(f"r{k}r" for k in np.arange(1,len(rkr_init)+1))
            rki_other = " - ".join(f"r{k}i" for k in np.arange(1,len(rki_init)+1))

            params.add(f"r{k}r", expr=f"-{rkr_other}")
            params.add(f"r{k}i", expr=f"C0-{rki_other}")
        
        params.add(f"z{k}r", value=zkr_init[k-1])
        params.add(f"z{k}i", value=zki_init[k-1], min=0)

    return params


def extract_params(
    params
):
    """
    Returns fitted parameter values from Parameters object.
    """
    N_exp = int((len(params.valuesdict()) - 1)/4)
    rkr = np.array([params[f"r{k}r"].value for k in np.arange(1,N_exp+1)])
    rki = np.array([params[f"r{k}i"].value for k in np.arange(1,N_exp+1)])
    zkr = np.array([params[f"z{k}r"].value for k in np.arange(1,N_exp+1)])
    zki = np.array([params[f"z{k}i"].value for k in np.arange(1,N_exp+1)])

    return rkr, rki, zkr, zki


def residuals(
    params,
    t,
    bcf,
    bcf_fit  
 ):
    """
    Evaluate error between bcf and bcf_fit.

    Parameters
    ----------
    params : Parameters
        Contains initial fitting parameter values.
    t : float, array_like
        Time or array of times t.
    bcf : complex, array_like
        Physical BCF evaluated at time(s) t. 
    bcf_fit : callable
        Fitted exponential BCF. 

    Returns
    -------
    Error between fitted and physical BCF.
    """
    # Unpack values of rkr, rki, zkr, zki from params
    rkr, rki, zkr, zki = extract_params(params)
    g = np.sqrt(-1j * (rkr + 1j * rki))
    Lmb = np.diag(zkr - 1j * zki)
    err = bcf - bcf_fit(t, c_vector=g, V_matrix=Lmb)

    return err.view(float)
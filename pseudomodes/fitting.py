"""
Module for fitting the correlation functions and power spectra of Gaussian baths, and extracting pseudomode parameters.
"""
import numpy as np
from numpy import ndarray
from numpy import linalg
from lmfit import Parameters, minimize
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
    gp = np.real(c0 * S.dot((1/(np.sqrt(2*(mu+1))))*(r + np.conj(r))))
    xi = S.dot(xi0.dot(S.T))
    gm = S.dot(gm0.dot(S.T))

    # Information on pm parameters
    info_dict = {
    'sys couplings': 'gp = \n {0}'.format(np.round(gp, 5)),
    'int couplings': 'xi = \n {0}'.format(np.round(xi, 5)),
    'decay rates': 'gm = \n {0}'.format(np.round(np.diag(gm), 5)),
    'all': 'gp = \n {0}\n\n'.format(np.round(np.real(gp), 5))+'xi = \n {0}\n\n'.format(np.round(xi, 5)) + 'gm = \n {0}'.format(np.round(np.diag(gm), 5)) 
    }

    return (gp, xi, gm, info_dict) if info==True else (gp, xi, gm)


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
        Time or array of times t>=0. 
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

    # Include possibility of negative times
    result = np.where(tlist>=0, 
                      (rk.T * rk * np.exp(-1j * np.outer(tlist, zk))).sum(axis=1), 
                      np.conj((rk.T * rk * np.exp(-1j * np.outer(-tlist, zk)))).sum(axis=1)
                      )

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


def ps_fit_gtr(
    w: float | list | ndarray,
    c_vector: list | ndarray,
    V_matrix: ndarray
) -> complex | ndarray:
    """
    Power spectrum (Fourier transform) of exponentially fitted BCF 
    
    .. math::
        \gamma'>(w) = 2 * Im[c_vector^T * (V_matrix - w)^{-1} * c_vector]
    
    where 'c_vector' is a complex array, and 'V_matrix' is a complex symmetric matrix.

    Use to fit greater component of fermionic power spectrum.  
    
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
    return ps_fit(w, c_vector, V_matrix)


def ps_fit_less(
    w: float | list | ndarray,
    c_vector: list | ndarray,
    V_matrix: ndarray
) -> complex | ndarray:
    """
    Power spectrum (Fourier transform) of exponentially fitted BCF 
    
    .. math::
        \gamma'<(w) = -2 * Im[conj(c_vector^T * (V_matrix + w)^{-1} * c_vector)]
    
    where 'c_vector' is a complex array, and 'V_matrix' is a complex symmetric matrix.

    Use to fit lesser component of fermionic power spectrum.  
    
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
    return ps_fit(-w, np.conj(c_vector), -np.conj(V_matrix))


def make_params(
    rkr_init: list | ndarray,
    rki_init: list | ndarray,
    zkr_init: list | ndarray,
    zki_init: list | ndarray,
    C0 : float | None = None
):
    """
    Create Parameters object initializing fitted parameters for arbitrary number of exponential terms. 

    NB: If C0 is provided, the fitted parameters rkr[-1] and rki[-1] are constrained such that 

    rkr[-1] = -sum([rkr[0], ..., rkr[-2]])

    rki[-1] =  C0 - sum([rki[0], ..., rki[-2]])

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
    C0 : float, optional 
        Normalization factor. 

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

    if C0 is None:
        params.add('C0', value=0, vary=False)
    else:
        params.add('C0', value=C0, vary=False)

    for k in np.arange(1,len(zkr_init)+1):
        if C0 is None:
            params.add(f"r{k}r", value=rkr_init[k-1])
            params.add(f"r{k}i", value=rki_init[k-1])

        else:
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
    N_exp = int((len(params.valuesdict()) - 1)/4) # Requires C0 added as parameter
    rkr = np.array([params[f"r{k}r"].value for k in np.arange(1,N_exp+1)])
    rki = np.array([params[f"r{k}i"].value for k in np.arange(1,N_exp+1)])
    zkr = np.array([params[f"z{k}r"].value for k in np.arange(1,N_exp+1)])
    zki = np.array([params[f"z{k}i"].value for k in np.arange(1,N_exp+1)])

    return rkr, rki, zkr, zki


def extract_params_fermion(
    params,
    num_terms_gtr,
    num_terms_less
):
    """
    Returns fitted parameters values from Parameters object.

    Use for fermionic PM environments.
    """
    num_terms_gtr = int(num_terms_gtr)
    num_terms_less = int(num_terms_less)

    # Greater
    rLkr = np.array([params[f"rL{k}r"].value for k in np.arange(1,num_terms_gtr+1)])
    rLki = np.array([params[f"rL{k}i"].value for k in np.arange(1,num_terms_gtr+1)])
    zLkr = np.array([params[f"zL{k}r"].value for k in np.arange(1,num_terms_gtr+1)])
    zLki = np.array([params[f"zL{k}i"].value for k in np.arange(1,num_terms_gtr+1)])

    # Lesser
    rRkr = np.array([params[f"rR{k}r"].value for k in np.arange(1,num_terms_less+1)])
    rRki = np.array([params[f"rR{k}i"].value for k in np.arange(1,num_terms_less+1)])
    zRkr = np.array([params[f"zR{k}r"].value for k in np.arange(1,num_terms_less+1)])
    zRki = np.array([params[f"zR{k}i"].value for k in np.arange(1,num_terms_less+1)])

    return (rLkr, rLki, zLkr, zLki), (rRkr, rRki, zRkr, zRki)


def bcf_residuals(
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


def bcf_fit_params(
    tlist: float | list | ndarray,
    bcf: ndarray,
    init_params: tuple[list],
    C0: float | None = None
) -> complex | ndarray:
    """
    Returns the parameters (g, Lmb) of a fitted BCF

    .. math::

        C^E(t) = \sum^N_{l=1} g^2_l * exp(-i * Lmb_l * t)

    Parameters
    ----------
    tlist : float | array_like
        Time or array of times t.
    bcf : ndarray
        Bath correlation function at times t. 
    init_params : 
        Initial fitting parameters in format (rkr_init, rki_init, zkr_init, zki_init).
        The lengths of rkr_init, rki_init must equal N-1. \\
        The lengths of zkr_init, zki_init must equal N.
    C0 : float, optional
        Normalization factor. 
    Returns
    -------
    Fitted parameters (g, Lmb). 
    """
    params = make_params(rkr_init=init_params[0], rki_init=init_params[1], zkr_init=init_params[2], zki_init=init_params[3], C0=C0)
    result = minimize(bcf_residuals, params, method='leastsq', args=(tlist, bcf, bcf_fit))

    rkr, rki, zkr, zki = extract_params(result.params)

    # Fitted parameters
    g = np.sqrt(-1j * (rkr + 1j*rki))
    Lmb = np.diag(zkr - 1j*zki)

    return (g, Lmb)


def bcf_fit_params_fermion(
    tlist: float | list | ndarray,
    bcf: ndarray,
    init_params: tuple[list],
    C0: float | None = None,
    bcf_type = 'gtr'
) -> complex | ndarray:
    """
    Returns the parameters (g, Lmb) of a fitted greater or lesser BCF.

    Greater fitting function:

    .. math::

        C^>E(t) = theta(t) * \sum^N_{l=1} g^2_l * exp(-i * Lmb_l * t) + theta(-t) * \sum^N_{l=1} (g^2_l)^* * exp(i * Lmb^*_l * |t|)

    Lesser fitting function:     

    .. math::

        C^<E(t) = theta(t) * \sum^N_{l=1} (g^2_l)^* * exp(i * Lmb^*_l * t) + theta(-t) * \sum^N_{l=1} g^2_l * exp(-i * Lmb_l * |t|)

    Parameters
    ----------
    tlist : float | array_like
        Array of times t.
    bcf : ndarray
        Bath correlation function at times t. 
    init_params : tuple
        Initial fitting parameters in format (rkr_init, rki_init, zkr_init, zki_init). \\
        The lengths of rkr_init, rki_init must equal N-1. \\
        The lengths of zkr_init, zki_init must equal N.
    C0 : float
        Normalization factor. 
    bcf_type : str, optional
        BCF type - use either 'gtr' or 'less'.
    Returns
    -------
    Fitted parameters (g, Lmb). 
    """
    if bcf_type not in ['gtr', 'less']:
            raise ValueError(f"Invalid BCF type '{bcf_type}'. Use either 'gtr' or 'less'.")
    
    params = make_params(rkr_init=init_params[0], rki_init=init_params[1], zkr_init=init_params[2], zki_init=init_params[3], C0=C0)

    if bcf_type == 'gtr':
        result = minimize(bcf_residuals, params, method='leastsq', args=(tlist, bcf, bcf_fit_gtr))
        rLkr, rLki, zLkr, zLki = extract_params(result.params)

        # Fitted parameters
        g = np.sqrt(-1j * (rLkr + 1j*rLki))
        Lmb = np.diag(zLkr - 1j*zLki)

    elif bcf_type == 'less':
        result = minimize(bcf_residuals, params, method='leastsq', args=(tlist, bcf, bcf_fit_less))
        rRkr, rRki, zRkr, zRki = map(np.conj, extract_params(result.params))
        
        # Fitted parameters 
        g = np.sqrt(-1j * (rRkr + 1j*rRki))
        Lmb = np.diag(zRkr - 1j*zRki)

    return (g, Lmb)


def bcf_fit_gtr(
    t: float | list | ndarray,
    c_vector: list | ndarray,
    V_matrix: ndarray[complex]
) -> ndarray[complex]:
    r"""
    Fitted exponential BCF of the form 

    .. math::
        C^>E(t) = c_vector^T * exp(-i * V_matrix * t) * c_vector

    for nonnegative t, where 'c_vector' is a complex array, and 'V_matrix' is a complex symmetric matrix. 

    For negative t, instead uses  

     .. math::
        C^>E(t) = conj(c_vector^T * exp(-i * V_matrix * |t|) * c_vector)

    Use to fit greater component of fermionic BCF.  

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
    return bcf_fit(t, c_vector, V_matrix)


def bcf_fit_less(
    t: float | list | ndarray,
    c_vector: list | ndarray,
    V_matrix: ndarray[complex]
) -> ndarray[complex]:
    r"""
    Fitted exponential BCF of the form 

    .. math::
        C^<E(t) = conj(c_vector^T * exp(-i * V_matrix * t) * c_vector)

    for nonnegative t, where 'c_vector' is a complex array, and 'V_matrix' is a complex symmetric matrix. 

    For negative t, instead uses  

     .. math::
        C^<E(t) = c_vector^T * exp(-i * V_matrix * |t|) * c_vector

    Use to fit lesser component of fermionic BCF.  

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
    return np.conj(bcf_fit(t, c_vector, V_matrix))


def make_params_fermion(
    rkr_init: tuple[list] | tuple[ndarray],
    rki_init: tuple[list] | tuple[ndarray],
    zkr_init: tuple[list] | tuple[ndarray],
    zki_init: tuple[list] | tuple[ndarray],
    C0 : tuple[float]
): 
    """
    Create Parameters object initializing fitted parameters for arbitrary number of exponential terms. 

    Each input tuple must be in format (greater, lesser). 
    
    NB: If C0 is provided, the fitted parameters (rLkr[-1], rRkr[-1]) and (rLki[-1], rRki[-1]) for greater/lesser
    components of BCF are constrained such that 
    
    rLkr[-1] = -sum([rLkr[0], ..., rLkr[-2]])
    rRkr[-1] = -sum([rLkr[0], ..., rLkr[-2]])
    
    rLki[-1] =  C0[0] - sum([rLki[0], ..., rLki[-2]])
    rRki[-1] =  C0[1] - sum([rRki[0], ..., rRki[-2]])
    
    Parameters
    ----------
    rkr_init : tuple
        Tuple of initial values for rLkr, rRkr in format (greater, lesser). 
    rki_init : tuple
        Tuple of initial values for rLki, rRki in format (greater, lesser).
    zkr_init : tuple
        Tuple of initial values of zLkr, zRkr in format (greater, lesser).
    zki_init : tuple
        Tuple of initial values of zLki, zRki in format (greater, lesser).
    C0 : tuple 
        Normalization factors for BCFs in format (greater, lesser). 

    Returns
    -------
    params : Parameters
    """
    # Error checks 
    for idx, env in enumerate(['L','R']):
        if not len(rkr_init[idx]) == len(rki_init[idx]):
            raise ValueError(f"r{env}kr_init and r{env}ki_init must have the same length.") 
        if not len(zkr_init[idx]) == len(zki_init[idx]):
            raise ValueError(f"z{env}kr_init and z{env}ki_init must have the same length.")
        if not len(rkr_init[idx]) == len(zkr_init[idx])-1 & len(rki_init[idx]) == len(zki_init[idx])-1:
            raise ValueError(f"Incorrect number of parameter values in r{env}kr_init, r{env}ki_init.")

    params = Parameters()

    params.add('CL0', value=C0[0], vary=False)
    params.add('CR0', value=C0[1], vary=False)

    for idx, env in enumerate(['L','R']):
        for k in np.arange(1,len(zkr_init[idx])+1):
            if k < max(np.arange(1,len(zkr_init[idx])+1)):
                params.add(f"r{env}{k}r", value=rkr_init[idx][k-1])
                params.add(f"r{env}{k}i", value=rki_init[idx][k-1])

            elif k == len(zkr_init[idx]):
                rkr_other = " - ".join(f"r{env}{k}r" for k in np.arange(1,len(rkr_init[idx])+1))
                rki_other = " - ".join(f"r{env}{k}i" for k in np.arange(1,len(rki_init[idx])+1))
            
                params.add(f"r{env}{k}r", expr=f"-{rkr_other}")
                params.add(f"r{env}{k}i", expr=f"C{env}0-{rki_other}")
            
            params.add(f"z{env}{k}r", value=zkr_init[idx][k-1])
            params.add(f"z{env}{k}i", value=zki_init[idx][k-1], min=0)

    return params 


def ps_residuals(
    params,
    w,
    T,
    mu,
    num_terms_gtr,
    num_terms_less,
    ps_gtr,
    ps_less):
    """
    Evaluates error between greater and lesser components of ps and ps_fit_gtr and ps_fit_less. 

    Penalizes fittings based on deviations from detailed balance constraint

    .. math::
    
        \gamma^>(w) / \gamma^<(w) = e^{\beta(w - \mu)}

    Parameters
    ----------
    params : Parameters
        Initial fitting parameter values. 
    w : float, array_like
        Frequency or array of frequenues w.
    T : float 
        Bath temperature. 
    mu : float
        Bath chemical potential.
    num_terms_gtr : int
        Number of terms used to fit ps_gtr. 
    num_terms_less : int
        Number of terms used to fit ps_less. 
    ps_fit_gtr : callable
        Fitted power spectrum (greater).
    ps_fit_lesser : callable
        Fitted power spectrum (lesser). 
    
    Returns
    -------
    Error between fitted and physical BCF.
    """
    # Unpack values of rkr, rki, zkr, zki from params
    # Greater (left)
    rLkr, rLki, zLkr, zLki = extract_params_fermion(params, num_terms_gtr, num_terms_less)[0]
    gL = np.sqrt(-1j * (rLkr + 1j * rLki))
    LmbL = np.diag(zLkr - 1j * zLki)

    # Lesser (right)
    rRkr, rRki, zRkr, zRki = extract_params_fermion(params, num_terms_gtr, num_terms_less)[1]
    gR = np.sqrt(-1j * (rRkr + 1j * rRki))
    LmbR = np.diag(zRkr - 1j * zRki)

    # Evaluate > and < spectrum components
    err_gtr = ps_gtr - ps_fit_gtr(w, c_vector=gL, V_matrix=LmbL)
    err_less = ps_less - ps_fit_less(w, c_vector=gR, V_matrix=LmbR)

    # # Detailed balance constraint
    # err_db = ps_fit_gtr(w, c_vector=gL, V_matrix=LmbL) - np.exp(1/T * (w - mu)) * ps_fit_less(w, c_vector=gR, V_matrix=LmbR)

    err = np.concatenate([err_gtr, err_less])

    return err.view(float)


def ps_fermion(
    w: float | list | ndarray,
    sd: ndarray,
    T: float,
    mu: float
) -> float | ndarray:
    """
    Greater and lesser components of bath power spectrum.

    Greater PS:

    .. math::

    \gamma^>(w) = J(w) * (1 - fd(w))

    Lesser PS:

    .. math::

    \gamma^<(w) = J(w) * fd(w)

    - J(w) is the bath spectral density.
    - fd(w) is the Fermi-Dirac distribution. 

    Parameters
    ----------
    w : float, array_like
        Frequency w or array of frequencies.
    sd : ndarray
        Bath spectral density at frequencies w.
    T : float
        Bath temperature.
    mu : float
        Bath chemical potential.

    Returns
    -------
    Power spectra (ps_gtr, ps_less)
    """
    def _fd(x):
        return 1./(np.exp(x) + 1)

    beta = 1./T
    
    ps_gtr = sd * (1 - _fd(beta * (w - mu)))
    ps_less = sd * _fd(beta * (w - mu))

    return (ps_gtr, ps_less)
    

def ps_fit_params_fermion(
    wlist: float | list | ndarray,
    sd: ndarray,
    T: float,
    mu: float,
    init_params_gtr: tuple[list] | tuple[ndarray],
    init_params_less: tuple[list] | tuple[ndarray],
    C0: tuple[float]
) -> complex | ndarray:
    """
    Fits the greater and lesser PS of a fermionic bath at temperature T 
    and chemical potential mu using the fitting functions 
    
    .. math::
    
        \gamma^>'(w) = 1j * \sum^N_{l=1} (g_Ll ** 2) / (w - z_Ll) + c.c.
    
    and 

    .. math::
    
        \gamma^<'(w) = -1j * \sum^N_{l=1} conj(g_Rl ** 2) / (w - conj(z_Rl)) + c.c. 
    
    Penalizes fitting based on deviations from the detailed balance constraint 

    .. math::

        \gamma^>(w) / \gamma^<(w) = e^{\beta(w - \mu)}

    Parameters
    ----------
    wlist : float, array_like
        Array of frequencies w. 
    sd : ndarray
        Bath spectral density at frequencies w.
    init_params_gtr : tuple
        Initial fitting parameters for \gamma^>'(w) in format (rkr_init, rki_init, zkr_init, zki_init). \\
        The lengths of rkr_init, rki_init must equal N-1. \\
        The lengths of zkr_init, zki_init must equal N.
    init_params_less : tuple
        Initial fitting parameters for \gamma^<'(w) in format (rkr_init, rki_init, zkr_init, zki_init). \\
        Same constraints apply as for init_params_gtr.
    C0 : tuple
        Normalization factors for BCFs in format (greater, lesser).

    Returns
    -------
    Fitted parameters (gL, LmbL) and (gR, LmbR) of greater and lesser parts of PS. 
    """
    ps_gtr, ps_less = ps_fermion(wlist, sd, T, mu)
        
    params = make_params_fermion((init_params_gtr[0], init_params_less[0]),
                                 (init_params_gtr[1], init_params_less[1]),
                                 (init_params_gtr[2], init_params_less[2]),
                                 (init_params_gtr[3], init_params_less[3]),
                                 C0)

    num_terms_gtr = len(max(init_params_gtr, key=len))
    num_terms_less = len(max(init_params_less, key=len))

    result = minimize(ps_residuals, params, method='leastsq', args=(wlist, 
                                                                    num_terms_gtr, 
                                                                    num_terms_less, 
                                                                    T,
                                                                    mu,
                                                                    ps_gtr, 
                                                                    ps_less)) # REQUIRE FITTING FUNCTIONS AS INPUT ARGUMENTS??? 
    rLkr, rLki, zLkr, zLki = extract_params_fermion(result.params, 
                                                    num_terms_gtr,     
                                                    num_terms_less)[0]
    rRkr, rRki, zRkr, zRki = extract_params_fermion(result.params, 
                                                    num_terms_gtr,     
                                                    num_terms_less)[1]

    # Fitted parameters
    gL = np.sqrt(-1j * (rLkr + 1j*rLki))
    gR = np.sqrt(-1j * (rRkr + 1j*rRki))
    LmbL = np.diag(zLkr - 1j*zLki)
    LmbR = np.diag(zRkr - 1j*zRki)

    return (gL, LmbL), (gR, LmbR)
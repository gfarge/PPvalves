""" Utility functions for PPvalves """

# Imports
# =======
import numpy as np

# Core
# ====

def calc_k(VALVES, PARAM, states_override=None):
    """Computes permeability k over the domain, according to valve distribution
    and state.

    Parameters
    ----------
    VALVES : dict.
        Valves parameters dictionnary.
    PARAM : dict.
        Physical parameters dictionnary.
    states_override : 1D array (default `states_override=None`)
        Valve states for which to plot the pore pressure equilibrium profile,
        overriding the valve states in `VALVES['open']`. `True` (or 1) is open,
        `False` (or 0) is closed, dimension of N_valves.

    Returns
    -------
    k : 1D array
        Updated permeability in space. Dimension Nx + 1.

    """
    # Unpack
    h = PARAM['h_']
    Nx = PARAM['Nx']
    k_bg = PARAM['k_bg']

    v_idx = VALVES['idx']
    v_wid = VALVES['width']
    v_k = VALVES['klo']

    if states_override is None:
        v_open = VALVES['open']
    else:
        v_open = states_override.astype(bool)

    # Visit valves and compute k
    #--> Init
    k = np.ones(Nx+2)*k_bg

    # Visit every valve that was active and change its permeability to its
    #  updated value
    v_iterable = zip(v_open, VALVES['idx'], VALVES['width'],\
                     VALVES['klo'])

    for v_is_open, idx, w, klo in v_iterable:
        k[idx+1 : int(idx+w/h+1)] = k_bg*v_is_open + ~v_is_open*klo

    return k

# ----------------------------------------------------------------------------

def calc_Q(P, k, PARAM):
    """Calculates q the massic flux in the domain.

    Parameters
    ----------
    P : 1D array
        Pore pressure in the domain, dimension Nx.
    k : 1D array
        Permeability in the domain, dimension Nx + 1 (in between pressure
        points).
    PARAM : 1D array
        Dictionnary of the system's physical parameters.

    Returns
    -------
    Q : 1D array
        Massic flux in space,  dimension Nx + 1 (in between pressure points).

    """

    # Unpack
    ## Physical and numerical parameters
    h = PARAM['h_']
    mu = PARAM['mu']
    rho = PARAM['rho']
    g = PARAM['g']
    alpha = PARAM['alpha']
    ## Scales
    q_scale = PARAM['q_scale']
    X_scale = PARAM['X_scale']
    P_scale = PARAM['P_scale']

    # Calculate Q
    dpdx = (P[1:] - P[:-1])/h
    # Q_ = -1*rho*k[1:-1]/mu / q_scale * (dpdx_ * P_scale/X_scale + rho*g*np.sin(alpha))
    Q = -1*rho*k[1:-1]/mu / q_scale * (dpdx * P_scale/X_scale) # Pression réduite

    return Q

# ----------------------------------------------------------------------------

def calc_bound_0(P0, PARAM):
    """
    Computes the value of in-flux (resp. input pressure) for fixed pressure
    (resp. fixed flux) at the input fictive point.

    Parameters
    ----------
    P0 : float or 1d array
        Pore pressure at first point of domain. Can be at a given time (float)
        or in time (1d array).
    PARAM : dict.
        Physical parameters dictionnary.

    Returns
    -------
    bound_0 : float or 1d array
        Value of in-flux (resp. input pressure) for fixed pressure (resp. fixed
        flux) at the input fictive point. Depending on input, can be a value at
        a given time, or in time. Same shape as input P0.

    Note:
    -----
    Only 'PP' and 'QP' boundary conditions are implemented.

    """
    # >> Dirichlet : pressure is fixed: bound_0 = qin
    isdir0 = (PARAM['qin_'] == -1)
    bound_dir0 = PARAM['k_bg']*PARAM['rho']/PARAM['mu'] \
                 * (PARAM['p0_'] - P0)/PARAM['hb_'] \
                 * PARAM['P_scale']/PARAM['X_scale']/PARAM['q_scale']

    # >> Neuman : flux is fixed: bound_0 = p0
    isneu0 = (PARAM['p0_'] == -1)
    bound_neu0 = P0 + PARAM['qin_'] * PARAM['hb_'] \
             * PARAM['mu']/PARAM['rho']/PARAM['k_bg'] \
             * PARAM['q_scale']*PARAM['X_scale']/PARAM['P_scale']

    # >> Check if boundary is correctly fixed
    if isdir0 == isneu0:
        raise ValueError("Boundary in 0 is wrongly fixed.")

    # >> Compute input bound
    bound_0 = isneu0 * bound_neu0 + isdir0 * bound_dir0

    return bound_0

# ----------------------------------------------------------------------------

def calc_bound_L(PL, PARAM):
    """
    Computes the value of out-flux (resp. output pressure) for fixed pressure
    (resp. fixed flux) at the output fictive point.

    Parameters
    ----------
    PL : float or 1d array
        Pore pressure at last point of domain. Can be at a given time (float)
        or in time (1d array).
    PARAM : dict.
        Physical parameters dictionnary.

    Returns
    -------
    bound_L : float or 1d array
        Value of out-flux (resp. output pressure) for fixed pressure (resp. fixed
        flux) at the output fictive point. Depending on input, can be a value at
        a given time, or in time. Same shape as input PL.

    Note:
    -----
    Only 'PP' and 'QP' boundary conditions are implemented.

    """
    # >> Dirichlet : pressure is fixed: bound_L = qout
    isdirL = (PARAM['qout_'] == -1)
    bound_dirL = PARAM['k_bg']*PARAM['rho']/PARAM['mu'] \
                * (PL - PARAM['pL_'])/PARAM['hb_'] \
                * PARAM['P_scale']/PARAM['X_scale']/PARAM['q_scale']

    # >> Neuman : input flux is fixed: bound_L = pL
    isneuL = (PARAM['pL_'] == -1)
    bound_neuL = PL - PARAM['qout_'] * PARAM['hb_'] \
                * PARAM['mu']/PARAM['rho']/PARAM['k_bg'] \
                * PARAM['q_scale']*PARAM['X_scale']/PARAM['P_scale']

    # >> Check if boundary is correctly fixed
    if isneuL == isdirL:
        raise ValueError("Boundary in L is wrongly fixed.")

    # >> Compute input bound
    bound_L = isneuL * bound_neuL + isdirL * bound_dirL

    return bound_L

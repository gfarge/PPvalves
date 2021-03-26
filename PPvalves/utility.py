""" Utility functions for PPvalves """

# Imports
# =======
import numpy as np

# Core
# ====

def calc_k(VALVES, PARAM, states_override=None):
    """
    Computes k, according to bg k and valve distribution and state.
    """

    # Unpack
    h = PARAM['h_']
    Nx = PARAM['Nx']
    k_bg = PARAM['k_bg']

    v_idx = VALVES['idx']
    v_wid = VALVES['width']
    v_k = VALVES['klo']

    if type(states_override) == type(None):
        v_open = VALVES['open']
    else:
        v_open = states_override.astype(bool)

    # Visit valves and compute k
    #--> Init
    k = np.ones(Nx+2)*k_bg

    #--> Loop
    for iv in range(len(v_idx)):
        if v_open[iv]:
            pass
        else:
    	    k[ v_idx[iv]+1 : int(v_idx[iv] + v_wid[iv]/h + 1) ] = v_k[iv]

    return k

# ----------------------------------------------------------------------------

def calc_Q(P, k, PARAM):
    """
    Calculates the massic flux in between the input P
    - Parameters
    	+ :param P: pore pressure
    	+ :param k: permeability distribution
    	+ :param PARAM: dictionnary of the system's characteristic parameters
    - Outputs
    	+ :return: Q: The flux calculated everywhere except at the boundaries,
    	in between values of P.

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
    # In the case where input pressure is fixed: bound_0 = q0
#    bound_0 = False * PARAM['k_bg']*PARAM['rho']/PARAM['mu'] \
#                 * (PARAM['p0_'] - P0)/PARAM['hb_'] \
#                 * PARAM['P_scale']/PARAM['X_scale']/PARAM['q_scale'] \
#             + True * (P0 + PARAM['qin_'] * PARAM['hb_'] \
#                  * PARAM['mu']/PARAM['rho']/PARAM['k_bg'] \
#                  * PARAM['q_scale']*PARAM['X_scale']/PARAM['P_scale'])

    if np.isnan(PARAM['qin_']):
        bound_0 = PARAM['k_bg']*PARAM['rho']/PARAM['mu'] \
                 * (PARAM['p0_'] - P0)/PARAM['hb_'] \
                 * PARAM['P_scale']/PARAM['X_scale']/PARAM['q_scale']

    # In the case where input flux is fixed: bound_0 = p0
    elif np.isnan(PARAM['p0_']):
        bound_0 = P0 + PARAM['qin_'] * PARAM['hb_'] \
                  * PARAM['mu']/PARAM['rho']/PARAM['k_bg'] \
                  * PARAM['q_scale']*PARAM['X_scale']/PARAM['P_scale']

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
    # In the case where output pressure is fixed: bound_L = qout_
    if np.isnan(PARAM['qout_']):
        bound_L = PARAM['k_bg']*PARAM['rho']/PARAM['mu'] \
                 * (PL - PARAM['pL_'])/PARAM['hb_'] \
                 * PARAM['P_scale']/PARAM['X_scale']/PARAM['q_scale']

    # In the case where input flux is fixed: bound_L = pL
    elif np.isnan(PARAM['pL_']):
        bound_L = PL - PARAM['qout_'] * PARAM['hb_'] \
                  * PARAM['mu']/PARAM['rho']/PARAM['k_bg'] \
                  * PARAM['q_scale']*PARAM['X_scale']/PARAM['P_scale']

    return bound_L

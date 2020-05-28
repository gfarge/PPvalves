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

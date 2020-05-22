""" Utility functions for PPvalves """

# Imports
# =======
import numpy as np

# Core
# ====

def calc_k(barriers, PARAM, state_override=None):
    """
    Computes k, according to bg k and barrier distribution and state.
    """

    # Unpack
    h = PARAM['h_']
    Nx = PARAM['Nx']
    k_bg = PARAM['k_bg']

    b_idx = barriers['idx']
    b_wid = barriers['width']
    b_k = barriers['klo']

    if type(state_override) == type(None):
        b_open = barriers['open']
    else:
        b_open = state_override.astype(bool)

    # Visit barriers and compute k
    #--> Init
    k = np.ones(Nx+2)*k_bg

    #--> Loop
    for bb in range(len(b_idx)):
        if b_open[bb]:
            pass
        else:
    	    k[ b_idx[bb]+1 : int(b_idx[bb] + b_wid[bb]/h + 1) ] = b_k[bb]

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

""" Study a system with valves around equilibrium """

# Imports
# =======
import numpy as np

# Core
# ====

# ------------------------------------------------------------------

def pp_profile(barriers, PARAM):

    """
    This functions calculates the equilibrium profile of reduced pore
    pressure for a given boundary conditions and permeability distribution.

    - Parameters:
    	+ :param PARAM: Parameters dictionnary.
    	+ :param barriers: barriers dictionnary.
    - Outputs:
    	+ :return Peq: 1D np array, reduced pore pressure equilibrium profile

    """

    # Unpack parameters
    q_scale = PARAM['q_scale']
    X_scale = PARAM['X_scale']
    P_scale = PARAM['P_scale']

    Nx = PARAM['Nx']
    h_ = PARAM['h_']
    hb_ = PARAM['hb_']
    X_ = np.linspace(0,Nx*h_,Nx+1)

    p0_ = PARAM['p0_']
    pL_ = PARAM['pL_']
    qin_ = PARAM['qin_']
    qout_ = PARAM['qout_']

    k_bg = PARAM['k'][0]
    mu = PARAM['mu']
    rho = PARAM['rho']

    # Unpack barriers
    b_close = [not o for o in barriers['open']]
    b_idx = barriers['idx'][b_close]
    b_wid = barriers['width'][b_close]
    b_k   = barriers['klo'][b_close]

    # Compute pressure profile
    if np.isnan(p0_) & np.isnan(qout_):
    # --> Flux boundary condition in 0
    #     pressure boundary condition in L
    #     -->> Define various parameters
        L_ = 1. + 2 * hb_
        dP_ = mu / rho * qin_ * ( (L_ - np.sum(b_wid))/k_bg + \
         np.sum(b_wid/b_k)) * q_scale*X_scale/P_scale
        p0_ = pL_ + dP_
        grad_bg = qin_ * mu / rho / k_bg * q_scale*X_scale/P_scale

        # -->> Initialize: fill the first empty segment of the domain
        Peq_ = np.zeros(Nx+1)
        Peq_[:b_idx[0]+1] = p0_ - (X_[:b_idx[0]+1]+hb_) * grad_bg

        for bb in range(len(b_idx)):
        #	-->> Compute a few quantities
            idx_bb = b_idx[bb] + 1  # first P pt in barrier
            idx_sb = int( b_idx[bb] + b_wid[bb]/h_)  # first P pt in next segment
            if bb+1 < len(b_idx): idx_se = b_idx[bb+1]  # last P pt of segment
            else: idx_se = len(X_)-1  # or last point of domain

            p0_ = Peq_[idx_bb - 1]
            b_X_ = np.linspace(h_, b_wid[bb], b_wid[bb]/h_)
            grad_b = qin_ * mu / rho / b_k[bb] * X_scale/P_scale*q_scale

            s_X_ = np.linspace(h_, (idx_se - idx_sb)*h_, idx_se - idx_sb )

        # -->> Fill a barrier
            Peq_[ idx_bb : idx_sb + 1] = p0_ - grad_b * b_X_
        # -->> Fill the following segment
            p0_ = Peq_[idx_sb]
            Peq_[ idx_sb + 1 : idx_se + 1] = p0_ - grad_bg * s_X_

    elif np.isnan(qin_) & np.isnan(qout_):
    # --> Pressure boundary conditions
    #	-->> Define various parameters
        DP_ = (p0_ - pL_)
        L_ = 1. + 2 * hb_
        grad_bg = DP_ / ( (L_ - np.sum(b_wid)) + k_bg * np.sum(b_wid/b_k) )

        # -->> Initialize: fill the first empty segment of the domain
        Peq_ = np.zeros(Nx+1)
        Peq_[:b_idx[0]+1] = p0_ - (X_[:b_idx[0]+1]+hb_) * grad_bg

        for bb in range(len(b_idx)):
            # -->> Compute a few quantities
            idx_bb = b_idx[bb] + 1  # first P pt in barrier
            idx_sb = int( b_idx[bb] + b_wid[bb]/h_)  # first P pt in next segment
            if bb+1 < len(b_idx): idx_se = b_idx[bb+1]  # last P pt of segment
            else: idx_se = len(X_)-1  # or last point of domain

            p0_ = Peq_[idx_bb-1]
            b_X_ = np.linspace(h_, b_wid[bb], b_wid[bb]/h_)
            grad_b = DP_ / ( b_k[bb]/k_bg * (L_ - np.sum(b_wid)) +\
            b_k[bb] * np.sum(b_wid/b_k) )

            s_X_ = np.linspace(h_,(idx_se - idx_sb)*h_,idx_se - idx_sb )

            # -->> Fill a barrier
            Peq_[ idx_bb : idx_sb + 1] = p0_ - grad_b * b_X_
            # -->> Fill the following segment
            p0_ = Peq_[idx_sb]
            Peq_[ idx_sb + 1 : idx_se + 1] = p0_ - grad_bg * s_X_

    return Peq_

#---------------------------------------------------------------------------------

def calc_k(barriers, PARAM, states=None):
    """
    This function computes the equivalent permeability of a domain with
    barriers, based on the equivalent resistivity formula.

    - Parameters:
    	+ :param barriers: barriers dictionnary
    	+ :param PARAM: physical parameters dictionnary
    	+ :param states: boolean array of barrier states: True is open,
    	False is closed. If states is given, it overrides states in
    	barriers. states can be 1D (N_barriers length), or 2D (N_time x
    	N_barriers). For the latter case, k_eq will be an array of
    	length N_time.
    - Output:
    	+ :return k_eq: float or 1D array, value of equivalent
    	permeability.


    """
    # --> Unpack physical parameters
    k_bg = PARAM['k_bg']
    L = 1.  # barriers widths and domain widths should have the same scale

    # --> Unpack barriers parameters
    L_b = barriers['width']
    R_b = L_b / barriers['klo']

    # -->> Get states
    if type(states)==type(None):
        b_close = np.bitwise_not(barriers['open'].astype(bool))
    else:
        if np.shape(states)[-1] != len(L_b):
    	    raise ValueError('states must have the following dimensions: N_times * N_barriers, but they are: {:}'.format(np.shape(states)))
        b_close = np.bitwise_not(states.astype(bool))

    # --> Compute equivalent resistivity
    if len(np.shape(b_close)) > 1:
        # -->> We compute R_eq at N_times time steps
        N_times = np.shape(b_close)[0]

        R_eq = [ np.sum( R_b[ b_close[tt,:] ] ) +  \
    	        (L - np.sum( L_b[ b_close[tt,:] ] ) ) / k_bg \
                    for tt in range(N_times)]
        R_eq = np.array(R_eq)
    else:
        # -->> We compute R_eq for given time step
        R_eq = np.sum(R_b[b_close]) + (L - np.sum(L_b[b_close])) / k_bg

    k = L / R_eq
    return k

"'"" Study a system with valves around equilibrium """

# Imports
# =======
import numpy as np

# Core
# ====

# ------------------------------------------------------------------

def calc_pp_inf(VALVES, PARAM, states_override=None):
    """
    This functions calculates the equilibrium profile of reduced pore
    pressure for a given boundary conditions and permeability distribution.

    Parameters
    ----------
    PARAM : dictionnary
    	Parameters dictionnary.
    VALVES : dictionnary
    	Valves dictionnary.
    states_override : 1D array (default=None)
        Valve states for which to plot the pore pressure equilibrium profile,
        overriding the valve states in VALVES['open']. True (or 1) is open,
        False (or 0) is closed, dimension of N_valves.
    Returns
    -------
    P_eq : 1D array
    	Reduced pore pressure equilibrium profile, dimension=PARAM['Nx']+1

    Note
    ----
    Not implemented yet: all valve open.

    """
    # Unpack parameters
    # -----------------
    q_scale = PARAM['q_scale']  # massic flux characteristic scale
    X_scale = PARAM['X_scale']  # characteristic length scale
    P_scale = PARAM['P_scale']  # pore pressure characteristic scale

    Nx = PARAM['Nx']  # Number of space steps in domain
    h_ = PARAM['h_']  # space step within domain
    hb_ = PARAM['hb_']  # space step between fict. pt. and domain
    X_ = np.linspace(0, Nx*h_, Nx+1)

    p0_ = PARAM['p0_']  # pore pressure in 0
    pL_ = PARAM['pL_']  # pore pressure in L
    qin_ = PARAM['qin_']  # input flux
    qout_ = PARAM['qout_']  # output flux

    k_bg = PARAM['k_bg']  # background channel permeability
    mu = PARAM['mu']  # dynamic viscosity of fluid
    rho = PARAM['rho']  # mass per unit volume of fluid

    # Unpack valves
    # -------------
    if states_override is None:
        closed_valves = np.bitwise_not(VALVES['open'].astype(bool))
    else:
        closed_valves = np.bitwise_not(states_override.astype(bool))

    # Deal with case where all valves are open
    # ----------------------------------------
    if not any(closed_valves):
        if np.isnan(qin_) & np.isnan(qout_):
            dP_ = (p0_ - pL_)  # pressure diff across domain
            L_ = 1. + 2 * hb_  # domain lenght
            grad_bg = dP_ / L_  # pressure gradient without a valve

            P_eq = p0_ - (X_+hb_) * grad_bg

        elif np.isnan(p0_) & np.isnan(qout_):
            print(pL_)
            L_ = 1. + 2 * hb_
            dP_ = mu / rho / k_bg * qin_ * L_ * q_scale*X_scale/P_scale
            p0_ = pL_ + dP_  # pore pressure in 0
            grad_bg = qin_ * mu / rho / k_bg * q_scale*X_scale/P_scale

            P_eq = p0_ - (X_+hb_)*grad_bg

        return P_eq
        # ------------------------------------

    # Let's continue when some valves are closed
    # ------------------------------------------
    idx_v = VALVES['idx'][closed_valves]  # array of valve space indices
    wid_v = VALVES['width'][closed_valves]  # array of valve width
    k_v   = VALVES['klo'][closed_valves]  # array of closed valve permeability
    R_v = wid_v/k_v  # array of valve resistivity

    # Compute pressure profile
    # ------------------------
    if np.isnan(p0_) & np.isnan(qout_):
    # --> Flux boundary condition in 0
    #     pressure boundary condition in L
    #     -->> Define various parameters
        L_ = 1. + 2 * hb_
        dP_ = mu / rho * qin_ * ((L_ - np.sum(wid_v))/k_bg \
              + np.sum(R_v)) * q_scale*X_scale/P_scale
        p0_ = pL_ + dP_  # pore pressure in 0
        grad_bg = qin_ * mu / rho / k_bg * q_scale*X_scale/P_scale

        # -->> Initialize: fill the first empty segment of the domain
        P_eq = np.zeros(Nx+1)
        P_eq[:idx_v[0]+1] = p0_ - (X_[:idx_v[0]+1]+hb_) * grad_bg

        for iv in range(len(idx_v)):
        #	-->> Compute a few quantities
            idx_v_0 = idx_v[iv] + 1  # first P pt in valve
            idx_s_0 = int(idx_v[iv] + wid_v[iv]/h_)  # first P pt in next segment
            if iv+1 < len(idx_v):
                idx_s_n = idx_v[iv+1]  # last P pt of segment
            else:
                idx_s_n = len(X_)-1  # or last point of domain

            p0_ = P_eq[idx_v_0 - 1]
            v_X_ = np.linspace(h_, wid_v[iv], int(wid_v[iv]/h_))
            grad_v = qin_ * mu / rho / k_v[iv] * X_scale/P_scale*q_scale

            s_X_ = np.linspace(h_, (idx_s_n - idx_s_0)*h_, idx_s_n - idx_s_0 )

        # -->> Fill a valve
            P_eq[ idx_v_0 : idx_s_0 + 1] = p0_ - grad_v * v_X_
        # -->> Fill the following segment
            p0_ = P_eq[idx_s_0]
            P_eq[ idx_s_0 + 1 : idx_s_n + 1] = p0_ - grad_bg * s_X_

    elif np.isnan(qin_) & np.isnan(qout_):
    # --> Pressure boundary conditions
    #	-->> Define various parameters
        dP_ = (p0_ - pL_)
        L_ = 1. + 2 * hb_
        grad_bg = dP_ / ( (L_ - np.sum(wid_v)) + k_bg * np.sum(wid_v/k_v) )

        # -->> Initialize: fill the first empty segment of the domain
        P_eq = np.zeros(Nx+1)
        P_eq[:idx_v[0]+1] = p0_ - (X_[:idx_v[0]+1]+hb_) * grad_bg

        for iv in range(len(idx_v)):
            # -->> Compute a few quantities
            idx_v_0 = idx_v[iv] + 1  # first P pt in valve
            idx_s_0 = int(idx_v[iv] + wid_v[iv]/h_)  # first P pt in next segment
            if iv+1 < len(idx_v): idx_s_n = idx_v[iv+1]  # last P pt of segment
            else: idx_s_n = len(X_)-1  # or last point of domain

            p0_ = P_eq[idx_v_0-1]
            v_X_ = np.linspace(h_, wid_v[iv], int(wid_v[iv]/h_))
            grad_v = dP_ / ( k_v[iv]/k_bg * (L_ - np.sum(wid_v)) +\
            k_v[iv] * np.sum(wid_v/k_v) )

            s_X_ = np.linspace(h_,(idx_s_n - idx_s_0)*h_,idx_s_n - idx_s_0 )

            # -->> Fill a valve
            P_eq[ idx_v_0 : idx_s_0 + 1] = p0_ - grad_v * v_X_
            # -->> Fill the following segment
            p0_ = P_eq[idx_s_0]
            P_eq[ idx_s_0 + 1 : idx_s_n + 1] = p0_ - grad_bg * s_X_

    return P_eq

#---------------------------------------------------------------------------------

def calc_k_eq(VALVES, PARAM, states_override=None, L=None):
    """
    This function computes the equivalent permeability of a domain with
    valves, based on the equivalent resistivity formula.

    Parameters
    ----------
    VALVES : dictionnary
        Valves dictionnary.
    PARAM: dictionnary
        Physical parameters dictionnary
    states_override : ndarray, boolean (default=None)
        Array of valves states: True is open,
    	False is closed. It overrides VALVES['open'].
    	VALVES. States can be 1D (N_valves length), or 2D (N_time x
    	N_valves). For the latter case, k_eq will be an array of
    	length N_time.
    L : float (default to `None`)
        Length of the domain to consider. If not specified, the whole domain
        length is taken.
    Returns
    -------
    k_eq : float or 1D array
        Value of equivalent permeability at given time, or in time, depending
        on shape of states_override.
    """
    # Unpack physical parameters
    # --------------------------
    k_bg = PARAM['k_bg']  # background channel permeability
    if L is None:
        L_ = 1. + 2*PARAM['hb_']  # valves widths and domain widths should have the
                                  # same scale
    else:
        L_ = L

    # Unpack valves parameters
    # ------------------------
    wid_v = VALVES['width']
    R_v =  wid_v / VALVES['klo']  # hydraulic resistance of the valve

    # Get states
    # ----------
    if states_override is None:
        closed_valves = np.bitwise_not(VALVES['open'].astype(bool))
    else:
        closed_valves = np.bitwise_not(states_override.astype(bool))

    # Compute equivalent hydraulic resistance (sum of restistance of each
    # segment and valve)
    # -------------------------------------------------------------------
    # --> At N_times time steps
    if len(np.shape(closed_valves)) > 1:
        N_times = np.shape(closed_valves)[0]

        R_eq = [np.sum(R_v[closed_valves[tt, :]]) \
    	        + (L_ - np.sum(wid_v[closed_valves[tt,:]])) / k_bg \
                for tt in range(N_times)]
        R_eq = np.array(R_eq)

    # --> For given time step
    else:
        R_eq = np.sum(R_v[closed_valves]) \
              + (L_ - np.sum(wid_v[closed_valves])) / k_bg

    keq = L_ / R_eq  # from equivalent resistance to equivalent permeability
    return keq


#---------------------------------------------------------------------------------

def calc_dP_inf(VALVES, PARAM, states_override=None):
    """
    Computes the differential of pore pressure between both ends of the domain,
    at equilibrium in the given valve state.

    Parameters
    ----------
    VALVES : dictionnary
        Valves dictionnary.
    PARAM: dictionnary
        Physical parameters dictionnary
    states_override : ndarray, boolean (default=None)
        Array of valves states: True is open,
    	False is closed. It overrides VALVES['open'].
    	VALVES. States can be 1D (N_valves length), or 2D (N_time x
    	N_valves). For the latter case, k_eq will be an array of
    	length N_time.
    Returns
    -------
    dP_inf : float
        Value of the differential of pore pressure at equilibrium in the input
        system. Important to note that dP is measured at fictive points.
    """
    # Unpack physical parameters
    # --------------------------
    p0_ = PARAM['p0_']  # pore pressure in 0
    pL_ = PARAM['pL_']  # pore pressure in L
    qin_ = PARAM['qin_']  # input flux
    qout_ = PARAM['qout_']  # output flux

    mu = PARAM['mu']  # dynamic viscosity of fluid
    rho = PARAM['rho']  # mass per unit volume of fluid
    k_bg = PARAM['k_bg']  # background channel permeability

    hb_ = PARAM['hb_']  # space step between fict. pt. and domain

    P_scale = PARAM['P_scale']  # pore pressure characteristic scale
    q_scale = PARAM['q_scale']  # massic flux characteristic scale
    X_scale = PARAM['X_scale']  # characteristic length scale

    # Unpack valve parameters
    # -----------------------
    if states_override is None:
        closed_valves = np.bitwise_not(VALVES['open'].astype(bool))
    else:
        closed_valves = np.bitwise_not(states_override.astype(bool))

    wid_v = VALVES['width'][closed_valves]
    R_v = wid_v/VALVES['klo'][closed_valves]

    # Compute dP_inf
    # --------------
    L_ = 1 + 2*hb_
    # --> For 'QP' boundary
    if np.isnan(p0_) & np.isnan(qout_):
        # -->> Equivalent hydraulic resistance of the system
        R_eq = [L_ - np.sum(wid_v[closed_valves])/k_bg + np.sum(R_v[closed_valves])]
        dP_inf = mu / rho * qin_ * R_eq * q_scale*X_scale/P_scale

    # --> For 'PP' boundary
    elif np.isnan(qin_) & np.isnan(qout_):
        dP_inf = p0_ - pL_

    return dP_inf

#---------------------------------------------------------------------------------

def calc_q_inf(VALVES, PARAM, states_override=None):
    """
    Computes the massic flux through the system at equilibrium in the given
    valve state.

    Parameters
    ----------
    VALVES : dictionnary
        Valves dictionnary.
    PARAM: dictionnary
        Physical parameters dictionnary
    states_override : ndarray, boolean (default=None)
        Array of valves states: True is open,
    	False is closed. It overrides VALVES['open'].
    	VALVES. States can be 1D (N_valves length), or 2D (N_time x
    	N_valves). For the latter case, k_eq will be an array of
    	length N_time.
    Returns
    -------
    q_inf : float
        Value of the flux at equilibrium in the input system.
    """
    # Unpack physical parameters
    # --------------------------
    p0_ = PARAM['p0_']  # pore pressure in 0
    pL_ = PARAM['pL_']  # pore pressure in L
    qin_ = PARAM['qin_']  # input flux
    qout_ = PARAM['qout_']  # output flux

    mu = PARAM['mu']  # dynamic viscosity of fluid
    rho = PARAM['rho']  # mass per unit volume of fluid
    k_bg = PARAM['k_bg']  # background channel permeability

    hb_ = PARAM['hb_']  # space step between fict. pt. and domain

    P_scale = PARAM['P_scale']  # pore pressure characteristic scale
    q_scale = PARAM['q_scale']  # massic flux characteristic scale
    X_scale = PARAM['X_scale']  # characteristic length scale

    # Unpack valve parameters
    # -----------------------
    if states_override is None:
        closed_valves = np.bitwise_not(VALVES['open'].astype(bool))
    else:
        closed_valves = np.bitwise_not(states_override.astype(bool))

    wid_v = VALVES['width'][closed_valves]
    R_v = wid_v/VALVES['klo'][closed_valves]

    # Compute dP_inf
    # --------------
    L_ = 1 + 2*hb_
    # --> For 'QP' boundary
    if np.isnan(p0_) & np.isnan(qout_):
        # -->> Equivalent hydraulic resistance of the system
        q_inf = qin_

    # --> For 'PP' boundary
    elif np.isnan(qin_) & np.isnan(qout_):
        dP_inf = p0_ - pL_
        R_eq = ((L_ - np.sum(wid_v))/k_bg + np.sum(R_v)) * X_scale
        q_inf = rho/mu * dP_inf / R_eq * P_scale/q_scale

    return q_inf

#---------------------------------------------------------------------------------

def calc_dP_crit(idx_v0, VALVES, PARAM, event='opening'):
    """
    Computes the critical value of the equilibrium pressure differential across
    a valve system (taken at fictive points) for which opening (resp. closing)
    threshold of a given valve v0 is reached. For now, this function is only
    implemented for event='opening' when all other valves are closed, and
    event='closing' when all other valves are open.

    Parameters
    ----------
    idx_v0 : int
        Index of the valve (python index, from 0 to N valves -1, starting downdip,
        growing updip).
    VALVES : dictionnary
        Valves dictionnary.
    PARAM: dictionnary
        Physical parameters dictionnary
    event : str
        Type of event for which to compute the critical pressure differential.
        'opening' for opening event, 'closing' for closing event.
    Returns
    -------
    dP_crit : float
        Critical value of the equilibrium pressure differential across the
        domain for which opening (resp. closing) threshold is reached.
    """
    # Unpack physical parameters
    # --------------------------
    k_bg = PARAM['k_bg']  # background channel permeability
    L_ = 1 + 2*PARAM['hb_']

    wid_v = VALVES['width']

    if event == 'opening':
        dP_thr_v0 = VALVES['dPhi'][idx_v0]
        R_v = wid_v/VALVES['klo']
    elif event == 'closing':
        dP_thr_v0 = VALVES['dPlo'][idx_v0]
        R_v = wid_v/k_bg

    # --> For the given valve v0
    R_v0 = R_v[idx_v0]

    # Actually compute it
    # -------------------
    R_bg = (L_ - np.sum(wid_v))/k_bg
    R_eq = np.sum(R_v) + R_bg  # Equivalent hydraulic resistance
    dP_crit = dP_thr_v0 * R_eq/R_v0

    return dP_crit

#---------------------------------------------------------------------------------

def calc_q_crit(idx_v0, VALVES, PARAM, event='opening'):
    """
    Computes the critical value of the equilibrium flux in a valve system, for
    which opening (resp.  closing) threshold of a given valve v0 is reached.

    Parameters
    ----------
    idx_v0 : int
        Index of given valve (python index, from 0 to N valves -1, starting
        downdip, growing updip).
    VALVES : dictionnary
        Valves dictionnary.
    PARAM: dictionnary
        Physical parameters dictionnary
    event : str (default='opening')
        Type of event for which to compute the critical flux. 'opening' for
        opening event, 'closing' for closing event.

    Returns
    -------
    q_crit : float
        Critical value of the equilibrium flux for which opening (resp.
        closing) threshold is reached.
    """
    # Unpack physical parameters
    # --------------------------
    q_scale = PARAM['q_scale']  # massic flux characteristic scale
    X_scale = PARAM['X_scale']  # characteristic length scale
    P_scale = PARAM['P_scale']  # pore pressure characteristic scale

    rho = PARAM['rho']  # mass per unit volume of fluid
    mu = PARAM['mu']  # dynamic viscosity of fluid


    # Unpack given valve parameters
    # -----------------------------
    wid_v0 = VALVES['width'][idx_v0]
    k_v0 = VALVES['klo'][idx_v0]

    if event == 'opening':
        dP_thr_v0 = VALVES['dPhi'][idx_v0]
        k_v0 = VALVES['klo'][idx_v0]
    elif event == 'closing':
        dP_thr_v0 = VALVES['dPlo'][idx_v0]
        k_v0 = PARAM['k_bg']

    # Actually compute it
    # -------------------
    q_crit = rho/mu * k_v0 / wid_v0 * dP_thr_v0 * P_scale/X_scale/q_scale

    return q_crit

#---------------------------------------------------------------------------------

def calc_k_eff(bounds_eff, PARAM):
    """
    Computes the effective permeability of the active system in permanent
    regime.

    Parameters
    ----------
    bounds_eff : list
        Effective value of the free variable (e.g. pressure if flux is fixed,
        and vice versa) at the input and output boundaries. First input then
        output in the list.
    PARAM : dictionnary
    	Parameters dictionnary.

    Returns
    -------
    k_eff : float
        Effective value of permeability.

    Note
    ----
    For now, only 'QP' and 'PP' are implemented.
    """
    # According to boundary conditions, get cross system delta_p, input flux
    # and length between boundaries
    if PARAM['bound'][0] == 'Q':
        q_in = PARAM['qin_']
        delta_p = bounds_eff[0]
    else:
        q_in = bounds_eff[0]
        delta_p = PARAM['p0_']

    # For now, length scale is going to be 1, no boundary depth taken into
    # account
    L = (1 + 2*PARAM['hb_'])

    # Compute effective permeability
    k_eff = PARAM['mu'] / PARAM['rho'] * L / delta_p * q_in \
           * PARAM['X_scale'] / PARAM['P_scale'] * PARAM['q_scale']

    return k_eff

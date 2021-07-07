""" Compute mass balance in the system, different ways."""

# - TODO -----------------------------------------------------------------
# - Archive function dflux, simply use in for dm(x), dmdt(x)
# ------------------------------------------------------------------------

# Imports
# =======

import numpy as np
from PPvalves.utility import calc_k, calc_Q

# Core
# ====

def in_out(IN, PARAM, bounds=False, int_t=True, verbose=False):
    r"""
    Computes the time derivative of the total mass (per unit area) along the
    domain, at time $t$.
    \[
    \frac{dM}{dt}(t) = q_{in}(t) - q_{out}(t).
    \]

    Parameters
    ----------
    IN : ndarray
        Default : pore-pressure history, dimensions 2D : Nt,Nx. If
        `bounds=True`, value of the free variable at each bound, dimensions 2D:
        Nt, 2 (in, out).
    PARAM : dict
    	Parameters dictionnary.
    bounds : bool (default=`False`)
        Option to input value of the free variable at each bound instead of
        full pore pressure history.
    int_t : bool, default = False
    	Set to True fo option to compute $\delta M(t)$ instead of
    	$\frac{dM}{dt}(t)$.
    verbose : bool, default = False
    	Make this function verbose.

    Returns
    -------
    deltaM : ndarray (default, for int_t=True)
    	Accumulation in time of total mass per unit area in domain,
    	dimensions: Nt
    dMdt : ndarray (for int_t=False)
    	Mass accumulation rate in the system in
    	time, dimension : Nt

    """
    # Unpack
    # ------
    # --> A clearer name for P/bounds
    if bounds:
        bounds_in_t = IN
    else:
        P = IN
    # --> Boundary conditions
    qin_ = PARAM['qin_']
    qout_ = PARAM['qout_']
    p0_ = PARAM['p0_']
    pL_ = PARAM['pL_']
    hb_ = PARAM['hb_']
    dt_ = PARAM['dt_']
    # --> Physical parameters
    rho = PARAM['rho']
    mu = PARAM['mu']
    k_bg = PARAM['k_bg']
    # --> Scales
    X_scale = PARAM['X_scale']
    P_scale = PARAM['P_scale']
    q_scale = PARAM['q_scale']
    T_scale = PARAM['T_scale']
    M_scale = q_scale*T_scale

    # According to boundary, compute in/out flux
    # ------------------------------------------
    if (qin_==-1) and (p0_!=-1):
        # --> Fixed pressure in 0
        if bounds:
            qin_ = bounds_in_t[:, 0]
        else:
            qin_ = rho*k_bg/mu * (p0_ - P[:, 0])/hb_ * P_scale/X_scale / q_scale
    elif (qin_!=-1) and (p0_==-1):
        # --> Fixed flux in 0
        pass
    if (qout_==-1) and (pL_!=-1):
        # --> Fixed pressure in L
        if bounds:
            qout_ = bounds_in_t[:, -1]
        else:
            qout_ = rho*k_bg/mu * (P[:, -1] - pL_)/hb_ * P_scale/X_scale / q_scale
    elif (qout_!=-1) and (pL_==-1):
        # --> Fixed flux in L
        pass

    # --> dMdt
    dMdt = (qin_ - qout_) * q_scale*T_scale/M_scale

    if int_t:
        # --> Compute mass evolution history
        deltaM = np.cumsum(dMdt*dt_, axis=0)
        if verbose:
            print('mb.in_out -- total mass accumulation in time, deltaM')
        return deltaM

    else:
        # --> Simply output mass derivative in time
        if verbose:
            print('mb.in_out -- total mass accumulation rate in time, dMdt')
        return dMdt

#---------------------------------------------------------------------------------

def in_pores(P, PARAM, int_x=True, int_t=True, verbose=False):
    r"""
    Computes the time derivative of volumic mass over the space domain, at
    time $t$, using:
    \[
    \frac{\partial m}{\partial t}(x,t) = \rho\beta\phi \frac{\partial p}{\partial t}(x,t).
    \]

    Parameters
    ----------
    P : ndarray
    	Pore-pressure history, dimensions 2D : Nt,Nx.
    PARAM : dict
    	Parameters dictionnary.
    int_x : bool (default=True)
    	Option to integrate over the space domain, to obtain
    	$\frac{dM}{dt}(t)$, the total mass derivative over the domain.
    	True activates it, False (default) deactivates it.
    int_t : bool (default=True)
    	Option to integrate over the time domain, to obtain
    	$\delta m(x,t)$ (or $\delta M(t)$ if int_x is True),
    	the total mass accumulated from the initial time.
    	True activates it, False (default) deactivates it.
    verbose : bool (default=False)
        Option to have the function print which variable it is computing.

    Returns
    -------
    deltaM : ndarray (default, for int_x=True,int_t=True)
    	Accumulation in time of total mass per unit area in domain,
    	dimensions: Nt
    dMdt : ndarray (for int_x=True,int_t=False)
    	Variation in time of total mass per unit area in domain,
    	dimensions: Nt
    deltam : ndarray (for int_x=False,int_t=True)
    	Accumulation of mass per unit area per unit length of space
    	domain, at each point of space domain, dimensions: Nt,Nx
    dmdt : ndarray (for int_x=False,int_t=True)
    	Variation in time mass per unit area per unit length of space
    	domain, at each point of space domain, dimensions: Nt,Nx

    """
    # --> Unpack parameters
    # -->> Physical parameters
    phi = PARAM['phi']
    rho = PARAM['rho']
    beta = PARAM['beta']
    # -->> Physical scales
    P_scale = PARAM['P_scale']
    X_scale = PARAM['X_scale']
    T_scale = PARAM['T_scale']
    q_scale = PARAM['q_scale']
    M_scale = q_scale * T_scale
    # -->> Numerical parameters
    h_ = PARAM['h_']
    hb_ = PARAM['hb_']
    dt_ = PARAM['dt_']

    # --> Compute time derivative of pressure
    dPdt = (P[1:, :] - P[:-1, :])/dt_

    # --> Compute mass
    dmdt = rho * beta * phi * dPdt * P_scale/M_scale

    if int_x:
    # --> Total over x-domain
        x_steps = np.ones(len(P[0, :])) * h_
        x_steps[0] = (h_ + hb_)/2
        x_steps[-1] = (h_ + hb_)/2
        dMdt = np.sum(dmdt*x_steps, axis=1) * X_scale
        if int_t:
        # -->> Total mass accumulation over domain
            if verbose: print('mb.in_pores -- computing total mass accumulation in time, deltaM')
            deltaM = np.cumsum(dMdt*dt_)
            return deltaM
        else:
        # -->> Variation of total mass over the domain
            if verbose: print('mb.in_pores -- computing total mass accumulation rate in time, dMdt')
            return dMdt
    else:
    # --> Mass x-profile
        if int_t:
        # -->> Mass accumulation per unit length for each x, at each t(.
            if verbose: print('mb.in_pores -- computing mass accumulation in time, at each point of domain, deltam')
            deltam = np.cumsum(dmdt*dt_, axis=0)
            return deltam

        else:
        # -->> Variation of mass per unit length for each x, at each t.
            if verbose: print('mb.in_pores -- computing mass accumulation rate in time, at each point of domain, dmdt')
            return dmdt

#---------------------------------------------------------------------------------

def dflux(P, VALVES, states, PARAM, int_x=True, int_t=True, verbose=False):
    r"""
    Computes the surfacic mass history, using mass balance at each point of
    the domain.

    Parameters
    ----------
    P : ndarray
    	Pore-pressure history, dimensions 2D : Nt,Nx.
    VALVES : dict
    	Valves dictionnary.
    states : ndarray
    	Valves state in time, array with boolean-like values: True is
    	open False is closed, dimensions : Nt, Nvalves.
    PARAM : dict
    	Parameters dictionnary.
    int_x : bool (default=False)
    	Option to integrate over the space domain, to obtain
    	$\frac{dM}{dt}(t)$, the total mass derivative over the domain.
    	True activates it, False (default) deactivates it.
    int_t : bool (default=False)
    	Option to integrate over the time domain, to obtain
    	$\delta m(x,t)$ (or $\delta M(t)$ if int_x is True),
    	the total mass accumulated from the initial time.
    	True activates it, False (default) deactivates it.
    verbose : bool (default=False)
        Option to have the function print which variable it is computing.

    Returns
    -------
    deltaM : ndarray (default, for int_x=True, int_t=True)
    	Mass accumulation in time over the whole space domain,
    	dimensions 1D : Nt
    dMdt : ndarray (default, for int_x=True, int_t=False)
    	Time derivative of total mass accumulation over the space
    	domain, in time, dimensions 1D : Nt
    deltam : ndarray (for int_x=False, int_t=True)
    	Accumulation of mass per unit length in time at each point of the
    	domain, dimensions 2D : Nt,Nx
    dmdt : ndarray (for int_x=False, int_t=False)
    	Time derivative of accumulation of mass per unit length in time
    	at each point of the domain, dimensions 2D : Nt,Nx

    Note:
    -----
        This function gives a wrong result: there is a problem with how the
        mass balance is actually calculated in our
    """
    # --> Unpack
    dt_ = PARAM['dt_']
    h_ = PARAM['h_']
    hb_ = PARAM['h_']
    X_scale = PARAM['X_scale']

    # --> Initialize
    Nt = np.shape(P)[0]

    # --> Through time, compute mass increment
    dmdt = np.zeros((Nt-1, np.shape(P)[1]))
    for tt in range(Nt-1):
        # -->> Compute permeability profile as a function of valves
        #      state
        k = calc_k(VALVES, PARAM, state_override=states[tt, :])

        # -->> Compute mass increment
        dmdt[tt,:] = calc_dmdt_dflux(P[tt], P[tt+1], k, PARAM)

    if int_x:
    # --> Total over x-domain
        x_steps = np.ones(len(P[0, :])) * h_
        x_steps[0] = (h_ + hb_)/2
        x_steps[-1] = (h_ + hb_)/2
        dMdt = np.sum(dmdt*x_steps, axis=1) * X_scale
        if int_t:
        # -->> Total mass accumulation over domain
            if verbose:
                print('mb.dflux -- computing total mass accumulation in time, deltaM')
            deltaM = np.cumsum(dMdt*dt_)
            return deltaM
        else:
        # -->> Variation of total mass over the domain
            if verbose:
                print('mb.dflux -- computing total mass accumulation rate in time, dMdt')
            return dMdt
    else:
    # --> Mass x-profile
        if int_t:
    	# -->> Mass accumulation per unit length for each x, at each t.
            if verbose:
                print('mb.dflux -- computing mass accumulation in time, at each point of domain, deltam')
            deltam = np.cumsum(dmdt*dt_, axis=0)
            return deltam

        else:
    	# -->> Variation of mass per unit length for each x, at each t.
            if verbose:
                print('mb.dflux -- computing mass accumulation rate in time, at each point of domain, dmdt')
            return dmdt


#---------------------------------------------------------------------------------


def calc_dmdt_dflux(Pprev, Pnext, k, PARAM):
    r"""
    Computes the time derivative of volumic mass over the space domain, at
    time $t$, using:
    \[
    \frac{\partial m}{\partial t}(x,t) = \frac{\partial q}{\partial x}(x,t).
    \]

    Parameters
    ----------
    P : ndarray
    	Pore-pressure history, dimensions 2D : Nt,Nx.
    k : ndarray
    	Permeability profile, dimension : Nx+1.
    PARAM : dict
    	Parameters dictionnary.

    Returns
    -------
    dmdt : ndarray
    	Time derivative of mass at time t, over the whole space domain,
    	dimension : Nx

    Notes
    -----
    	This method of computing the mass_balance is much more expensive than
        the previous ones. It is more or less equivalent to re-run the
        simulation, and should exactly be equal to the previous functions. Use
        only for comparison purposes.
        Implementation of boundary is approximative: only FCTS, not
        Crank-Nicholson. Difference is marginal as there are no sudden
        variations in time near boundaries.

    """
    # Unpack parameters
    # -----------------
    # --> Physical parameters
    mu = PARAM['mu']
    rho = PARAM['rho']
    # --> Boundary conditions
    p0_ = PARAM['p0_']
    pL_ = PARAM['pL_']
    qin_ = PARAM['qin_']
    qout_ = PARAM['qout_']
    # -->> Physical scales
    P_scale = PARAM['P_scale']
    X_scale = PARAM['X_scale']
    T_scale = PARAM['T_scale']
    q_scale = PARAM['q_scale']
    M_scale = q_scale * T_scale
    # -->> Numerical parameters
    h_ = PARAM['h_']
    hb_ = PARAM['hb_']

    # Compute dmdt over the domain
    # ----------------------------
    dmdt = np.zeros(len(Pnext))
    # --> Within the domain
    Qprev = calc_Q(Pprev, k, PARAM)
    Qnext = calc_Q(Pnext, k, PARAM)
    Q = (Qnext + Qprev)/2
    dmdt[1:-1] = (Q[:-1] - Q[1:])/h_ * q_scale/X_scale * T_scale/M_scale

    # --> At boundaries (pbs with variable space step)
    # -->> Fixed flux in 0
    if p0_ == -1 and (qin_ != -1):
        p0_ = Pnext[0] + qin_ * hb_ * mu / rho / k[0] * q_scale*X_scale / P_scale
        d2pdx2_0 = (h_*p0_ - (h_+hb_)*Pnext[0] + hb_*Pnext[1])\
                / (h_*hb_ * (h_+hb_)/2)
        dmdt[0] = d2pdx2_0 * k[0] * rho / mu\
                  * P_scale/X_scale**2 * T_scale/M_scale

    # -->> Fixed pressure in 0
    elif (p0_ != -1) and (qin_==-1):
        d2pdx2_0 = (h_*p0_ - (h_+hb_)*Pnext[0] + hb_*Pnext[1])\
                / (h_*hb_ * (h_+hb_)/2)
        dmdt[0] = d2pdx2_0 * k[0] * rho / mu\
                  *P_scale/X_scale**2 * T_scale/M_scale

    # -->> Fixed flux in L
    if (pL_ == -1) and (qout_ != -1):
        pL_ = Pnext[-1] - qout_ * hb_ * mu / rho / k[-1] * q_scale*X_scale / P_scale
        d2pdx2_L = (hb_*Pnext[-2] - (h_+hb_)*Pnext[-1] + h_*pL_)\
                / (h_*hb_ * (h_+hb_)/2)
        dmdt[-1] = d2pdx2_L * k[-1] * rho / mu\
                  * P_scale/X_scale**2 * T_scale/M_scale
    # -->> Fixed pressure in L
    elif (pL_ != -1) and (qout_==-1):
        d2pdx2_L = (hb_*Pnext[-2] - (h_+hb_)*Pnext[-1] + h_*pL_)\
                / (h_*hb_ * (h_+hb_)/2)
        dmdt[-1] = d2pdx2_L * k[-1] * rho / mu\
                  * P_scale/X_scale**2 * T_scale/M_scale

    return dmdt

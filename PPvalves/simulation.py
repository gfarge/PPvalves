#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Module to run PPvalves in different ways, and save the results.

Solves for the pore pressure diffusion in time, and communicates with
`PPvalves.mat_math`, `PPvalves.valves` and `PPvalves.initialize` to handle
initializing the numerical system, valve evolution, and inversion of the
matrix.
"""


## >> Imports
import copy
import time
import numpy as np
import pickle

import PPvalves.mat_math as mat
import PPvalves.initialize as init
import PPvalves.valves as valv
import PPvalves.utility as util


def run_nov(P, PARAM, verbose=True):
    """Solves fluid pressure diffusion without valves.

    Propagates an initial state of the pore-pressure in time. Permeability can
    be heterogeneous, but will not be dynamic.

    Parameters
    ----------
    P : 2D array
        Initialized pore pressure array `P[0, :] = P0`. First dimension should
        be time, second is space.
    PARAM : dict
        Physical parameters dictionnary. Permeability in space can be input as
        an array in `PARAM['k']` --- as it is defined in between and around
        pressure points, its space dimension is 1 element longer than that of
        the pore pressure array.
    verbose : bool, optional
        Option to have the function print what it's doing.

    Returns
    -------
    P : 2D array
        Pore pressure evolution in time and space (same dimension as input).
    """
    # >> Unpack
    Nt = PARAM['Nt']

    # >> Set up initial system
    if verbose: print('simulation.run_nov -- sytsem setup...')
    A, B, b = init.build_sys(PARAM)

    # >> Loop through time
    if verbose: print('simulation.run_nov -- starting run...')
    for tt in range(Nt):
        d = mat.product(B, P[tt,:]) + b  # compact form of knowns
        P[tt+1,:] = mat.TDMAsolver(A,d)  # solving the system for t+1

    if verbose: print('simulation.run_nov -- Done !'
    return P

#---------------------------------------------------------------------------------

def run_light(P0, PARAM, VALVES, verbose=True):
    r"""Runs PPv, without saving the full pressure history.

    Solves diffusion equation and actions valves. Does not store
    pressure history, simply previous and next times. Used to deal with memory
    issues.

    Parameters
    ----------
    P0 : 1D array
        Initial state of pore pressure in the system, dimension
        `PARAM['Nx'] + 1`.
    VALVES : dict
        Valves parameters dictionnary.
    PARAM : dict.
        Dictionnary of physical parameters describing the system.
    verbose : bool, optional
        Have the function print what it's doing.

    Returns
    -------
    Plast : 1D array
        Last state of pore pressure across the domain, dimension
        `PARAM['Nx'] + 1`.
    bounds_in_t : 2D array
        Free bound variable (*e.g.* flux if pressure is fixed) at in- and
        output fictive points in time, dimension (`PARAM['Nt'] + 1`, 2).
    v_activity : 3D array
        Valve state and pressure differential for each valves, at all times.
        Dimensions : (`PARAM['Nt'] + 1`, 2, Nvalves). First column:
        valve states in time (`True` or `1` is open, `False` or `0` is closed),
        second column: $\delta p$ across valve (taken at pressure points
        right outside low permeability zone). Used to compute catalogs of
        events.
    trun : dict.
        Dictionnary with the details of the time spent on computing the product
        for the knowns (`trun['prod']`), the time spent on solving for next
        state of pressure (`trun['solve']`), the time spent on actionning
        valves (`trun['valve']`).  The sum of all three is the complete
        runtime.
    """
    if verbose: print('simulation.run_light -- initialization...')
    # Create variables for useful values
    # ==================================
    Nt = PARAM['Nt']  # number of time steps for this simulation
    h = PARAM['h_']  # space step

    # Initialization steps
    # ====================
    trun = {'prod' : 0, 'solve' : 0, 'valve' : 0}  # runtime dictionnary

    # --> Locate valves, initialize valve activity
    v_activity = np.zeros((Nt+1,2, len(VALVES['idx'])))
    v_activity[0, 0, :] = VALVES['open']  # initialize valve states

    v_id1 = VALVES['idx']  # Pressure pt. right before valves
    v_id2 = VALVES['idx'] + VALVES['width']/h  # Pr. pt. right after valves
    v_id2 = v_id2.astype(int)

    v_activity[0, 1, :] = P0[v_id1] - P0[v_id2]

    # Initialize bound 0
    # ------------------
    bounds_in_t = np.zeros((Nt+1, 2))
    bounds_in_t[0, 0] = util.calc_bound_0(P0[0], PARAM)
    bounds_in_t[0, 1] = util.calc_bound_L(P0[-1], PARAM)

    # Set up matrix system
    # --------------------
    if verbose: print('simulation.run_light -- building system...')
    A, B, b = init.build_sys(PARAM)

    # Solving in time
    # ===============
    if verbose: print('simulation.run_light -- starting run...')
    Pnext = P0
    for tt in range(Nt):
        Pprev = Pnext  # Stepping forward

        # Compute knowns (context)
        # ------------------------
        tprod0 = time.time()  # start timer for product

        r = mat.product(B, Pprev) + b # calc knowns (right hand side)

        trun['prod'] = trun['prod'] + time.time() - tprod0  # add elapsed t

        # Solve for next time
        # -------------------
        tsolve0 = time.time()  # start timer for solver

        Pnext = mat.TDMAsolver(A,r) # solve system
        bounds_in_t[tt+1, 0] = util.calc_bound_0(Pnext[0], PARAM)  # get bound 0
        bounds_in_t[tt+1, 1] = util.calc_bound_L(Pnext[-1], PARAM) # get bound L

        trun['solve'] = trun['solve'] + time.time() - tsolve0 # add elapsed t

        # Manage valve evolution
        # ----------------------
        tvalve0 = time.time()  # start timer for valves

        VALVES, active_valves = valv.evolve(Pnext, h, VALVES)

        #--> Build new system according to new valve states
        if np.any(active_valves):
            PARAM['k'] = valv.update_k(VALVES, active_valves, PARAM)
            A, B, b = init.build_sys(PARAM) # update system with new permeab.
        trun['valve'] = trun['valve'] + time.time() - tvalve0  # add elapsed t
        # --> Update v_activity
        v_activity[tt+1, 0, :] = VALVES['open']
        v_activity[tt+1, 1, :] = VALVES['dP']
    if verbose: print('simulation.run_light -- Done!')

    Plast = Pnext

    return Plast, bounds_in_t, v_activity, trun

# ----------------------------------------------------------------------------

def run(P, PARAM, VALVES, verbose=True):
    """Runs PPv.

    Solves diffusion equation and actions valves. Stores and returns full
    pressure history.

    Parameters
    ----------
    P : 2D array
        Initialized matrix of pore pressures. Dimensions (`PARAM['Nt'] + 1`,
        `PARAM['Nx'] + 1`).
    VALVES : dict.
        Valves parameters dictionnary.
    PARAM : dict.
        Dictionnary of physical parameters describing the system.

    Returns
    -------
    P : 2d array
        Pore pressure history, same shape as input P.
    v_activity : 3D array
        Valve state and pressure differential for each valves, at all times.
        Dimensions : (`PARAM['Nt'] + 1`, 2, Nvalves). First column:
        valve states in time (`True` or `1` is open, `False` or `0` is closed),
        second column: $\delta p$ across valve (taken at pressure points
        right outside low permeability zone). Used to compute catalogs of
        events.
    trun : dict.
        Dictionnary with the details of the time spent on computing the product
        (trun['prod']), the time spent on solving for next state of pressure
        (trun['solve']), the time spent on actionning valves (trun['valve']).
        The sum of all three is the complete runtime.
    """
    if verbose: print('simulation.run -- initialization...')
    # Create variables for useful values
    # ==================================
    Nt = PARAM['Nt']  # number of time steps for this simulation
    h = PARAM['h_']  # space step

    # Initialization steps
    # ====================
    trun = {'prod' : 0, 'solve' : 0, 'valve' : 0}  # runtime dictionnary

    # --> Locate valves, initialize valve activity
    v_activity = np.zeros((Nt+1,2, len(VALVES['idx'])))
    v_activity[0, 0, :] = VALVES['open']  # initialize valve states

    v_id1 = VALVES['idx']  # Pressure pt. right before valves
    v_id2 = VALVES['idx'] + VALVES['width']/h  # Pr. pt. right after valves
    v_id2 = v_id2.astype(int)

    v_activity[0, 1, :] = P[0, v_id1] - P[0, v_id2]

    # Set up matrix system
    # --------------------
    A, B, b = init.build_sys(PARAM)

    # Solving in time
    # ===============
    if verbose: print('simulation.run -- starting run...')
    for tt in range(Nt):
        # Compute knowns (context)
        # ------------------------
        tprod0 = time.time()  # start timer for product

        r = mat.product(B, P[tt, :]) + b # calc knowns (right hand side)

        trun['prod'] = trun['prod'] + time.time() - tprod0  # add elapsed t

        # Solve for next time
        # -------------------
        tsolve0 = time.time()  # start timer for solver

        P[tt+1, :] = mat.TDMAsolver(A, r) # solve system

        trun['solve'] = trun['solve'] + time.time() - tsolve0 # add elapsed t

        # Manage valve evolution
        # ----------------------
        tvalve0 = time.time()  # start timer for valves

        VALVES, active_valves = valv.evolve(P[tt+1, :], h, VALVES)

        #--> Build new system according to new valve states
        if np.any(active_valves):
            PARAM['k'] = valv.update_k(VALVES, active_valves, PARAM)
            A, B, b = init.build_sys(PARAM) # update system with new permeab.
        trun['valve'] = trun['valve'] + time.time() - tvalve0  # add elapsed t
        # --> Update v_activity
        v_activity[tt+1, 0, :] = VALVES['open']
        v_activity[tt+1, 1, :] = VALVES['dP']

    if verbose: print('simulation.run -- Done!')
    return P, v_activity, trun

#---------------------------------------------------------------------------------

def run_time(PARAM):
    r"""Rough estimation of PPv run time for given physical parameters.

    This gives a rough, slightly overestimated approximation. It does not
    account for the specificity of valve distribution and characteristics.

    Parameters
    ----------
    PARAM : dict
        Physical parameters dictionnary.

    Returns
    -------
    trun : float
        Estimated time in second taken for a run with such parameters.

    Notes
    -----
    `trun` is proportionnal to $N_x$ (because of the computation of matrix and
    product) and to $N_t$, which is itself proportionnal to $N_x^2$ and `Ttot`,
    the total physical run time. Then, `trun` $\sim N_x^3$ x `Ttot`.
    Experimentally, the proportionnality constant is around 8e-6.

    """
    # >> Unpack
    Nx = PARAM['Nx']
    Nt = PARAM['Nt']
    Ttot_ = PARAM['dt_']*Nt
    A = 8.e-6  # empirical proportionnality constant

    # >> Compute
    trun = A * Nx**3 * Ttot_

    print('simulation.run_time -- This run should be approximately {:.2e} seconds long'.format(trun))
    return trun

# -----------------------------------------------------------------------------

def save(path, dic, verbose=True):
    """Saves the output of a simulation.

    Output should be packaged in a dictionnary first. It is then saved using the
    pickle package. Simple wrapper function.

    Parameters
    ----------
    path : str
       Path and filename where to save the file. No extension needed.
    dic : dictionnary
        Dictionnary packaging all output to be saved. Be careful to use
        stereotypical names for the keys: 'PARAM', 'VALVES', 'v_activity',
        'P0', 'Plast', 'bounds', 't_ev', 'x_ev', 'k_eq'...
    verbose : bool, optional
        Option to have the function print what it's doing.

    """
    if path[-4:] != '.pkl':
        path += '.pkl'  # add extension if not present

    if verbose : print('simulation.save -- saving at {:}...'.format(path))
    pickle.dump(dic, open(path, 'wb'))

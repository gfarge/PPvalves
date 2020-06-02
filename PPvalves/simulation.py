#!/usr/bin/python
# -*- coding: utf-8 -*-


""" pp_valves solves for the pore pressure diffusion in time, and handles valves
opening and closing. Set up is handled by  setup.py. Matrix computations are handled
by matmath.py."""


## Imports
import copy
import time
import numpy as np
import pickle

import PPvalves.mat_math as mat
import PPvalves.initialize as init
import PPvalves.valves as valv
import PPvalves.utility as util


def run_nov(P,PARAM):
    """
    Propagates an initial state of the pore-pressure in time. valves but no
    dynamic valves.
    valves.
    """
    # Unpack
    Nt = PARAM['Nt']

    # Set up initial system
    print('propagate -- sytsem setup...')
    A, B, b = init.build_sys(PARAM)

    # Go through time
    print('propagate -- starting run...')
    for tt in range(Nt):
        d = mat.product(B,P[tt,:]) + b  # compact form of knowns
        P[tt+1,:] = mat.TDMAsolver(A,d)  # solving the system

    print('propagate -- Done !')
    return P

#---------------------------------------------------------------------------------

def run_light(P0, PARAM, VALVES):
    """
    Runs PPV, solving diffusion equation and actionning valves. Does not store
    pressure history, simply previous and next times. If pressure is fixed
    downdip, returns in-flux, if flux is fixed, returns entry pressure. This
    serves as an effective diagnostic of the system.

    Parameters
    ----------
    P0 : 1d array
        Initial state of pore pressure in the system. Dimension PARAM['Nx'] +
        1.
    VALVES : dict.
        Valves parameters dictionnary.
    PARAM : dict.
        Dictionnary of physical parameters describing the system.

    Returns
    -------
    bound_in_t : 1d array
        Flux (resp. pressure) at entry fictive point in time, dimension
        PARAM['Nt'] + 1.
    v_activity : 3d array
        Indicators of valve state and pressure differential witnessed at all
        times. Used to compute catalogs of events. Dimensions : PARAM['Nt'] + 1
        * 2 * N_valves. First column: states (True or 1 is open, False or 0 is
        closed), second column: dP across valve (taken at pressure points
        right outside low permeability zone).
    trun : dict.
        Dictionnary with the details of the time spent on computing the product
        (trun['prod']), the time spent on solving for next state of pressure
        (trun['solve']), the time spent on actionning valves (trun['valve']).
        The sum of all three is the complete runtime.
    """
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
    bound_in_t = np.zeros(Nt+1)
    bound_in_t[0] = util.calc_bound_0(P0[0], PARAM)

    # Set up matrix system
    # --------------------
    A, B, b = init.build_sys(PARAM)

    # Solving in time
    # ===============
    Pnext = P0
    for tt in range(Nt):
        Pprev = Pnext  # Stepping forward

        # Compute knowns (context)
        # ------------------------
        tprod0 = time.time()  # start timer for product

        r = mat.product(B,Pprev) + b # calc knowns (right hand side)

        trun['prod'] = trun['prod'] + time.time() - tprod0  # add elapsed t

        # Solve for next time
        # -------------------
        tsolve0 = time.time()  # start timer for solver

        Pnext = mat.TDMAsolver(A,r) # solve system
        bound_in_t[tt+1] = util.calc_bound_0(Pnext[0], PARAM)  # get bound 0

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

    return bound_in_t, v_activity, trun

# ----------------------------------------------------------------------------

def run(P, PARAM, VALVES, verbose=False):
    """
    Runs PPV, solving diffusion equation and actionning valves. Stores and
    returns pressure history.

    Parameters
    ----------
    P : 2d array
        Initialized matrix of pore pressures. Dimensions PARAM['Nt'] + 1 *
        PARAM['Nx'] + 1.
    VALVES : dict.
        Valves parameters dictionnary.
    PARAM : dict.
        Dictionnary of physical parameters describing the system.

    Returns
    -------
    P : 2d array
        Pore pressure history, same shape as input P.
    v_activity : 3d array
        Indicators of valve state and pressure differential witnessed at all
        times. Used to compute catalogs of events. Dimensions : PARAM['Nt'] + 1
        * 2 * N_valves. First column: states (True or 1 is open, False or 0 is
        closed), second column: dP across valve (taken at pressure points
        right outside low permeability zone).
    trun : dict.
        Dictionnary with the details of the time spent on computing the product
        (trun['prod']), the time spent on solving for next state of pressure
        (trun['solve']), the time spent on actionning valves (trun['valve']).
        The sum of all three is the complete runtime.
    """
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
    if verbose: print('run_ppv -- starting run...')
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

    if verbose: print('run_ppv -- Done!')
    return P, v_activity, trun

#---------------------------------------------------------------------------------

def run_time(PARAM):
    """
    Calculation of PPV run time. Trun is proportionnal to Nx (inverse
    and product) and to Nt, which is itself proportionnal to Nx**2 and to
    Ttot. So: Trun ~ Nx**3 * Ttot. From a few experiments, the
    proportionnality constant is around 8e-6.

    - Parameters:
    	+ :param PARAM: parameters dictionnary
    - Outputs:
    	+ :return trun: float, time in second taken by run with those
    	parameters.

    """

    # Unpack
    Nx = PARAM['Nx']
    Nt = PARAM['Nt']
    Ttot_ = PARAM['dt_']*Nt
    A = 8.e-6  # proportionnality constant

    # Compute
    trun = A * Nx**3 * Ttot_

    print('This run would be {:}s long'.format(trun))
    return trun

# -----------------------------------------------------------------------------

def save(filename, P_, v_activity, VALVES, PARAM, full=False):
    """
    Save the results and parameters of a simulation. Options to save only valve
    activity or add in full pore pressure history. The results are saved as a
    dictionnary, using pickle.

    Parameters
    ----------
    filename : str
       Path and filename where to save the file. No extension needed.
    P_ : 2d array
        Pressure history, dimensions: Nt * Nx. Only saved if full = `True`.
    PARAM : dict
        Parameters dictionnary.
    VALVES : dict
        Valves parameters dictionnary.
    v_activity : 3d array
        Valve activity array, dimensions: Nt * 2 * Nvalves. On the second
        dimension, stores *(a)* the valve state (1 is open, 0 is closed),
        *(b)* the pressure differential across the valve.
    full : bool (default=`False`)
        Option to save full pressure history (if full=`True`).

    """
    # Build output dictionnary
    # ------------------------
    out = {}
    out['PARAM'] = PARAM
    out['VALVES'] = VALVES
    out['v_activity'] = v_activity
    if full:
        # --> option to save full pressure history
        out['P_'] = P_
        filename += '.full'
        print('ppv.save -- saving pore pressure history too')

    filename += '.pkl'
    # Actually saving
    # ---------------
    print('Saving at {:}...'.format(filename))
    pickle.dump(out, open(filename, 'wb'))
    print('\nDone!')

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
import tables as tb

import PPvalves.initialize as init
import PPvalves.valves as valv
from PPvalves.utility import calc_bound_0, calc_bound_L
import PPvalves.trid_math.tmath as tm
from PPvalves.equilibrium import calc_k_eq


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
        d = tm.prod(B[0], B[1], B[2], P[tt,:], len(B[0])) + b  # compact form of knowns
        P[tt+1,:] = tm.solve(A[0], A[1], A[2], d, len(d))  # solving the system for t+1

    if verbose: print('simulation.run_nov -- Done !')
    return P

#---------------------------------------------------------------------------------

def run_nov_light(P, PARAM, verbose=True):
    """Solves fluid pressure diffusion without valves, light memory version.

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
    Plast : 2D array
        Final pore pressure state.
    """
    # >> Unpack
    Nt = PARAM['Nt']

    # >> Set up initial system
    if verbose: print('simulation.run_nov -- sytsem setup...')
    A, B, b = init.build_sys(PARAM)

    # >> Loop through time
    if verbose: print('simulation.run_nov -- starting run...')
    for tt in range(Nt):
        d = tm.prod(B[0], B[1], B[2], P[tt,:], len(B[0])) + b  # compact form of knowns
        P[tt+1,:] = tm.solve(A[0], A[1], A[2], d, len(d))  # solving the system for t+1

    if verbose: print('simulation.run_nov -- Done !')
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
        second column: :math:`\delta p` across valve (taken at pressure points
        right outside low permeability zone). Used to compute catalogs of
        events.
    trun : dict.
        Dictionnary with the details of the time spent on computing the product
        for the knowns (`trun['prod']`), the time spent on solving for next
        state of pressure (`trun['solve']`), the time spent on actionning
        valves (`trun['valve']`).  The sum of all three is the complete
        runtime.
    """
    trun = {'total' : -time.time(), 'prod' : 0, 'solve' : 0,
            'valves_inner' : 0, 'valves' : 0, 'sys_build' : 0}  # runtime dictionnary

    if verbose: print('simulation.run_light -- initialization...')
    # Create variables for useful values
    # ==================================
    Nt = PARAM['Nt']  # number of time steps for this simulation
    h = PARAM['h_']  # space step

    # Initialization steps
    # ====================

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
    bounds_in_t[0, 0] = calc_bound_0(P0[0], PARAM)
    bounds_in_t[0, 1] = calc_bound_L(P0[-1], PARAM)

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
        r = tm.prod(B[0], B[1], B[2], Pprev, len(Pprev)) + b # calc knowns (right hand side)
        trun['prod'] += time.time() - tprod0  # add elapsed t

        # Solve for next time
        # -------------------
        tsolve0 = time.time()  # start timer for solver
        Pnext = tm.solve(A[0], A[1], A[2], r, len(r)) # solve system
        trun['solve'] += time.time() - tsolve0 # add elapsed t


        # Manage valve evolution
        # ----------------------
        tvalve0 = time.time()  # start timer for valves

        VALVES, opening, closing = valv.evolve(Pnext, h, VALVES)

        #--> Build new system according to new valve states
        if np.any(opening | closing):  # (much more efficient to do it only when
                                   #  needed, but should be optimized...)
            tin0 = time.time()
            PARAM['k'] = valv.update_k(VALVES, opening | closing, PARAM)
            tbuild0 = time.time()
#            A, B, b = init.build_sys(PARAM) # update system with new permeab.
            A, B = init.update_sys(A, B, PARAM) # update system with new permeab.
            trun['sys_build'] += time.time() - tbuild0  # add elapsed t
            trun['valves_inner'] += time.time() - tin0  # add elapsed t

        trun['valves'] += time.time() - tvalve0  # add elapsed t

        # --> Update v_activity and bounds
        v_activity[tt+1, 0, :] = VALVES['open']
        v_activity[tt+1, 1, :] = VALVES['dP']
        bounds_in_t[tt+1, 0] = calc_bound_0(Pnext[0], PARAM)  # get bound 0
        bounds_in_t[tt+1, 1] = calc_bound_L(Pnext[-1], PARAM) # get bound L

    if verbose: print('simulation.run_light -- Done!')

    Plast = Pnext
    trun['total'] += time.time()

    return Plast, bounds_in_t, v_activity, trun

# ----------------------------------------------------------------------------
def run_light_memory(P0, PARAM, VALVES, outpath, chunk=1e5, verbose=True):
    r"""Runs PPv, and uses HDF5 structures to save results to manage even the
    largest system sizes.

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
    outpath : str
        Absolute path and filename for the hdf5 output file, containing time
        series of `valve_states` and `valve_dP` for all valves, `k_eq` and
        `bounds`.
    chunk : int (default `chunk = 1e5`)
        Number of timesteps to wait before saving chunks of results. For 260
        valves, we have something of around 500Mb with the default, which is
        good.
    verbose : bool, optional
        Have the function print what it's doing.

    Returns
    -------
    Plast : 1D array
        Last state of pore pressure across the domain, dimension
        `PARAM['Nx'] + 1`.
    trun : dict.
        Dictionnary with the details of the time spent on computing the product
        for the knowns (`trun['prod']`), the time spent on solving for next
        state of pressure (`trun['solve']`), the time spent on actionning
        valves (`trun['valve']`).  The sum of all three is the complete
        runtime.
    """
    trun = {'total' : -time.time(), 'prod' : 0, 'solve' : 0,
            'valves_inner' : 0, 'valves' : 0, 'sys_build' : 0,
            'save_events' : 0}  # runtime dictionnary

    if verbose: print('simulation.run_light -- initialization...')
    # Create variables for useful values
    # ==================================
    Nt = PARAM['Nt']  # number of time steps for this simulation
    h = PARAM['h_']  # space step

    # Initialization steps
    # ====================
    # -> Create output data structure
    outvar = np.dtype([ ("bounds", np.float64, (2,)), ("k_eq", np.float64)])
    catalog = np.dtype([("op_t", np.float64), ("op_i", np.uint8)])
    fileh = tb.open_file(outpath+'.h5', mode="w")  # h5file
    outvar = fileh.create_table(fileh.root, "outvar", outvar, "Output variables",
                                expectedrows=PARAM['Nt']+1) # variables table
    catalog = fileh.create_table(fileh.root, "catalog", catalog, "Event catalog",
                                 expectedrows=PARAM['Nt']*2) # catalog table

    # -> Initialize valve activity, bounds, and k_eq
    outvar.row['bounds'] = [calc_bound_0(P0[0], PARAM), calc_bound_L(P0[-1], PARAM)]
    outvar.row['k_eq'] = calc_k_eq(VALVES, PARAM)
    outvar.row.append()  # write values

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
        r = tm.prod(B[0], B[1], B[2], Pprev, len(Pprev)) + b # calc knowns (right hand side)
        trun['prod'] += time.time() - tprod0  # add elapsed t

        # Solve for next time
        # -------------------
        tsolve0 = time.time()  # start timer for solver
        Pnext = tm.solve(A[0], A[1], A[2], r, len(r)) # solve system
        trun['solve'] += time.time() - tsolve0 # add elapsed t

        # Manage valve evolution
        # ----------------------
        tvalve0 = time.time()  # start timer for valves

        VALVES, opening, closing = valv.evolve(Pnext, h, VALVES)

        #--> Build new system according to new valve states
        if np.any(opening | closing):  # (much more efficient to do it only when
                                   #  needed, but should be optimized...)
            tin0 = time.time()
            PARAM['k'] = valv.update_k(VALVES, opening | closing, PARAM)
            tbuild0 = time.time()
            A, B = init.update_sys(A, B, PARAM) # update system with new permeab.
            trun['sys_build'] += time.time() - tbuild0  # add elapsed t

            # -> Save events
            tevents0 = time.time()
            for ii in range(np.sum(opening)):
                catalog.row['op_t'] = (tt+1) * PARAM['dt_']
                catalog.row['op_i'] = np.where(opening)[0][ii]
                catalog.row.append()
            trun['save_events'] += time.time() - tevents0  # add elapsed t
            trun['valves_inner'] += time.time() - tin0  # add elapsed t
        trun['valves'] += time.time() - tvalve0  # add elapsed t

        # Save outputs and flush buffer if too big
        # ----------------------------------------
        outvar.row['bounds'] = [calc_bound_0(Pnext[0], PARAM),
                               calc_bound_L(Pnext[-1], PARAM)]
        outvar.row['k_eq'] = calc_k_eq(VALVES, PARAM)
        outvar.row.append()

        if (tt%int(chunk) == 0) & (tt != 0): # flush in chunks
            outvar.flush() ; outvar = fileh.root.outvar
            catalog.flush() ; ocatalog= fileh.root.catalog

    outvar.flush() ; catalog.flush() ; fileh.close()  # flush and close results

    if verbose: print('simulation.run_light -- Done!')

    Plast = Pnext
    trun['total'] += time.time()

    return Plast, trun

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
        second column: :math:`\delta p` across valve (taken at pressure points
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
    trun = {'total': -time.time(), 'prod' : 0, 'solve' : 0, 'valves' : 0}  # runtime dictionnary

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
        r = tm.prod(B[0], B[1], B[2], P[tt, :], len(B[0])) + b # calc knowns (right hand side)
        trun['prod'] += time.time() - tprod0  # add elapsed t

        # Solve for next time
        # -------------------
        tsolve0 = time.time()  # start timer for solver
        P[tt+1, :] = tm.solve(A[0], A[1], A[2], r, len(r)) # solve system
        trun['solve'] += time.time() - tsolve0 # add elapsed t

        # Manage valve evolution
        # ----------------------
        tvalve0 = time.time()  # start timer for valves

        VALVES, opening, closing = valv.evolve(P[tt+1, :], h, VALVES)

        #--> Build new system according to new valve states
        if np.any(opening | closing):
            PARAM['k'] = valv.update_k(VALVES, opening | closing, PARAM)
            A, B = init.update_sys(A, B, PARAM) # update system with new permeab.
        trun['valves'] += time.time() - tvalve0  # add elapsed t

        # --> Update v_activity
        v_activity[tt+1, 0, :] = VALVES['open']
        v_activity[tt+1, 1, :] = VALVES['dP']

    if verbose: print('simulation.run -- Done!')
    return P, v_activity, trun

#---------------------------------------------------------------------------------

def run_time(PARAM):
    r""" DEPRECATED : worked for previous versions.
    Rough estimation of PPv run time for given physical parameters.

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
    `trun` is proportionnal to `N_x` (because of the computation of matrix and
    product) and to `N_t`, which is itself proportionnal to `N_x^2` and `Ttot`,
    the total physical run time. Then, `trun` `\sim N_x^3` x `Ttot`.
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

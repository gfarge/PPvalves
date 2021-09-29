#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Module to run PPvalves in different ways, and save the results.

Solves for the pore pressure diffusion in time, and communicates with
`PPvalves.mat_math`, `PPvalves.valves` and `PPvalves.initialize` to handle
initializing the numerical system, valve evolution, and inversion of the
matrix.
"""


# >> Imports
import time
import pickle
import numpy as np
import tables as tb

import PPvalves.initialize as init
import PPvalves.valves as valv
from PPvalves.utility import calc_bound_0, calc_bound_L
import PPvalves.trid_math.tmath as tm
from PPvalves.equilibrium import calc_k_eq


# ---------------------------------------------------------------------------
#                           BIGGER RUNS
# ---------------------------------------------------------------------------

def run_light(P0, PARAM, VALVES, save_which, outpath, verbose=True):
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
    save_which : dictionnary
        Dictionnary of boolean values for large variable that we might want to
        track and save: 'P' for full pressure history, 'dP_valves' for dP at
        valves. Variables always saved are: k_eq (channel equivalent
        permeability), bounds (free boundary variables), catalog (opening valve
        index, and time at which it opens).
    outpath : str
        Absolute path and filename for the hdf5 output file, containing time
        series of `valve_states` and `valve_dP` for all valves, `k_eq` and
        `bounds`.
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
    trun = {'total': -time.time(), 'prod': 0, 'solve': 0,
            'valves': 0, 'save': 0}  # runtime dictionnary

    if verbose: print('simulation.run_light -- initialization...')
    # Create variables for useful values
    # ==================================
    Nt = PARAM['Nt']  # number of time steps for this simulation
    Nx = PARAM['Nx']  # number of space steps in domain
    h = PARAM['h_']  # space step

    Nv = len(VALVES['idx']) # number of valves

    # Initialization steps
    # ====================
    fileh = tb.open_file(outpath+'.h5', mode='w')  # opens h5file

    # -> Create events catalog table
    catalog_dt = np.dtype([("op_t", np.float64), ("op_i", np.uint8)])
    catalog = fileh.create_table(fileh.root, "catalog", catalog_dt, "catalog",
                                 expectedrows=PARAM['Nt']+1)

    # -> Create output variables data structure and tables
    data_structure = []
    data_structure.append(("bounds", np.float64, (2,)))
    data_structure.append(("k_eq", np.float64))

    # optionnal output variables
    if save_which['P']:
        data_structure.append(("P", np.float64, (Nx+1,)))
    if save_which['dP_valves']:
        data_structure.append(("dP_valves", np.float64, (Nv,)))

    data_structure = np.dtype(data_structure)
    outvar = fileh.create_table(fileh.root, "outvar", data_structure,
                                "outvar", expectedrows=PARAM['Nt']+1)

    # initialize the outputs with initial state
    outvar.row['bounds'] = [calc_bound_0(P0[0], PARAM),
                            calc_bound_L(P0[-1], PARAM)]
    outvar.row['k_eq'] = calc_k_eq(VALVES, PARAM)
    if save_which['P']:
        outvar.row['P'] = P0
    if save_which['dP_valves']:
        v_id1 = VALVES['idx']  # Pressure pt. right before valves
        v_id2 = VALVES['idx'] + VALVES['width']/h  # Pr. pt. right after valves
        v_id2 = v_id2.astype(int)
        outvar.row['dP_valves'] = P0[v_id1] - P0[v_id2]

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
        r = tm.prod(B[0], B[1], B[2], Pprev, len(Pprev)) + b  # calc knowns (right hand side)
        trun['prod'] += time.time() - tprod0  # add elapsed t

        # Solve for next time
        # -------------------
        tsolve0 = time.time()  # start timer for solver
        Pnext = tm.solve(A[0], A[1], A[2], r, len(r))  # solve system
        trun['solve'] += time.time() - tsolve0  # add elapsed t

        # Manage valve evolution
        # ----------------------
        tvalve0 = time.time()  # start timer for valves

        VALVES, opening, closing = valv.evolve(Pnext, h, VALVES)

        # --> Build new system according to new valve states
        if np.any(opening | closing):
            PARAM['k'] = valv.update_k(VALVES, opening | closing, PARAM)
            A, B = init.update_sys(A, B, PARAM)  # update system with new permeab.

            # --> Save events
            tsave0 = time.time()
            for ii in range(np.sum(opening)):
                catalog.row['op_t'] = (tt+1) * PARAM['dt_']
                catalog.row['op_i'] = np.where(opening)[0][ii]
                catalog.row.append()
            trun['save'] += time.time() - tsave0  # add elapsed t
        trun['valves'] += time.time() - tvalve0  # add elapsed t

        # Save outputs and flush buffer if too big
        # ----------------------------------------
        outvar.row['bounds'] = [calc_bound_0(Pnext[0], PARAM),
                                calc_bound_L(Pnext[-1], PARAM)]
        outvar.row['k_eq'] = calc_k_eq(VALVES, PARAM)
        # -> Optionnal outputs
        if save_which['P']:
            outvar.row['P'] = Pnext
        if save_which['dP_valves']:
            outvar.row['dP_valves'] = VALVES['dP']
        outvar.row.append()

        if (tt % int(1e5) == 0) & (tt != 0):  # flush in chunks
            outvar.flush(); outvar = fileh.root.outvar
            catalog.flush(); catalog= fileh.root.catalog

    outvar.flush() ; catalog.flush() ; fileh.close()  # flush and close results

    if verbose: print('simulation.run_light -- Done!')

    Plast = Pnext
    trun['total'] += time.time()

    return Plast, trun

# ---------------------------------------------------------------------------
#                           SMALLER RUNS
# ---------------------------------------------------------------------------

def run(P0, PARAM, VALVES, save_P=False, verbose=True):
    """Runs PPv, for small runs (memory and storage wise).

    Solves diffusion equation and actions valves.

    Parameters
    ----------
    P0 : 1D array
        Initial state of pore pressure in the system, dimension
        `PARAM['Nx'] + 1`.
    VALVES : dict
        Valves parameters dictionnary.
    PARAM : dict.
        Dictionnary of physical parameters describing the system.
    save_P : bool (default `save_P = False`)
        Option to save the pressure history in time and space.
    verbose : bool (default `verbose = False`)
        Have the function print what it's doing.

    Returns
    -------
    Plast or P: 1D array or 2D array
        Last state of pore pressure across the domain, dimension `PARAM['Nx'] +
        1`. If `save_P = True`, full time-space history of pressure.
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
    trun = {'total': -time.time(), 'prod': 0, 'solve': 0,
            'valves': 0}  # runtime dictionnary

    if verbose: print('simulation.run_light -- initialization...')
    # Create variables for useful values
    # ==================================
    Nt = PARAM['Nt']  # number of time steps for this simulation
    Nx = PARAM['Nx']  # number of time steps for this simulation
    h = PARAM['h_']  # space step

    # Initialization steps
    # ====================
    # --> Locate valves, initialize valve activity
    v_activity = np.zeros((Nt+1, 2, len(VALVES['idx'])))
    v_activity[0, 0, :] = VALVES['open']  # initialize valve states

    v_id1 = VALVES['idx']  # Pressure pt. right before valves
    v_id2 = VALVES['idx'] + VALVES['width']/h  # Pr. pt. right after valves
    v_id2 = v_id2.astype(int)

    v_activity[0, 1, :] = P0[v_id1] - P0[v_id2]

    # --> Initialize P
    if save_P:
        P = np.zeros((Nt+1, Nx+1))
        P[0, :] = P0

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
        r = tm.prod(B[0], B[1], B[2], Pprev, len(Pprev)) + b  # calc knowns
        trun['prod'] += time.time() - tprod0  # add elapsed t

        # Solve for next time
        # -------------------
        tsolve0 = time.time()  # start timer for solver
        Pnext = tm.solve(A[0], A[1], A[2], r, len(r))  # solve system
        trun['solve'] += time.time() - tsolve0  # add elapsed t

        if save_P:
            P[tt+1, :] = Pnext

        # Manage valve evolution
        # ----------------------
        tvalve0 = time.time()  # start timer for valves

        VALVES, opening, closing = valv.evolve(Pnext, h, VALVES)

        # --> Build new system according to new valve states
        # much more efficient to do it only when needed should be optimized
        if np.any(opening | closing):
            PARAM['k'] = valv.update_k(VALVES, opening | closing, PARAM)
            A, B = init.update_sys(A, B, PARAM)  # update system with new permeab.
        trun['valves'] += time.time() - tvalve0  # add elapsed t

        # --> Update v_activity and bounds
        v_activity[tt+1, 0, :] = VALVES['open']
        v_activity[tt+1, 1, :] = VALVES['dP']
        bounds_in_t[tt+1, 0] = calc_bound_0(Pnext[0], PARAM)  # get bound 0
        bounds_in_t[tt+1, 1] = calc_bound_L(Pnext[-1], PARAM)  # get bound L

    trun['total'] += time.time()
    if verbose: print('simulation.run_light -- Done!')

    if save_P:
        return P, bounds_in_t, v_activity, trun

    else:
        Plast = Pnext
        return Plast, bounds_in_t, v_activity, trun
# ---------------------------------------------------------------------------

def run_no_valves(P, PARAM, verbose=True):
    """Solves fluid pressure diffusion without valving action. For a small
    system, with not many timesteps.

    Propagates an initial state of the pore-pressure in time. Permeability can
    be heterogeneous, but will not be dynamic. It is specified in `PARAM`.

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

# ---------------------------------------------------------------------------------
#                           HELPER FUNCTIONS
# ---------------------------------------------------------------------------------

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

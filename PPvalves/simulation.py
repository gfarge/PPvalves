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
import PPvalves.valves as va


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

def run(P, PARAM, VALVES, cheap=False, verbose=False):
    """
    Solves the diffusion eqution in time.

    - Parameters:
    	+ :param P: if cheap=False, P is a NtimexNspace 2D np.array. First line is initialized
    	 with initial conditions. If cheap=False, P is a Nspace 1D array initialized with
    	 initial conditions.
    	+ :param PARAM: dictionnary description of the physical and numerical
    	 variables of the system. k(x), beta, mu, phi, rho, h, dt_, and
    	 boundary conditions (adim) must figure in it:
    		- if p0_ (resp pL_) is NaN and qin_ (resp qout_) is not, then
    		 apply Neuman boundary conditions
    		- if qin_ (resp qout_) is NaN and p0_ (resp pL_) is not, then
    		 apply Dirichlet boundary conditions
    		- if both qout_and pL_ are NaN, then apply "stable flux"
    		 (dq/dx = 0) boundary conditions
    	+ :param VALVES: dictionnary description of valves. Necessary fields
    	 are 'state' (0 for closed, 1 for open), idx (first space index of
    	 k_valve), dPhi, dPlo (threshold values for opening and closing), dP
    	 (pore pressure differential between each end of the valve, along
    	 time). Best to use the output of make_valves.
    	+ :param cheap: boolean, if True, run_ppv only saves P_previous and P_next,
    	 and does not output P in time, but only v_activity

    - Output:
    	+ :return: (only if cheap=False) P as a NtimexNspace 2D np.array,
    	 with the full solution for
    	 pore pressure.
    	+ :return: v_activity, valve activity: column1: valve state,
    	 column2: valve dP. 3D array: Ntime*2*Nvalve, valves are
    	 ordered with growing position along X.

    """
    # Init time
    trun = {}
    trun['prod'] = 0
    trun['solve'] = 0
    trun['valve'] = 0

    # Unpack
    Nt = PARAM['Nt']
    h = PARAM['h_']

    # Initialize v_activity: valve activity
    v_activity = np.zeros((Nt+1,2,len(VALVES['idx'])))
    v_activity[0,0,:] = VALVES['open']
    v_id1 = VALVES['idx']
    v_id2 = VALVES['idx'] + VALVES['width']/h
    v_id2 = v_id2.astype(int)

    if cheap:
        v_activity[0, 1, :] = P[v_id1] - P[v_id2]
    else:
        v_activity[0, 1, :] = P[0, v_id1] - P[0, v_id2]

    # Set up initial system
    if verbose: print('run_ppv -- sytsem setup...')
    A,B,b = init.build_sys(PARAM)
    if cheap:
        Pprev = copy.copy(P)
        Pnext = copy.copy(P)

    # Go through time
    if verbose: print('run_ppv -- starting run...')
    for tt in range(Nt):
        if cheap:
            Pprev = Pnext
        #--> Solve for p(tt+1)
        tprod0 = time.time()

        if cheap:
            r = mat.product(B,Pprev) + b # calc knowns (right hand side)
        else:
            r = mat.product(B,P[tt,:]) + b # calc knowns (right hand side)

        trun['prod'] = trun['prod'] + time.time() - tprod0
        tsolve0 = time.time()

        if cheap:
            Pnext = mat.TDMAsolver(A,r) # solve system
        else:
            P[tt+1,:] = mat.TDMAsolver(A,r) # solve system
        trun['solve'] = trun['solve'] + time.time() - tsolve0
        #--> valve evolution
        tvalve0 = time.time()

        if cheap:
            VALVES, active_v = va.evolve(Pnext, h, VALVES)
        else:
            VALVES, active_v = va.evolve(P[tt+1,:], h, VALVES)

        #--> Build new system accordingly
        if np.any(active_v):
            PARAM['k'] = va.update_k(VALVES, active_v, PARAM)
            A,B,b = init.build_sys(PARAM) # update system with new k
        trun['valve'] = trun['valve'] + time.time() - tvalve0
        # Update v_activity
        v_activity[tt+1,0,:] = VALVES['open']
        v_activity[tt+1,1,:] = VALVES['dP']
    if verbose: print('run_ppv -- Done!')

    if cheap:
        return Pnext, v_activity, trun
    else:
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

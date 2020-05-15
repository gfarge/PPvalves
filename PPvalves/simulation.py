#!/usr/bin/python
# -*- coding: utf-8 -*-


""" pp_valves solves for the pore pressure diffusion in time, and handles barriers
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
    Propagates an initial state of the pore-pressure in time. Barriers but no
    dynamic valves.
    barriers.
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

def run(P, PARAM, barriers, cheap=False, verbose=False):
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
    	+ :param barriers: dictionnary description of barriers. Necessary fields
    	 are 'state' (0 for closed, 1 for open), idx (first space index of
    	 k_barrier), dPhi, dPlo (threshold values for opening and closing), dP
    	 (pore pressure differential between each end of the barrier, along
    	 time). Best to use the output of make_barriers.
    	+ :param cheap: boolean, if True, run_ppv only saves P_previous and P_next,
    	 and does not output P in time, but only BA

    - Output:
    	+ :return: (only if cheap=False) P as a NtimexNspace 2D np.array,
    	 with the full solution for
    	 pore pressure.
    	+ :return: BA, barrier activity: column1: barrier state,
    	 column2: barrier dP. 3D array: Ntime*2*Nbarrier, barriers are
    	 ordered with growing position along X.

    """
    # Init time
    trun = {}
    trun['prod'] = 0
    trun['solve'] = 0
    trun['barrier'] = 0

    # Unpack
    Nt = PARAM['Nt']
    h = PARAM['h_']

    # Initialize BA: barrier activity
    BA = np.zeros((Nt+1,2,len(barriers['idx'])))
    BA[0,0,:] = barriers['open']
    b_id1 = barriers['idx']
    b_id2 = barriers['idx'] + barriers['width']/h
    b_id2 = b_id2.astype(int)

    if cheap:
        BA[0,1,:] = P[b_id1] - P[b_id2]
    else:
        BA[0,1,:] = P[0,b_id1] - P[0,b_id2]

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
        #--> Barrier evolution
        tbarrier0 = time.time()

        if cheap:
            barriers, b_activity = va.evolve(Pnext, h, barriers)
        else:
            barriers, b_activity = va.evolve(P[tt+1,:], h, barriers)

        #--> Build new system accordingly
        if np.any(b_activity):
            PARAM['k'] = va.update_k(barriers, b_activity, PARAM)
            A,B,b = init.build_sys(PARAM) # update system with new k
        trun['barrier'] = trun['barrier'] + time.time() - tbarrier0
        # Update BA
        BA[tt+1,0,:] = barriers['open']
        BA[tt+1,1,:] = barriers['dP']
    if verbose: print('run_ppv -- Done!')

    if cheap:
        return Pprev,Pnext,BA,trun
    else:
        return P, BA, trun


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

def save(filename, P_, BA, barriers, PARAM, full=False):
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
    barriers : dict
        Valves parameters dictionnary.
    BA : 3d array
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
    out['barriers'] = barriers
    out['BA'] = BA
    if full:
        # --> option to save full pressure history
        out['P_'] = P_
        filename += '.full'

    filename += '.pkl'
    # Actually saving
    # ---------------
    print('Saving at {:}...'.format(filename))
    pickle.dump(out, open(filename, 'wb'))
    print('\nDone!')

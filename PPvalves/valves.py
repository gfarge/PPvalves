#""" A module to manage valve creation, distribution and action """

# Imports
# =======

import numpy as np


# Core
# ====

def scatter(scale, b_wid, PARAM):
    """
    Scatters barriers on the domain, according to an exponential law for their
    distance. End and beginning of barriers (in terms of k) have to be at
    least h apart, they cannot be on the first or last 2 pts of
    domain.
    Barriers all have a fixed width.

    - Parameters:
    	+ :param b_wid: width of the barrier as a fraction of the total
    	+ :param scale: mean distance between barriers
    	+ :param PARAM: parameters dictionnary
    - Outputs:
    	+ :return b_idx: width of the barrier as a fraction of the total
    """
    # Unpack
    Nx = PARAM['Nx']
    h_ = PARAM['h_']

    # Initialization
    prev_end = 0 # last
    b_id = prev_end # b_id+1 is first index of low k
    b_idx = []

    # Draw interbarrier distances while they do not cross domain end
    while b_id + b_wid/h_ < Nx:
        dist = np.random.exponential(scale)

        b_id = np.floor(dist/h_) + prev_end

        b_idx.append(b_id)
        prev_end = b_id + b_wid/h_

    b_idx = np.array(b_idx[:-1]).astype(int)

    return b_idx

# ----------------------------------------------------------------------------

def make(idx, dPhi, dPlo, width, klo, PARAM, verbose=True):
    """
    Makes barriers dictionnary, describing the state of all barriers at one
    moment in time. Used in propagation.

    - Parameters:
    	+ :param idx: 1D array, index of P(x) just before barrier.
    	 A barrier has to start at 1 at least, and it has to end before
    	 the end of the domain: idx < Nx+2
    	+ :param dPhi, dPlo: 1D array, same length as idx, value of dP
    	 for opening/closing the corresponding valve.
    	+ :param width: width of each barrier, in unit or non dim.
    	 Corresponds to width between P(x) point just before and just after
    	 low k barrier
    	+ :param klo: 1D np.array, same length as idx, values are each
    	 barrier's permeability when closed.

    - Outputs:
    	+ :return: barriers: barriers dictionnary description. Fields as
    	 input,
    	 all closed initially.
    """
    # Unpack
    h = PARAM['h_']
    Nx = PARAM['Nx']

    # Check for invalid barriers:
    #--> Barriers out of domain (low perm starting at 0 or ending at/after Nx)
    if np.any(idx+1<=1) | np.any((idx + width/h) >= Nx):
        raise ValueError('A barrier\'s boundaries exceed allowed domain')
    #--> Barriers one above another
    i_sort = np.argsort(idx)
    idx = idx[i_sort]
    width = width[i_sort]
    if np.any(idx[:-1] + width[:-1]/h >= idx[1:]+1):
        raise ValueError('Some barriers are ontop of one another')

    #--> Barriers too thin (less than 2*h_ wide)
    if np.any((width - 2*h) < 0):
        raise ValueError('Some barriers are thinner than 2*h, minimum width recquired.')

    # Build dictionnary
    barriers = {}

    Nb = len(idx) # number of barriers

    #--> Order barriers in increasing X
    i_sort = np.argsort(idx)

    barriers['idx'] = idx
    barriers['dP'] = np.zeros(Nb) # delta pore p seen by one barrier
    barriers['open'] = np.zeros(len(idx)).astype(bool) # open=True when open
    barriers['dPhi'] = dPhi[i_sort]
    barriers['dPlo'] = dPlo[i_sort]
    barriers['width'] = width
    barriers['klo'] = klo[i_sort]

    if verbose:
        print('Barriers idxs: \n {:}'.format(barriers['idx']))
        print('Opening dP (nd): \n {:}'.format(barriers['dPhi']))
        print('Closing dP (nd): \n {:}'.format(barriers['dPlo']))
        print('Barriers state (True=open): \n {:}'.format(barriers['open']))

    return barriers

# ----------------------------------------------------------------------------

def X2idx(X, d):
    """
    A function to calculate the barrier index in X
    for a given x coordinate, d.

    - Parameters
            + :param X: 1D array of regularly spaced space coordinates.
            + :type X: ndarray of floats
            + :param d: x coordinate of the barrier, preferably closest to a multiple
            of the spacing of X
            + :type d: float

    - Outputs
            + :rtype: idx : integer
            + :return: idx : index of X corresponding to
            the input d.
    """

    # Get closest index to distance you set
    idx = np.argmin(np.abs(X-d))

    return idx

# ----------------------------------------------------------------------------

def dist2idx(X, d, w):
    """
    A function to calculate the barrier indices on each side of the mid point of X
    for a given distance. In this second version, d is the distance between last point
	of low_k of previous barrier and first point of low_k of next.

    - Parameters
            + :param X: 1D array of regularly spaced space coordinates.
            + :type X: ndarray of floats
            + :param d: distance between the barriers, preferably as a multiple of
            the spacing of X
            + :type d: float
            + :param w: width of barriers
            + :type w: float

    - Outputs
        + :rtype: idx1,idx2 : tuple of integers.
        + :return: idx1,idx2 : tuple of the indices in X corresponding to
        the input d, indices are on each side of the mid-point of X.
    """

    # Get space increment
    h = X[1]-X[0]

    # Get mid-point of X
    mid_idx = len(X)/2. # we are using float to diminish the number of cases

    # Convert distance to index distance
    idx_d = d/h

    # Calculate indices
    idx1 = int(np.floor(mid_idx - idx_d/2.).astype(int) - w/h)
    idx2 = int(np.floor(mid_idx + idx_d/2.).astype(int) - 1)

    return idx1, idx2

# -----------------------------------------------------------------------------

def evolve(P, h, barriers):
    """
    Checks the new pore pressure state at barriers. Open/Close them accordingly.
    Computes the new permeability profile associated.

    - Parameters:
    	+ :param P: pore pressure profile at time t as a 1D np array.
    	+ :param barriers: barriers dict. description at t-dt (before
    	update)

    - Output:
    	+ :return: barriers, as input, but open and dP evolved to be
    	consistent with input P at time t.
    	+ :return: b_activity: boolean, True if OPEN or CLOSE activity.
    	Used to check if we have to modify the permeability profile.

    """
    # Visit every barrier
    #--> Initialize barrier activity: no activity a priori
    b_activity = np.zeros(len(barriers['idx'])).astype(bool)

    for ib, (idx, wb) in enumerate(zip(barriers['idx'], barriers['width'])):
        #--> Upddate pressure diff
        barriers['dP'][ib] = P[idx] - P[int(idx+wb/h)]

        #--> Open or Close?
        if (barriers['dP'][ib] > barriers['dPhi'][ib]) & \
         (not barriers['open'][ib]):
        #-->> if barrier is closed, and dP above thr: OPEN
            barriers['open'][ib] = True
            b_activity[ib] = True

        elif (barriers['dP'][ib] < barriers['dPlo'][ib]) &\
         (barriers['open'][ib]):
        #-->> if barrier is open, and dP below thr: CLOSE
            barriers['open'][ib] = False
            b_activity[ib] = True

    return barriers, b_activity

#---------------------------------------------------------------------------------

def update_k(barriers, b_act, PARAM):
    """ Computes permeability profile according to barrier distribution and
    choice of background/barrier permeability."""

    # Unpack
    h = PARAM['h_']
    k = PARAM['k']
    k_bg = k[0] # ensure that your k is always k_bg at the 2 1st and 2 last pts

    # Visit every barrier that was active and change its permeability to its
    #  updated value
    b_iterable = zip(barriers['open'][b_act], barriers['idx'][b_act],\
     barriers['width'][b_act], barriers['klo'][b_act])

    for b_is_open, idx, w, klo in b_iterable:
        if b_is_open:
        #--> if barrier is open: background permeability
            k[idx+1 : int(idx+w/h+1)] = k_bg
        else:
        #--> if barrier is closed: lower permeability
            k[idx+1 : int(idx+w/h+1)] = klo
    return k

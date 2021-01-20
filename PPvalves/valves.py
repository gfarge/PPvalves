#""" A module to manage valve creation, distribution and action """

# Imports
# =======

import numpy as np


# Core
# ====
def comb(dx, v_wid, PARAM):
    """
    Scatters valve at a constant distance along the domain.

    Parameters
    ----------
    dx : float
        Inter-valve distance to maintain across the domain, in non-dimensional
        unit of length (fraction of total domain length).
    v_wid : float
        Width of the valve as a fraction of the total length of the domain
        (non-dimensional length of valve).
    PARAM : dict
        Parameters dictionnary.

    Returns
    -------
    v_idx : 1D array
        Array of valve indices, index of P(x) just before valve.

    """
    v_idx_0 = 1  # first possible valve index
    v_idx_N = PARAM['Nx'] - int(v_wid/PARAM['h_'])  # last possible valve index
    d_idx = int((dx + v_wid)/PARAM['h_'])  # distance between valve indices

    v_idx = np.arange(v_idx_0, v_idx_N, d_idx)

    return v_idx

def scatter(scale, v_wid, PARAM):
    """
    Scatters valves on the domain, according to an exponential law for their
    distance. End and beginning of valves (in terms of k) have to be at
    least h apart, they cannot be on the first or last 2 pts of
    domain.
    valves all have a fixed width.

    - Parameters:
    	+ :param v_wid: width of the valve as a fraction of the total
    	+ :param scale: mean distance between valves
    	+ :param PARAM: parameters dictionnary
    - Outputs:
    	+ :return v_idx: width of the valve as a fraction of the total
    """
    # Unpack
    Nx = PARAM['Nx']
    h_ = PARAM['h_']

    # Initialization
    prev_end = 1 # last
    v_id = prev_end # v_id+1 is first index of low k
    v_idx = []

    # Draw intervalve distances while they do not cross domain end
    while v_id + v_wid/h_ < Nx - 1:
        dist = np.random.exponential(scale)

        v_id = np.floor(dist/h_) + prev_end

        v_idx.append(v_id)
        prev_end = v_id + v_wid/h_

    v_idx = np.array(v_idx[:-1]).astype(int)

    return v_idx

# ----------------------------------------------------------------------------

def make(idx, dPhi, dPlo, width, klo, PARAM, verbose=True):
    """
    Makes VALVES dictionnary, describing the state of all valves at one
    moment in time. Used in propagation.

    - Parameters:
    	+ :param idx: 1D array, index of P(x) just before valve.
    	 A valve has to start at 1 at least, and it has to end before
    	 the end of the domain: idx < Nx+2
    	+ :param dPhi, dPlo: 1D array, same length as idx, value of dP
    	 for opening/closing the corresponding valve.
    	+ :param width: width of each valve, in unit or non dim.
    	 Corresponds to width between P(x) point just before and just after
    	 low k valve
    	+ :param klo: 1D np.array, same length as idx, values are each
    	 valve's permeability when closed.

    - Outputs:
    	+ :return: VALVES: valves dictionnary description. Fields as
    	 input,
    	 all closed initially.
    """
    # Unpack
    h = PARAM['h_']
    Nx = PARAM['Nx']

    # Check for invalid valves:
    #--> valves out of domain (low perm starting at 0 or ending at/after Nx)
    if np.any(idx+1<=1) | np.any((idx + width/h) >= Nx):
        raise ValueError('A valve\'s boundaries exceed allowed domain')
    #--> valves one above another
    i_sort = np.argsort(idx)
    idx = idx[i_sort]
    width = width[i_sort]
    if np.any(idx[:-1] + width[:-1]/h >= idx[1:]+1):
        raise ValueError('Some valves are ontop of one another')

    #--> valves too thin (less than 2*h_ wide)
    if np.any((width - 2*h) < 0):
        raise ValueError('Some valves are thinner than 2*h, minimum width recquired.')

    # Build dictionnary
    VALVES = {}

    Nv = len(idx) # number of valves

    #--> Order valves in increasing X
    i_sort = np.argsort(idx)

    VALVES['idx'] = idx
    VALVES['dP'] = np.zeros(Nv) # delta pore p seen by one valve
    VALVES['open'] = np.zeros(len(idx)).astype(bool) # open=True when open
    VALVES['dPhi'] = dPhi[i_sort]
    VALVES['dPlo'] = dPlo[i_sort]
    VALVES['width'] = width
    VALVES['klo'] = klo[i_sort]

    if verbose:
        print('valves idxs: \n {:}'.format(VALVES['idx']))
        print('Opening dP (nd): \n {:}'.format(VALVES['dPhi']))
        print('Closing dP (nd): \n {:}'.format(VALVES['dPlo']))
        print('valves state (True=open): \n {:}'.format(VALVES['open']))

    return VALVES

# ----------------------------------------------------------------------------

def X2idx(X, d):
    """
    A function to calculate the valve index in X
    for a given x coordinate, d.

    - Parameters
            + :param X: 1D array of regularly spaced space coordinates.
            + :type X: ndarray of floats
            + :param d: x coordinate of the valve, preferably closest to a multiple
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
    A function to calculate the valve indices on each side of the mid point of X
    for a given distance. In this second version, d is the distance between last point
	of low_k of previous valve and first point of low_k of next.

    - Parameters
            + :param X: 1D array of regularly spaced space coordinates.
            + :type X: ndarray of floats
            + :param d: distance between the valves, preferably as a multiple of
            the spacing of X
            + :type d: float
            + :param w: width of valves
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

def evolve(P, h, VALVES):
    """
    Checks the new pore pressure state at valves. Open/Close them accordingly.
    Computes the new permeability profile associated.

    - Parameters:
    	+ :param P: pore pressure profile at time t as a 1D np array.
    	+ :param VALVES: VALVES dict. description at t-dt (before
    	update)

    - Output:
    	+ :return: VALVES, as input, but open and dP evolved to be
    	consistent with input P at time t.
    	+ :return: active_v: boolean, True if OPEN or CLOSE activity.
    	Used to check if we have to modify the permeability profile.

    """
    # Visit every valve
    #--> Initialize valve activity: no activity a priori
    active_v = np.zeros(len(VALVES['idx'])).astype(bool)

    for iv, (idx, wv) in enumerate(zip(VALVES['idx'], VALVES['width'])):
        #--> Upddate pressure diff
        VALVES['dP'][iv] = P[idx] - P[int(idx+wv/h)]

        #--> Open or Close?
        if (VALVES['dP'][iv] > VALVES['dPhi'][iv]) & \
         (not VALVES['open'][iv]):
        #-->> if valve is closed, and dP above thr: OPEN
            VALVES['open'][iv] = True
            active_v[iv] = True

        elif (VALVES['dP'][iv] < VALVES['dPlo'][iv]) &\
         (VALVES['open'][iv]):
        #-->> if valve is open, and dP below thr: CLOSE
            VALVES['open'][iv] = False
            active_v[iv] = True

    return VALVES, active_v

#---------------------------------------------------------------------------------

def update_k(VALVES, active_v, PARAM):
    """ Computes permeability profile according to valve distribution and
    choice of background/valve permeability."""

    # Unpack
    h = PARAM['h_']
    k = PARAM['k']
    k_bg = k[0] # ensure that your k is always k_bg at the 2 1st and 2 last pts

    # Visit every valve that was active and change its permeability to its
    #  updated value
    v_iterable = zip(VALVES['open'][active_v], VALVES['idx'][active_v],\
     VALVES['width'][active_v], VALVES['klo'][active_v])

    for v_is_open, idx, w, klo in v_iterable:
        if v_is_open:
        #--> if valve is open: background permeability
            k[idx+1 : int(idx+w/h+1)] = k_bg
        else:
        #--> if valve is closed: lower permeability
            k[idx+1 : int(idx+w/h+1)] = klo
    return k

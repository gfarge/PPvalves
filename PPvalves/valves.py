""" A module to manage valve creation, distribution and action """

# Imports
# =======

import numpy as np
from scipy.stats import weibull_min
from scipy.special import gamma


# Core
# ====
def weibull(u, N, v_wid, PARAM, xvmin=0, xvmax=1):
    """
    Distributes valves according to a Weibull distribution.

    Parameters
    ----------
    u : float
        Shape parameter of the Weibull distribution, controlling how clustered
        (`0 < u < 1`), Poissonian (`u = 1`), or regular (`u > 1`) the
        distribution is.
    N : float
        Number of valves in the distribution.
    v_wid : float
        Valve width.
    PARAM : dict
        Parameters dictionnary.
    xvmin, xvmax : floats (default `xvmin = 0`, `xvmax = 1`)
        Position of the extrema of the distribution.

    Returns
    -------
    v_idx : 1D array
        Array of valve indices, index of P(x) just before valve.

    """
    # >> Unpack
    Nx = PARAM['Nx']
    h_ = PARAM['h_']

    # >> Build distribution
    loc = 0
    iv_mean = (xvmax - xvmin - N*v_wid)/(N-1) # mean intervalve distance
    x0 = iv_mean*u / gamma(1/u)
    W = weibull_min(u, loc=loc, scale=x0)

    # >> Draw intervalve distances until it fits
    fits = False
    off = True
    while (not fits) or (off):
        iv_d = W.rvs(N)  # intervalve distance
        iv_d[0] = xvmin  # first valve at xvmin
        iv_d[1:] = iv_d[1:] + v_wid  # distance between valve centers
        xv = np.cumsum(iv_d)

        # >> Update fitting and off criterion
        fits = (xv[-1] + v_wid + h_) < xvmax
        off = (np.mean(iv_d[1:]) - iv_mean) < 0.1*iv_mean


    v_idx = np.round(xv / h_).astype(int)

    return v_idx

# ----------------------------------------------------------------------------

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
    v_idx_0 = 10  # first possible valve index is 1
    v_idx_N = PARAM['Nx'] - int(v_wid/PARAM['h_'])  # last possible valve index
    d_idx = int((dx + v_wid)/PARAM['h_'])  # distance between valve indices

    v_idx = np.arange(v_idx_0, v_idx_N, d_idx)

    return v_idx

# ----------------------------------------------------------------------------

def scatter(scale, v_wid, PARAM):
    """
    Scatters valves on the domain, according to an exponential law for their
    distance. End and beginning of valves (in terms of k) have to be at
    least h apart, they cannot be on the first or last 2 pts of
    domain.
    valves all have a fixed width.

    Parameters
    ----------
    v_wid : float
        Width of the valve in physical unit.
    scale : float
        Mean distance between valves (parameter of exponential law that dictate
        inter-valve distances).
    PARAM : dictionnary
        Parameters dictionnary

    Returns
    -------
    v_idx : 1D array
        Array of the valves indices for the discretization used in `PARAM`.
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

def patch_scatter(patch, scale, v_wid, PARAM):
    """
    Scatters valves in the domain, but only in a patch.

    Distributed according to an exponential law for their distance. End and
    beginning of valves (in terms of k) have to be at least `h` apart. First
    valve starts exactly where the patch starts, last valve finishes exactly
    where the patch finishes. Valves all have a fixed width.

    Parameters
    ----------
    patch : tuple
        Extrema of the patch, in physical unit.
    v_wid : float
        Width of the valve in physical unit.
    scale : float
        Mean distance between valves (parameter of exponential law that dictate
        inter-valve distances).
    PARAM : dictionnary
        Parameters dictionnary

    Returns
    -------
    v_idx : 1D array
        Array of the valves indices for the discretization used in `PARAM`.
    """
    # Unpack
    Nx = PARAM['Nx']
    h_ = PARAM['h_']
    X = np.arange(0, 1+h_, h_)

    pid1 = np.where(X == min(patch))[0][0] # x index where patch starts
    pid2 = np.where(X == max(patch))[0][0] # x index where patch ends
    print(pid1, X[pid1])
    print(pid2, X[pid2])

    # Initialization
    v_id = pid1  # first valve starts the patch
    v_idx = [v_id]
    prev_end = v_id + v_wid/h_ # end index of the (first) previous valve

    while v_id + 2*v_wid/h_ < pid2: # last valve must be before the closing valve
        print(v_idx)
        dist = np.random.exponential(scale)  # draw intervalve distance

        v_id = np.floor(dist/h_) + prev_end  # add valve at dist from end of
                                             # previous valve
        v_idx.append(v_id)          # add a new valve to our list
        prev_end = v_id + v_wid/h_  # change previous valve end to the new one

    v_idx.remove(v_idx[-1]) # remove the last valve

    last_v_id = pid2 - v_wid/h_
    v_idx.append(last_v_id) # add the closing valve
    v_idx = np.array(v_idx).astype(int)

    return v_idx

# ----------------------------------------------------------------------------

def make(idx, dPhi, dPlo, width, klo, PARAM, verbose=True):
    """Creates valve distribution based on input parameters

    Parameters
    ----------
    idx : 1D integer array
        X indices just before valve. A valve has to start at 1 at least, and it
        has to end before the end of the domain: idx < Nx+2.
    dPhi, dPlo : 1D arrays
        Threshold of `dP` for opening and closing. Same length as `idx`.
    width : 1D array
        Width of each valve, in units. Corresponds to width between X point
        just before and just after low permeability domain. Same length as `idx`.
    klo : 1D array
    	Permeability of valves when closed. Same length as `idx`.

    Returns
    -------
    VALVES : dictionnary
        Valves dictionnary description. Fields are similar to input, additional
        fields for storing valve state (closed initially) and dP.

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
    """A function to calculate the valve index for a given x coordinate.

    Parameters
    ----------
    X : 1D array
        Space vector.
    d : float
        `X` coordinate of the updip end of the valve, preferably closest to a
        multiple of the spacing of X for no surprises...

    Returns
    -------
    idx : integer
        Indices of X corresponding to the input `d`.
    """
    # Get closest index to distance you set, what a function
    idx = np.argmin(np.abs(X-d))

    return idx

# ----------------------------------------------------------------------------

def dist2idx(X, d, w):
    """Compute valve indices for two valves at a given distance, centered in
    the domain.
In this second version, d is the distance between last point
	of low_k of previous valve and first point of low_k of next.

    Parameters
    ----------
    X : 1D array
        Space vector.
    d : float
        The distance between last point of low permeability of first valve and
        first point of low permeability of next.
    w : float
        Valves' width

    Returns
    -------
    idx1, idx2 : integers.
        Indices of `X` corresponding to the input `d`, indices are on each side
        of the mid-point of `X`.
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
    """Checks the new pore pressure state at valves and opens/closes them
    accordingly.

    Parameters
    ----------
    P : 1D array
        Pore pressure profile that will define valve states.
    VALVES : dict.
        Valve parameters dictionnary.

    Returns
    -------
    VALVES : dict.
        Updated valve parameters dictionnary: current `'dP'` and `'state'`.
    active_valves : boolean array
        Which valve has either opened or closed.

    """
    # >> Update pressure differential
    VALVES['dP'] = P[VALVES['idx']] - P[(VALVES['idx']+VALVES['width']/h).astype(int)]

    # >> Compute boolean for above/below opening/closing criteria
    above_dpop = VALVES['dP'] > VALVES['dPhi']
    below_dpcl = VALVES['dP'] < VALVES['dPlo']

    # >> Compute new states
    # opening
    opening = ~VALVES['open'] & above_dpop
    VALVES['open'] = VALVES['open'] | above_dpop

    # closing
    closing = VALVES['open'] & below_dpcl
    VALVES['open'] = VALVES['open'] & ~below_dpcl

    # NB : condition to update state is different than opening (resp. closing) detector
    # because the first one accounts for when an open valve stays open (resp. a
    # closed valve stays closed) and not the latter.

    return VALVES, opening, closing

#---------------------------------------------------------------------------------

def update_k(VALVES, active_valves, PARAM):
    """Computes permeability profile according to valve new valve states.

    Parameters
    ----------
    VALVES : dictionnary
        Valves dictionnary description. Fields are similar to input, additional
    active_valves : boolean array
        Which valve has either opened or closed.
    PARAM : dict
        Parameters dictionnary.

    Returns
    -------
    k : 1D array
        Updated permeability.

    """
    # Visit every valve that was active and change its permeability to its
    #  updated value
    v_iterable = zip(VALVES['open'][active_valves], VALVES['idx'][active_valves],\
            VALVES['width'][active_valves], VALVES['klo'][active_valves])

    for v_is_open, idx, w, klo in v_iterable:
        PARAM['k'][idx+1 : int(idx+w/PARAM['h_']+1)] = PARAM['k'][0]*v_is_open + ~v_is_open*klo

    return PARAM['k']

# Alternate, to avoid to loop, but not really more efficient at first sight,
# maybe when more valves ?
#
#     >> Make slices of space domain to isolate their permeability
#    slices = [np.arange(VALVES['idx'][active_valves][ii]+1,
#              int(VALVES['idx'][active_valves][ii]+VALVES['width'][active_valves][ii]/h) + 1)
#              for ii in range(len(VALVES['idx'][active_valves]))]
#
#    # >> For each valve, associate its new permeability to its state
#    kval = [slices[ii]*0 +
#            ~VALVES['open'][active_valves][ii]*VALVES['klo'][active_valves][ii] +
#            VALVES['open'][active_valves][ii]*k_bg
#            for ii in range(len(VALVES['idx'][active_valves]))]
#
#    # >> Update permeability domain using slices and permeability in slices
#    kval = np.concatenate(kval)
#    slices = np.concatenate(slices)
#    k[slices] = kval

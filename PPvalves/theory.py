""" A module to compare PPvalves numerical resolution of diffusion processes
with theoretical ones """

# Imports
# =======

import numpy as np
import scipy.special as ssp

# Core
# ====


def Perf(X, T, X0, D, dP):
    """ Calculate the P error function for all times on the whole domain (inf domain)"""

    print('Perf -- Computing infinite domain solution...')
    erfP = np.zeros((len(T),len(X)))

    for itt, tt in enumerate(T):
        v = (X - X0) / np.sqrt(4. * D * tt)
        erfP[itt, :] = dP / 2. * ssp.erf(v)

    print('Perf -- Done !\n')

    return erfP

# -----------------------------------------------------------------------------



def Perf_bounded(X,T,X0,L,D,dP,n=5):
    """
    Computes the solution for Pp in a bounded domain, with no-flux boundaries.

    To cancel the flux at each boundary, the domain is considered cyclic and
    infinite with mirrored pressure across each boundary.
    Based on the method described in:
    http://web.mit.edu/1.061/www/dream/FOUR/FOURTHEORY.PDF
    http://www.dartmouth.edu/~cushman/courses/engs43/Diffusion-variations.pdf

    - Parameters
    	+ :param X: the space domain (numpy array 1D)
    	+ :param T: time vector (numpy array 1D)
    	+ :param X0: step location (float)
    	+ :param L: the domain extent in space (float).
    	+ :param D: diffusivity coefficient (float)
    	+ :param dP: step amplitude (float)
    	+ :param n: number of mirror solutions, keep odd, less than 7 is always
    	enough.
    - Outputs
    	+ :return: erfP: numpy array of the theoretical solution for bounded
    	diffusion.

    """
    print('Perf_bounded -- Computing no-flux boundaries solution...')
    m = n//2
    if m > 0:
    	sign_x = np.ones(m)
    	sign_x[::2] = -1

    erfP = np.zeros((len(T),len(X)))
    for itt,tt in enumerate(T):
        v = (X - X0) / np.sqrt(4*D*tt)
        erfP[itt,:] = dP/2. * ssp.erf(v)
        if m > 0:
            for mm,sign in zip(range(m),sign_x):
                m1 = (mm+1)//2
                v1 = sign*(X - (sign*X0 - 2*m1*L)) / np.sqrt(4*D*tt)
                m2 = (mm+2)//2
                v2 = sign*(X - (sign*X0 + 2*m2*L)) / np.sqrt(4*D*tt)
                erfP[itt, :] = erfP[itt, :] + dP/2.*ssp.erf(v1) + dP/2.*ssp.erf(v2)

    print('Perf_bounded -- Done')

    return erfP
# ------------------------------------------------------------------

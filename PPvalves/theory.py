""" A module to compare PPvalves numerical resolution of diffusion processes
with theoretical ones """

# Imports
# =======

import numpy as np
import scipy.special as ssp

# Core
# ====

def calc_trans_ramp(t, X, x1, x2, D):
    """
    Computes the solution of the diffusion of a ramp: a linear step from x1 to
    x2. Before x1, the initial condition is 1, after x2, it's 0.

    Derivation of this solution is presented in Exact_diffusion_solutions
    notebook.

    Parameters
    ----------
    t : float
        Given time to compute the solution at.
    X : 1D array
        Space array to compute the solution on.
    x1 : float
        Location of the beginning of the ramp.
    x2 : float
        Location of the end of the ramp.

    Returns
    -------
    p : 1D array
        Diffusive transient at time t, over X. Same dimension as X.
    """
    # Compute transient for step
    # --------------------------
    S = 1/2 * (1 - ssp.erf((X - x1)/(np.sqrt(4*D*t))))

    # Compute transient for ramp
    # --------------------------
    I0 = x2 * 1/2 * (ssp.erf((X - x1)/np.sqrt(4*D*t)) \
                     - ssp.erf((X - x2)/np.sqrt(4*D*t)))
    I1 = np.sqrt(D*t/np.pi) * np.exp(-1*(X - x1)**2/(4*D*t)) \
         + X * 1/2 * (1 + ssp.erf((X - x1)/np.sqrt(4*D*t)))
    I2 = np.sqrt(D*t/np.pi) * np.exp(-1*(X - x2)**2/(4*D*t)) \
         + X * 1/2 * (1 + ssp.erf((X - x2)/np.sqrt(4*D*t)))

    R = 1 / (x2 - x1) * (I0 - (I1 - I2))

    p = S + R

    return p


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

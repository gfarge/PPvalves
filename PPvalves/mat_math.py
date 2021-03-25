#!/usr/bin/python
# -*- coding: utf-8 -*-


"""mat_math handles tridiagonal matrix inversion and multiplications used in
solving pore pressure evolution in time.
"""

## Imports
import numpy as np

## Core
#--------------------------------------------------------------------------------

def product(M, v):
    r"""Tridiagonal product.

    Calculates the dot product `p = M . v` where
    `M` is a tridiagonal square matrix, and `v` a vector.

    Parameters
    ----------
    M : list
         Matrix `M` tridiagonal square, as a list of its diagonals
         vectors `(a, b,c)`, themselves as arrays.
    v : 1D array
        Vector `v`.
    Returns
    -------
    p : 1D array
        Product `p = M . v` as a numpy array.

    Notes
    -----
    When using `[a, b, c]` to represent a tridiagonal matrix, by convention:
    `len(a) = len(b) = len(c)`, `a[0] = 0`, `c[-1] = 0`.
    """
    # Initialize
    p = np.zeros(len(v))
    a, b, c = M
    # Core

    #--> For every elements except boundaries
    for ii in range(1, len(p)-1):
        p[ii] = a[ii]*v[ii-1] + b[ii]*v[ii] + c[ii]*v[ii+1]

    #--> For boundaries
    p[0] = b[0]*v[0] + c[0]*v[1]
    p[-1] = b[-1]*v[-1] + a[-1]*v[-2]

    return p
#--------------------------------------------------------------------------------

def ma2TD(A):
    """Converts a tridiagonal matrix to tridiagonal representation.

    Parameters
    ----------
    A : 2D array
        Square, tridiagonal matrix.

    Returns
    -------
    M : list
        List of three arrays `(a, b, c)`, representing the lower diagonal, the
        diagonal and the upper diagonal of the tridiagonal matrix.

    Notes
    -----
    When using `[a, b, c]` to represent a tridiagonal matrix, by convention:
    `len(a) = len(b) = len(c)`, `a[0] = 0`, `c[-1] = 0`.
    """
    # Initialize
    a = np.zeros(np.shape(A)[0])
    b = np.zeros(np.shape(A)[0])
    c = np.zeros(np.shape(A)[0])

    # Run through the lines of the matrix
    for ii in range(np.shape(A)[0]):
        b[ii] = A[ii, ii]
        if ii < (np.shape(A)[0]-1):
            c[ii] = A[ii, ii+1]
        if ii > 0:
            a[ii] = A[ii, ii-1]
    M = [a, b, c]
    return M

#--------------------------------------------------------------------------------

def TD2ma(A):
    """Converts a tridiagonal matrix A in simplified form to a numpy 2D array
    M.

    Parameters
    ----------
    A : 2D array
        List of three arrays `(a, b, c)`, representing the lower diagonal, the
        diagonal and the upper diagonal of the tridiagonal matrix.

    Returns
    -------
    M : list
        Square, tridiagonal matrix.

    Notes
    -----
    When using `[a, b, c]` to represent a tridiagonal matrix, by convention:
    `len(a) = len(b) = len(c)`, `a[0] = 0`, `c[-1] = 0`.
    """
    a, b, c = A
    M = np.zeros((len(b), len(b)))
    # Run through lines of the matrix
    for ii in range(len(b)):
        M[ii, ii] = b[ii]
        if ii<len(b)-1:
            M[ii, ii+1] = c[ii]
        if ii>0:
            M[ii, ii-1] = a[ii]
    return M


#--------------------------------------------------------------------------------

def TDMAsolver(A, d):
    """Tri diagonal system `d = A.X` solver. refer to

    Parameters
    ----------
    A : list
        List of arrays `(a, b, c)` representing the three diagonals
    	of the tridiagonal matrix of the linear system.
    d : array_like
        Vector of the knowns.

    Returns
    -------
    xc : 1D array
        Solution of the system.

    Notes
    -----
    To learn more on the tridiagonal matrix algorithm (or Thomas algorithm):
    `wikipedia <http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm>`_ or
    `here
    <http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)>`_.
    This functions was copied from this bit of code:
    `code <https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9>`_.
    """
    nf = len(d) # number of equations
    a, b, c = A
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]

    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]
    return xc

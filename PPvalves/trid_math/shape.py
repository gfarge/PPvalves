#!/usr/bin/python
# -*- coding: utf-8 -*-


"""Handles tridiagonal matrix conversion from complete to simplified forms:
only diagonals, using different paddings.
"""

## Imports
import numpy as np
import scipy.sparse

## Core
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

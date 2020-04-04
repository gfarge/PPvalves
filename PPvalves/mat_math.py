#!/usr/bin/python
# -*- coding: utf-8 -*-


""" matmath handles tridiagonal matrix inversion and multiplications """


## Imports 
import numpy as np
import matplotlib.pyplot as plt


## Core

#--------------------------------------------------------------------------------
def product(M,v):
	"""
	Calculates the dot product M.v where M is a tridiagonal square matrix, and v
	a vector.
	- Parameters
		+ :param M: matrix M tridiagonal square, as a list of its diagonal 
		vectors a,b,c, themselves as arrays, len(a) = len(b) = len(c), a1=0,
		cn=0
		+ :param v: vector v, as a np.array.
	- Outputs
		+ :return: p : M.v product as a numpy array.

	"""
	# Initialize
	p = np.zeros(len(v))
	a,b,c = M
	# Core

	#--> For every elements except boundaries
	for ii in range(1,len(p)-1):
		p[ii] = a[ii]*v[ii-1] + b[ii]*v[ii] + c[ii]*v[ii+1]

	#--> For boundaries
	p[0] = b[0]*v[0] + c[0]*v[1]
	p[-1] = b[-1]*v[-1] + a[-1]*v[-2]
	
	return p
#--------------------------------------------------------------------------------

def ma2TD(A):
	"""
	Converts a tridiagonal matrix to TD representation: 3 lists representing
	the 3 diagonals.
	a,b,c : len(a) = len(b) = len(c), a0=0, cn=0

	- Parameters:
		+ :param A: 2D, square, tridiagonal matrix.
	- Outputs:
		+ :return: M, a list of 3 arrays a,b,c representing the three
		diagonals of the matrix.
	
	"""
	# Initialize
	a = np.zeros(np.shape(A)[0])
	b = np.zeros(np.shape(A)[0])
	c = np.zeros(np.shape(A)[0])

	# Run through the lines of the matrix
	for ii in range(np.shape(A)[0]):
		b[ii] = A[ii,ii]
		if ii < (np.shape(A)[0]-1):
			c[ii] = A[ii,ii+1] 
		if ii > 0:
			a[ii] = A[ii,ii-1] 
	M = [a,b,c]
	return M

#--------------------------------------------------------------------------------

def TD2ma(A):
	"""
	Converts a tridiagonal matrix A in a,b,c form to a numpy 2D array M.
	a,b,c : len(a) = len(b) = len(c), a0 = 0, cn = 0

	- Parameters
		+ :param A: list of 3 lists or numpy arrays a,b,c representing the
		three diagonals of the matrix representing the system. 
	- Outputs
		+ :return: M, 2D numpy array

	"""
	a, b, c = A
	M = np.zeros((len(b),len(b)))
	# Run through lines of the matrix
	for ii in range(len(b)):
		M[ii,ii] = b[ii]
		if ii<len(b)-1:
			M[ii,ii+1] = c[ii]
		if ii>0:
			M[ii,ii-1] = a[ii]
	return M


#--------------------------------------------------------------------------------

def TDMAsolver(A,d):
	"""
	Tri diagonal system solver refer to 
	http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
	and to 
	http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
	
	- Parameters
		+ :param A: list of 3 lists a,b,c or numpy arrays representing the
		three diagonals
		of the matrix representing the system. len(a) = len(b) =
		len(c), cn = a0 = 0
		+ :param d: list or numpy array representing the "knowns" side of the
		system
	- Outputs
		+ :return: xc : numpy array of the solved unknowns.

	/////////////////////////////
	source: https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
	
	"""
	a, b, c = A
	nf = len(d) # number of equations
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

#--------------------------------------------------------------------------------
def product0(M,v):
	"""
	Calculates the dot product M.v where M is a tridiagonal square matrix, and v
	a vector.
	- Parameters
		+ :param M: matrix M tridiagonal square, as a list of its diagonal 
		vectors a,b,c, themselves as arrays, len(a) = len(b) - 1 = len(c)
		+ :param v: vector v, as a np.array.
	- Outputs
		+ :return: p : M.v product as a numpy array.

	"""
	p = np.zeros(len(v))
	a,b,c = M
	# Core elements
	for ii in range(1,len(v)-1):
		p[ii] = a[ii-1]*v[ii-1] + b[ii]*v[ii] + c[ii]*v[ii+1]

	# Boundaries
	p[0] = b[0]*v[0] + c[0]*v[1]
	p[-1] = a[-1]*v[-2] + b[-1]*v[-1]

	return p

#--------------------------------------------------------------------------------

def TDMAsolver0(A,d):
	"""
	Tri diagonal system solver refer to 
	http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
	and to 
	http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
	
	- Parameters
		+ :param A: list of 3 lists a,b,c or numpy arrays representing the
		three diagonals
		of the matrix representing the system. len(a) = len(b) - 1 =
		len(c).
		+ :param d: list or numpy array representing the "knowns" side of the
		system
	- Outputs
		+ :return: xc : numpy array of the solved unknowns.

	/////////////////////////////
	source: https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
	
	"""
	a, b, c = A
	nf = len(d) # number of equations
	ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
	for it in range(1, nf):
		mc = ac[it-1]/bc[it-1]
		bc[it] = bc[it] - mc*cc[it-1] 
		dc[it] = dc[it] - mc*dc[it-1]
	    	    
	xc = bc
	xc[-1] = dc[-1]/bc[-1]
	
	for il in range(nf-2, -1, -1):
		xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]
	return xc


#--------------------------------------------------------------------------------

def TD2ma0(A):
	"""
	Converts a tridiagonal matrix A in a,b,c form to a numpy 2D array M.
	a,b,c : len(a) = len(b) - 1 = len(c)

	- Parameters
		+ :param A: list of 3 lists or numpy arrays a,b,c representing the
		three diagonals of the matrix representing the system. 
	- Outputs
		+ :return: M, 2D numpy array

	"""
	a, b, c = A
	M = np.zeros((len(b),len(b)))
	# Run through lines of the matrix
	for ii in range(len(b)):
		M[ii,ii] = b[ii]
		if ii<len(b)-1:
			M[ii,ii+1] = c[ii]
		if ii>0:
			M[ii,ii-1] = a[ii-1]
	return M


#--------------------------------------------------------------------------------

def ma2TD0(A):
	"""
	Converts a tridiagonal matrix to TD representation: 3 lists representing
	the 3 diagonals.
	a,b,c : len(a) = len(b) - 1 = len(c)

	- Parameters:
		+ :param A: 2D, square, tridiagonal matrix.
	- Outputs:
		+ :return: M, a list of 3 arrays a,b,c representing the three
		diagonals of the matrix.
	
	"""
	# Initialize
	a = np.zeros(np.shape(A)[0]-1)
	b = np.zeros(np.shape(A)[0])
	c = np.zeros(np.shape(A)[0]-1)

	# Run through the lines of the matrix
	for ii in range(np.shape(A)[0]):
		b[ii] = A[ii,ii]
		if ii < (np.shape(A)[0]-1):
			c[ii] = A[ii,ii+1] 
		if ii > 0:
			a[ii-1] = A[ii,ii-1] 
	M = [a,b,c]
	return M

#--------------------------------------------------------------------------------

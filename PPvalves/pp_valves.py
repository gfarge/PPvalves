#!/usr/bin/python
# -*- coding: utf-8 -*-


""" pp_valves solves for the pore pressure diffusion in time, and handles barriers
opening and closing. Set up is handled by  setup.py. Matrix computations are handled
by matmath.py."""


## Imports
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.special as ssp
import copy

import PPvalves.mat_math as mat



## Paths

## Core


# ////////////////////////////////////////////////////////////////////////////////
# 	PP_valves setting up barriers/initcond/parameters
# ////////////////////////////////////////////////////////////////////////////////

#---------------------------------------------------------------------------------

def b_scat(scale,b_wid,PARAM):
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
	h_ = PARAM['h_']
	Nx = PARAM['Nx']

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

#---------------------------------------------------------------------------------
def make_barriers(idx,dPhi,dPlo,width,klo,PARAM,verbose=True):
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
	if np.any(idx[:-1] + width[:-1]/h >= idx[1:]+1) :
		raise ValueError('Some barriers are ontop of one another')

	#--> Barriers too thin (less than 2*h_ wide)
	if np.any((width - 2*h)<0):
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

#---------------------------------------------------------------------------------

def initcond(Z,PARAM,option):
	"""
	Set up initial conditions of reduced pore pressure.

	- Parameters
		+ :param Z: vector of depths, as a numpy array.
		+ :param PARAM: dictionnary of the physical and numerical
		parameters describing the system.
		+ :param option: initial conditions option:
			- option=0 : Pr is set to hydrostatic gradient (0) and
			  shallowest point at lithostatic pressure
			- option=1 : Pr is set to lithostatic gradient and shallowest
			point at lithostatic pressure
			- option=2 : Pr is set to a step function, with step
			characteristics set in PARAM['step'] (dP amplitude and
			step index).
			- option=3 : Pr is set to a cosine function, with characteristics
			taken in PARAM['cos_P'] (number of wavelength and amplitude).
	- Outputs
		+ :return: Pr:  initialized vector of reduced pore-pressure, as a
		numpy array.

	"""
	# Unpack
	## Physical and numerical parameters
	rho = PARAM['rho']
	rho_r = PARAM['rho_r']
	g = PARAM['g']
	hb_ = PARAM['hb_']
	alpha = PARAM['alpha']
	Nx = PARAM['Nx']

	## Scales
	P_scale = PARAM['P_scale']
	Z_scale = PARAM['Z_scale']

	Pr = np.zeros(len(Z))

	# Hydrostatic gradient
	if option==0:
#		print("initcond -- Hydrostatic gradient, no flux in the system")
		Ztop = Z[-1]
		Pr[-1] = rho_r*g*Ztop * Z_scale/P_scale # Ptop
#		Pr[:-1] = Pr[-1] + rho*g*(Z[:-1]-Ztop) * Z_scale/P_scale # All pressures
		# in between
		Pr[:] = (rho_r-rho)*g*Ztop * Z_scale/P_scale
	# Step function
	if option==2:
		dP, iX0 = PARAM['step']
#		print("initcond -- Erf test: initial step, dP = {:.2f}".format(dP))
		Ztop = Z[-1]
#		P[-1] = rho_r*g*Ztop * Z_scale/P_scale # Ptop
#		P[:-1] = P[-1] + rho*g*(Z[:-1]-Ztop) * Z_scale/P_scale # All pressures
		# in between
		Pr[:] = (rho_r-rho)*g*Ztop * Z_scale/P_scale
		step = np.zeros(len(Pr))
		step[:iX0] = step[:iX0] + 0.5*dP*np.ones(iX0)
		step[iX0:] = step[iX0:] - 0.5*dP*np.ones(len(Pr)-iX0)
		Pr = Pr + step
	# Lithostatic gradient
	if option==1:
		Ztop=Z[-1]
#		print("initcond -- Lithostatic gradient, qlitho in the system")
#		Pr = (rho_r-rho)*g*(Z-Ztop+hb_*np.sin(alpha)) * Z_scale/P_scale
		Pr = np.linspace(1,0,num=Nx+1)
	# Cosine function
	if option==3:
		h_= PARAM['h_']
		Nx = PARAM['Nx']
		nb_wavelength, amp = PARAM['cos_P']
#		print("initcond -- Cosine profile, {0:d} wavelengths, amp = {1:.2f}".format(nb_wavelength, amp))
		X_cos = 2*np.pi*np.linspace(0.,Nx*h_,num=Nx+1)
		cos_P = amp*np.cos(X_cos*nb_wavelength)
		Pr = cos_P

	return Pr

# ////////////////////////////////////////////////////////////////////////////////
# 	PP_valves building numerical system
# ////////////////////////////////////////////////////////////////////////////////

"""
This function sets up the elements for the numerical resolution of
a diffusion problem, with Crank Nicolson method.

The problem is expressed as the linear system A . p_n+1 = B . p_n + b_n+1 + b_n,
this module builds each matrix and vector in the system.

"""
def buildsys(PARAM):
	"""
	This function builds the A,B and b matrices and returns a storage efficient
	structure only keeping the three diagonal vectors of the tridiagonal matrices.

	- Parameters
	        + :param PARAM: dictionnary description of the physical and
	        numerical variables of the system. k(x), beta, mu, phi, h, dt_ must
	        figure in it.
	- Outputs
	        + :return: matrix A and B, in the form of 2 lists of their three diagonal
	        vectors.
	        + :return: vector b, as a 1D np.array.


	"""

	# Unpack physical parameters
	h = PARAM['h_']
	dt = PARAM['dt_']
	k = PARAM['k']
	beta = PARAM['beta']
	mu = PARAM['mu']
	phi = PARAM['phi']

	T_scale = PARAM['T_scale']
	X_scale = PARAM['X_scale']


	# Build A and B core structure
	delta_minus = k[:-1]/phi/mu/beta * dt/2./h**2 * T_scale/X_scale**2
	delta_plus = k[1:]/phi/mu/beta * dt/2./h**2 * T_scale/X_scale**2

	aa = np.zeros(len(delta_minus))
	aa[1:] = -1. * delta_minus[1:]
	ab = 1. + delta_plus + delta_minus
	ac = np.zeros(len(delta_plus))
	ac[:-1] = -1. * delta_plus[:-1]

	ba = np.zeros(len(delta_minus))
	ba[1:] = delta_minus[1:]
	bb = 1. - delta_plus - delta_minus
	bc = np.zeros(len(delta_plus))
	bc[:-1] = delta_plus[:-1]

	A = [aa,ab,ac]
	B = [ba,bb,bc]

	# Apply boundary conditions to A and B and create b
	A,B,b = boundary(A,B,PARAM)

	return A,B,b
#-----------------------------------------------------------------------------

def boundary(A,B,PARAM):
	"""
	Apply limit conditions to the system equations, but you can determine
	the depth of the boundary hb_, that is, the distance between first/last
	points and fictive points.

	- Parameters
	        + :param A,B: A and B matrices in the form of a list of their three
	        diagonal vectors.
	        + :param PARAM: dictionnary description of the physical and
	        numerical variables of the system. k(x), beta, mu, phi, rho, h_,
	        hb_, dt_, and
	        boundary conditions (adim) specifications must figure in it:
	                - if p0_ (resp pL_) is NaN and qin_ (resp qout_) is not, then apply
	                Neuman boundary conditions to A, B, b
	                - if qin_ (resp qout_) is NaN and p0_ (resp pL_) is not, then apply
	                Dirichlet boundary conditions to A, B, b
	                - if both qout_ and pL_ are NaN, then apply "stable flux"
	                (dq/dx = 0) boundary conditions to A, B, b
	- Outputs
	        + :return: matrix A and B, in the form of 2 lists of their three diagonal
	        vectors, with applied boundary conditions.
	        + :return: vector b, as a 1D np.array, with applied boundary conditions.

	"""
	# Unpack PARAM
	## boundary conditions
	p0 = PARAM['p0_']  # already adim
	pL = PARAM['pL_']  # already adim
	qin = PARAM['qin_']  # already adim
	qout = PARAM['qout_']  # already adim
	## physical and numerical parameters
	h = PARAM['h_']  # already adim
	hb = PARAM['hb_']  # already adim
	dt = PARAM['dt_']  # already adim

	k = PARAM['k']
	beta = PARAM['beta']
	mu = PARAM['mu']
	phi = PARAM['phi']
	rho = PARAM['rho']
	alpha = PARAM['alpha']
	g = PARAM['g']
	## scales
	X_scale = PARAM['X_scale']
	T_scale = PARAM['T_scale']
	P_scale = PARAM['P_scale']
	q_scale = PARAM['q_scale']


	# Initialize b and deltas
	D = k[0]/phi/mu/beta # permeability should be the same at -1, 0, N and N+1
	b = np.zeros(len(A[1]))

	# x = 0 boundary
	## Neuman
	if (np.isnan(p0)) & (not np.isnan(qin)):
		#print('Neuman in 0: qin_ = {:}'.format(qin_))
		## factors of p0_ : Ab[0] and Bb[0]
		A[1][0] = 1. + D * dt / hb * (1./h - 1./(hb+h)) * T_scale/X_scale**2
		B[1][0] = 1. - D * dt / hb * (1./h - 1./(hb+h)) * T_scale/X_scale**2

		## factors of p1_ : Ac[0] and Bc[0]
		A[2][0] = - 1.* D * dt / (h * (hb+h)) * T_scale/X_scale**2
		B[2][0] = D * dt / (h * (hb+h)) * T_scale/X_scale**2

		## constant terms (b = b_n - b_n+1)
		b[0] = 2. * D * dt / (h + hb) * T_scale/X_scale**2 *(mu/rho/k[0] * qin * q_scale) * X_scale/P_scale

	## Dirichlet
	elif (np.isnan(qin)) & (not np.isnan(p0)):
		#print('Dirichlet in 0, p0_={:.2f}'.format(p0_))
		## factors of p0_ : Ab[0] and Bb[0]
		A[1][0] = 1. + D*dt/h/hb * T_scale/X_scale**2
		B[1][0] = 1. - D*dt/h/hb * T_scale/X_scale**2

		## factors of p1_ : Ac[0] and Bc[0]
		A[2][0] =  -1. *  D*dt / (h * (h + hb)) * T_scale/X_scale**2
		B[2][0] = D*dt / (h * (h + hb)) * T_scale/X_scale**2
		## constant terms (b_n = b_n+1, hence the 2)
		b[0] = 2. * p0 * D*dt / (hb * (h + hb)) * T_scale/X_scale**2 # * P_scale/P_scale

	else:
		raise ValueError("boundary -- /!\\ x = 0 boundary conditions wrongly set")

	# x = L boundary
	## Neuman
	if (np.isnan(pL)) & (not np.isnan(qout)):
		#print('Neuman in L: qout_ = {:}'.format(qout_))
		## factors of pN_: Ab[-1] and Bb[-1]
		A[1][-1] = 1. + D * dt / hb * (1./h - 1./(hb+h)) * T_scale/X_scale**2
		B[1][-1] = 1. - D * dt / hb * (1./h - 1./(hb+h)) * T_scale/X_scale**2

		## factors of pN-1_: Aa[-1] and Ba[-1]
		A[0][-1] = -1. * D * dt / (h * (hb+h)) * T_scale/X_scale**2
		B[0][-1] = D * dt / (h * (hb+h)) * T_scale/X_scale**2

		## constant terms (b_n = b_n+1, hence the 2)
		b[-1] = -2. * D * dt / (h + hb) * T_scale/X_scale**2 *(mu/rho/k[-1] * qout * q_scale) * X_scale/P_scale

	## Dirichlet
	elif (np.isnan(qout)) & (not np.isnan(pL)):
		#print('Dirichlet in L, pL_={:.2f}'.format(pL_))
		## factors of pN_: Ab[-1] and Bb[-1]
		A[1][-1] = 1. + D * dt / hb / h * T_scale/X_scale**2
		B[1][-1] = 1. - D * dt / hb / h * T_scale/X_scale**2

		## factors of pN-1_: Aa[-1] and Ba[-1]
		A[0][-1] = -1. * D * dt / (h * (hb+h)) * T_scale/X_scale**2
		B[0][-1] = D * dt / (h * (hb+h)) * T_scale/X_scale**2
		## constant terms (b_n = b_n+1, hence the 2)
		b[-1] = 2. * pL * D*dt / (hb * (h+hb)) * T_scale/X_scale**2 # * P_scale/P_scale

	## "stable flux", dq/dx(xL) = 0 <=> dp/dt_(xL) = 0
	elif (np.isnan(qout)) & (np.isnan(pL)):
		#print('Stable flux in L')
		## factors of pL_
		A[1][-1] = 1.
		B[1][-1] = 1.
		## factors of p(L-h)
		A[0][-1] = 0.
		B[0][-1] = 0.
		## constant terms (b_n = b_n+1, hence the 2): None

	else:
		raise ValueError("boundary -- /!\\ x = L boundary conditions wrongly set")

	return A, B, b


# ////////////////////////////////////////////////////////////////////////////////
# 	PP_valves solving numerical system, evolving barriers
# ////////////////////////////////////////////////////////////////////////////////


def run_static_ppv(P,PARAM):
	"""
	Propagates an initial state of the pore-pressure in time. Static
	barriers.
	"""
	# Unpack
	Nt = PARAM['Nt']

	# Set up initial system
	print('propagate -- sytsem setup...')
	A,B,b = buildsys(PARAM)

	# Go through time
	print('propagate -- starting run...')
	for tt in range(Nt):

		d = mat.product(B,P[tt,:]) + b  # compact form of knowns
		P[tt+1,:] = mat.TDMAsolver(A,d)  # solving the system

	print('propagate -- Done !')
	return P

#---------------------------------------------------------------------------------

def run_ppv(P,PARAM,barriers,cheap=False,verbose=False):
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
	A,B,b = buildsys(PARAM)
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
			barriers, b_activity = barrier_evolve(Pnext,h,barriers)
		else:
			barriers, b_activity = barrier_evolve(P[tt+1,:],h,barriers)

		#--> Build new system accordingly
		if np.any(b_activity):
			PARAM['k'] = update_k_b(barriers,b_activity,PARAM)
			A,B,b = buildsys(PARAM) # update system with new k
		trun['barrier'] = trun['barrier'] + time.time() - tbarrier0
		# Update BA
		BA[tt+1,0,:] = barriers['open']
		BA[tt+1,1,:] = barriers['dP']
	if verbose: print('run_ppv -- Done!')

	if cheap:
		return Pprev,Pnext,BA,trun
	else:
		return P,BA,trun


#---------------------------------------------------------------------------------

def barrier_evolve(P,h,barriers):
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

	for ib, (idx,wb) in enumerate(zip(barriers['idx'],barriers['width'])):
		#--> Upddate pressure diff
		barriers['dP'][ib] = P[idx] - P[int(idx+wb/h)]

		#--> Open or Close?
		if (barriers['dP'][ib]>barriers['dPhi'][ib]) & \
		 (not barriers['open'][ib]) :
		#-->> if barrier is closed, and dP above thr: OPEN
			barriers['open'][ib]=True
			b_activity[ib] = True

		elif (barriers['dP'][ib]<barriers['dPlo'][ib]) &\
		 (barriers['open'][ib]) :
		#-->> if barrier is open, and dP below thr: CLOSE
			barriers['open'][ib]=False
			b_activity[ib] = True

	return barriers, b_activity

#---------------------------------------------------------------------------------

def update_k_b(barriers,b_act,PARAM):
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

#---------------------------------------------------------------------------------

def k_init(barriers,PARAM,state_override=None):
	"""
	Computes k, according to bg k and barrier distribution and state.
	"""

	# Unpack
	h = PARAM['h_']
	Nx = PARAM['Nx']
	k_bg = PARAM['k_bg']

	b_idx = barriers['idx']
	b_wid = barriers['width']
	b_k = barriers['klo']
	if type(state_override) == type(None):
		b_open = barriers['open']
	else:
		b_open = state_override.astype(bool)

	# Visit barriers and compute k

	#--> Init
	k = np.ones(Nx+2)*k_bg

	#--> Loop
	for bb in range(len(b_idx)):
		if b_open[bb]: continue
		else:
			k[ b_idx[bb]+1 : int(b_idx[bb] + b_wid[bb]/h + 1) ] = b_k[bb]

	return k

#---------------------------------------------------------------------------------

# ////////////////////////////////////////////////////////////////////////////////
# 	PHYSICS
# ////////////////////////////////////////////////////////////////////////////////

def mass_balance_inout(P,PARAM,int_t=True):
	r"""
	Computes the time derivative of the total mass (per unit area) along the
	domain, at time $t$.
	\[
	\frac{dM}{dt}(t) = q_{in}(t) - q_{out}(t).
	\]

	Parameters
	----------
	P : ndarray
		Pore-pressure history, dimensions 2D : Nt,Nx.
	PARAM : dict
		Parameters dictionnary. 
	int_t : bool, default = False
		Set to True fo option to compute $\delta M(t)$ instead of
		$\frac{dM}{dt}(t)$.

	Returns
	-------
	deltaM : ndarray (default, for int_t=True)
		Accumulation in time of total mass per unit area in domain,
		dimensions: Nt
	dMdt : ndarray (for int_t=False)
		Time derivative of total mass accumulation in the system in
		time, dimension : Nt
	
	Note
	----
		Check how we deal with boundaries here.

	"""
	# --> Unpack
	# -->> Boundary conditions
	qin_ = PARAM['qin_']
	qout_ = PARAM['qout_']
	p0_ = PARAM['p0_']
	pL_ = PARAM['pL_']
	hb_ = PARAM['hb_']
	dt_ = PARAM['dt_']
	# -->> Physical parameters
	rho = PARAM['rho']
	mu = PARAM['mu']
	k_bg = PARAM['k_bg']
	# -->> Scales
	X_scale = PARAM['X_scale']
	P_scale = PARAM['P_scale']
	q_scale = PARAM['q_scale']
	T_scale = PARAM['T_scale']
	M_scale = q_scale*T_scale

	# --> According to boundary, compute in/out flux
	if np.isnan(qin_) and (not np.isnan(p0_)):
		# --> Fixed pressure in 0
		qin_ = rho*k_bg/mu * (p0_ - P[:,0])/hb_ * P_scale/X_scale / q_scale
	elif (not np.isnan(qin_)) and np.isnan(p0_):
		# --> Fixed flux in 0
		pass
	if np.isnan(qout_) and (not np.isnan(pL_)):
		# --> Fixed pressure in L
		qout_ = rho*k_bg/mu * (P[:,-1] - pL_)/hb_ * P_scale/X_scale / q_scale
	elif (not np.isnan(qout_)) and np.isnan(pL_):
		# --> Fixed flux in L
		pass

	# --> dMdt
	dMdt = (qin_ - qout_) * q_scale*T_scale/M_scale

	if int_t:
		# --> Compute mass evolution history
		deltaM = np.cumsum(dMdt*dt_,axis=0)

		return deltaM
	else:
		# --> Simply output mass derivative in time
		return dMdt

#---------------------------------------------------------------------------------

def mass_balance_inpores(P,PARAM,int_x=True,int_t=True):
	r"""
	Computes the time derivative of volumic mass over the space domain, at
	time $t$, using:
	\[
	\frac{\partial m}{\partial t}(x,t) = \rho\beta\phi \frac{\partial p}{\partial t}(x,t).
	\]

	Parameters
	----------
	P : ndarray
		Pore-pressure history, dimensions 2D : Nt,Nx.
	PARAM : dict
		Parameters dictionnary. 
	int_x : bool (default=True)
		Option to integrate over the space domain, to obtain
		$\frac{dM}{dt}(t)$, the total mass derivative over the domain.
		True activates it, False (default) deactivates it.
	int_t : bool (default=True)
		Option to integrate over the time domain, to obtain
		$\delta m(x,t)$ (or $\delta M(t)$ if int_x is True),
		the total mass accumulated from the initial time.
		True activates it, False (default) deactivates it.

	Returns
	-------
	deltaM : ndarray (default, for int_x=True,int_t=True)
		Accumulation in time of total mass per unit area in domain,
		dimensions: Nt
	dMdt : ndarray (for int_x=True,int_t=False)
		Variation in time of total mass per unit area in domain,
		dimensions: Nt
	deltam : ndarray (for int_x=False,int_t=True)
		Accumulation of mass per unit area per unit length of space
		domain, at each point of space domain, dimensions: Nt,Nx
	dmdt : ndarray (for int_x=False,int_t=True)
		Variation in time mass per unit area per unit length of space
		domain, at each point of space domain, dimensions: Nt,Nx
	
	"""
	# --> Unpack parameters
	# -->> Physical parameters
	phi = PARAM['phi']
	rho = PARAM['rho']
	beta = PARAM['beta']
	# -->> Physical scales
	P_scale = PARAM['P_scale']
	X_scale = PARAM['X_scale']
	T_scale = PARAM['T_scale']
	q_scale = PARAM['q_scale']
	M_scale = q_scale * T_scale
	# -->> Numerical parameters
	h_ = PARAM['h_']
	dt_ = PARAM['dt_']

	# --> Compute time derivative of pressure
	dPdt = (P[1:,:] - P[:-1,:])/dt_

	# --> Compute mass
	dmdt = rho * beta * phi * dPdt * P_scale/M_scale
	
	if int_x:
	# --> Total over x-domain
		dMdt = np.sum(dmdt*h_,axis=1) * X_scale
		if int_t:
		# -->> Total mass accumulation over domain
			deltaM = np.cumsum(dMdt*dt_)
			return deltaM
		else:
		# -->> Variation of total mass over the domain
			return dMdt
	else:
	# --> Mass x-profile
		if int_t:
		# -->> Mass accumulation per unit length for each x, at each t(.
			deltam = np.cumsum(dmdt*dt_,axis=1)
			return deltam

		else:
		# -->> Variation of mass per unit length for each x, at each t.
			return dmdt
#---------------------------------------------------------------------------------

def mass_balance_dflux(P,barriers,states,PARAM,int_x=True,int_t=True):
	"""
	Computes the surfacic mass history, using mass balance at each point of
	the domain.

	Parameters
	----------
	P : ndarray
		Pore-pressure history, dimensions 2D : Nt,Nx.
	barriers : dict
		Barriers dictionnary.
	states : ndarray
		Valves state in time, array with boolean-like values: True is
		open False is closed, dimensions : Nt, Nbarriers.
	PARAM : dict
		Parameters dictionnary. 
	int_x : bool (default=False)
		Option to integrate over the space domain, to obtain
		$\frac{dM}{dt}(t)$, the total mass derivative over the domain.
		True activates it, False (default) deactivates it.
	int_t : bool (default=False)
		Option to integrate over the time domain, to obtain
		$\delta m(x,t)$ (or $\delta M(t)$ if int_x is True),
		the total mass accumulated from the initial time.
		True activates it, False (default) deactivates it.

	Returns
	-------
	deltaM : ndarray (default, for int_x=True, int_t=True)
		Mass accumulation in time over the whole space domain, 
		dimensions 1D : Nt
	dMdt : ndarray (default, for int_x=True, int_t=False)
		Time derivative of total mass accumulation over the space
		domain, in time, dimensions 1D : Nt
	deltam : ndarray (for int_x=False, int_t=True)
		Accumulation of mass per unit length in time at each point of the
		domain, dimensions 2D : Nt,Nx
	dmdt : ndarray (for int_x=False, int_t=False)
		Time derivative of accumulation of mass per unit length in time
		at each point of the domain, dimensions 2D : Nt,Nx

	Note:
	-----
		Computation of M at boundarie is not implemented yet.
	"""
	# --> Unpack
	dt_ = PARAM['dt_']
	h_ = PARAM['h_']
	X_scale = PARAM['X_scale']

	# --> Initialize
	Nt = np.shape(P)[0]

	# --> Through time, compute mass increment
	dmdt = np.zeros((Nt,np.shape(P)[1]-2))
	for tt in range(Nt):
		# -->> Compute permeability profile as a function of valves
		#      state
		k = k_init(barriers,PARAM,state_override=states[tt,:])

		# -->> Compute mass increment
		dmdt[tt,:] = calc_dmdt_dflux(P[tt],k,PARAM)

	if int_x:
	# --> Total over x-domain
		dMdt = np.sum(dmdt*h_,axis=1) * X_scale
		if int_t:
		# -->> Total mass accumulation over domain
			deltaM = np.cumsum(dMdt*dt_)
			return deltaM
		else:
		# -->> Variation of total mass over the domain
			return dMdt
	else:
	# --> Mass x-profile
		if int_t:
		# -->> Mass accumulation per unit length for each x, at each t(.
			deltam = np.cumsum(dmdt*dt_,axis=1)
			return deltam

		else:
		# -->> Variation of mass per unit length for each x, at each t.
			return dmdt


#---------------------------------------------------------------------------------


def calc_dmdt_dflux(P,k,PARAM):
	r"""
	Computes the time derivative of volumic mass over the space domain, at
	time $t$, using:
	\[
	\frac{\partial m}{\partial t}(x,t) = \frac{\partial q}{\partial x}(x,t).
	\]

	Parameters
	----------
	P : ndarray
		Pore-pressure history, dimensions 2D : Nt,Nx.
	k : ndarray
		Permeability profile, dimension : Nx+1.
	PARAM : dict
		Parameters dictionnary. 

	Returns
	-------
	dmdt : ndarray
		Time derivative of mass at time t, over the whole space domain,
		dimension : Nx
	
	Note
	----
		Boundaries are not implemented yet, dmdt is actually Nx-2	

	"""
	# --> Unpack parameters
	# -->> Physical parameters
	mu = PARAM['mu']
	rho = PARAM['rho']
	# -->> Boundary conditions
	p0_ = PARAM['p0_']
	pL_ = PARAM['pL_']
	qin_ = PARAM['qin_']
	qout_ = PARAM['qout_']
	# -->> Physical scales
	P_scale = PARAM['P_scale']
	X_scale = PARAM['X_scale']
	T_scale = PARAM['T_scale']
	q_scale = PARAM['q_scale']
	M_scale = q_scale * T_scale
	# -->> Numerical parameters
	h_ = PARAM['h_']

	# --> Compute first derivative of pressure
	# -->> Within the domain
	dPdx = np.zeros(len(k))
	dPdx[1:-1] = (P[1:] - P[:-1])/h_

#	# -->> At boundaries (pbs with variable space step)
#	if np.isnan(p0_) and (not np.isnan(qin_)):
#		# -->> Fixed flux in 0
#		dPdx[0] = -1. * qin_ * mu / rho / k[0] * h_/hb_
#	elif (not np.isnan(p0_)) and (np.isnan(qin_)):
#		# -->> Fixed pressure in 0
#		dPdx[0] = (P[0] - p0_) / hb_
#
#	if np.isnan(pL_) and (not np.isnan(qout_)):
#		# -->> Fixed flux in L
#		dPdx[-1] = -1. * qout_ * mu / rho / k[-1] 
#	elif (not np.isnan(pL_)) and (np.isnan(qout_)):
#		# -->> Fixed pressure in L
#		dPdx[-1] = (pL_ - P[-1]) / h_

	# --> Compute dM
	kdPdx = k * dPdx
	dmdt = rho/mu * (kdPdx[2:-1] - kdPdx[1:-2])/h_ * P_scale/X_scale**2/(M_scale/T_scale)

	return dmdt

# ---------------------------------------------------------------------
	

def calcQ(P,k,PARAM):
	"""
	Calculates the massic flux in between the input P
	- Parameters
		+ :param P: pore pressure
		+ :param k: permeability distribution
		+ :param PARAM: dictionnary of the system's characteristic parameters
	- Outputs
		+ :return: Q: The flux calculated everywhere except at the boundaries,
		in between values of P.

	"""

	# Unpack
	## Physical and numerical parameters
	h = PARAM['h_']
	mu = PARAM['mu']
	rho = PARAM['rho']
	g = PARAM['g']
	alpha = PARAM['alpha']
	## Scales
	q_scale = PARAM['q_scale']
	X_scale = PARAM['X_scale']
	P_scale = PARAM['P_scale']

	# Calculate Q
	Q = np.zeros(len(k-2))
	dpdx = (P[1:] - P[:-1])/h
#	Q_ = -1*rho*k[1:-1]/mu / q_scale * (dpdx_ * P_scale/X_scale + rho*g*np.sin(alpha))
	Q = -1*rho*k[1:-1]/mu / q_scale * (dpdx * P_scale/X_scale) # Pression réduite
	return Q

#---------------------------------------------------------------------------------


def Perf(X,T,X0,D,dP):
    """ Calculate the P error function for all times on the whole domain (inf domain)"""

    print('Perf -- Computing infinite domain solution...')
    erfP = np.zeros((len(T),len(X)))

    for itt, tt in enumerate(T):
        v = (X-X0) / np.sqrt(4.*D*tt)
        erfP[itt,:] = dP / 2. * ssp.erf(v)

    print('Perf -- Done !\n')

    return erfP

#---------------------------------------------------------------------------------



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
				erfP[itt,:] = erfP[itt,:] + dP/2.*ssp.erf(v1) + dP/2.*ssp.erf(v2)

	print('Perf_bounded -- Done')

	return erfP

#---------------------------------------------------------------------------------
def Pred2P(Pr,PARAM):
	"""
	Computes the real pore pressure, using the reduced pore pressure profile.
	Input and output variables are either all adim or all dim.

	- Parameters:
		+ :param Pr: nd array 1D, reduced pore pressure.
		+ :param PARAM: dictionnary of parameters.
	- Outputs:
		+ :return: P, nd array 1D, same shape as Pr, real pore pressure.

	"""
	# Unpack
	rho = PARAM['rho']
	alpha = PARAM['alpha']
	g = PARAM['g']
	Nx = PARAM['Nx']
	h = PARAM['h_']
	Z0 = PARAM['Z0_']
	X_scale = PARAM['X_scale']
	Z_scale = PARAM['Z_scale']
	P_scale = PARAM['P_scale']
	print(Z0_,Nx,h_)


	X = np.linspace(0,Nx*h,num=Nx+1) #* X_scale/X_scale
	Xref = Z0/np.sin(alpha) * Z_scale/X_scale

	# Core
	P = Pr - rho*g*np.sin(alpha)*(X-Xref) * X_scale/P_scale

	return P
#---------------------------------------------------------------------------------

def P2Pred(P,PARAM):
	"""
	Computes the reduced pore pressure, using the real pore pressure profile.
	Input and output variables are either all adim or all dim.

	- Parameters:
		+ :param P: nd array 1D, real pore pressure.
		+ :param PARAM: dictionnary of parameters.
	- Outputs:
		+ :return: Pr, nd array 1D, same shape as P, reduced pore pressure.

	"""
	# Unpack
	rho = PARAM['rho']
	alpha = PARAM['alpha']
	g = PARAM['g']
	Nx = PARAM['Nx']
	h = PARAM['h_']
	Z0 = PARAM['Z0_']
	X_scale = PARAM['X_scale']
	Z_scale = PARAM['Z_scale']
	P_scale = PARAM['P_scale']

	print(Z0_,Nx,h_)

	X = np.linspace(0,Nx*h,num=Nx+1) #* X_scale/X_scale
	Xref = Z0/np.sin(alpha) * Z_scale/X_scale
	# Core
	Pr = P + rho*g*np.sin(alpha)*(X-Xref) * X_scale/P_scale

	return Pr

#---------------------------------------------------------------------------------

def bX2idx(X,d):
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

#---------------------------------------------------------------------------------

def bdist2idx(X,d,w):
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
        idx2 = int(np.floor(mid_idx + idx_d/2.).astype(int) - 1 )

        return idx1,idx2

#---------------------------------------------------------------------------------

def pp_eq_profile(barriers, PARAM):

	"""
	This functions calculates the equilibrium profile of reduced pore
	pressure for a given boundary conditions and permeability distribution.

	- Parameters:
		+ :param PARAM: Parameters dictionnary.
		+ :param barriers: barriers dictionnary.
	- Outputs:
		+ :return Peq: 1D np array, reduced pore pressure equilibrium profile

	"""

	# Unpack parameters
	q_scale = PARAM['q_scale']
	X_scale = PARAM['X_scale']
	P_scale = PARAM['P_scale']

	Nx = PARAM['Nx']
	h_ = PARAM['h_']
	hb_ = PARAM['hb_']
	X_ = np.linspace(0,Nx*h_,Nx+1)

	p0_ = PARAM['p0_']
	pL_ = PARAM['pL_']
	qin_ = PARAM['qin_']
	qout_ = PARAM['qout_']

	k_bg = PARAM['k'][0]
	mu = PARAM['mu']
	rho = PARAM['rho']

	# Unpack barriers
	b_close = [not o for o in barriers['open']]
	b_idx = barriers['idx'][b_close]
	b_wid = barriers['width'][b_close]
	b_k   = barriers['klo'][b_close]

	# Compute pressure profile
	if np.isnan(p0_) & np.isnan(qout_):
	# --> Flux boundary condition in 0
	#     pressure boundary condition in L
		# -->> Define various parameters
		L_ = 1. + 2 * hb_
		dP_ = mu / rho * qin_ * ( (L_ - np.sum(b_wid))/k_bg + \
		 np.sum(b_wid/b_k)) * q_scale*X_scale/P_scale
		p0_ = pL_ + dP_
		grad_bg = qin_ * mu / rho / k_bg * q_scale*X_scale/P_scale

		# -->> Initialize: fill the first empty segment of the domain
		Peq_ = np.zeros(Nx+1)
		Peq_[:b_idx[0]+1] = p0_ - (X_[:b_idx[0]+1]+hb_) * grad_bg

		for bb in range(len(b_idx)):
		#	-->> Compute a few quantities
			idx_bb = b_idx[bb] + 1  # first P pt in barrier
			idx_sb = int( b_idx[bb] + b_wid[bb]/h_)  # first P pt in next segment
			if bb+1 < len(b_idx): idx_se = b_idx[bb+1]  # last P pt of segment
			else: idx_se = len(X_)-1  # or last point of domain

			p0_ = Peq_[idx_bb - 1]
			b_X_ = np.linspace(h_, b_wid[bb], b_wid[bb]/h_)
			grad_b = qin_ * mu / rho / b_k[bb] * X_scale/P_scale*q_scale

			s_X_ = np.linspace(h_,(idx_se - idx_sb)*h_,idx_se - idx_sb )

	#	-->> Fill a barrier
			Peq_[ idx_bb : idx_sb + 1] = p0_ - grad_b * b_X_
	#	-->> Fill the following segment
			p0_ = Peq_[idx_sb]
			Peq_[ idx_sb + 1 : idx_se + 1] = p0_ - grad_bg * s_X_

	elif np.isnan(qin_) & np.isnan(qout_):
	# --> Pressure boundary conditions
	#	-->> Define various parameters
		DP_ = (p0_ - pL_)
		L_ = 1. + 2 * hb_
		grad_bg = DP_ / ( (L_ - np.sum(b_wid)) + k_bg * np.sum(b_wid/b_k) )

	#	-->> Initialize: fill the first empty segment of the domain
		Peq_ = np.zeros(Nx+1)
		Peq_[:b_idx[0]+1] = p0_ - (X_[:b_idx[0]+1]+hb_) * grad_bg


		for bb in range(len(b_idx)):
	#		-->> Compute a few quantities
			idx_bb = b_idx[bb] + 1  # first P pt in barrier
			idx_sb = int( b_idx[bb] + b_wid[bb]/h_)  # first P pt in next segment
			if bb+1 < len(b_idx): idx_se = b_idx[bb+1]  # last P pt of segment
			else: idx_se = len(X_)-1  # or last point of domain

			p0_ = Peq_[idx_bb-1]
			b_X_ = np.linspace(h_, b_wid[bb], b_wid[bb]/h_)
			grad_b = DP_ / ( b_k[bb]/k_bg * (L_ - np.sum(b_wid)) +\
			 b_k[bb] * np.sum(b_wid/b_k) )

			s_X_ = np.linspace(h_,(idx_se - idx_sb)*h_,idx_se - idx_sb )

	#	-->> Fill a barrier
			Peq_[ idx_bb : idx_sb + 1] = p0_ - grad_b * b_X_
	#	-->> Fill the following segment
			p0_ = Peq_[idx_sb]
			Peq_[ idx_sb + 1 : idx_se + 1] = p0_ - grad_bg * s_X_

	return Peq_

#---------------------------------------------------------------------------------

def k_eq(barriers,PARAM,states=None):
	"""
	This function computes the equivalent permeability of a domain with
	barriers, based on the equivalent resistivity formula.

	- Parameters:
		+ :param barriers: barriers dictionnary
		+ :param PARAM: physical parameters dictionnary
		+ :param states: boolean array of barrier states: True is open,
		False is closed. If states is given, it overrides states in
		barriers. states can be 1D (N_barriers length), or 2D (N_time x
		N_barriers). For the latter case, k_eq will be an array of
		length N_time.
	- Output:
		+ :return k_eq: float or 1D array, value of equivalent
		permeability.
		

	"""
	# --> Unpack physical parameters
	k_bg = PARAM['k_bg']
	L = 1.  # barriers widths and domain widths should have the same scale

	# --> Unpack barriers parameters
	L_b = barriers['width']
	R_b = L_b / barriers['klo']

	# -->> Get states
	if type(states)==type(None):
		b_close = np.bitwise_not(barriers['open'].astype(bool))
	else:
		if np.shape(states)[-1] != len(L_b):
			raise ValueError('states must have the following dimensions: N_times * N_barriers, but they are: {:}'.format(np.shape(states)))
		b_close = np.bitwise_not(states.astype(bool))

	# --> Compute equivalent resistivity
	if len(np.shape(b_close))>1:
		# -->> We compute R_eq at N_times time steps
		N_times = np.shape(b_close)[0]
	
		R_eq = [ np.sum( R_b[ b_close[tt,:] ] ) +  \
		        (L - np.sum( L_b[ b_close[tt,:] ] ) ) / k_bg \
	                for tt in range(N_times)]
		R_eq = np.array(R_eq)
	else:
		# -->> We compute R_eq for given time step
		R_eq = np.sum(R_b[b_close]) + (L - np.sum(L_b[b_close])) / k_bg

	k_eq = L / R_eq
	return k_eq

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



""" Initialize a PPvalve run: initial conditions, boundary conditions, and
building the system to solve. """

# Imports
# =======
import numpy as np

# Core
# ====

# ----------------------------------------------------------------------------

def init_cond(Z, PARAM, option):
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
        #print("initcond -- Hydrostatic gradient, no flux in the system")
        Ztop = Z[-1]
        Pr[-1] = rho_r*g*Ztop * Z_scale/P_scale # Ptop
        #Pr[:-1] = Pr[-1] + rho*g*(Z[:-1]-Ztop) * Z_scale/P_scale # All pressures
        # in between
        Pr[:] = (rho_r-rho)*g*Ztop * Z_scale/P_scale
    # Step function
    if option==2:
        dP, iX0 = PARAM['step']
        #print("initcond -- Erf test: initial step, dP = {:.2f}".format(dP))
        Ztop = Z[-1]
        #P[-1] = rho_r*g*Ztop * Z_scale/P_scale # Ptop
        #P[:-1] = P[-1] + rho*g*(Z[:-1]-Ztop) * Z_scale/P_scale # All pressures
        # in between
        Pr[:] = (rho_r-rho)*g*Ztop * Z_scale/P_scale
        step = np.zeros(len(Pr))
        step[:iX0] = step[:iX0] + 0.5*dP*np.ones(iX0)
        step[iX0:] = step[iX0:] - 0.5*dP*np.ones(len(Pr)-iX0)
        Pr = Pr + step
    # Lithostatic gradient
    if option == 1:
        Ztop=Z[-1]
        #print("initcond -- Lithostatic gradient, qlitho in the system")
        #Pr = (rho_r-rho)*g*(Z-Ztop+hb_*np.sin(alpha)) * Z_scale/P_scale
        Pr = np.linspace(1,0,num=Nx+1)
    # Cosine function
    if option==3:
        h_= PARAM['h_']
        Nx = PARAM['Nx']
        nb_wavelength, amp = PARAM['cos_P']
        # print("initcond -- Cosine profile, {0:d} wavelengths, amp = {1:.2f}".format(nb_wavelength, amp))
        X_cos = 2*np.pi*np.linspace(0.,Nx*h_,num=Nx+1)
        cos_P = amp*np.cos(X_cos*nb_wavelength)
        Pr = cos_P

    return Pr


def build_sys(PARAM):
    """
    This function sets up the elements for the numerical resolution of
    a diffusion problem, with Crank Nicolson method.

    The problem is expressed as the linear system A . p_n+1 = B . p_n + b_n+1 + b_n,
    this function builds A, B, and b matrices, and returns a storage efficient
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

    A = [aa, ab, ac]
    B = [ba, bb, bc]

    # Apply boundary conditions to A and B and create b
    A, B, b = boundary(A, B, PARAM)

    return A, B, b

#-----------------------------------------------------------------------------

def boundary(A, B, PARAM):
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

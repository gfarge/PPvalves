""" Initialize a PPvalve run: initial conditions, boundary conditions, and
building the system to solve. """

# Imports
# =======
import copy
import numpy as np

import PPvalves.equilibrium as equil
import PPvalves.utility as util

# Core
# ====

def read_input(input_args, verbose=False):
    """
    Parses input arguments and processes them into the PARAM dictionnary
    (physical and numerical parameters).  When a certain parameter is not
    specified, its default value is used.

    Parameters
    ----------
    input_args : list
        List of strings of characters, that will be parsed and interpreted.
        Example: to choose the value 6.8e-6 for parameter Xi, "Xi=6.8e-6" must
        be present in the list. Any parameter for which no value is specified
        will be set to default.
    verbose : bool (default `False`)
        Option to make it speak to you.

    Returns
    -------
    PARAM : dictionnary
        Numerical and physical parameters dictionnary. Full description in
        README.
    """
    # Default parameters
    # ------------------
    # NB: here, only the parameters that do not depend on others. With those,
    # scales, numerical parameters etc are defined.

    PARAM = {
        'g' : 9.81,  # gravity acceleration (m.s-2)
        'rho' : 1000, 'rho_r' : 2850,  # fluid and rock density (kg.m-3)

        'alpha' : 10 / 180 * np.pi,  # fault dip angle
        'A' : 1.e3,  # in m2, fault section area (1e3 m2)
        'X0' : 0, 'Xtop' : 20 * 1000,  # extremities of the channel (m)
        'Z0' : 40 * 1000,  # depth of deep end of channel (m)

        'mu' : 1.e-3, # fluid viscosity, Pa.s (1e-3)
        'phi': 0.05,  # rock porosity, between 0 and 1 (0.05, max 0.1)
        'k_bg' : 6.e-12,  # rock permeability, m2 (3e-12 m2)
        'beta' : 2.2e-8,  # pore/fluid compressibility, in Pa-1 (2.2 GPa-1)

        'Nx' : 500,  # number of space increments in domain (number of pnts -1)
        'hb_' : 1 / 100,  # depth of boundary (last --> fictive pnts)
        'Ttot_' : 0.05,  # physical duration of the run, scaled

        'bound' : 'QP',  # boundary conditions code ('PP', 'QP')
        'bound_value' : 1.0,  # scaled value of fixed variable at boundary in 0

        'init_v_state' : 'open'  # initial valve state
        }

    for arg_str in input_args:
        arg, value = arg_str.split('=')
        if arg in PARAM.keys():
            if verbose: print('Got {:}, puts it in'.format(arg))
            if isinstance(PARAM[arg], int):
                PARAM[arg] = int(value)
            elif isinstance(PARAM[arg], float):
                PARAM[arg] = float(value)
            elif isinstance(PARAM[arg], str):
                PARAM[arg] = value

    # Compute the rest of parameters
    # ------------------------------
    # (i) Misc
    PARAM['Ztop'] = PARAM['Z0'] \
                   - (PARAM['Xtop'] - PARAM['X0'])*np.sin(PARAM['alpha'])

    PARAM['D'] = PARAM['k_bg'] / PARAM['phi']/PARAM['mu']/PARAM['beta']

    # (ii) Dimensions
    PARAM['X_scale'] = PARAM['Xtop'] - PARAM['X0']
    PARAM['Z_scale'] = PARAM['X_scale'] * np.sin(PARAM['alpha'])
    PARAM['T_scale'] = PARAM['X_scale']**2 / PARAM['D']
    PARAM['P_scale'] = (PARAM['rho_r'] - PARAM['rho']) * PARAM['g']\
                      * PARAM['X_scale'] * np.sin(PARAM['alpha'])
    PARAM['q_scale'] = PARAM['k_bg'] * PARAM['rho'] / PARAM['mu']\
                      * PARAM['P_scale'] / PARAM['X_scale']
    PARAM['M_scale'] = PARAM['q_scale'] * PARAM['T_scale']

    PARAM['Z0_'] = PARAM['Z0']/PARAM['X_scale']

    # (iii) Discretization
    PARAM['h_'] = 1 / PARAM['Nx']
    PARAM['dt_'] = 0.5 * PARAM['h_']**2
    PARAM['Nt'] = int(np.ceil(PARAM['Ttot_']/PARAM['dt_']))


    # Set up boundary conditions
    # --------------------------
    PARAM = boundary(PARAM['bound'], PARAM['bound_value'], PARAM)

    return PARAM

# ----------------------------------------------------------------------------

def boundary(bound, bound_value, PARAM, verbose=False):
    """
    Set up the boundary condition parameters depending on the option you
    choose.

    Parameters
    ----------
    bound : str
        Boundary option: 'PP', 'QP', 'QQ' implemented for now. First letter
        represent downdip boundary, second letter updip boundary. 'P' is for
        fixed pore pressure (lithostatic value at corresponding depth), 'Q' is
        for fixed flux (flux maintaining lithostatic gradient without
        valves).
    bound_value : float
        Boundary condition value. Value of imposed dP if bound is 'PP', q if
        bound is 'QP'.
    PARAM : dictionnary
        Dictionnary of the system parameters
    verbose : boolean (default verbose=False)
        Make the function talk.

    Returns
    -------
    PARAM : dictionnary
        Updated dictionnary of the system parameters

    Note:
    -----
    For now, only implemented to manage adimensionnalized variables.

    """

    # Fix boundary condition
    # ----------------------
    if bound == 'PP':
        # p0_ = 1 + PARAM['hb_']
        # pL_ = 0 - PARAM['hb_']
        p0_ = bound_value
        pL_ = 0
        qin_ = np.nan
        qout_ = np.nan
        if verbose:
            print('init.boundary -- Border conditions : p0_ = {0:.4f}, pL_ = {1:.4f}'.format(p0_, pL_))

    elif bound == 'QQ':
        p0_ = np.nan
        pL_ = np.nan
        qin_ = bound_value
        qout_ = bound_value
        if verbose:
            print('init.boundary -- Border conditions : qin = {0:.4f}, qout = {1:.4f}'.format(qin_, qout_))

    elif bound == 'QP':
        p0_ = np.nan
        pL_ = 0
        qin_ = bound_value
        qout_ = np.nan
        if verbose:
            print('init.boundary -- Border conditions : qin = {0:.4f}, pL = {1:.4f}'.format(qin_, pL_))

    else:
        raise ValueError("bound can only be 'PP', 'QP', 'QQ'.")

    # Package it
    # ----------
    PARAM['p0_'] = p0_
    PARAM['pL_'] = pL_
    PARAM['qin_'] = qin_
    PARAM['qout_'] = qout_

    return PARAM


# ----------------------------------------------------------------------------

def init_cond(VALVES, PARAM, q0=1, dp0=None, states_override=None):
    """
    Sets up initial conditions of pore pressure, permeability and valve states.
    The initial conditions correspond to the state of a given valve system at
    equilibrium, under a flux q0 or a cross-system pressure differential dp0.
    The valve opening/closing state is either given by VALVES['open'] or by the
    states_override option.

    Parameters
    ----------
    VALVES : dict.
        Valves paramaters dictionnary. If states_override is not used,
        VALVES['open'] is used to define the states of valves.
    PARAM : dictionnary
        Parameters dictionnary.
    q0 : float (default q0 = 1, lithostatic flux)
        The flux the whole system should witness in its initial condition.
        Only q0 or dp0 should be defined, the other is set to None.
    dp0 : float (default dp0 = None)
        The cross-system pore pressure differential the system should witness
        in its initial condition. Only q0 or dp0 should be defined, the other
        is set to None.
    states_override : 1D array (default=None)
        Valve states for which to set initial conditions, overriding the valve
        states in VALVES['open']. True (or 1) is open, False (or 0) is closed,
        dimension of N_valves.

    Returns
    -------
    P : 1d array
        Pore pressure initial state, dimension PARAM['Nx'] + 1
    VALVES : dict.
        Valves dictionnary with input initial states.
    PARAM : dict.
        Physical parameters dictionnary, with k updated.

    Examples
    --------
    - For a flat lithostatic pressure: input dp0=None, q0=1 and all valves
      open.
    - For a flat hydrostatic pressure: input dp0=None, q0=0 and all valves
      open.

    """
    # Compute permeability
    # --------------------
    PARAM['k'] = util.calc_k(VALVES, PARAM, states_override=states_override)

    # Compute pore pressure profile
    # -----------------------------
    # --> First use PARAM structure but using q0/dp0 as boundary conditions
    PARAM_pp_prof = copy.copy(PARAM)
    if (q0 is None) and (dp0 is not None):
        PARAM_pp_prof['qin_'] = np.nan
        PARAM_pp_prof['qout_'] = np.nan
        PARAM_pp_prof['p0_'] = dp0
        PARAM_pp_prof['pL_'] = 0
    elif (q0 is not None) and (dp0 is None):
        PARAM_pp_prof['qin_'] = q0
        PARAM_pp_prof['qout_'] = np.nan
        PARAM_pp_prof['p0_'] = np.nan
        PARAM_pp_prof['pL_'] = 0
    else:
        raise ValueError('One and only one of dp0/q0 should be specified, the other left to None')

    P = equil.calc_pp_inf(VALVES, PARAM_pp_prof, \
                          states_override=states_override)

    # Update valve states if states are overridden
    # --------------------------------------------
    if states_override is not None:
        VALVES['open'] = states_override

    return P, VALVES, PARAM

# -----------------------------------------------------------------------------

def test_init_cond(option, init_param, PARAM):
    """
    Set up initial conditions of reduced pore pressure for testing.

    Parameters
    ----------
    option : float
        Initial condition option
    		- option=1 : Initial conditions are set to a ramp function.
              init_params must contain x1 and x2, the first point of the ramp,
              x2 the last point. It is a linear ramp step from the region
              before x1 at 1, to the region after x2 at 0.
    		- option=2 : Initial conditions are set to a cosine function.
              init_params must contain n_wv and amp, number of wavelengths and
              amplitude.
    init_param : tuple
        Initial conditions parameters. See above for details.
    PARAM : dictionnary
        Dictionnary of the system's physical parameters.

    Returns
    -------
    p0 : ndarray
        Initialized vector of pore-pressures.

    """
    X = np.linspace(0, 1, num=PARAM['Nx']+1)
    p0 = np.zeros(PARAM['Nx'] + 1)

    # First option: the step
    # ----------------------
    if option == 1:
        x1, x2 = init_param

        for ii in range(PARAM['Nx']+1):
            if X[ii] <= x1:
                p0[ii] = 1
            elif (X[ii] < x2) & (X[ii] > x1):
                p0[ii] = (x2 - X[ii]) * 1/(x2 - x1)
            elif X[ii] >= x2:
                p0[ii] = 0

    # Second option: the cosine
    # -------------------------
    elif option == 2:
        n_wv, amp = init_param
        p0 = amp * np.cos(X * n_wv)

    return p0

# ----------------------------------------------------------------------------

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
    A, B, b = sys_boundary(A, B, PARAM)

    return A, B, b

#-----------------------------------------------------------------------------

def sys_boundary(A, B, PARAM):
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
    	raise ValueError("sys_boundary -- /!\\ x = L boundary conditions wrongly set")

    return A, B, b

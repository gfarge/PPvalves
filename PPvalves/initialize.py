"""Functions to initialize a PPvalve run.

Manages initial conditions, boundary conditions, and building the matricial
system to solve.
"""

# Imports
# =======
import copy
import numpy as np

import PPvalves.equilibrium as equil
import PPvalves.utility as util

# Core
# ====

def read_input(input_args, verbose=False):
    """Reads input of PPv and assigns default values.

    Parses the string of input arguments and processes them into the `PARAM`
    dictionnary (physical and numerical parameters). When a parameter
    is not specified, its default value is used.

    Parameters
    ----------
    input_args : list
        List of strings of characters, that will be parsed and interpreted.
        To choose the value 6.8e-6 for parameter Xi, "Xi=6.8e-6" must
        be present in the list. Any parameter for which no value is specified
        will be set to default.
    verbose : bool, optional
        Option to have the function print what it's doing.

    Returns
    -------
    PARAM : dictionnary
        Numerical and physical parameters dictionnary.
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

        'init_v_state' : 'closed',  # initial valve state

        'vdist' : 'first'  # valve distribution, either a file _name_,
                           # or 'first' for the first distribution we tested
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

    # (ii) Scales
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
    """Sets boundary conditions.

    Parameters
    ----------
    bound : str
        Boundary option: 'PP', 'QP' implemented for now. First letter
        represent downdip boundary, second letter updip boundary. 'P' is for
        fixed pore pressure, 'Q' is for fixed flux.
    bound_value : float
        Boundary condition value. Value of imposed input pressure if bound is
        'PP', and imposed input flux if bound is 'QP'. Output pressure is fixed
        to 0.
    PARAM : dictionnary
        Physical parameters dictionnary.
    verbose : boolean, optional
        Have the function print what it's doing.

    Returns
    -------
    PARAM : dictionnary
        Updated dictionnary of the system's parameters.

    Note:
    -----
    For now, only implemented to manage adimensionnalized variables.
    """

    # Fix boundary condition
    # ----------------------
    if bound == 'PP':
        p0_ = bound_value
        pL_ = 0
        qin_ = -1 #np.nan
        qout_ = -1 #np.nan
        if verbose:
            print('init.boundary -- Border conditions : p0_ = {0:.4f}, pL_ = {1:.4f}'.format(p0_, pL_))

    elif bound == 'QQ':
        p0_ = -1 #np.nan
        pL_ = -1 #np.nan
        qin_ = bound_value
        qout_ = bound_value
        if verbose:
            print('init.boundary -- Border conditions : qin = {0:.4f}, qout = {1:.4f}'.format(qin_, qout_))

    elif bound == 'QP':
        p0_ = -1 #np.nan
        pL_ = 0
        qin_ = bound_value
        qout_ = -1 #np.nan
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
    r"""Sets up default initial conditions of fluid pressure, permeability and
    valve states.

    The default initial conditions correspond to the state of the input valve
    system at equilibrium, under a the given input flux or a cross-system pressure
    differential.

    Parameters
    ----------
    VALVES : dict.
        Valves paramaters dictionnary. If `states_override` is not used,
        `VALVES['open']` is used to define the states of valves.
    PARAM : dictionnary
        Physical parameters dictionnary.
    q0 : float (default `q0 = 1`, lithostatic flux)
        The flux the whole system should witness in its initial condition.
        Only one of `q0` or `dp0` should be defined at a time, the other is set
        to `None`.
    dp0 : float (default `dp0 = None`)
        The cross-system pore pressure differential the system should witness
        in its initial condition. Only `q0` or `dp0` should be defined, the other
        is set to `None`.
    states_override : 1D array (default : `None`)
        Valve states for which to set initial conditions, overriding the valve
        states in `VALVES['open']`. `True` (or `1`) is open, `False` (or `0`)
        is closed, dimension N_valves.

    Returns
    -------
    P : 1d array
        Pore pressure initial state, dimension `PARAM['Nx'] + 1`.
    VALVES : dict.
        Valves dictionnary updated with input initial states.
    PARAM : dict.
        Physical parameters dictionnary, with permeability `k` updated.
    """
    # Compute permeability
    # --------------------
    PARAM['k'] = util.calc_k(VALVES, PARAM, states_override=states_override)

    # Compute pore pressure profile
    # -----------------------------
    # --> First use PARAM structure but using q0/dp0 as boundary conditions
    PARAM_pp_prof = copy.copy(PARAM)
    if (q0 is None) and (dp0 is not None):
        PARAM_pp_prof['qin_'] = -1 #np.nan
        PARAM_pp_prof['qout_'] = -1 #np.nan
        PARAM_pp_prof['p0_'] = dp0
        PARAM_pp_prof['pL_'] = 0
    elif (q0 is not None) and (dp0 is None):
        PARAM_pp_prof['qin_'] = q0
        PARAM_pp_prof['qout_'] = -1 #np.nan
        PARAM_pp_prof['p0_'] = -1 #np.nan
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
    """Set up initial conditions test situations.

    Sets up cosinusoidal pressure, or a pressure ramp like the one in closed
    valves at equilibrium. It can be used to test the numerical convergence and
    accuracy of the fluid pressure diffusion.

    Parameters
    ----------
    option : float
        Initial condition option:
    		- `option=1` : Initial conditions are set to a ramp function.
               `init_params` must be `(x1, x2)`, the first point of the ramp,
              `x2` the last point. It is a linear ramp step of pressure from
              the region before `x1` at 1, to the region after `x2` at 0.
    		- `option=2` : Initial conditions are set to a cosine function.
              `init_params` must be  `(n_wv, amp)`, number of wavelengths and
              amplitude.
    init_param : tuple
        Initial conditions parameters. See above for details on option specific
        input.
    PARAM : dictionnary
        Dictionnary of the system's physical parameters.

    Returns
    -------
    p0 : ndarray
        Initialized vector of pore-pressure in space.

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
    r"""Builds the matrix system for a given permeability structure in space.

    Diffusion of pore pressure in time is obtained by solving the linear
    system: `A . p_n+1 = B . p_n + b_n+1 + b_n`. This function builds `A`, `B`,
    and `b = b_n+1 + b_n`, and returns a storage efficient structure only
    keeping the three diagonal vectors of the tridiagonal matrices.

    Parameters
    ----------
    PARAM : dictionnary
        Physical parameters dictionnary. Recquired keys: `'h_'` (scaled space
        step), `'dt_'` (scaled time step), `'k'` (vector of permeability in
        space), `'beta'` (matrix-fluid compressibility), 'mu' (fluid
        viscosity), `'phi'`.  (porosity), `'T_scale'` and `'X_scale'` (time and
        space scales).

    Returns
    -------
    A, B : lists
        List for of `A` and `B` matrices: `[a, b, c]`, where `b` is
        diagonal, `a` and `b` lower and upper diagonals, all as arrays.
    b : 1D array
        Vector `b`.

    Notes
    -----
        - When using `[a, b, c]` to represent a tridiagonal matrix, by convention:
        `len(a) = len(b) = len(c)`, `a[0] = 0`, `c[-1] = 0`.
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

    # >> Build A and B core structure
    delta_minus = k[:-1]/phi/mu/beta * dt/2./h**2 * T_scale/X_scale**2
    delta_plus = k[1:]/phi/mu/beta * dt/2./h**2 * T_scale/X_scale**2

    # > A lower diag
    aa = np.zeros(len(delta_minus))
    aa[1:] = -1. * delta_minus[1:]
    # > A diag
    ab = 1. + delta_plus + delta_minus
    # > A upper diag
    ac = np.zeros(len(delta_plus))
    ac[:-1] = -1. * delta_plus[:-1]

    # > A lower diag
    ba = np.zeros(len(delta_minus))
    ba[1:] = delta_minus[1:]
    # > A diag
    bb = 1. - delta_plus - delta_minus
    # > A upper diag
    bc = np.zeros(len(delta_plus))
    bc[:-1] = delta_plus[:-1]

    A = [aa, ab, ac]
    B = [ba, bb, bc]

    # >> Apply boundary conditions to A and B and create b
    A, B, b = sys_boundary(A, B, PARAM)

    A = [np.asfortranarray(d) for d in A]
    B = [np.asfortranarray(d) for d in B]
    b = np.asfortranarray(b)

    return A, B, b

# ------------------------------------------------------------------------------
def update_sys(A, B, PARAM):
    """Updates the matrices after a change.

    Parameters
    ----------
    A, B : lists
        List for of `A` and `B` matrices: `[a, b, c]`, where `b` is
        diagonal, `a` and `b` lower and upper diagonals, all as arrays.
    PARAM : dictionnary
        Physical parameters dictionnary. Recquired keys: `'h_'` (scaled space
        step), `'dt_'` (scaled time step), `'k'` (vector of permeability in
        space), `'beta'` (matrix-fluid compressibility), 'mu' (fluid
        viscosity), `'phi'`.  (porosity), `'T_scale'` and `'X_scale'` (time and
        space scales).

    Returns
    -------
    A, B : lists
        List for of `A` and `B` matrices: `[a, b, c]`, where `b` is
        diagonal, `a` and `b` lower and upper diagonals, all as arrays.

    Notes
    -----
        - When using `[a, b, c]` to represent a tridiagonal matrix, by convention:
        `len(a) = len(b) = len(c)`, `a[0] = 0`, `c[-1] = 0`.
    """
    # >> Build A and B core structure
    delta_minus = PARAM['k'][:-1]/PARAM['phi']/PARAM['mu']/PARAM['beta'] \
                * PARAM['dt_']/2./PARAM['h_']**2  \
                * PARAM['T_scale']/PARAM['X_scale']**2
    delta_plus = PARAM['k'][1:]/PARAM['phi']/PARAM['mu']/PARAM['beta'] \
                * PARAM['dt_']/2./PARAM['h_']**2 \
                * PARAM['T_scale']/PARAM['X_scale']**2

    # > A lower diag
    A[0][1:-1] = -1. * delta_minus[1:-1]
    # > A diag
    A[1][1:-1] = 1. + delta_plus[1:-1] + delta_minus[1:-1]
    # > A upper diag
    A[2][1:-1] = -1. * delta_plus[1:-1]

    # > A lower diag
    B[0][1:-1] = delta_minus[1:-1]
    # > A diag
    B[1][1:-1] = 1. - delta_plus[1:-1] - delta_minus[1:-1]
    # > A upper diag
    B[2][1:-1] = delta_plus[1:-1]

    return A, B

#-----------------------------------------------------------------------------

def sys_boundary(A, B, PARAM):
    """Apply limit conditions to the system equations.

    Parameters
    ----------
        A, B : lists
            `A` and `B` matrices in the form of a list of their three
            diagonal vectors (lower diagonal, diagonal, upper diagonal).
        PARAM : dictionnary
            Physical parameters of the system. Boundary conditions
            specifications must figure in it:
                - if `PARAM['p0_']` (resp `PARAM['pL_']`) is -1 and
                `PARAM['qin_']` (resp `PARAM['qout_']) is not, then apply
                Neuman boundary conditions in matrices.
                - if `PARAM['qin_'] (resp `PARAM['qout_']) is -1 and
                  `PARAM['p0_']` (resp `PARAM['pL_']`) is not, then apply
                Dirichlet boundary conditions in matrices.

    Returns
    -------
        A, B, : lists
            `A` and `B` matrices in the form of a list of their three
            diagonal vectors (lower diagonal, diagonal, upper diagonal), with
            updated boundary conditions.
        b : 1D array
            Vector `b`, with updated boundary conditions.

    Notes
    -----
        - You can determine the depth of the boundary `PARAM['hb_']`, that is, the
        distance between first/last points and fictive points at which boundary
        conditions are fixed.

        - When using `[a, b, c]` to represent a tridiagonal matrix, by convention:
        `len(a) = len(b) = len(c)`, `a[0] = 0`, `c[-1] = 0`.
    """
    # >> Unpack PARAM
    # --> boundary conditions
    p0 = PARAM['p0_']  # already adim
    pL = PARAM['pL_']  # already adim
    qin = PARAM['qin_']  # already adim
    qout = PARAM['qout_']  # already adim
    # --> numerical parameters
    h = PARAM['h_']  # already adim
    hb = PARAM['hb_']  # already adim
    dt = PARAM['dt_']  # already adim

    # --> physical parameters
    k = PARAM['k']
    beta = PARAM['beta']
    mu = PARAM['mu']
    phi = PARAM['phi']
    rho = PARAM['rho']
    alpha = PARAM['alpha']
    g = PARAM['g']
    # --> scales
    X_scale = PARAM['X_scale']
    T_scale = PARAM['T_scale']
    P_scale = PARAM['P_scale']
    q_scale = PARAM['q_scale']

    # >> Initialize b and deltas
    D = k[0]/phi/mu/beta # permeability should be the same at -1, 0, N and N+1
    b = np.zeros(len(A[1]))

#    # -----------------------  x = 0 boundary --------------------------------
#    # >> Neuman : fixing flux
#    isneu0 = (p0 == -1) & (qin != -1)
#    # -> factors of p0_ : Ab[0] and Bb[0]
#    Ab_neu0 = 1. + D * dt / hb * (1./h - 1./(hb+h)) * T_scale/X_scale**2
#    Bb_neu0 = 1. - D * dt / hb * (1./h - 1./(hb+h)) * T_scale/X_scale**2
#    # ->  factors of p1_ : Ac[0] and Bc[0]
#    Ac_neu0 = - 1.* D * dt / (h * (hb+h)) * T_scale/X_scale**2
#    Bc_neu0 = D * dt / (h * (hb+h)) * T_scale/X_scale**2
#    # -> constant terms (b_n = b_n+1, hence the 2)
#    b_neu0 = 2. * D * dt / (h + hb) * T_scale/X_scale**2 *(mu/rho/k[0] * qin * q_scale) * X_scale/P_scale
#
#    # >> Dirichlet : fixing pressure
#    isdir0 = (qin == -1) & (p0 != -1)
#    # -> factors of p0_ : Ab[0] and Bb[0]
#    Ab_dir0 = 1. + D*dt/h/hb * T_scale/X_scale**2
#    Bb_dir0 = 1. - D*dt/h/hb * T_scale/X_scale**2
#    # ->  factors of p1_ : Ac[0] and Bc[0]
#    Ac_dir0 =  -1. *  D*dt / (h * (h + hb)) * T_scale/X_scale**2
#    Bc_dir0 = D*dt / (h * (h + hb)) * T_scale/X_scale**2
#    # -> constant terms (b_n = b_n+1, hence the 2)
#    b_dir0 = 2. * p0 * D*dt / (hb * (h + hb)) * T_scale/X_scale**2 # * P_scale/P_scale
#
#    # >> Check if boundary is correctly set
#    if (isneu0 == isdir0):
#    	raise ValueError("boundary -- /!\\ x = 0 boundary conditions wrongly set")
#
#    # >> Set boundary
#    # -> factors of p0_ : Ab[0] and Bb[0]
#    A[1][0] = isneu0 * Ab_neu0 + isdir0 * Ab_dir0
#    B[1][0] = isneu0 * Bb_neu0 + isdir0 * Bb_dir0
#    # -> factors of p1_ : Ac[0] and Bc[0]
#    A[2][0] = isneu0 * Ac_neu0 + isdir0 * Ac_dir0
#    B[2][0] = isneu0 * Bc_neu0 + isdir0 * Bc_dir0
#    # -> constant terms (b = b_n - b_n+1)
#    b[0] = isneu0 * b_neu0 + isdir0 * b_dir0
#
#    # -----------------------  x = L boundary --------------------------------
#    # >> Neuman : fixing flux
#    isneuL =  (pL == -1) & (qout != -1)
#    # -> factors of pN_: Ab[-1] and Bb[-1]
#    Ab_neuL = 1. + D * dt / hb * (1./h - 1./(hb+h)) * T_scale/X_scale**2
#    Bb_neuL = 1. - D * dt / hb * (1./h - 1./(hb+h)) * T_scale/X_scale**2
#    # -> factors of pN-1_: Aa[-1] and Ba[-1]
#    Aa_neuL = -1. * D * dt / (h * (hb+h)) * T_scale/X_scale**2
#    Ba_neuL = D * dt / (h * (hb+h)) * T_scale/X_scale**2
#    # -> constant terms (b_n = b_n+1, hence the 2)
#    b_neuL = -2. * D * dt / (h + hb) * T_scale/X_scale**2 *(mu/rho/k[-1] * qout * q_scale) * X_scale/P_scale
#
#    # >> Dirichlet : fixing pressure
#    isdirL =  (pL != -1) & (qout == -1)
#    # -> factors of pN_: Ab[-1] and Bb[-1]
#    Ab_dirL = 1. + D * dt / hb / h * T_scale/X_scale**2
#    Bb_dirL = 1. - D * dt / hb / h * T_scale/X_scale**2
#    # -> factors of pN-1_: Aa[-1] and Ba[-1]
#    Aa_dirL = -1. * D * dt / (h * (hb+h)) * T_scale/X_scale**2
#    Ba_dirL = D * dt / (h * (hb+h)) * T_scale/X_scale**2
#    # -> constant terms (b_n = b_n+1, hence the 2)
#    b_dirL = 2. * pL * D*dt / (hb * (h+hb)) * T_scale/X_scale**2 # * P_scale/P_scale
#
#    # >> Check if boundary is correctly set
##    if (isneuL == isdirL):
##    	raise ValueError("sys_boundary -- /!\\ x = L boundary conditions wrongly set")
#
#    # >> Set boundary
#    # --> factors of pN_: Ab[-1] and Bb[-1]
#    A[1][-1] = isneuL * Ab_neuL + isdirL * Ab_dirL
#    B[1][-1] = isneuL * Bb_neuL + isdirL * Bb_dirL
#    # --> factors of pN-1_: Aa[-1] and Ba[-1]
#    A[0][-1] = isneuL * Aa_neuL + isdirL * Aa_dirL
#    B[0][-1] = isneuL * Ba_neuL + isdirL * Ba_dirL
#    # --> constant terms (b_n = b_n+1, hence the 2)
#    b[-1] = isneuL * b_neuL + isdirL * b_dirL
#
#    return A, B, b

    if (p0==-1) & (qin!=-1):
    	# --> factors of p0_ : Ab[0] and Bb[0]
    	A[1][0] = 1. + D * dt / hb * (1./h - 1./(hb+h)) * T_scale/X_scale**2
    	B[1][0] = 1. - D * dt / hb * (1./h - 1./(hb+h)) * T_scale/X_scale**2

    	# --> factors of p1_ : Ac[0] and Bc[0]
    	A[2][0] = - 1.* D * dt / (h * (hb+h)) * T_scale/X_scale**2
    	B[2][0] = D * dt / (h * (hb+h)) * T_scale/X_scale**2

    	# --> constant terms (b = b_n - b_n+1)
    	b[0] = 2. * D * dt / (h + hb) * T_scale/X_scale**2 *(mu/rho/k[0] * qin * q_scale) * X_scale/P_scale

    # --> Dirichlet : fixing pressure
    elif (qin==-1) & (p0!=-1):
    	# --> factors of p0_ : Ab[0] and Bb[0]
    	A[1][0] = 1. + D*dt/h/hb * T_scale/X_scale**2
    	B[1][0] = 1. - D*dt/h/hb * T_scale/X_scale**2

    	# -->  factors of p1_ : Ac[0] and Bc[0]
    	A[2][0] =  -1. *  D*dt / (h * (h + hb)) * T_scale/X_scale**2
    	B[2][0] = D*dt / (h * (h + hb)) * T_scale/X_scale**2

    	# --> constant terms (b_n = b_n+1, hence the 2)
    	b[0] = 2. * p0 * D*dt / (hb * (h + hb)) * T_scale/X_scale**2 # * P_scale/P_scale

    else:
    	raise ValueError("boundary -- /!\\ x = 0 boundary conditions wrongly set")

    # >> x = L boundary
    # -->  Neuman : fixing flux
    if (pL==-1) & (qout!=-1):
    	# --> factors of pN_: Ab[-1] and Bb[-1]
    	A[1][-1] = 1. + D * dt / hb * (1./h - 1./(hb+h)) * T_scale/X_scale**2
    	B[1][-1] = 1. - D * dt / hb * (1./h - 1./(hb+h)) * T_scale/X_scale**2

    	# --> factors of pN-1_: Aa[-1] and Ba[-1]
    	A[0][-1] = -1. * D * dt / (h * (hb+h)) * T_scale/X_scale**2
    	B[0][-1] = D * dt / (h * (hb+h)) * T_scale/X_scale**2

    	# --> constant terms (b_n = b_n+1, hence the 2)
    	b[-1] = -2. * D * dt / (h + hb) * T_scale/X_scale**2 *(mu/rho/k[-1] * qout * q_scale) * X_scale/P_scale

    # --> Dirichlet : fixing pressure
    elif (qout==-1) & (pL!=-1):
    	# --> factors of pN_: Ab[-1] and Bb[-1]
    	A[1][-1] = 1. + D * dt / hb / h * T_scale/X_scale**2
    	B[1][-1] = 1. - D * dt / hb / h * T_scale/X_scale**2

    	# --> factors of pN-1_: Aa[-1] and Ba[-1]
    	A[0][-1] = -1. * D * dt / (h * (hb+h)) * T_scale/X_scale**2
    	B[0][-1] = D * dt / (h * (hb+h)) * T_scale/X_scale**2

    	# --> constant terms (b_n = b_n+1, hence the 2)
    	b[-1] = 2. * pL * D*dt / (hb * (h+hb)) * T_scale/X_scale**2 # * P_scale/P_scale

    return A, B, b



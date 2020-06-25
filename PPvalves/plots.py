""" A module to plot PPvalves stuff """

# IMPORTS
# =======

# General modules
# ---------------
import numpy as np

# Graphical modules
# -----------------
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# My modules
# ----------
from PPvalves.utility import calc_Q, calc_k
import PPvalves.equilibrium as equi


# FUNCTIONS
# =========

def set_plot_params(input_params, needed_params):
    """
    Plot parameters are given default values if the input plot_params do not
    specify them.

    Parameters
    ----------
    input_params : dict
        Dictionnary of input plot parameters.
    needed_params : list
        List of plot parameters that need to be specified.

    Returns
    -------
    plot_params : dict
        Dictionnary of all the necessary plot parameters.
    """
    # Default parameters values
    # -------------------------
    default_params = {
        'v_op_fc' : 'whitesmoke',
        'v_cl_fc' : 'silver',
        'v_fc_sch' : None,
        'v_zorder' : 1,

        'pp_lc' : 'crimson',
        'pp_lw' : 1.5,
        'pp_zorder' : 10,

        'q_lc' : 'teal',
        'q_lw' : 1,
        'q_zorder' : 9,

        'pp_b_m' : 'x',
        'pp_b_ms' : 8,
        'pp_b_mew' : 1.5,

        'q_b_m' : '>',
        'q_b_ms' : 8,
        'q_b_mew' : .5,

        'b_zorder' : 20
              }

    # Specify value for missing input
    # -------------------------------
    plot_params = {}
    for key in needed_params:
        if key in input_params.keys():
            plot_params[key] = input_params[key]
        else:
            plot_params[key] = default_params[key]

    return plot_params

# ----------------------------------------------------------------------------

def valves(X, VALVES, states_override=None, fig=None, ax=None, X_axis='x', plot_params={}):
    """
    A function to plot valves in existing figure/axes.

    Parameters
    ----------
    X : 1D array
        Space position array.
    VALVES : dictionnary
        Valve parameters dictionnary.
    states_override : 1D array (default to None)
        Boolean array overriding VALVE['open'] to plot states. Each element is
        the state of a valve, True is open, False is closed.
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.
    X_axis : str (default is 'x')
        Which axis corresponds to physical X axis. 'x' for horizontal axis, 'y'
        for vertical axis
    plot_params : dictionnary (default is {}, empty dic.)
        a dictionnary of plotting parameters for the valves. Implemented:
        facecolor (any matplotlib ways of indicating color) for both open and
        close state (default = 'v_op_fc' : 'lightgray', 'v_cl_fc' :
        'darkgray'), zorder (default 'zorder' : 0).

    Returns
    -------
    valves_pc : matplotlib PatchCollection object
        Valves patches.
    """
    # As a function of input, point to correct objects for valve states,
    # figure and axis
    # ------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    if X_axis == 'x':
        zlim = ax.get_ylim()
    elif X_axis == 'y':
        zlim = ax.get_xlim()

    if states_override is None:
        open_valves = VALVES['open'].astype(bool)
    else:
        open_valves = states_override.astype(bool)

    # Check which parameters are default and which are defined by user
    # ----------------------------------------------------------------
    needed_params = ['v_op_fc', 'v_cl_fc', 'v_fc_sch', 'v_zorder']
    plot_params = set_plot_params(plot_params, needed_params)

    # Build valve patches and colors as a function of valve states
    # ------------------------------------------------------------
    if X_axis == 'x':
        valve_patches = [Rectangle((x_v, zlim[0]), w_v, zlim[1]-zlim[0])
                         for x_v, w_v in zip(X[VALVES['idx']], VALVES['width'])]
    elif X_axis == 'y':
        valve_patches = [Rectangle((zlim[0], x_v), zlim[1]-zlim[0], w_v)
                         for x_v, w_v in zip(X[VALVES['idx']], VALVES['width'])]
    def set_valve_fc(v_open):
        if v_open:
            color = plot_params['v_op_fc']
        else:
            color = plot_params['v_cl_fc']
        return color

    valve_facecolors = [set_valve_fc(op) for op in open_valves]

    # Wrap patches in collection and plot it
    # --------------------------------------
    valves_pc = PatchCollection(valve_patches,
                                facecolor=valve_facecolors,
                                edgecolor=None,
                                axes=ax, zorder=plot_params['v_zorder'])
    ax.add_collection(valves_pc)

    return valves_pc

# ----------------------------------------------------------------------------

def pp_profile(X, P, fig=None, ax=None, plot_params={}):
    """
    Plots a pore pressure profile on an existing figure.

    Parameters
    ----------
    X : 1D array
        Space position array.
    P : 1D array
        Pore pressure in space array, same dimension as X.
    fig : matplotlib figure object (default to None)
        Figure where to plot the profile. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.
    params : dictionnary (default is {}, empty dic.)
        a dictionnary of plotting parameters. Implemented:
        line color (any matplotlib ways of indicating color, default 'pp_lc' :
        'teal'), line width (default 'pp_lw' : 1.5), zorder (default
        'pp_zorder' : 10, over everything).

    Returns
    -------
    pp_line : matplotlib line object
        Line object for the pore pressure profile.
    """
    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Check which parameters are default and which are defined by user
    # ----------------------------------------------------------------
    needed_params = ['pp_lc', 'pp_zorder', 'pp_lw']
    plot_params = set_plot_params(plot_params, needed_params)

    # Plot pore pressure profile
    # --------------------------
    pp_line = ax.plot(X, P,
                      lw=plot_params['pp_lw'], c=plot_params['pp_lc'],
                      zorder=plot_params['pp_zorder'],
                      label='Pore pressure')
    ax.set_ylabel('Pore pressure', color=plot_params['pp_lc'])
    ax.set_xlabel('<-- Downdip X Updip -->')

    return pp_line

# ----------------------------------------------------------------------------

def q_profile(X, Q, fig=None, ax=None, plot_params={}):
    """
    Plots a flux profile on an existing figure.

    Parameters
    ----------
    X : 1D array
        Space position array.
    Q : 1D array
        Pore pressure in space array, should be dimension of X -1.
    fig : matplotlib figure object (default to None)
        Figure where to plot the profile. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the profile. If not specified, takes output of
        plt.gca(): current active axes.
    params : dictionnary (default is {}, empty dic.)
        a dictionnary of plotting parameters. Implemented:
        line color (any matplotlib ways of indicating color, default 'q_lc' :
        'aquamarine'), line width (default 'q_lw' : 1.5), zorder (default
        'q_zorder' : 10, over everything).

    Returns
    -------
    q_line : matplotlib line object
        Line object for the flux profile.
    """
    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Check which parameters are default and which are defined by user
    # ----------------------------------------------------------------
    needed_params = ['q_lc', 'q_zorder', 'q_lw']
    plot_params = set_plot_params(plot_params, needed_params)

    # Plot massic flux profile
    # ------------------------
    Xq = (X[1:] + X[:-1])/2  # Q is defined in between pressure points
    q_line = ax.plot(Xq, Q,
                     lw=plot_params['q_lw'], c=plot_params['q_lc'],
                     zorder=plot_params['q_zorder'],
                     label='Massic flux')
    ax.set_ylabel('Massic flux', color=plot_params['q_lc'])
    ax.set_xlabel('<-- Downdip X Updip -->')

    ax.set_ylim((0-0.1*np.max(Q), np.max(Q)*1.1))

    return q_line

# ----------------------------------------------------------------------------

def x_profile(X, P, PARAM, VALVES, states_override=None, fig=None, ax=None, plot_params={}):
    """
    Plot profile of pore pressure, flux and valve states and positions at a
    given time on an existing figure and axes.

    Parameters
    ----------
    X : 1D array
        Space position array.
    P : 1D array
        Pore pressure in space array, at plot time, same dimension as X.
    PARAM : dictionnary
        Dictionnary of physical parameters set for the system.
    VALVES : dictionnary
        Valve parameters dictionnary. VALVES['open'] is used for valve states
        if states_override is not specified.
    states_override : 1D array (default to None)
        Boolean array overriding VALVE['open'] to plot states. Each element is
        the state of a valve, True is open, False is closed. Dimension Nvalves.
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.
    params : dictionnary (default is {}, empty dic.)
        a dictionnary of plotting parameters for the pore pressure profile,
        flux profile and valves. See pp_profile, q_profile, and valves
        functions of this module for respective paramaters.

    Returns
    -------
    g_objs : list
        List of matplotlib objects corresponding to pore pressure profile line,
        flux profile line and valves patch collection.
    """
    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax_q = plt.gca()
    else:
        ax_q = ax

    # --> Add another axis for p, and switch location of labels
    ax_p = ax_q.twinx()
    ax_p.tick_params(left=True, labelleft=True,
                     right=False, labelright=False)
    ax_q.tick_params(left=False, labelleft=False,
                     right=True, labelright=True)
    ax_p.yaxis.set_label_position("left")

    # Plot pore pressure profile
    # --------------------------
    pp_line = pp_profile(X, P, fig=fig, ax=ax_p,
                         plot_params=plot_params)

    # Compute and plot flux profile
    # -----------------------------
    k = calc_k(VALVES, PARAM, states_override=states_override)
    Q = calc_Q(P, k, PARAM)
    q_line = q_profile(X, Q, fig=fig, ax=ax_q,
                       plot_params=plot_params)
    ax_q.yaxis.set_label_position("right")

    # Plot valves
    # -----------
    valves_pc = valves(X, VALVES, states_override=states_override, fig=fig,
                       ax=ax_q, plot_params=plot_params)

    #    ax_p.set_title('State of the system at t={:.2f}'.format(t_plot))

    axes = [ax_p, ax_q]
    g_objs = [pp_line, q_line, valves_pc]

    return axes, g_objs

# ----------------------------------------------------------------------------

def bounds(p0, PARAM, fig=None, ax=None, plot_params={}):
    """
    Plot profile of pore pressure, flux and valve states and positions at a
    given time on an existing figure and axes.

    Parameters
    ----------
    p0 : float
        Y-coordinate (pore pressure possibly?) for input point of domain.
    PARAM : dictionnary
        Dictionnary of physical parameters set for the system.
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.
    params : dictionnary (default is {}, empty dic.)
        a dictionnary of plotting parameters for the pore pressure profile,
        flux profile and valves. See pp_profile, q_profile, and valves
        functions of this module for respective paramaters.

    Returns
    -------
    ax : axes object from matplotlib.
        Updated ax object.

    g_objs : list
        List of matplotlib objects corresponding to pore pressure profile line,
        flux profile line and valves patch collection.
    """
    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Check which parameters are default and which are defined by user
    # ----------------------------------------------------------------
    needed_params = ['q_b_m', 'q_b_ms', 'q_b_mew', 'q_lc',
                     'pp_b_m', 'pp_b_ms', 'pp_b_mew', 'pp_lc',
                     'b_zorder']
    plot_params = set_plot_params(plot_params, needed_params)

    # Check boundary conditions
    # -------------------------
    if np.isnan(PARAM['qin_']):
        # --> Fixed pressure
        p0 = PARAM['p0_']
        x_in = 0 - PARAM['hb_']

        str_in = r'$p_{{in}}={0:.2f}$'.format(PARAM['p0_'])

        mark_in = plot_params['pp_b_m']
        mark_in_ec = plot_params['pp_lc']
        mark_in_fc = None
        mark_in_ew = plot_params['pp_b_mew']
        mark_in_s = plot_params['pp_b_ms']

    else:
        # --> Fixed flux
        x_in = 0
        str_in = r'$q_{{in}}={0:.2f}$'.format(PARAM['qin_'])

        mark_in = plot_params['q_b_m']
        mark_in_ec = 'k'
        mark_in_fc = plot_params['q_lc']
        mark_in_ew = plot_params['q_b_mew']
        mark_in_s = plot_params['q_b_ms']

    if np.isnan(PARAM['qout_']):
        # --> Fixed pressure
        x_out = 1 + PARAM['hb_']
        pL = PARAM['pL_']

        str_out = r'$p_{{out}}={0:.2f}$'.format(PARAM['pL_'])

        mark_out = plot_params['pp_b_m']
        mark_out_ec = plot_params['pp_lc']
        mark_out_fc = None
        mark_out_ew = plot_params['pp_b_mew']
        mark_out_s = plot_params['pp_b_ms']

    else:
        # --> Fixed flux
        x_in = 1
        pL = 0  # to be changed

        str_out = r'$q_{{out}}={0:.2f}$'.format(PARAM['qout_'])

        mark_out = plot_params['pp_b_m']
        mark_out_ec = plot_params['pp_lc']
        mark_out_fc = None
        mark_out_ew = plot_params['pp_b_mew']
        mark_out_s = plot_params['pp_b_ms']


    # Plot the boundary conditions markers
    # ------------------------------------
    l_mark_in = ax.plot(x_in, p0, ls=None, marker=mark_in,
                        ms=mark_in_s, mew=mark_in_ew,
                        mfc=mark_in_fc, mec=mark_in_ec,
                        zorder=plot_params['b_zorder'])
    l_mark_out = ax.plot(x_out, pL, ls=None, marker=mark_out,
                         ms=mark_out_s, mew=mark_out_ew,
                         mfc=mark_out_fc, mec=mark_out_ec,
                         zorder=plot_params['b_zorder'])

    # Plot the boundary conditions text
    # ---------------------------------
    txt_in = ax.text(0.05, p0*1.02, str_in,
                     fontsize=10,
                     va='center', ha='left',
                     bbox={'boxstyle' : "square, pad=0.1",
                           'alpha' : 0.5, 'facecolor':'w', 'edgecolor':'w'},
                     zorder=9)
    txt_out = ax.text(0.95, 0, str_out,
                      fontsize=10,
                      va='center', ha='right',
                      bbox={'boxstyle' : "square, pad=0.1",
                            'alpha' : 0.5, 'facecolor':'w', 'edgecolor':'w'},
                      zorder=9)

    g_objs = [l_mark_in, l_mark_out, txt_in, txt_out]
    return ax, g_objs


# ----------------------------------------------------------------------------

def init(X, p0, states0, VALVES, PARAM, plot_params={}, save_name=None):
    """
    Plot the initial and boundary conditions: pore pressure profile, boundary
    conditions values, valve states at t=0 and gauge of input boundary value.

    Parameters
    ----------
    X : 1D array
        Space array, dimension PARAM['Nx'] + 1.
    p0 : 1D array
        Initial profile of pore pressure. Same dimension as X.
    states0 : 1D array, boolean
        Initial state of valves: True is open, False is closed. Dimension is
        the number of valves.
    VALVES : dictionnary
        Valve parameters dictionnary.
    PARAM : dictionnary
        Dictionnary of physical parameters set for the system.
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.
    plot_params : dictionnary (default is {}, empty dic.)
        A dictionnary of plotting parameters.
    save_name : str or None (default)
        Path for the figure to save.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    axes : axes object from matplotlib.
        The axes created in this function.
    """
    # Initialize figure layout
    # ------------------------
    fig = plt.figure(figsize=(7, 3))
    gs = fig.add_gridspec(1, 7)
    fig.subplots_adjust(wspace=0.05)

    ax_pp = fig.add_subplot(gs[:, :6])
    ax_b = fig.add_subplot(gs[:, 6:])

    ax_pp.set_title('Initial and boundary conditions')


    # Ticks and label parameters
    # --------------------------
    ax_b.tick_params(left=False, labelleft=False,
                     right=True, labelright=True,
                     bottom=False, labelbottom=False)

    # Plot inital system state
    # -------------------------
    pp_profile(X, p0, fig=fig, ax=ax_pp)  # pore pressure profile

    # --> tweak plot limits so that boundaries labels fit
    ylim_pp = ax_pp.get_ylim()
    dy = abs(ylim_pp[1] - ylim_pp[0])
    ax_pp.set_ylim((ylim_pp[0] - 0.05*dy, ylim_pp[1] + 0.05*dy))

    valves(X, VALVES, states_override=states0, fig=fig, ax=ax_pp,
           plot_params=plot_params)  # valves

    bounds(p0[0], PARAM, fig=fig, ax=ax_pp)

    # Plot boundary condition gauge
    # -----------------------------
    if np.isnan(PARAM['qin_']):
        in_bound = 'p'
    else:
        in_bound = 'q'

    bound_gauge(in_bound, VALVES, PARAM, states_override=states0, fig=fig, ax=ax_b)

    plt.tight_layout()

    # Saving?
    # -------
    if save_name is not None:
        plt.savefig(save_name)
    else:
        plt.show()

    axes = [ax_pp, ax_b]

    return fig, axes

# ----------------------------------------------------------------------------

def bound_gauge(bound, VALVES, PARAM, states_override=None, fig=None, ax=None, plot_params={}):
    """
    Plots a gauge to compare the critical opening and closing flux (or dP)
    to the boundary conditions of the system.

    Parameters
    ----------
    bound : str
        Information about the type of boundary condition.
    VALVES : dictionnary
        Valve parameters dictionnary.
    PARAM : dictionnary
        Dictionnary of physical parameters set for the system.
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.
    plot_params : dictionnary (default is {}, empty dic.)
        A dictionnary of plotting parameters.

    Returns
    -------
    ax : axis object from matplotlib.
        The axis created in this function.
    g_objs : list
        List of graphical objects created by this function.
    """
    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Get parameters of interest
    # --------------------------
    needed_params = ['v_cl_fc', 'v_op_fc', 'q_lc', 'pp_lc']
    plot_params = set_plot_params(plot_params, needed_params)

    # Compute critical thresholds
    # ---------------------------
    if bound == 'q':
        bound = 'flux'
        bound_value = PARAM['qin_']
        v_op_crit = equi.calc_q_crit(0, VALVES, PARAM, event='opening')
        v_cl_crit = equi.calc_q_crit(0, VALVES, PARAM, event='closing')
        bound_value_c = plot_params['q_lc']

    elif bound == 'p':
        bound = 'pressure'
        bound_value = PARAM['p0_']
        v_op_crit = equi.calc_dP_crit(0, VALVES, PARAM, event='opening')
        v_cl_crit = equi.calc_dP_crit(0, VALVES, PARAM, event='closing')
        bound_value_c = plot_params['pp_lc']

    # Define plot limits, ticks, tick labels
    # --------------------------------------
    ax.set_xlim((0, 1))

    # --> Equivalent domains, or shape them
    y_lim = [min(v_op_crit, v_cl_crit) - abs(v_op_crit - v_cl_crit),
             max(v_op_crit, v_cl_crit) + abs(v_op_crit - v_cl_crit)]
    if bound_value > y_lim[1]:
        y_lim[1] = 1.05 * bound_value
    elif bound_value < y_lim[0]:
        y_lim[0] = 0.95 * bound_value
    ax.set_ylim(y_lim)

    ax.set_yticks([v_cl_crit, v_op_crit])
    ax.set_yticklabels(["{:.2f}".format(v_cl_crit), "{:.2f}".format(v_op_crit)])

    ax.set_ylabel('Crit. input {:}'.format(bound), rotation=270, labelpad=15)
    ax.yaxis.set_label_position("right")

    # Plot thresholds and domains
    # ---------------------------
    y_cl_dom = ax.get_ylim()[0]  # bottom left of closed domain
    y_op_dom = max(v_cl_crit, v_op_crit)  # bottom left of open domain
    h_cl_dom = min(v_cl_crit, v_op_crit) - y_cl_dom  # height of closed domain
    h_op_dom = ax.get_ylim()[1] - y_op_dom  # height of open domain
    y_trans_dom = y_cl_dom + h_cl_dom  # bottom left of transition domain
    h_trans_dom = y_op_dom - y_trans_dom  # height of transition domain

    ax.axhline(v_cl_crit, c='k', ls='-', lw=2, zorder=10)
    ax.axhline(v_op_crit, c='k', ls='-', lw=2, zorder=10)

    p_op_dom = Rectangle((0, y_op_dom), 1, h_op_dom,
                         ec=None, fc=plot_params['v_op_fc'], zorder=0)
    txt_op = ax.text(0.05, y_op_dom + 0.5 * h_op_dom, 'Op',
                     ha='left', va='center', rotation=270)
    p_cl_dom = Rectangle((0, y_cl_dom), 1, h_cl_dom,
                         ec=None, fc=plot_params['v_cl_fc'], zorder=0)
    txt_cl = ax.text(0.05, y_cl_dom + 0.5 * h_cl_dom, 'Cl',
                     ha='left', va='center', rotation=270)
    ax.add_patch(p_op_dom)
    ax.add_patch(p_cl_dom)

    if v_op_crit < v_cl_crit:
        p_trans_dom = Rectangle((0, y_trans_dom), 1, h_trans_dom,
                                ec=plot_params['v_op_fc'], fc=plot_params['v_cl_fc'],
                                hatch='..', lw=1)
        txt_trans = ax.text(0.05, y_trans_dom + 0.5 * h_trans_dom, 'Act',
                            ha='left', va='center', rotation=270)
    else:
        p_trans_dom = Rectangle((0, y_trans_dom), 1, h_trans_dom,
                                ec=plot_params['v_op_fc'], fc=plot_params['v_cl_fc'],
                                hatch='///', lw=1)
        txt_trans = ax.text(0.05, y_trans_dom + 0.5 * h_trans_dom, 'CI',
                            ha='left', va='center', rotation=270)
    ax.add_patch(p_trans_dom)

    # Plot boundary conditions
    # ------------------------
    l_bound_value = ax.plot(0.7, bound_value, ls=None,
                            marker='+',  ms=10, mew=2,
                            mec=bound_value_c, zorder=11)

    g_objs = [p_op_dom, p_cl_dom, p_trans_dom,
              l_bound_value,
              txt_op, txt_cl, txt_trans]

    return ax, g_objs

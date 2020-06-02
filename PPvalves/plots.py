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


# Some parameters
# ===============

conv_P = 1.e-9 # for conversion to GPa
conv_Z = 1.e-3 # for conversion to km
conv_X = 1.e-3 # for conversion to km


# FUNCTIONS
# =========

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
    if ('v_op_fc' in plot_params.keys()) and ('v_cl_fc' in plot_params.keys()):
        v_op_fc = plot_params['v_op_fc']
        v_cl_fc = plot_params['v_cl_fc']
    elif 'v_fc_sch' in plot_params.keys():
            v_op_fc = 'palegreen'
            v_cl_fc = 'tomato'
    else:
        v_op_fc = 'whitesmoke'
        v_cl_fc = 'silver'

    if 'zorder' in plot_params.keys():
        zorder = plot_params['zorder']
    else:
        zorder = 1

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
            color = v_op_fc
        else:
            color = v_cl_fc
        return color

    valve_facecolors = [set_valve_fc(op) for op in open_valves]

    # Wrap patches in collection and plot it
    # --------------------------------------
    valves_pc = PatchCollection(valve_patches,
                                facecolor=valve_facecolors,
                                edgecolor=None,
                                axes=ax, zorder=zorder)
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
    if 'pp_lc' in plot_params.keys():
        lc = plot_params['pp_lc']
    else:
        lc = 'crimson'

    if 'pp_zorder' in plot_params.keys():
        zorder = plot_params['pp_zorder']
    else:
        zorder = 10
    if 'pp_lw' in plot_params.keys():
        lw = plot_params['pp_lw']
    else:
        lw = 1.5

    # Plot pore pressure profile
    # --------------------------
    pp_line = ax.plot(X, P,
                      lw=lw, c=lc,
                      zorder=zorder,
                      label='Pore pressure')
    ax.set_ylabel('Pore pressure', color=lc)
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
    if 'q_lc' in plot_params.keys():
        lc = plot_params['q_lc']
    else:
        lc = 'teal'

    if 'q_zorder' in plot_params.keys():
        zorder = plot_params['q_zorder']
    else:
        zorder = 10
    if 'q_lw' in plot_params.keys():
        lw = plot_params['q_lw']
    else:
        lw = 1.

    # Plot massic flux profile
    # ------------------------
    Xq = (X[1:] + X[:-1])/2  # Q is defined in between pressure points
    q_line = ax.plot(Xq, Q,
                     lw=lw, c=lc,
                     zorder=zorder,
                     label='Massic flux')
    ax.set_ylabel('Massic flux', color=lc)
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
        ax_q=ax

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

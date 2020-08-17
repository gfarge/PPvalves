""" Plot full figures """

# Imports
# =======
# Built-in packages
# -----------------
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# My packages
# -----------
from PPvalves.plots.utility import set_plot_params
from PPvalves.plots.elements import valves, q_profile, pp_profile, bounds, bound_gauge

from PPvalves.utility import calc_k, calc_Q
import PPvalves.equilibrium as equi


# Core
# ====

# ------------------------------------------------------------------

def x_profile(X, P, PARAM, VALVES, states_override=None, fig=None, ax=None, plot_params={}, save_name=None):
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
    plot_params : dictionnary (default is {}, empty dic.)
        a dictionnary of plotting parameters for the pore pressure profile,
        flux profile and valves. See pp_profile, q_profile, and valves
        functions of this module for respective paramaters.
    save_name : str or None (default)
        Path for the figure to save.

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
    valves_pc, v_op_pc, v_cl_pc = valves(X, VALVES,
                                         states_override=states_override,
                                         fig=fig, ax=ax_q,
                                         plot_params=plot_params)

    #    ax_p.set_title('State of the system at t={:.2f}'.format(t_plot))

    # Saving?
    # -------
    if save_name is not None:
        print('Saving figure at {:}'.format(save_name))
        plt.savefig(save_name, facecolor=[0, 0, 0, 0])
    else:
        plt.show()

    axes = [ax_p, ax_q]
    g_objs = [pp_line, q_line, valves_pc, v_op_pc, v_cl_pc]

    return axes, g_objs


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
    fig = plt.figure(figsize=(8, 3.5))
    gs = fig.add_gridspec(1, 8)
    fig.subplots_adjust(wspace=0.05)

    ax_pp = fig.add_subplot(gs[:, :7])
    ax_b = fig.add_subplot(gs[:, 7:])

    ax_pp.set_title('Initial state and boundary conditions')


    # Ticks and label parameters
    # --------------------------
    ax_b.tick_params(left=False, labelleft=False,
                     right=True, labelright=True,
                     bottom=False, labelbottom=False)

    # Plot inital system state
    # -------------------------
    pp_profile(X, p0, fig=fig, ax=ax_pp)  # pore pressure profile

    bounds(p0[0], PARAM, fig=fig, ax=ax_pp)

    valves(X, VALVES, states_override=states0, fig=fig, ax=ax_pp,
           plot_params=plot_params)  # valves

    # Plot boundary condition gauge
    # -----------------------------
    if np.isnan(PARAM['qin_']):
        in_bound = 'p'
    else:
        in_bound = 'q'

    bound_gauge(in_bound, VALVES, PARAM, fig=fig, ax=ax_b)

    plt.tight_layout()

    # Saving?
    # -------
    if save_name is not None:
        print('Saving figure at {:}'.format(save_name))
        plt.savefig(save_name, facecolor=[0, 0, 0, 0])
    else:
        plt.show()

    axes = [ax_pp, ax_b]

    return fig, axes

# ------------------------------------------------------------------

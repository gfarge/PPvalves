""" Plot full figures """

# Imports
# =======
# Built-in packages
# -----------------
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import ticker

# My packages
# -----------
from PPvalves.plots.utility import set_plot_params
from PPvalves.plots.elements import valves, q_profile, pp_profile, bounds, \
                                    bound_gauge, recurrence, activity_dip, \
                                    perm_eq, activity_rate, mass_balance, \
                                    bound_in

from PPvalves.utility import calc_k, calc_Q
import PPvalves.equilibrium as equi


# Core
# ====

# ------------------------------------------------------------------

def x_profile(X, P, PARAM, VALVES, states_override=None, plot_params={}, save_name=None):
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
    fig, ax_q = plt.subplots(figsize=(8, 3.5))

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

    plt.tight_layout()

    # Saving?
    # -------
    if save_name is not None:
        print('Saving figure at {:}'.format(save_name))
        plt.savefig(save_name, facecolor=[0, 0, 0, 0])
    else:
        plt.show()

    axes = [ax_p, ax_q]
    g_objs = [pp_line, q_line, valves_pc, v_op_pc, v_cl_pc]

    return fig, axes, g_objs


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

def recurrence_fig(event_t, rec, log=True, tlim=None, save_name=None):
    """
    Plots events recurrence interval in time.

    Parameters
    ----------
    event_t : 1D array
        Array of event times, dimension is N_ev, the recurrence intervals are
        counted at each events, for the next interval: last event is excluded.
    rec : 1D array
        Array of events recurrence interval, time before the next event.
        Dimension is N_ev - 1
    log : bool (default=`True`)
        Option to have the y-axis (recurrence intervals) in log scale. Set to
        logarithmic (`True`) by default, to linear otherwise.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    save_name : str or None (default)
        Path for the figure to save.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    axes : axes object from matplotlib.
        The axes created in this function.
    """
    # Initialize the function
    # -----------------------
    fig, ax = plt.subplots(figsize=(8, 3.5))

    fig, ax, g_objs = recurrence(rec, event_t, log=log, tlim=tlim,
                                 fig=fig, ax=ax)
    plt.tight_layout()

    # Saving?
    # -------
    if save_name is not None:
        print('Saving figure at {:}'.format(save_name))
        plt.savefig(save_name, facecolor=[0, 0, 0, 0])
    else:
        plt.show()

    return fig, ax, g_objs

# ------------------------------------------------------------------

def activity_dip_fig(event_t, event_x, tlim=None, save_name=None):
    """
    Plots activity across dip in time.

    Parameters
    ----------
    event_t : 1D array
        Array of event times, dimension is N_event.
    event_x : 1D array
        Array of event locations, dimension is N_event.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    save_name : str or None (default)
        Path for the figure to save.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    axes : axes object from matplotlib.
        The axes created in this function.
    """
    # Initialize the function
    # -----------------------
    fig, ax = plt.subplots(figsize=(8, 3.5))

    fig, ax, g_objs = activity_dip(event_t, event_x, tlim=tlim, fig=fig, ax=ax)
    plt.tight_layout()

    # Saving?
    # -------
    if save_name is not None:
        print('Saving figure at {:}'.format(save_name))
        plt.savefig(save_name, facecolor=[0, 0, 0, 0])
    else:
        plt.show()

    return fig, ax, g_objs


# ------------------------------------------------------------------

def activity_rate_fig(rate_time, rate, tlim=None, save_name=None):
    """
    Plots activity rate in time.

    Parameters
    ----------
    rate_time : 1D array
        Array of times for which the activity rate is computed, dimension is
        N_times.
    rate : 1D array
        Array of activity rate in time, dimension is N_times.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    save_name : str or None (default)
        Path for the figure to save.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    axes : axes object from matplotlib.
        The axes created in this function.
    """
    # Initialize the function
    # -----------------------
    fig, ax = plt.subplots(figsize=(8, 3.5))

    fig, ax, g_objs = activity_rate(rate_time, rate, tlim=tlim, fig=fig, ax=ax)
    plt.tight_layout()

    # Saving?
    # -------
    if save_name is not None:
        print('Saving figure at {:}'.format(save_name))
        plt.savefig(save_name, facecolor=[0, 0, 0, 0])
    else:
        plt.show()

    return fig, ax, g_objs

# ----------------------------------------------------------------------------

def perm_eq_fig(T, k_eq, tlim=None, log=True, save_name=None):
    """
    Plots equivalent permeability in time.

    Parameters
    ----------
    T : 1D array
        Array of times, dimension is N_times.
    k_eq : 1D array
        Array of equivalent permeability in time, dimension is N_times.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    log : bool (default=`True`)
        Option to have the y-axis (permeability) in log-scale. Set to
        logarithmic (`True`) by default, to linear otherwise.
    save_name : str or None (default)
        Path for the figure to save.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    ax : ax object from matplotlib.
        The axis created in this function.
    g_objs : line object from matplotlib
        The line object created in this function.
    """
    # Initialize the function
    # -----------------------
    fig, ax = plt.subplots(figsize=(8, 3.5))

    fig, ax, g_objs = perm_eq(T, k_eq, tlim=tlim, log=log, fig=fig, ax=ax)
    plt.tight_layout()

    # Saving?
    # -------
    if save_name is not None:
        print('Saving figure at {:}'.format(save_name))
        plt.savefig(save_name, facecolor=[0, 0, 0, 0])
    else:
        plt.show()

    return fig, ax, g_objs

# ----------------------------------------------------------------------------

def mass_balance_fig(T, deltaM, tlim=None, save_name=None):
    """
    Plots mass balance in system in time.

    Parameters
    ----------
    T : 1D array
        Array of times, dimension is N_times.
    deltaM : 1D array
        Array of mass balance in time, dimension is N_times.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    save_name : str or None (default)
        Path for the figure to save.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    ax : ax object from matplotlib.
        The axis created in this function.
    g_objs : line object from matplotlib
        The line object created in this function.
    """
    # Initialize the function
    # -----------------------
    fig, ax = plt.subplots(figsize=(8, 3.5))

    fig, ax, g_objs = mass_balance(T, deltaM, tlim=tlim, fig=fig, ax=ax)
    plt.tight_layout()

    # Saving?
    # -------
    if save_name is not None:
        print('Saving figure at {:}'.format(save_name))
        plt.savefig(save_name, facecolor=[0, 0, 0, 0])
    else:
        plt.show()

    return fig, ax, g_objs

# ----------------------------------------------------------------------------

def bound_in_fig(T, bound_0, PARAM, tlim=None, save_name=None):
    """
    Plots values of pp/q the in bound in time.

    Parameters
    ----------
    T : 1D array
        Array of times, dimension is N_times.
    bound_0 : 1D array
        Array of values taken by the in bound in time, dimension is N_times.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    save_name : str or None (default)
        Path for the figure to save.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    ax : ax object from matplotlib.
        The axis created in this function.
    g_objs : line object from matplotlib
        The line object created in this function.
    """
    # Initialize the function
    # -----------------------
    fig, ax = plt.subplots(figsize=(8, 3.5))

    fig, ax, g_objs = bound_in(T, bound_0, PARAM, txt=True, tlim=tlim, fig=fig,
                               ax=ax)
    plt.tight_layout()

    # Saving?
    # -------
    if save_name is not None:
        print('Saving figure at {:}'.format(save_name))
        plt.savefig(save_name, facecolor=[0, 0, 0, 0])
    else:
        plt.show()

    return fig, ax, g_objs

# ----------------------------------------------------------------------------

def tseries_comp(T, rate_time, bound_0, deltaM, k_eq, rate, PARAM, tlim=None,\
                 show_T=None, save_name=None):
    """
    Parameters
    ----------
    T : 1D array
        Array of times, dimension is N_times.
    rate_time : 1D array
        Array of times for which the activity rate is computed, dimension is
        N_rate_times.
    bound_0 : 1D array
        Array of values taken by the in bound in time, dimension is N_times.
    deltaM : 1D array
        Array of mass balance in time, dimension is N_times.
    k_eq : 1D array
        Array of equivalent permeability in time, dimension is N_times.
    rate : 1D array
        Array of activity rate in time, dimension is N_rate_times.
    PARAM : dictionnary
        Dictionnary of physical parameters set for the system.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    show_T : list/tuple/array (default `None`)
        Option to highlight some region. If specified as an array-like object
        (2 elements long) the range in between each element will be
        highlighted.
    save_name : str or None (default)
        Path for the figure to save.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    axes : list ax object from matplotlib.
        The axes created in this function.
    """
    def show_period(show_T, ax, with_txt=False):
        """
        Function to show period span
        """
        T = show_T[1] - show_T[0]
        T1, T2 = show_T

        ax.axvspan(T1, T2, fc='0.9', ec='0.9')

        if with_txt:
            ax.text(T1 + 0.1*T, ax.get_ylim()[0],
                    r"$T_{{cycle}} = {:.2e}$".format(T),
                    ha = 'left', va='bottom',
                    bbox={'boxstyle' : 'square, pad=0.1', 'alpha' : 0.5,
                          'facecolor' : 'w', 'edgecolor' : 'w'}, zorder=11)

    # Set up figure
    # -------------
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 7))

    # Adjust plot setup
    # -----------------
    for ax, v in zip(axes, [bound_0, deltaM, k_eq, rate]):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.tick_params(bottom=True, labelbottom=False, direction='in',
                       which='both')

        ax.grid(lw=.5, axis='x')

#        dy = max(v) - min(v)
#        ylim = (0.95*min(v), max(v) + 0.01*dy)
#        ax.set_ylim(ylim)

    # Plot each subplot
    # -----------------
    # a/ In-bound
    _ = bound_in(T, bound_0, PARAM, tlim=tlim, fig=fig, ax=axes[0])
    axes[0].set_xlabel("")
    axes[0].set_title("")

    if show_T is not None:
        show_period(show_T, axes[0], with_txt=True)

    # b/ Mass balance
    _ = mass_balance(T, deltaM, tlim=tlim, fig=fig, ax=axes[1])
    axes[1].set_xlabel("")
    axes[1].set_title("")

    if show_T is not None:
        show_period(show_T, axes[1])

    # c/ Equivalent permeability
    _ = perm_eq(T, k_eq, tlim=tlim, fig=fig, ax=axes[2])
    axes[2].set_xlabel("")
    axes[2].set_title("")

    if show_T is not None:
        show_period(show_T, axes[2])

    # d/ Activity rate
    _ = activity_rate(rate_time, rate, tlim=tlim, fig=fig, ax=axes[3])
    axes[3].set_title("")
    axes[3].tick_params(bottom=True, labelbottom=True)
    axes[3].spines['bottom'].set_visible(True)

    if show_T is not None:
        show_period(show_T, axes[3])

    fig.subplots_adjust(hspace=0.02, bottom=0.1)

    # Saving?
    # -------
    if save_name is not None:
        print('Saving figure at {:}'.format(save_name))
        plt.savefig(save_name, facecolor=[0, 0, 0, 0])
    else:
        plt.show()

    return fig, axes

# ----------------------------------------------------------------------------

def phase_diagram(T, variables, labels, scales=('linear','linear'), tlim=None, \
        cycle=None, plot_params={}, save_name=None):
    """
    Plots the phase diagram of two variables in time.

    Parameters
    ----------
    T : 1D array
        Time array, both variables should be defined on the same time steps,
        dimension N_times.
    variables : list, tuple, array-like
        Array of the two variables to plot against each other, as 1D arrays of
        the same dimension, N_times.
    labels : list, tuple
        The variable labels, as strings of characters..
    scales : list, tuple (default = `('linear', 'linear')`)
        The scales used to plot the variables. Default is set to both linear
        scale. `linear` for linear scale, `log` for logarithmic scale.
    tlim : tuple (default = `None`)
        Option to plot in between specific time limits, specified as a tuple.
    cycle : float (default = `None`)
        Use a cyclic colormap for time to highlight cycles, cycle is the period
        of the cycle.
    plot_params : dic (default=`{}`)
        A dictionnary of plotting parameters.
    save_name : str
        Path for the figure to save.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    ax : ax object from matplotlib.
        The axis created in this function.
    """
    # Unpack
    # ------
    v_x, v_y = variables
    l_x, l_y = labels
    sc_x, sc_y = scales

    period = (T <= max(tlim)) & (T >= min(tlim))
    T = T[period]
    v_x = v_x[period]
    v_y = v_y[period]

    # Get defaults
    # ------------
    if cycle is None:
        needed_params = ['time_cmap']
        plot_params = set_plot_params(plot_params, needed_params)
        cmap = plot_params['time_cmap']
    else:
        needed_params = ['cyclic_cmap']
        plot_params = set_plot_params(plot_params, needed_params)
        cmap = plot_params['cyclic_cmap']
        T = (T%cycle) * 2*np.pi / cycle  # transform T in phase

    # Set up figure and axis
    # ----------------------
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Plot the scatter
    # ----------------
    coll = ax.scatter(v_x, v_y, c=T, cmap=cmap, s=5)

    ax.set_xscale(sc_x)
    ax.set_yscale(sc_y)

    ax.set_xlabel(l_x)
    ax.set_ylabel(l_y)

    # Set up colorbar
    # ---------------
    if cycle is None:
        cb = plt.colorbar(coll, label='Time (scaled)')
    else:
        cb = plt.colorbar(coll, label='Cycle phase',
                          ticks=[0, np.pi/2, np.pi, 3/2*np.pi, 2*np.pi],
                          format=ticker.FixedFormatter([r'$0$', r'$\pi/2$',
                                                        r'$\pi$', r'$3\pi/2$',
                                                        r'$2\pi$']))
    plt.tight_layout()

    # Saving?
    # -------
    if save_name is not None:
        print('Saving figure at {:}'.format(save_name))
        plt.savefig(save_name, facecolor=[0, 0, 0, 0])
    else:
        plt.show()

    return fig, ax

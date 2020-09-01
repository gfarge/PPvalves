""" Plots figure elements, to be assembled """

# Imports
# =======
# Built-in packages
# -----------------
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

# My packages
# -----------
from PPvalves.plots.utility import set_plot_params, set_valve_fc, set_valve_mc
import PPvalves.equilibrium as equi


# Core
# ====

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

    if states_override is None:
        open_valves = VALVES['open'].astype(bool)
    else:
        open_valves = states_override.astype(bool)

    # Check which parameters are default and which are defined by user
    # ----------------------------------------------------------------
    needed_params = ['v_op_fc', 'v_cl_fc', 'v_st_marker', 'v_zorder']
    plot_params = set_plot_params(plot_params, needed_params)

    # Add state markers before plotting patches underneath
    # ----------------------------------------------------
    if plot_params['v_st_marker']:
        v_op_pc, v_cl_pc = valve_markers(X, VALVES,
                                         states_override=states_override,
                                         fig=fig, ax=ax, X_axis=X_axis,
                                         plot_params=plot_params)

    # Check which axis is physical X_axis
    # -----------------------------------
    if X_axis == 'x':
        zlim = ax.get_ylim()
#        print(zlim)
    elif X_axis == 'y':
        zlim = ax.get_xlim()
#        print(zlim)

    # Build valve patches and colors as a function of valve states
    # ------------------------------------------------------------
    if X_axis == 'x':
        valve_patches = [Rectangle((x_v, zlim[0]), w_v, zlim[1]-zlim[0])
                         for x_v, w_v in zip(X[VALVES['idx']], VALVES['width'])]
    elif X_axis == 'y':
        valve_patches = [Rectangle((zlim[0], x_v), zlim[1]-zlim[0], w_v)
                         for x_v, w_v in zip(X[VALVES['idx']], VALVES['width'])]

    valve_facecolors = set_valve_fc(open_valves, plot_params)

    # Wrap patches in collection and plot it
    # --------------------------------------
    valves_pc = PatchCollection(valve_patches,
                                facecolor=valve_facecolors,
                                edgecolor=None,
                                axes=ax, zorder=plot_params['v_zorder'])
    ax.add_collection(valves_pc)

    if plot_params['v_st_marker']:
        return valves_pc, v_op_pc, v_cl_pc

    return valves_pc

# ----------------------------------------------------------------------------

def valve_markers(X, VALVES, states_override=None, fig=None, ax=None, X_axis='x', plot_params={}):
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
    valves_op_m : matplotlib PathCollection object
        Valves open markers.
    valves_cl_m : matplotlib PathCollection object
        Valves closed markers.
    """
    # As a function of input, point to correct objects for valve states,
    # figure and axis
    # ------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    if states_override is None:
        open_valves = VALVES['open'].astype(bool)
    else:
        open_valves = states_override.astype(bool)

    # Tweak boundaries to make space for markers
    # ------------------------------------------
    dy_bottom = -0.21
    if X_axis == 'x':
        ax.set_ylim(dy_bottom, ax.get_ylim()[1])
    elif X_axis == 'y':
        ax.set_xlim(dy_bottom, ax.get_xlim()[1])

    # Set up default parameters when no input
    # ---------------------------------------
    needed_params = ['v_op_mc', 'v_cl_mc', 'v_zorder']
    plot_params = set_plot_params(plot_params, needed_params)

    # Set valve marker colors, and positions
    # --------------------------------------
    m_op_c, m_cl_c = set_valve_mc(open_valves, plot_params)
    X_marker = X[VALVES['idx']] + .5 * VALVES['width']  # center of valve
    Y_cl_marker = np.ones(len(VALVES['idx'])) * 1/3*dy_bottom  # bottom
    Y_op_marker = np.ones(len(VALVES['idx'])) * 2/3*dy_bottom  # bottom +

    # Build valve patches and colors as a function of valve states
    # ------------------------------------------------------------
    if X_axis == 'x':
        valves_cl_pc = ax.scatter(X_marker, Y_cl_marker, marker='x', ec=m_cl_c,
                                  zorder=plot_params['v_zorder']+1)
        valves_op_pc = ax.scatter(X_marker, Y_op_marker,
                                  marker=[(-1, -1), (1, 0), (-1, 1)], lw=1.5,
                                  ec=m_op_c, zorder=plot_params['v_zorder']+1,
                                  fc=[0, 0, 0 ,0])
    elif X_axis == 'y':
        valves_cl_pc = ax.scatter(Y_cl_marker, X_marker, marker='x', ec=m_cl_c,
                                  zorder=plot_params['v_zorder']+1)
        valves_op_pc = ax.scatter(Y_op_marker, X_marker,
                                  marker=[(-1, -1), (1, 0), (-1, 1)], lw=1.5,
                                  ec=m_op_c, zorder=plot_params['v_zorder']+1,
                                  fc=[0, 0, 0 ,0])

    return valves_op_pc, valves_cl_pc

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
    txt_in = ax.text(0.05, p0*1.0, str_in,
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

    # Arrange the limits to fit the text box
    # --------------------------------------
    ax.set_ylim(min(-0.05, ax.get_ylim()[0]), max(1.05, ax.get_ylim()[1]))

    g_objs = [l_mark_in, l_mark_out, txt_in, txt_out]
    return ax, g_objs

# ----------------------------------------------------------------------------

def bound_gauge(bound, VALVES, PARAM, fig=None, ax=None, plot_params={}):
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

    # Plot initial state and boundary conditions
    # ------------------------------------------
    # --> Boundary condition
    l_bound_value = ax.plot(0.5, bound_value, ls=None,
                            marker='o', ms=12, mew=2, mec=bound_value_c,
                            mfc=[0, 0, 0, 0], zorder=11)
    l_bound_value = ax.plot(0.5, bound_value, ls=None,
                            marker='+', ms=12, mew=2, mec=bound_value_c,
                            zorder=11)
    ax.text(0.7, bound_value, r'$q_{{in}}$',
            va='bottom', ha='left',
            bbox={'boxstyle' : "square, pad=0.1", 'alpha' : 0.5,
                  'facecolor':'w', 'edgecolor':'w'}, zorder=12)
    # --> Initial state
    l_bound_value = ax.plot(0.5, v_op_crit, ls=None,
                            marker='o', ms=12, mew=2, mec='crimson',
                            mfc=[0, 0, 0, 0], zorder=11)
    l_bound_value = ax.plot(0.5, v_op_crit, ls=None,
                            marker='+', ms=12, mew=2, mec='crimson',
                            zorder=11)
    ax.text(0.7, v_op_crit, r'$q_{{0}}$',
            va='bottom', ha='left',
            bbox={'boxstyle' : "square, pad=0.1", 'alpha' : 0.5,
                  'facecolor':'w', 'edgecolor':'w'}, zorder=12)
    # --> t0 arrow
    ax.annotate('', (0.5, bound_value), (0.5, v_op_crit),
                arrowprops={'arrowstyle' : '->', 'linewidth' : 2},
                zorder=12)
    ax.text(0.75, (bound_value + v_op_crit)/2, r'$t_0$',
            va='center', ha='center',
            bbox={'boxstyle' : "square, pad=0.1", 'alpha' : 0.5,
                  'facecolor':'w', 'edgecolor':'w'})

    g_objs = [p_op_dom, p_cl_dom, p_trans_dom,
              l_bound_value,
              txt_op, txt_cl, txt_trans]

    return ax, g_objs

# ------------------------------------------------------------------

def recurrence(event_t, rec, log=True, tlim=None, plot_params={}, fig=None, ax=None):
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
    plot_params : dictionnary (default is {}, empty dic.)
        A dictionnary of plotting parameters. Implemented:
        linecolor (any matplotlib ways of indicating color) default = 'act_lc'
        : 'k')
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    ax : ax object from matplotlib.
        The axis created in this function.
    g_objs : line object from matplotlib
        The line object created in this function.
    """
    # Define plot time window
    # -----------------------
    if tlim is not None:
        t_win = (tlim[0] < event_t) & (event_t < tlim[1])
        event_t = event_t[t_win]
        rec = rec[t_win]

    if len(event_t) > 1e4:
        rasterize=True

    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Check which parameters are default and which are defined by user
    # ----------------------------------------------------------------
    needed_params = ['act_lc']
    plot_params = set_plot_params(plot_params, needed_params)

    # Plot the events
    # ---------------
    ev_l, = ax.plot(event_t[:-1], rec, 'o', c=plot_params['act_lc'], ms=1,
                    rasterized=rasterize)
    if log:
        ax.set_yscale('log')

    if tlim is not None:
        ax.set_xlim(tlim)

    # Set labels and title
    # --------------------
    ax.set_title("Events recurrence interval (time before next event)")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel("Rec. interval (scaled)", c=plot_params['act_lc'])

    plt.tight_layout()

    g_objs = ev_l

    return fig, ax, g_objs

# ----------------------------------------------------------------------------

def activity_dip(event_t, event_x, tlim=None, plot_params={}, fig=None, ax=None):
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
    plot_params : dictionnary (default is {}, empty dic.)
        A dictionnary of plotting parameters. Implemented:
        linecolor (any matplotlib ways of indicating color) default = 'act_lc'
        : 'k')
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    ax : ax object from matplotlib.
        The axis created in this function.
    g_objs : line object from matplotlib
        The line object created in this function.
    """
    # Define plot time window
    # -----------------------
    if tlim is not None:
        t_win = (tlim[0] < event_t) & (event_t < tlim[1])
        event_t = event_t[t_win]
        event_x = event_x[t_win]

    if len(event_t) > 1e4:
        rasterize=True

    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Check which parameters are default and which are defined by user
    # ----------------------------------------------------------------
    needed_params = ['act_lc']
    plot_params = set_plot_params(plot_params, needed_params)

    # Plot the events
    # ---------------
    ev_l, = ax.plot(event_t, event_x, 'o', c=plot_params['act_lc'], ms=1,
                    rasterized=rasterize)

    if tlim is not None:
        ax.set_xlim(tlim)

    # Set labels and title
    # --------------------
    ax.set_title("Activity across dip in time")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel("Along-dip X (scaled)", c=plot_params['act_lc'])

    plt.tight_layout()

    g_objs = ev_l

    return fig, ax, g_objs

# ----------------------------------------------------------------------------

def activity_rate(rate_time, rate, tlim=None, plot_params={}, fig=None, ax=None):
    """
    Plots equivalent permeability in time.

    Parameters
    ----------
    rate_time : 1D array
        Array of times for which the activity rate is computed, dimension is
        N_times.
    rate : 1D array
        Array of activity rate in time, dimension is N_times.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    plot_params : dictionnary (default is {}, empty dic.)
        A dictionnary of plotting parameters. Implemented:
        linecolor (any matplotlib ways of indicating color) default = 'k_eq_lc'
        : 'darkturquoise')
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    ax : ax object from matplotlib.
        The axis created in this function.
    g_objs : line object from matplotlib
        The line object created in this function.
    """
    # Define plot time window
    # -----------------------
    if tlim is not None:
        t_win = (tlim[0] < T) & (T < tlim[1])
        rate_time = rate_time[t_win]
        rate = rate [t_win]

    if len(rate_time) > 1e4:
        rasterize=True

    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Check which parameters are default and which are defined by user
    # ----------------------------------------------------------------
    needed_params = ['act_lc']
    plot_params = set_plot_params(plot_params, needed_params)

    # Plot
    # ----
    act_r_l, = ax.plot(rate_time, rate, ls='-', lw=1.5, c=plot_params['act_lc'],
                       rasterized=rasterize)

    if tlim is not None:
        ax.set_xlim(tlim)

    # Labels and title
    # ----------------
    ax.set_title("Activity rate in time")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel(r"Activity rate", c=plot_params['act_lc'])
    plt.tight_layout()

    g_objs = act_r_l

    return fig, ax, g_objs

# ----------------------------------------------------------------------------

def perm_eq(T, k_eq, tlim=None, log=True, plot_params={}, fig=None, ax=None):
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
    plot_params : dictionnary (default is {}, empty dic.)
        A dictionnary of plotting parameters. Implemented:
        linecolor (any matplotlib ways of indicating color) default = 'k_eq_lc'
        : 'darkturquoise')
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    ax : ax object from matplotlib.
        The axis created in this function.
    g_objs : line object from matplotlib
        The line object created in this function.
    """
    # Define plot time window
    # -----------------------
    if tlim is not None:
        t_win = (tlim[0] < T) & (T < tlim[1])
        T = T[t_win]
        k_eq = k_eq[t_win]

    if len(T) > 1e4:
        rasterize=True

    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Check which parameters are default and which are defined by user
    # ----------------------------------------------------------------
    needed_params = ['k_eq_lc']
    plot_params = set_plot_params(plot_params, needed_params)

    # Plot
    # ----
    k_eq_l, = ax.plot(T, k_eq, ls='-', lw=1.5, c=plot_params['k_eq_lc'],
                      rasterized=rasterize)

    if log:
        ax.set_yscale('log')

    if tlim is not None:
        ax.set_xlim(tlim)

    # Labels and title
    # ----------------
    ax.set_title("System's equivalent permeability in time")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel(r"$k_{eq}$ ($m^2$)", c=plot_params['k_eq_lc'])
    plt.tight_layout()

    g_objs = k_eq_l

    return fig, ax, g_objs

# ----------------------------------------------------------------------------

def mass_balance(T, deltaM, tlim=None, plot_params={}, fig=None, ax=None):
    """
    Plots mass balance in time.

    Parameters
    ----------
    T : 1D array
        Array of times, dimension is N_times.
    deltaM : 1D array
        Array of mass balance in time, dimension is N_times.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    plot_params : dictionnary (default is {}, empty dic.)
        A dictionnary of plotting parameters. Implemented:
        linecolor (any matplotlib ways of indicating color) default = 'mass_lc'
        : 'darkturquoise')
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    ax : ax object from matplotlib.
        The axis created in this function.
    g_objs : line object from matplotlib
        The line object created in this function.
    """
    # Define plot time window
    # -----------------------
    if tlim is not None:
        t_win = (tlim[0] < T) & (T < tlim[1])
        T = T[t_win]
        bound_0 = bound_0[t_win]

    if len(T) > 1e4:
        rasterize=True

    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Check which parameters are default and which are defined by user
    # ----------------------------------------------------------------
    needed_params = ['mass_lc']
    plot_params = set_plot_params(plot_params, needed_params)

    # Plot
    # ----
    mass_b_l, = ax.plot(T, deltaM, ls='-', lw=1.5, c=plot_params['mass_lc'],
                        rasterized=rasterize)

    if tlim is not None:
        ax.set_xlim(tlim)

    # Labels and title
    # ----------------
    ax.set_title("Mass balance in the system")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel(r"$\delta M$ (scaled)", c=plot_params['mass_lc'])
    plt.tight_layout()

    g_objs = mass_b_l

    return fig, ax, g_objs

# ----------------------------------------------------------------------------

def bound_in(T, bound_0, PARAM, tlim=None, txt=False, plot_params={}, fig=None, ax=None):
    """
    Plots pp/q value of in-bound in time.

    Parameters
    ----------
    T : 1D array
        Array of times, dimension is N_times.
    bound_0 : 1D array
        Array of values taken by the in bound in time, dimension is N_times.
    PARAM : dictionnary
        Physical and numerical parameters dictionnary to determine which
        variable is plotted.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    txt : bool (default `False`)
        Option to represent and print the effective value of the input boundary
        variable.
    plot_params : dictionnary (default is {}, empty dic.)
        A dictionnary of plotting parameters. Implemented:
        linecolor (any matplotlib ways of indicating color) default = 'pp_lc'
        : 'crimson', 'q_lc' : 'teal')
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    ax : ax object from matplotlib.
        The axis created in this function.
    g_objs : line object from matplotlib
        The line object created in this function.
    """
    # Define plot time window
    # -----------------------
    if tlim is not None:
        t_win = (tlim[0] < T) & (T < tlim[1])
        T = T[t_win]
        bound_0 = bound_0[t_win]

    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Check which parameters are default and which are defined by user
    # ----------------------------------------------------------------
    if PARAM['bound'][0]=='Q':
        needed_params = ['pp_lc']
    else:
        needed_params = ['q_lc']
    plot_params = set_plot_params(plot_params, needed_params)

    if PARAM['bound'][0] == 'Q':
        lc = plot_params['pp_lc']
        ylabel = r'$p_{in}$ (scaled)'
        title = 'Input pressure in time'
    else:
        lc = plot_params['q_lc']
        ylabel = r'$q_{in}$ (scaled)'
        title = 'Input flux in time'

    # Plot
    # ----
    bound_0_l, = ax.plot(T, bound_0, ls='-', lw=1.5, c=lc, rasterized=rasterize)

    if tlim is not None:
        ax.set_xlim(tlim)

    # Add value of effective bound?
    # -----------------------------
    if txt:
        v_eff = np.mean(bound_0[len(2*bound_0)//3:])
        ax.axhline(v_eff, lw=.8, c='k', zorder=0)

        if PARAM['bound'][0] == 'Q':
            label = r"$\overline{{p_{{in}}}}={:.2f}$".format(v_eff)
        else:
            label = r"$\overline{{q_{{in}}}}={:.2f}$".format(v_eff)

        ax.text(ax.get_xlim()[0]+0.01, v_eff, label, ha='left', va='bottom',
                bbox={'boxstyle' : 'square, pad=0.1', 'alpha' : 0.5,
                      'facecolor' : 'w', 'edgecolor' : 'w'}, zorder=11)

    # Labels and title
    # ----------------
    ax.set_title(title)
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel(ylabel, c=lc)
    plt.tight_layout()

    g_objs = bound_0_l

    return fig, ax, g_objs

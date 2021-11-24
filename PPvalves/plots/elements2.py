""" --- Empty description --- """

# Imports
# =======
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as tk
from matplotlib.colors import to_rgba

import PPvalves.equilibrium as equi
import PPvalves.utility as util

# General parameters
# ==================
# >> General plot parameters
c_k = '#00bdaa'
c_p = '#fe346e'
c_q = '#400082'

c_act = 'k'

c_mig = '#f5b461'  # color for migrations
alpha_mig = 0.6

c_op = '#9ad3bc'
c_cl = '#ec524b'
c_int = '#f5b461'

# Functions
# =========

# -----------------------------------------------------------------------------
def valve_system(X, VALVES, fc='k', ec=[0, 0, 0, 0], fs=None, fig=None, ax=None):
    """
    Plot simple valve system (code bar).

    Parameters
    ----------
    X : 1D array
        Space position array.
    VALVES : dictionnary
        Valve parameters dictionnary.
    fc : color (default `fc = 'k'`)
        Valve patch facecolor.
    ec : color (default `ec = [0, 0, 0, 0]`)
        Valve patch edgecolor.
    fs : int (default `fs = None`)
        Controls fontsizes.
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
    g_objs : patch collection
        The patch collection of valves.

    """
    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    if fs is None:
        fs = 9

    # Fix bounds of plot
    # ------------------
    dX = X.max() - X.min()
    ax.set_xlim(X.min() - 0.02*dX, X.max() + 0.02*dX)

    # Build valve patch collection
    # ----------------------------
    xv = X[VALVES['idx']]
    wv = VALVES['width']

    v_pcoll = []
    for x, w in zip(xv, wv):
        p = ax.axvspan(x, x+w, fc=fc, ec=ec)
        v_pcoll.append(p)

    # Labels and axes
    # ---------------
    ax.set_xlabel(r"$x$", fontsize=fs)
    ax.tick_params(left=False, labelleft=False, labelsize=fs-1)
    # --> Better ticks
    ax.xaxis.set_major_locator(tk.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(tk.MultipleLocator(0.1))

    for tlabels in [ax.get_xticklabels(which='both'), ax.get_yticklabels(which='both')]:
        for txt in tlabels:
            txt.set_fontname('Roboto Condensed')

    plt.tight_layout()

    g_objs = v_pcoll

    return fig, ax, g_objs

# -----------------------------------------------------------------------------
def flux_space(q, PARAM, VALVES, fs=None, fig=None, ax=None):
    """
    Plots flux space.

    Parameters
    ----------
    q : float or array
        Fluxes to represent in the flux space. If `q` is a scalar, it is
        interpreted as an input flux, and plotted alongside the initial flux.
        If `q` is an array-like, it is interpreted as all the input fluxes of a
        simulation, and plotted alone.
    PARAM : dictionnary
        Dictionnary of physical parameters set for the system.
    VALVES : dictionnary
        Valve parameters dictionnary.
    fs : int (default `fs = None`)
        Controls fontsizes.
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
    g_objs : list
        List of graphical objects in the axes.

    """
    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    if fs is None:
        fs = 9

    # >> Compute the reference fluxes
    # ===============================
    q_c_op = equi.calc_q_crit(0, VALVES, PARAM, 'opening')
    q_c_cl = equi.calc_q_crit(0, VALVES, PARAM, 'closing')

    # >> Arange the axes
    # ==================
    ax.set_facecolor([0, 0, 0, 0])

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # >> Arange the ticks and labels
    # ==============================
    ax.xaxis.set_minor_locator(tk.MultipleLocator(0.5))
    ax.xaxis.set_minor_formatter(tk.ScalarFormatter())
    for tick in ax.get_xticklabels(which='minor'):
        tick.set_fontsize(fs-1)
    ax.tick_params(which='minor', rotation=90)

    ax.xaxis.set_major_locator(tk.FixedLocator([q_c_op, q_c_cl]))
    ax.xaxis.set_major_formatter(tk.FixedFormatter([r"$q_{{c}}^{{break}} = {:.2f}$".format(q_c_op),
                                 r"$q_{{c}}^{{clog}} = {:.2f}$".format(q_c_cl)]))
    ax.tick_params(left=False, labelleft=False)
    for tick in ax.get_xticklabels(which='major'):
        tick.set_ha('left')
        tick.set_fontsize(fs)

    ax.tick_params(which='minor', rotation=45)

    ax.set_xlabel(r"Input flux $q_{in}$", fontsize=fs)
    ax.set_xlim((0, 1.5*q_c_cl))
    ax.set_ylim((-1, 1))

    ax.axvspan(q_c_op, q_c_cl, fc=to_rgba(c_int, 0.5), ec=to_rgba(c_int, 0))
    ax.axvspan(q_c_cl, ax.get_xlim()[1], fc=to_rgba(c_op, 0.5),
               ec=to_rgba(c_op, 0))
    ax.axvspan(ax.get_xlim()[0], q_c_op, fc=to_rgba(c_cl, 0.5),
               ec=to_rgba(c_cl, 0))
    ax.axvline(q_c_op, c='k')
    ax.axvline(q_c_cl, c='k')

    if not hasattr(q, "__len__"):
        ax.plot(PARAM['criticality'] * q_c_op, 0, ls='', marker='x', ms=10,
                c='.5', label=r'$q_0 = 0.99\,q_c^{break}$')
        ax.plot(q, 0, ls='', marker='+', mfc=to_rgba('k', 0), mec='k', ms=15,
                mew=2, label='$q_{{in}} = {:.2f}$'.format(q))
    else:
        ax.plot(q, np.zeros(len(q)), ls='', marker='+', mfc=to_rgba('k', 0),
                mec='k', ms=15, mew=1, label='Simulation fluxes')

    ax.legend(bbox_to_anchor=(1, 1.2), loc='upper left', framealpha=0,
              fontsize=fs)

    return fig, ax

# -----------------------------------------------------------------------------
def q_t_serie(time, q, q_smooth, q_mean, q_eff, tlim=None, fs=None, fig=None, ax=None):
    """
    Plots a flux time series.

    Parameters
    ----------
    time : 1D array
        Array of times.
    q : 1D array
        Array of fluxes, same dimension as `time`.
    q_smooth : 1D array
        Smoothed fluxes, same dimension as `time`.
    q_mean : float
        Average value for `q` over the permanent regime.
    q_eff : float
        Effective value of the input boundary variable. If specified it is
        plotted as a horizontal line.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    fs : float (default `fs = None`)
        Fontsize.
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.

    Returns
    -------
    fig : matplotlib figure object
        The input or current figure.
    ax : matplotlib axes object
        The input or current axes.

    """
    # Define plot time window
    # -----------------------
    if tlim is not None:
        t_win = (tlim[0] < time) & (time < tlim[1])
        time = time[t_win]
        q = q[t_win]
        q_smooth = q_smooth[t_win]

    rasterize = bool(len(time) > 1e4)

    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Fontsize
    # --------
    if fs is None:
        fs = 9

    # Label the axes
    # --------------
    ax.set_ylabel('Output flux (scaled)', fontsize=fs, c=c_q)
    ax.set_xlabel('Time (scaled)', fontsize=fs)

    # Plotting
    # --------
    ax.plot(time, q, c=to_rgba(c_q, 0.3), lw=1, rasterized=rasterize, zorder=0,
            label='$q_{out}$')
    ax.plot(time, q_smooth, c=c_q, rasterized=rasterize, zorder=2,
            label='Smoothed $q_{out}$')

    ax.axhline(q_eff, c='.5', lw=2, ls='--', label='Effective flux $q_{eff}$',
               zorder=1)
    ax.axhline(q_mean, c=to_rgba(c_q, .5), lw=2, ls='--',
               label='Average flux $q_{\infty}$', zorder=1)

    # Legend
    # ------
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1.05), fontsize=fs,
              framealpha=0)

    # Set axes limits
    # ---------------
    if tlim is not None:
        ax.set_xlim(tlim)

    dy = 0.08 * (q.max() - q.min())
    ylim = (q.min()-dy, q.max()+dy)
    ax.set_ylim(ylim)

    # Play with ticks
    # ---------------
    for tlabels in [ax.get_xticklabels(), ax.get_yticklabels()]:
        for txt in tlabels:
            txt.set_fontsize(fs-1)

    return fig, ax

# -----------------------------------------------------------------------------
def k_t_serie(time, k, k_smooth, k_mean, k_eff, PARAM, VALVES, tlim=None, fs=None, fig=None, ax=None):
    """
    Plots a permeability time series (input dimensionalized permeabilities).

    Parameters
    ----------
    time : 1D array
        Array of times.
    k : 1D array
        Array of permeability time series, same dimension as `time`.
    k_smooth : 1D array
        Smoothed time serie, same dimension as `time`.
    k_mean : float
        Average permeability over permanent regime.
    k_eff : float
        Effective value of the permeability. If specified it is
        plotted as a horizontal line.
    PARAM : dict
        Dictionnary of physical parameters.
    VALVES : dict
        Dictionnary of valves parameters.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    fs : float (default `fs = None`)
        Fontsize.
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.

    Returns
    -------
    fig : matplotlib figure object
        The input or current figure.
    ax : matplotlib axes object
        The input or current axes.

    """
    # Define plot time window
    # -----------------------
    if tlim is not None:
        t_win = (time > tlim[0]) & (time < tlim[1])
        time = time[t_win]
        k = k[t_win]
        k_smooth = k_smooth[t_win]

    rasterize = bool(len(time) > 1e4)

    # All valves closed, all valves open permeabilities
    # -------------------------------------------------
    kop = equi.calc_k_eq(VALVES, PARAM, states_override=np.ones(len(VALVES['idx'])))

    # Convert them to opening ratios
    # ------------------------------
    op_rat = util.keq2opratio(k, VALVES, PARAM)
    op_rat_smooth = util.keq2opratio(k_smooth, VALVES, PARAM)
    op_rat_eff = util.keq2opratio(k_eff, VALVES, PARAM)
    op_rat_mean = util.keq2opratio(k_mean, VALVES, PARAM)

    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Fontsize
    # --------
    if fs is None:
        fs = 9

    # Label the axes
    # --------------
    ax.set_ylabel('Proportion of open valves', fontsize=fs, c=c_k)
    ax.set_xlabel('Time (scaled)', fontsize=fs)

    # Plotting
    # --------
    ax.plot(time, op_rat, c=to_rgba(c_k, 0.3), lw=1, rasterized=rasterize,
            zorder=0, label='$k_{eq}$')
    ax.plot(time, op_rat_smooth, c=c_k, rasterized=rasterize, zorder=2,
            label='Smoothed $k_{eq}$')

    ax.axhline(op_rat_eff, c='.5', lw=2, ls='--',
               label='Eff. perm. $k_{eff}$', zorder=1)
    ax.axhline(op_rat_mean, c=to_rgba(c_k, .5), lw=2, ls='--',
               label='Average perm. $k_{\infty}$', zorder=1)

    ax.axhline(1, c='k', ls=':', lw=1, label='All valves open', zorder=3)
    ax.axhline(0, c='k', ls='--', lw=1, label='All valves closed', zorder=3)

    # Set y axis
    # ----------
    dy = 0.08 * (op_rat.max() - op_rat.min())
    ylim = (op_rat.min()-dy, op_rat.max()+dy)
    ax.set_ylim(ylim)

    # Open ratio axis
    # ---------------
    ax_k = ax.twinx()
    ax_k.set_ylabel('$k_{eq}$ (scaled)', c=c_k, fontsize=fs)

    ticks = ax.get_yticks()
    ax_k.yaxis.set_ticks(ticks)
    tlabels = ['{:.2f}'.format(util.opratio2keq(t, VALVES, PARAM)/kop) for t in ticks]
    ax_k.yaxis.set_ticklabels(tlabels)
    ax_k.set_ylim(ax.get_ylim())

    # Legend
    # ------
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.05), fontsize=fs,
              framealpha=0)

    # Set axes limits
    # ---------------
    if tlim is not None:
        ax.set_xlim(tlim)

    # Play with ticks
    # ---------------
    for tlabels in [ax.get_xticklabels(), ax.get_yticklabels(),
                    ax_k.get_yticklabels()]:
        for txt in tlabels:
            txt.set_fontsize(fs-1)

    return fig, ax

# -----------------------------------------------------------------------------
def dp_t_serie(time, dp, dp_smooth, dp_mean, dp_eff, tlim=None, fs=None, fig=None, ax=None):
    """
    Plots a pressure loss across the system time series.

    Parameters
    ----------
    time : 1D array
        Array of times.
    dp : 1D array
        Array of pressure loss time series, same dimension as `time`.
    dp_smooth : 1D array
        Smoothed time serie, same dimension as `time`.
    dp_mean : float
        Average of pressure loss over the permanent regime.
    dp_eff : float
        Effective value of the pressure loss. If specified it is
        plotted as a horizontal line.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    fs : float (default `fs = None`)
        Fontsize.
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.

    Returns
    -------
    fig : matplotlib figure object
        The input or current figure.
    ax : matplotlib axes object
        The input or current axes.

    """
    # Define plot time window
    # -----------------------
    if tlim is not None:
        t_win = (tlim[0] < time) & (time < tlim[1])
        time = time[t_win]
        dp = dp[t_win]
        dp_smooth = dp_smooth[t_win]

    rasterize = bool(len(time) > 1e4)

    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Fontsize
    # --------
    if fs is None:
        fs = 9

    # Label the axes
    # --------------
    ax.set_ylabel('Pressure loss (scaled)', fontsize=fs, c=c_p)
    ax.set_xlabel('Time (scaled)', fontsize=fs)

    # Plotting
    # --------
    ax.plot(time, dp, c=to_rgba(c_p, 0.3), lw=1, rasterized=rasterize, zorder=0,
            label=r'$\Delta p$')
    ax.plot(time, dp_smooth, c=c_p, rasterized=rasterize, zorder=2,
            label=r'Smoothed $\Delta p$')

    ax.axhline(dp_eff, c='.5', lw=2, ls='--',
               label=r'Effective flux $\Delta p_{eff}$', zorder=1)
    ax.axhline(dp_mean, c=to_rgba(c_p, .5), lw=2, ls='--',
               label=r'Average $\Delta p_{\infty}$', zorder=1)

    # Legend
    # ------
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1.05), fontsize=fs,
              framealpha=0)

    # Set axes limits
    # ---------------
    if tlim is not None:
        ax.set_xlim(tlim)

    dy = 0.08 * (dp.max() - dp.min())
    ylim = (dp.min()-dy, dp.max()+dy)
    ax.set_ylim(ylim)

    # Play with ticks
    # ---------------
    for tlabels in [ax.get_xticklabels(), ax.get_yticklabels()]:
        for txt in tlabels:
            txt.set_fontsize(fs-1)

    return fig, ax

# -----------------------------------------------------------------------------
def act_t_serie(time, rate, rate_smooth, tlim=None, period=None, fs=None, fig=None, ax=None):
    """
    Plots a time serie of activity rate.

    Parameters
    ----------
    time : 1D array
        Array of times.
    rate : 1D array
        Array of time series of activity rate, same dimension as `time`.
    rate_smooth : 1D array
        Array of time series of activity rate, same dimension as `time`.
    period : float (default to `None`)
        Displays detected period as a range on the plot.
    tlim : tuple (default `None`)
        Option to plot in between specific time limits, specified as a tuple.
    fs : float (default `fs = None`)
        Fontsize.
    fig : matplotlib figure object (default to None)
        Figure where to plot the valves. If not specified, takes output of
        plt.gcf(): current active figure.
    ax : matplotlib axes object (default to None)
        Axes where to plot the valves. If not specified, takes output of
        plt.gca(): current active axes.

    Returns
    -------
    fig : matplotlib figure object
        The input or current figure.
    ax : matplotlib axes object
        The input or current axes.

    """
    # Define plot time window
    # -----------------------
    if tlim is not None:
        t_win = (tlim[0] < time) & (time < tlim[1])
        time = time[t_win]
        rate = rate[t_win]
        rate_smooth = rate_smooth[t_win]

    if len(time) > 1e4:
        rasterize=True
    else:
        rasterize=False

    # As a function of input, point to correct objects for figure and axis
    # --------------------------------------------------------------------
    if fig is None:
        fig = plt.gcf()

    if ax is None:
        ax = plt.gca()

    # Fontsize
    # --------
    if fs is None:
        fs = 9

    # Label the axes
    # --------------
    ax.set_ylabel('Activity rate (scaled)', fontsize=fs, c=c_act)
    ax.set_xlabel('Time (scaled)', fontsize=fs)

    # Plotting
    # --------
    ax.plot(time, rate, c=to_rgba(c_act, 0.3), lw=1, rasterized=rasterize,
            zorder=0, label='Activity rate')
    ax.plot(time, rate_smooth, c=c_act, rasterized=rasterize, zorder=2,
            label='Smoothed act. rate')

    if period is not None:
        t0_per_patch = time[0] + 0.7*(time[1]-time[0])
        ax.axvspan(t0_per_patch, t0_per_patch + period, fc=to_rgba('r', 0.3),
                   ec='#00000000', zorder=-1, label='Detected period')

    # Legend
    # ------
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1.05), fontsize=fs,
              framealpha=0)

    # Set axes limits
    # ---------------
    if tlim is not None:
        ax.set_xlim(tlim)

    dy = 0.08 * (rate.max() - rate.min())
    ylim = (rate.min()-dy, rate.max()+dy)
    ax.set_ylim(ylim)

    # Play with ticks
    # ---------------
    for tlabels in [ax.get_xticklabels(), ax.get_yticklabels()]:
        for txt in tlabels:
            txt.set_fontsize(fs-1)

    return fig, ax

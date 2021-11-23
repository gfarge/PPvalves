""" Elementary plots, to be integrated in a figure """

# Imports
# =======
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as tk
from matplotlib.colors import to_rgba

from PPvalves.plots.elements2 import valve_system, flux_space, dp_t_serie, \
                                     q_t_serie, k_t_serie, act_t_serie


# General parameters (colors)
# ===========================
c_k = '#00bdaa'
c_p = '#fe346e'
c_q = '#400082'

c_act = 'k'

c_mig = '#f5b461' # color for migrations
alpha_mig = 0.6

c_op = '#9ad3bc'
c_cl = '#ec524b'
c_int = '#f5b461'

# Functions
# =========

def valve_system_fig(X, VALVES, fc='k', ec=[0, 0, 0, 0], figsize=None, fs=None, figpath=None):
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
    figpath : str or None (default)
        Path for the figure to save.

    Returns
    -------
    fig : figure object from matplotlib.
        The figure created in this function.
    ax : ax object from matplotlib.
        The axis created in this function.
    g_objs : patch collection
        The patch collection of valves.

    """
    # Defaults
    # --------
    if figsize is None:
        figsize=(10/2.54, 2/2.54)

    # Make figure
    # -----------
    fig, ax = plt.subplots(figsize=figsize)
    fig, ax, g_objs = valve_system(X, VALVES, fc=fc, ec=ec, fs=fs, fig=fig, ax=ax)

    # Saving?
    # -------
    if figpath is not None:
        print('Saving figure at {:}'.format(figpath))
        plt.savefig(figpath, facecolor=[0, 0, 0, 0])
    else:
        plt.show()

    return fig, ax, g_objs


# -----------------------------------------------------------------------------
def flux_space_fig(q, PARAM, VALVES, fs=None, figsize=None, figpath=None):
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
    # Defaults
    # --------
    if figsize is None:
        figsize = np.array([14, 3.5])/2.54

    # Make figure
    # -----------
    fig, ax = plt.subplots(figsize=figsize)
    fig, ax = flux_space(q, PARAM, VALVES, fs=fs, fig=fig, ax=ax)
    plt.tight_layout()

    # Saving?
    # -------
    if figpath is not None:
        print('Saving figure at {:}'.format(figpath))
        plt.savefig(figpath, facecolor=[0, 0, 0, 0])
    else:
        plt.show()

    return fig, ax

# -----------------------------------------------------------------------------
def summary_t_series(time, rate_time, Vars, Smooth_vars, Mean_vars, Eff_vars, PARAM, VALVES, period=None, tlim=None, fs=None, figsize=(8.1, 11.5), figpath=None):
    """
    Plots a summary of time series of output flux, equivalent permeability, pressure loss and activity.

    Parameters
    ----------
    time : 1D array
        Time array.
    rate_time : 1D array
        Time array for activity rate.
    Vars : dict
        Dictionnary of time series: keys should be `q`, `k`, `dp`, `rate`. Each one should be a 1D-array of the same dimension as `time`
    Smooth_vars : dict
        Dictionnary of smoothed time series. Same keys and individual types as `Vars`.
    Mean_vars : dict
        Dictionnary of average over the permanent regime of the variables, as `float` for each. Same keys as `Vars`.
    Eff_vars : dict
        Dictionnary of average effective variables in the permanent regime, as `float` for each. Same keys as `Vars`.
    period : float (default `period = None`)
        Highlight period on the activity plot.
    PARAM : dict
        Dictionnary of physical parameters.
    VALVES : dict
        Dictionnary of valve parameters.
    tlim : tuple or array-like (default `tlim = None`)
        Impose a time span to the plot.
    fs : float (default `fs = None`)
        Impose the reference fontsize.
    figsize : tuple or array-like (default `figsize = (8.1, 11.5)`, in cm, roughly A4)
        Impose figure size.
    figpath : (default `figpath = None`)
        If specified, the figure is not shown, and directly saved, using `figpath` as the path.

    Returns
    -------
    fig : matplotlib figure object
        The matplotlib figure object created.

    """
    # >> Unpack variables
    # -------------------
    q = Vars['q']; k = Vars['k']; dp = Vars['dp']; rate = Vars['rate']

    q_smooth = Smooth_vars['q']; k_smooth = Smooth_vars['k']
    dp_smooth = Smooth_vars['dp']; rate_smooth = Smooth_vars['rate']

    q_mean = Mean_vars['q']; k_mean = Mean_vars['k']; dp_mean = Mean_vars['dp']
    q_eff = Eff_vars['q']; k_eff = Eff_vars['k']; dp_eff = Eff_vars['dp']

    # >> Define the figure
    # --------------------
    fig, axes = plt.subplots(4, 1, figsize=figsize)

    # >> Plots
    # --------
    # --> Flux
    ax = axes[0]
    fig, ax = q_t_serie(time, q, q_smooth, q_mean, q_eff, tlim=tlim, fs=fs,
                        fig=fig, ax=ax)

    # --> Permeability
    ax = axes[1]
    fig, ax = k_t_serie(time, k, k_smooth, k_mean, k_eff, PARAM, VALVES,
                        tlim=tlim, fs=fs, fig=fig, ax=ax)

    # --> Pressure loss
    ax = axes[2]
    fig, ax = dp_t_serie(time, dp, dp_smooth, dp_mean, dp_eff, tlim=tlim,
                         fs=fs, fig=fig, ax=ax)

    # --> Activity rate
    ax = axes[3]
    fig, ax = act_t_serie(rate_time, rate, rate_smooth, period=period,
                          tlim=tlim, fs=fs, fig=fig, ax=ax)

    # >> Format the plot
    # ------------------
    fig.subplots_adjust(left=0.08, right=0.75, top=0.99, bottom=0.05,
                        hspace=0.2)

    # >> Save?
    # --------
    if figpath is not None:
        print('Saving at {:}'.format(figpath))
        plt.savefig(figpath)
    else:
        plt.show()

    return fig

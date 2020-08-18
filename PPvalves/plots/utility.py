""" Utility functions for ppv_plt"""

# Imports
# =======
import numpy as np

# Core
# ====

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
        'v_st_marker' : True,
        'v_op_mc' : 'lime',
        'v_cl_mc' : 'red',
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

        'b_zorder' : 20,

        'act_lc' : 'k',
        'k_eq_lc' : 'darkturquoise',
        'mass_lc' : 'purple'

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

def set_valve_fc(v_open, plot_params):
    """
    Sets valve patch (or else) facecolor as a function of preset colors and
    valve state.

    Parameters
    ----------
    v_open : 1D array
        Valve states, as a boolean array. Dimension Nvalves.
    plot_params : dictionnary
        Dictionnary. Necessary entries : 'v_op_fc' and 'v_cl_fc'.

    Returns
    -------
    color : list
        List of object facecolors. Same dimension as v_open.

    """
    color = []
    for ii in range(len(v_open)):
        if v_open[ii]:
            color.append(plot_params['v_op_fc'])
        else:
            color.append(plot_params['v_cl_fc'])

    return color

# ----------------------------------------------------------------------------

def set_valve_mc(v_open, plot_params):
    """
    Sets valve marker (or else) color as a function of preset colors and
    valve state.

    Parameters
    ----------
    v_open : 1D array
        Valve states, as a boolean array. Dimension Nvalves.
    plot_params : dictionnary
        Dictionnary. Necessary entries : 'v_op_mc' and 'v_cl_mc'.

    Returns
    -------
    m_op_color : list
        List of open marker colors. Same dimension as v_open.
    m_cl_color : list
        List of closed marker colors. Same dimension as v_open.

    """
    m_op_color = []
    m_cl_color = []

    for ii in range(len(v_open)):
        if v_open[ii]:
            m_op_color.append(plot_params['v_op_mc'])
            m_cl_color.append([0, 0, 0, .1])
        else:
            m_op_color.append([1, 1, 1, .5])
            m_cl_color.append(plot_params['v_cl_mc'])

    return m_op_color, m_cl_color

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

# ----------------------------------------------------------------------------

def bg_profileZ(P, Z, PARAM):
    """ Create background figure and axes for the pressure profile in Z """

    # Unpack
    g = PARAM['g']
    rho = PARAM['rho']
    rho_r = PARAM['rho_r']
    P_scale = PARAM['P_scale']
    Z_scale = PARAM['Z_scale']

    L = Z[0] - Z[-1]

    # Set ylim, on Z
    Zlow = Z[0] + 0.1*L # lowest point of the graph
    Zhi = Z[-1] - 0.2*L # highest point of the graph /!\ axis reversed
    ylim = np.array([Zhi, Zlow])

    Pflow = rho*g*Zlow * Z_scale/P_scale # fluid pressure for lower bound
    Pfhi = rho*g*Zhi * Z_scale/P_scale # fluid pressure for higher bound
    Pf = np.array([Pfhi,Pflow])

    Prlow = rho_r*g*Zlow * Z_scale/P_scale # lithostatic pressure for lower bound
    Prhi = rho_r*g*Zhi * Z_scale/P_scale # lithostatic pressure for higher bound
    Pr = np.array([Prhi,Prlow])

    # Set
    z = np.array([Zhi,Zlow])
    #xlim = (Prhi, Prlow)
    xlim = np.array([min([Prhi,min(P)-0.1*(max(P)-Prhi)]), max(P)+0.1*(max(P)-Prhi)])

    #### plot
    fig, ax = plt.subplots(figsize=(4,4))
    plt.subplots_adjust(left=0.15)

    # plot hydrostatic gradient grid
    Ngrid = 10 # nb of gradient copies for grid
    dP = (xlim[1] - xlim[0]+(Pflow-Pfhi))/float(Ngrid) # N-1 so that it reaches the end
    Pf = Pf + xlim[0] - Pf[1] # set first line of grid at beg of plot
    for ii in range(Ngrid):
        if (P_scale == 1.) & (Z_scale == 1.):
            ax.plot(Pf*conv_P,z*conv_Z,'--',lw=0.5,c='lightgray')
        else:
            ax.plot(Pf,z,'--',lw=0.5,c='lightgray')
        Pf = Pf + dP

    if (P_scale == 1.) & (Z_scale ==1.):
        # lithostatic pressure
        ax.plot(Pr*conv_P,z*conv_Z,'-.',lw=1,c='gray')
        # refine axes
        ax.set_xlabel('Pressure (GPa)')
        ax.set_ylabel('Depth (km)')
        ax.set_xlim(xlim*conv_P)
        ax.set_ylim(ylim*conv_Z)
    else:
        # lithostatic pressure
        ax.plot(Pr,z,'-.',lw=1,c='gray')
        # refine axes
        ax.set_xlabel('Pressure (N/D)')
        ax.set_ylabel('Depth (N/D)')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.tick_params(top=True,bottom=False,right=False,labeltop=True,labelleft=True,labelbottom=False,labelright=False,direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return fig, ax

##################################################################

def profileZ(P,Z,PARAM):
    ## plot profile Z from bg_profileZ background
    P_scale = PARAM['P_scale']
    Z_scale = PARAM['Z_scale']

    #### plot
    fig, ax = bg_profileZ(P,Z,PARAM)
    if (P_scale == 1.) & (Z_scale == 1.):
        ax.plot(P*conv_P,Z*conv_Z,'-',lw=1.8,c='magenta')
    else:
        ax.plot(P,Z,'-',lw=1.8,c='magenta')
    plt.show()

##################################################################

def bg_profileX(P,X,Z,PARAM):
    ## Create background figure and axes for the pressure profile in X

    # Unpack
    rho = PARAM['rho']
    rho_r = PARAM['rho_r']
    g = PARAM['g']
    alpha = PARAM['alpha']
    P_scale = PARAM['P_scale']
    X_scale = PARAM['X_scale']
    Z_scale = PARAM['Z_scale']

    L = X[-1] - X[0]

    # set plot limits
    ## Xlims: min and max of X

    Xleft = X[0]-0.1*L
    Xright = X[-1]+0.1*L
    xlim = np.array([Xleft,Xright])

    ## Points for where to draw lith gradient: Z corresp to xlim
    Zleft = Z[0] - np.sin(alpha)*Xleft * X_scale/Z_scale
    Zright = Z[0] - np.sin(alpha)*Xright * X_scale/Z_scale
    Prleft = rho_r*g*Zleft * Z_scale / P_scale
    Prright = rho_r*g*Zright * Z_scale / P_scale
    Pr = np.array([Prleft,Prright]) # pressure for lith gradient

    ## Reduce pressure, remove hydro gradient
    X0 = X[-1] # arbitrarily set a level to calculate pressure differences
    dPred = rho*g*np.sin(alpha)*(X0-X) * X_scale/P_scale
    #dPred = rho*g*Z * Z_scale/P_scale
    #Pr = Pr - rho*g*np.array([Zleft,Zright]) * Z_scale/P_scale
    DP = max(P) - min(P)

    ## Set Ylims
    Phi = max(P) + 0.2*DP
    Plo = min(P) - 0.2*DP
    ylim = np.array([Plo, Phi])


    #### plot

    fig, ax_P = plt.subplots(figsize=(7,3.5))
    plt.subplots_adjust(left=0.15,right=0.85,bottom=0.2)
    ax_Q = ax_P.twinx()

    if (P_scale == 1.) & (X_scale == 1.):
        # lithostatic pressure
        ax_P.plot(xlim*conv_X,Pr*conv_P,'-.',lw=1,c='gray')

        # refine axes
        ax_P.set_xlabel('X (km)')
        ax_P.set_ylabel('P (GPa)')
        ax_Q.set_ylabel('Q (kg.s-1)')
        ax_P.set_xlim(xlim*conv_X)
        ax_P.set_ylim(ylim*conv_P)
    else:
        # lithostatic pressure
        ax_P.plot(xlim,Pr,'-.',lw=1,c='gray')

        # refine axes
        ax_P.set_xlabel('X (N/D)')
        ax_P.set_ylabel('P (N/D)')
        ax_Q.set_ylabel('Q (N/D)')
        ax_P.set_xlim(xlim)
        ax_P.set_ylim(ylim)

    ax_P.tick_params(top=False,bottom=True,right=False,labeltop=False,labelleft=True,labelbottom=True,labelright=False,direction='in')
    ax_P.patch.set_visible(False)
    ax_Q.set(zorder=-1)


    return fig, ax_P, ax_Q

##################################################################

def profileX(P,X,Z,k,PARAM):
    ## From the background of bg_profileX, plot Pred and Q

    # Unpack
    rho = PARAM['rho']
    g = PARAM['g']
    alpha = PARAM['alpha']
    P_scale = PARAM['P_scale']
    X_scale = PARAM['X_scale']
    Z_scale = PARAM['Z_scale']

    # Calculate Q and Pred
    dx = X[1] - X[0]
    Q = calc_Q(P,k,PARAM)
    XQ = (X[1:] + X[:-1])/2.

    # plot
    fig, ax_P, ax_Q = bg_profileX(P,X,Z,PARAM)
    ax_P.patch.set_visible(False)

    # plot lines
    if (P_scale == 1.) & (X_scale == 1.):
        ax_P.plot(X*conv_X,P*conv_P,lw=2.,c='magenta',label='Pore pressure')
        ax_Q.fill_between(XQ*conv_X,Q,np.zeros(len(Q)-2),color='cyan',alpha=0.5,label='Fluid flux')
    else:
        ax_P.plot(X,P,lw=2.,c='magenta',label='Pore pressure')
        ax_Q.fill_between(XQ,Q,np.zeros(len(Q)),color='cyan',alpha=0.5,label='Fluid flux')

    return fig,ax_P,ax_Q
#     plt.show()


##################################################################

def animDiff1DX(P,X,T,PARAM,fig,ax_P,ax_Q,l,fp,writer,anim_filename):
    """
    Function to animate the pressure gradient diffusing through time
    P(X)
    P is the pressure in space-time (2d array), X is space domain, T
    is time domain, PARAM the parameter dictionnary, fig, ax_P, ax_Q
    the figure and axes of pressure and flux, l is the pressure line, fp
    the fictive point object to set the data, writer is the ffmpeg writer,
    anim_filename the name of the animation file to save.
    """

    # Unpack variables
    k = PARAM['k']
    g = PARAM['g']
    A = PARAM['A']
    mu = PARAM['mu']
    rho = PARAM['rho']
    dx = PARAM['h_']
    alpha = PARAM['alpha']
    X_scale = PARAM['X_scale']
    Z_scale = PARAM['Z_scale']
    P_scale = PARAM['P_scale']

    X0 = X[-1]

    print('## Writing frames ##\n')
    with writer.saving(fig, anim_filename, 400):
        for ii,tt in enumerate(range(len(T)-2)):
            #Calculate Q and Pred
            Q = calc_Q(P[tt,:], k, PARAM)
            XQ = (X[1:] + X[:-1])/2.
            Pred = P[tt,:]
            if (P_scale == 1.) & (X_scale == 1.):
                # plot the Pressure and Debit curves for each frame
                l.set_data(X[1:-1]*conv_X,Pred[1:-1]*conv_P)
                fp.set_data([X[0]*conv_X,X[-1]*conv_X],[Pred[0]*conv_P,Pred[-1]*conv_P])
                Qarea = ax_Q.fill_between(XQ*conv_X,Q,np.zeros(len(Q)),color='cyan',alpha=0.5)
            else:
                # plot the Pressure and Debit curves for each frame
                l.set_data(X[1:-1],Pred[1:-1])
                fp.set_data([X[0],X[-1]],[Pred[0],Pred[-1]])
                Qarea = ax_Q.fill_between(X[1:-1],Q[1:-1],np.zeros(len(Q)-2),color='cyan',alpha=0.5)
            # grab frame
            writer.grab_frame()
            # remove the Q plot
            ax_Q.collections.remove(Qarea)

    print('## Saved ' + anim_filename + ' ##\n')
    print('## Done ##\n')


##################################################################

def animDiff1DZ(P,Z,T,PARAM,fig,ax,l,fp,writer,anim_filename):
    ## Function to animate the pressure gradient diffusing through time
    ## P(Z)

    # Unpack variables
    X_scale = PARAM['X_scale']
    Z_scale = PARAM['Z_scale']
    P_scale = PARAM['P_scale']

    print('## Writing frames ##\n')
    with writer.saving(plt.gcf(), anim_filename, 400):
        for tt in range(len(T)-1):
        	if (P_scale == 1.) & (Z_scale == 1.):
        	    # plot the Pressure curve for each frame
        	    l.set_data(P[tt,1:-1]*conv_P,Z[1:-1]*conv_Z)
        	    fp.set_data([P[tt,0]*conv_P,P[tt,-1]*conv_P],[Z[0]*conv_Z,Z[-1]*conv_Z])
        	else:
        	    # plot the Pressure curve for each frame
        	    l.set_data(P[tt,1:-1],Z[1:-1])
        	    fp.set_data([P[tt,0],P[tt,-1]],[Z[0],Z[-1]])
        	# grab frame
        	writer.grab_frame()

    print('## Saved ' + anim_filename + ' ##\n')
    print('## Done ##\n')




##################################################################



def plot_barriers(barriers,X,PARAM):
    """
    Plots the domain with barriers and thresholds
    """
    plt.close()
    # Unpack variables
    h_ = PARAM['h']
    g = PARAM['g']
    rho = PARAM['rho']
    rho_r = PARAM['rho_r']
    alpha = PARAM['alpha']
    X_scale = PARAM['X_scale']
    P_scale = PARAM['P_scale']
    dim = (X_scale == 1.) & (P_scale == 1.)

    conv_P = 1.e-6 # to MPa

    idx = barriers['idx']
    dPhi = barriers['dPhi']/h_
    dPlo = barriers['dPlo']/h_

    # compute lithostatic gradient
    lith = rho_r*g*np.sin(alpha)*X_scale/P_scale

    fig, ax = plt.subplots(figsize = (10,4))
    if dim:
        ax.set_xlabel('X (km)')
        ax.set_ylabel('barrier dP/dx (MPa/m)')

        # Set ylim
        ylimlo = (lith - 0.55*(max(dPhi) - lith))*conv_P
        ylimhi = (max(dPhi) + 0.1*(max(dPhi) - lith))*conv_P
        ax.set_ylim([ylimlo,ylimhi])

        # plot lithostatic gradient
        l_lith, = ax.plot(np.array([X[0],X[-1]])*conv_X,np.array([lith,lith])*conv_P,c='k',ls='-',marker='+',mec='b',mew=2,lw=2)

        ficpts, = ax.plot(np.array([X[0],X[-1]])*conv_X,np.array([lith,lith])*conv_P,'b+',ms=10.,mew=2.)

        # plot domain limit
        ax.axvline(X[1]*conv_X, c='gray', ls='--',lw=2.)
        ax.axvline(X[-2]*conv_X, c='gray', ls='--',lw=2.)

        for ii,hi,lo in zip(idx,dPhi,dPlo):
            l_hi, = ax.plot(np.array(1,1)*(X[ii]+X[ii-1])/2*conv_X,np.array([lith,hi])*conv_P,'r-',lw=2)
            l_lo, = ax.plot(np.array(1,1)*(X[ii]+X[ii-1])/2*conv_X,np.array([lith,lo])*conv_P,'g-',lw=4)
    else:
        ax.set_xlabel('X (N/D)')
        ax.set_ylabel('barrier dP/dx (N/D)')
        ax.set_title('Gradient values are scaled to lithostatic gradient')

        # Set ylim
        ylimlo = lith - 0.55*(max(dPhi) - lith)
        ylimhi = max(dPhi) + 0.1*(max(dPhi) - lith)
        ax.set_ylim([ylimlo,ylimhi])

        # plot lithostatic gradient
        l_lith, = ax.plot(np.array([X[1],X[-2]]),np.array([lith,lith]),c='k',ls='-',lw=2)
        ficpts, = ax.plot(np.array([X[0],X[-1]]),np.array([lith,lith]),'b+',ms=10.,mew=2.)

        # plot domain limit
        ax.axvline(X[1], c='gray', ls='--',lw=2.)
        ax.axvline(X[-2], c='gray', ls='--',lw=2.)

        for ii,hi,lo in zip(idx,dPhi,dPlo):
            l_hi, = ax.plot(np.array([1,1])*(X[ii]+X[ii-1])/2.,[lith,hi],'r-',lw=2)
            l_lo, = ax.plot(np.array([1,1])*(X[ii]+X[ii-1])/2,[lith,lo],'g-',lw=4)
        ax.legend((l_lith,ficpts,l_hi,l_lo),('Lith. gradient.','Fict. points','Opening dP/dx','Closing dP/dx'),loc='lower right')
    plt.grid()




#################################################################################


def plot_dPhist(dP,T,style='k-',lw=1):

    """
    Function to plot the history of pressure delta across barriers.

    # inputs:

    _ dP : array of pressure history of one barrier
    _ T : array of times, same dimension as dP.
    _ style : str, defines the style of the line: eg. 'k.-'

    # outputs:

    _ l : line object

    """

    # Get axes
    global ax
    ax = plt.gca()

    # Plot dP history
    l, = ax.plot(T,dP,style,lw=lw)

    # Labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Pressure diff. across barriers')

    # Handle plot lims
    ## xlims
    ax.set_xlim(min(l.get_xdata()),max(l.get_xdata()))

    ## ylims
    ylim = ax.get_ylim()
    ymin = 0.8*min(dP)
    ymax = 1.1*max(dP)

    if ylim[0]<ymin:
    	ymin = ylim[0]
    if ylim[1]>ymax:
    	ymax = ylim[1]



    # Tick params
    ax.tick_params(top=True,bottom=True,left=True,right=True,direction='in',which='both')

    return l

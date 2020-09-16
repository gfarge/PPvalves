""" Module used to produce and analyse activity catalogs, adapted to analyse
the results of PPV runs but also of real catalogs."""

# /// Imports
import numpy as np


# Core

#---------------------------------------------------------------------------------

def event_count(ev_type, states, time, catalog=False, v_x=None):
    """
    Gives the time at which an event occurs, that is,
    the index when the just calculated dP crosses the threshold
    corresponding to the event type selected.

    Parameters
    ----------
    ev_type : str
        type of event to count, either "close", or "open".
    states : nd array
        Valve states history: N_times * N_valves.  False is closed, True is
            open. Output of run_ppv, v_activity[times,0,valve_idx],
            where valve_idx is the idxs of the valves you want the
            event count for.
    time : nd array
        Array of times, same shape as first dimension of valve states.
    catalog : bool (default `False`)
        if turned to `True`, the catalog option allows close_count to return
        x_events in addition to t_event.  When turned on, this option recquires
        that v_x is specified.
    v_x : 1d array,
        Location of each valve, same size as 2nd dimension of states.

    Returns
    -------
    events_t : nd array
        Event times.
    events_x : nd array
        Event locations, if catalog is True.

    """

    # Check if v_x is present if catalog optin is True
    if catalog & isinstance(v_x, type(None)):
        raise ValueError("When catalog option is turned on, the valves'" + \
         'location needs to be specified with the v_x argument.')

    # Get opening time idx for each valve in one array

    # Test if we have 1 or several valves here
    if len(states.shape) > 1:
        n_valves = states.shape[1] # valve dimension

        all_events_i = []  # time indices of events
        if catalog:
            all_events_x = []

        for ibarr in range(n_valves):
            # --> First get the time indices of events
            # for each history, get the time when states switches from 0 (closed)
            # to 1 (open)

            b_states = states[:, ibarr]

            if ev_type == 'close':
                b_events_i = np.where(b_states[1:].astype(int) -\
                b_states[:-1].astype(int) == -1)[0]+1
            elif ev_type == 'open':
                b_events_i = np.where(b_states[1:].astype(int) -\
                b_states[:-1].astype(int) == 1)[0]+1

            b_events_i = b_events_i.tolist()
            all_events_i.extend(b_events_i)

            # --> Then build location vector
            if catalog:
                b_events_x = [v_x[ibarr] for ii in\
            	 range(len(b_events_i))]
                all_events_x.extend(b_events_x)

        # Convert idx to time
        events_t = time[all_events_i]
        if catalog:
            events_x = np.array(all_events_x)

        # Sort array
        if catalog:
            id_sort = np.argsort(events_t)
            events_t = events_t[id_sort]
            events_x = events_x[id_sort]
            return events_t, events_x

        else:
            events_t = np.sort(events_t)
            return events_t

    else:
        if ev_type == 'close':
            events_i = np.where(states[1:].astype(int) -\
                        states[:-1].astype(int) == -1)[0] + 1
        if ev_type == 'open':
            events_i = np.where(states[1:].astype(int) -\
    	                states[:-1].astype(int) == 1)[0] + 1
        events_i = events_i.tolist()
        t_events = time[events_i]
        return t_events

#---------------------------------------------------------------------------------

def open_count(states, time, catalog=False, v_x=None):
    """
    Gives the time at which the valves open, that is,
    the index when the just calculated dP is higher than dPhi for
    the valve. Calls event_count, same parameters.

    Parameters:
    -----------
    states: nd array
        Valve states history, dimension Ntimes * Nvalves.  False is closed, True is
        open. Output of run_ppv, v_activity[times,0,valve_idx], where valve_idx
        is the idxs of the valves you want the event count for.
    time : nd array
        Array of times, same shape as first dimension of valve states.
    catalog : bool (default `False`)
        if turned to `True`, the catalog option allows close_count to return
        x_events in addition to t_event.  When turned on, this option recquires
        that v_x is specified.
    v_x : 1d array,
        Location of each valve, same size as 2nd dimension of states.

    Returns
    -------
    events_t : nd array
        Event times.
    events_x : nd array
        Event locations, if catalog is True.


    """

    # Call event count
    out = event_count('open', states, time, catalog,v_x)

    return out

#---------------------------------------------------------------------------------

def close_count(states, time, catalog=False, v_x=None):
    """
    Gives the time at which the valves open, that is,
    the index when the just calculated dP is higher than dPhi for
    the valve. Calls event_count, same parameters.

    Parameters:
    -----------
    states: nd array
        Valve states history, dimension Ntimes * Nvalves.  False is closed, True is
        open. Output of run_ppv, v_activity[times,0,valve_idx], where valve_idx
        is the idxs of the valves you want the event count for.
    time : nd array
        Array of times, same shape as first dimension of valve states.
    catalog : bool (default `False`)
        if turned to `True`, the catalog option allows close_count to return
        x_events in addition to t_event.  When turned on, this option recquires
        that v_x is specified.
    v_x : 1d array,
        Location of each valve, same size as 2nd dimension of states.

    Returns
    -------
    events_t : nd array
        Event times.
    events_x : nd array
        Event locations, if catalog is True.

    """

    # Call event count
    out = event_count('close', states, time, catalog,v_x)

    return out

#---------------------------------------------------------------------------------

def recurrence(event_time):
    """
    Calculates time before next event for an event sequence.


    Parameters
    ----------
    event_time : 1D array
        Event times.

    Returns
    -------
    time_before_next
        Time before next event for the all the events but the last one. Same
        unit as the event times.
    """

    time_before_next = event_time[1:] - event_time[:-1]

    return time_before_next

#---------------------------------------------------------------------------------

def event_count_signal(event_time, dt, t0=0., tn=None, norm=False):
    """
    Calculates the event count signal: returns an evenly spaced array
    of the binned count of events.

    Parameters
    ----------
    event_time : 1D array
        Event times.
    dt : float
        Binning length to compute activity rate, final discretization of the
        event count signal, in seconds.
    t0 : float (default = `0`)
        Beginning time of the first bin, in seconds.
    tn : float (default `None`)
        Beginning time of the final bin, in seconds. Default is time of last event.
    norm : bool (default = `False`)
        Option to normalize by norm of vector.

    Returns
    -------
    event_count_signal : 1D array
        Evenly spaced count of events in bins of dt. Activity rate.
    bin_edges : 1D array
        Boundaries of all bins, in second.
    """

    # Check if tn is 'max' and if so set it to its value
    if tn is None:
        tn = max(event_time)

    # The evenly spaced count is in fact a simple histogram
    ev_count, bin_edges = np.histogram(event_time,\
            bins=int((tn-t0)//dt+2), range=(t0, tn+dt))

    # Normalize with norm of vector
    if norm:
        ev_count = ev_count.astype(float)/np.linalg.norm(ev_count)

    # /!\ bin_edges length is 1 unit longer than event_count

    return ev_count, bin_edges

#---------------------------------------------------------------------------------

def autocorr(sig, dt, norm=True):
    """
    Calculates the autocorrelation of a signal.

    - Parameters
            + :param sig: 1D array of signal to compute the autocorrelation of.
            + :type sig: ndarray
            + :param dt: time step in the signal, in seconds. If sig = binned count, use bin size
            + :type dt: float
            + :param norm: autocorrelation normalized by the square of the norm
              of the signal (default is True).
            + :param norm: bool

    - Output
            + :return: :autocorr: 1D array containing autocorrelation value
            + :rtype: :autocorr: ndarray
            + :return: :lag: 1D array of time lag of autocorrelation, in seconds
            + :rtype: :lag: ndarray
    """

    # Normalize
    if norm:
        sig1 = (sig - np.mean(sig)) / (np.std(sig)*len(sig))
        sig2 = (sig - np.mean(sig)) / np.std(sig)

    # Calculate correlation
    autocorr_s = np.correlate(sig1.astype(float), sig2.astype(float), 'full')

    # Compute time lag of autocorrelation
    lag = np.arange(0, len(sig1)+len(sig2)-1)-(len(sig1)-1)
    lag = lag.astype(float)*dt
    return autocorr_s, lag

#---------------------------------------------------------------------------------

def crosscorr(sig1, sig2, dt, norm=True):
    """
    Calculates the crosscorrelation of two signal. Can be of different sizes.

    - Parameters
            + :param sig1, sig2: 1D arrays of signal.
            + :type sig1, sig2: ndarrays
            + :param dt: time step in the input signal, in seconds. If sig = binned count, use bin size.
            + :type dt: float
            + :param norm: if True crosscorrelation normalized by the square of the norm of the signal (default True).
            + :type norm: bool

    - Output
            + :return: corr: 1D array containing cross-correlation values.
            + :rtype: corr: ndarray
            + :return: lag : 1D array of time lag of correlation, in seconds. Lag may be strange if sig1 and sig2 do not have the same length
            + :rtype: lag: ndarray
    """

    # Normalize
    if norm:
        sig1 = (sig1 - np.mean(sig1)) / (np.std(sig1)*len(sig1))
        sig2 = (sig2 - np.mean(sig2)) / np.std(sig2)

    # Calculate correlation
    corr = np.correlate(sig1.astype(float), sig2.astype(float),'full')

    # Normalize
    if norm:
        corr = corr.astype(float) / (np.linalg.norm(sig1)*np.linalg.norm(sig2))

    # Compute time lag of autocorrelation
    lag = np.arange(0,len(sig1)+len(sig2)-1)-(len(sig1)-1)
    lag = lag.astype(float)*dt

    return corr, lag


#---------------------------------------------------------------------------------

def corr_coeff(sig1, sig2, dt, norm=True):
    """
    Calculates the correlation coefficient between two signals.

    The correlation coefficient is taken as the maximum of the cross-correlation
    of the two signals, the corresponding lag is also returned.

    - Parameters
        + :param sig1, sig2: 1D arrays of evenly spaced input signal.
        + :type sig1, sig2: ndarrays
        + :param dt: spacing of the values of the input signal, in seconds.
        + :type dt: float
        + :param norm: If True, normalisation is applied when calculating the cross correlation: the cross-correlation is normalized by the product of the signals' norms (default True).
        + :type norm: bool

    - Output
        + :return: cc: correlation coefficient.
        + :rtype: cc: bool
        + :return: lag : lag in seconds.  If positive (resp. negative), sig2 was shifted forward (backwards) in time to correspond with sig1. In other words, if positive (resp. negative), sig1 events are  after sig2 events (resp before).
        + :rtype: lag : float
    """

    # Calculates the cross correlation
    corr, corr_lag = crosscorr(sig1, sig2, dt, norm=norm)
    # Get correlation coefficient and time lag
    cc = max(abs(corr))
    cc_lag = corr_lag[np.argmax(abs(corr))]

    return cc, cc_lag

#---------------------------------------------------------------------------------


def act_interval_frac(event_t, tau):
    """
    This function computes the fraction of intervals of lengths `tau[ii]` that
    contain at least an event.

    Parameters
    ----------
    event_t : 1D array
        Array of event times of occurence, dimension N_ev.
    tau : 1D array
        Array of interval lengths to test, dimension N_tau.

    Returns
    -------
    X_tau : 1D array
        Fraction of intervals of corresponding lengths `tau` that contain at
        least an event, dimension N_tau
    tau : 1D array
        Input tau adjusted so as to match the actual value of `tau[ii]` that is
        tested, which has to be a multiple of the min of `tau`, dimension N_tau.

    """
    tau_min = min(tau)  # lowest tau is taken as sweeping window time increment
    count, _ = event_count_signal(event_t, tau_min, t0=min(event_t))

    X_tau = np.zeros(len(tau))

    for ii, tt in enumerate(tau):
        # First: build sweeping window
        len_tau_win = int(np.round(tt/tau_min))
        tau_win = np.ones(len_tau_win)  # sweeping window to count events
        tau[ii] = len_tau_win * tau_min

        # Second: correlate and count intervals with activity
        corr = np.correlate(count, tau_win, mode='valid')

        X_tau[ii] = 1 - np.sum(corr == 0) / len(corr)

    return X_tau, tau


#---------------------------------------------------------------------------------

def correlation_matrix(event_t, event_x, dt, X):
    """
    Computes the correlation of activity in different zones.

    Parameters
    ----------
    event_t : 1D array
        Array of event times, dimension N_ev.
    event_x : 1D array
        Array of event positions, dimension N_ev.
    dt : float
        Time step to compute the activity of the different zones.
    X : 1D array
        Array of the positions of the activity zones boundaries, dimension N_X

    Returns
    -------
    corr_mat : 2D array
        Cross-correlation coefficient matrix for the activity of the different
        regions specified by X, dimensions N_X - 1, N_X - 1.
    lag_mat : 2D array
        Lag time matrix corresponding to the maximum cross-correlation for the
        activity of different regions specified by X, dimensions N_X - 1, N_X -
        1.

    """
    # Compute the activity in each region
    # -----------------------------------
    activities = []  # The array of regional activities
    for ix in range(len(X)-1):
        region = (event_x >= X[ix]) & (event_x < X[ix + 1])
        regional_act, _ = event_count_signal(event_t[region], dt,
                                             t0=min(event_t))
        activities.append(regional_act)

    # Perform the cross-correlation and build the corr and lag matrices
    # -----------------------------------------------------------------
    corr_mat = np.zeros((len(X) - 1, len(X) - 1))
    lag_mat = np.zeros((len(X) - 1, len(X) - 1))

    for ix in range(len(X) - 1):
        for jx in range(ix, len(X) - 1):
            cc, cc_lag = corr_coeff(activities[ix], activities[jx],
                                              dt, norm=True)
            corr_mat[ix, jx] = corr_mat[jx, ix] = cc
            lag_mat[ix, jx] = lag_mat[jx, ix] = cc_lag

    return corr_mat, lag_mat

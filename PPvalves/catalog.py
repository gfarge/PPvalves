""" Module used to produce and analyse activity catalogs, adapted to analyse
the results of PPV runs but also of real catalogs."""

# /// Imports
import numpy as np


# Core

#---------------------------------------------------------------------------------

def event_count(ev_type,states,time,catalog=False,b_x=None):
    """
    Gives the time at which an event occurs, that is,
    the index when the just calculated dP crosses the threshold
    corresponding to the event type selected.

    - Parameters:
            + :param ev_type: type of event to count, either "close", or
    	"open"
            + :param states: is a nd array of states history: Nt * Nbarr.
            False is closed, True is
            open. Output of run_ppv, BA[times,0,barrier_idx],
            where barrier_idx is the idxs of the barriers you want the
            event count for.
            + :param time: is the time vector, np array, same shape as first
    	+ :param catalog: boolean, if turned True, the catalog option
    	allows close_count to return x_events in addition to t_event.
    	When turned on, this option recquires that b_x is specified.
    	+ :param b_x: np array, 1D, same size as 2nd dimension of
    	states, location of each barrier.

    - Outputs:
            + :return: events_t, np array of event times.
            + :return: events_x, np array of event times, if catalog is True.

    """

    # Check if b_x is present if catalog optin is True
    if catalog & isinstance(b_x, type(None)):
        raise ValueError("When catalog option is turned on, the barriers'" + \
         'location needs to be specified with the b_x argument.')

    # Get opening time idx for each barrier in one array

    # Test if we have 1 or several barriers here
    if len(states.shape) > 1:
        n_barriers = states.shape[1] # barrier dimension

        all_events_i = []  # time indices of events
        if catalog:
            all_events_x = []

        for ibarr in range(n_barriers):
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
                b_events_x = [b_x[ibarr] for ii in\
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

def open_count(states, time, catalog=False, b_x=None):
    """
    Gives the time at which the barriers open, that is,
    the index when the just calculated dP is higher than dPhi for
    the barrier. Calls event_count, same parameters.

    - Parameters:
            + :param states: is a nd array of states history: Nt * Nbarr.
            False is closed, True is
            open. Output of run_ppv, BA[times,0,barrier_idx],
            where barrier_idx is the idxs of the barriers you want the
            event count for.
            + :param time: is the time vector, np array, same shape as first
    	+ :param catalog: boolean, if turned True, the catalog option
    	allows close_count to return x_events in addition to t_event.
    	When turned on, this option recquires that b_x is specified.
    	+ :param b_x: np array, 1D, same size as 2nd dimension of
    	states, location of each barrier.

    - Outputs:
            + :return: events_t, np array of event times.
            + :return: events_x, np array of event times, if catalog is True.

    """

    # Call event count
    out = event_count('open', states, time, catalog,b_x)

    return out

#---------------------------------------------------------------------------------

def close_count(states, time, catalog=False, b_x=None):
    """
    Gives the time at which the barriers open, that is,
    the index when the just calculated dP is higher than dPhi for
    the barrier. Calls event_count, same parameters.

    - Parameters:
            + :param states: is a nd array of states history: Nt * Nbarr.
            False is closed, True is
            open. Output of run_ppv, BA[times,0,barrier_idx],
            where barrier_idx is the idxs of the barriers you want the
            event count for.
            + :param time: is the time vector, np array, same shape as first
    	+ :param catalog: boolean, if turned True, the catalog option
    	allows close_count to return x_events in addition to t_event.
    	When turned on, this option recquires that b_x is specified.
    	+ :param b_x: np array, 1D, same size as 2nd dimension of
    	states, location of each barrier.

    - Outputs:
            + :return: events_t, np array of event times.
            + :return: events_x, np array of event times, if catalog is True.

    """

    # Call event count
    out = event_count('close', states, time, catalog,b_x)

    return out

#---------------------------------------------------------------------------------

def recurrence(event_time):
    """
    Calculates time before next event for an event sequence.


    - Parameters
            + :param event_time: 1D array of event times, in seconds
            + :type event_time: ndarray

    - Output
            + :rtype: ndarray
            + :return: time before next event, in seconds
    """

    time_before_next = event_time[1:] - event_time[:-1]

    return time_before_next

#---------------------------------------------------------------------------------

def event_count_signal(event_time, dt, t0=0., tn='max', norm=False):
    """
    Calculates the event count signal: returns an evenly spaced array
    of the binned count of events.

    - Parameters
            + :param event_time: 1D array of event times, in seconds.
            + :type event_time: ndarray
            + :param dt: duration of bins, final discretization of the event
               count signal, in seconds.
            + :type dt: float
            + :param t0: float, beginning time of the first bin, in seconds.
             Default is 0.
            + :type t0: float
            + :param tn: float, beginning time of the final bin, in seconds.
              Default is time of last event.
            + :type tn: float
            + :param norm: default is False, option to normalize by norm of vector.
            + :type norm: bool

    - Output
            + :return:  event_count_signal : 1D array of evenly spaced count of
              events in bins of dt.
            + :rtype:  event_count_signal : ndarray
            + :return: bin_edges : 1D array of boundaries of all bins, in second.
            + :rtype: bin_edges : ndarray
    """

    # Check if tn is 'max' and if so set it to its value
    if tn == 'max':
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

""" Module used to produce and analyse activity catalogs, adapted to analyse
the results of PPV runs but also of real catalogs."""

# /// Imports
import numpy as np
from mtspec import mtspec
from scipy.special import erfinv
from scipy.signal import savgol_filter
from scipy.stats import chisquare, chi2, poisson


# Core

#------------------------------------------------------------------------------

def open_ratio(states):
    """
    Computes the ratio of open valves to total number of valves in time.

    Parameters
    ----------
    states: nd array
        Valve states history, dimension Ntimes * Nvalves.  False is closed, True is
        open. Output of run_ppv, v_activity[times,0,valve_idx], where valve_idx
        is the idxs of the valves you want the event count for.

    Returns
    -------
    ratio : 1D array
        At each time states is specified, the proportion of open valves is
        returned.

    """
    Nv = np.shape(states)[1]  # Total number of valves
    ratio = np.sum(states, axis=1) / Nv

    return ratio

#------------------------------------------------------------------------------

def event_count(ev_type, states, time, catalog=False, VALVES=None, X=None):
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
        that VALVES and X are specified.
    VALVES : dictionnary (default to `None`)
        Valve characteristics dictionnary. Needed for `catalog` option
    X : 1D array (default to `None`)
        Space array. Needed for `catalog` option


    Returns
    -------
    events_t : nd array
        Event times.
    events_x : nd array
        Event locations, if catalog is True.

    """

    # Check if v_x is present if catalog optin is True
    if catalog & isinstance(VALVES, type(None)) & isinstance(X, type(None)):
        raise ValueError("When catalog option is turned on, the valves'" + \
         'location needs to be specified with the v_x argument.')

    # Get opening time idx for each valve in one array

    # Test if we have 1 or several valves here
    if len(states.shape) > 1:
        n_valves = states.shape[1] # valve dimension

        all_events_i = []  # time indices of events
        if catalog:
            v_x = X[VALVES['idx']] + VALVES['width']/2
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

#------------------------------------------------------------------------------

def open_count(states, time, catalog=False, VALVES=None, X=None):
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
        that `VALVES` and `X` are specified.
    VALVES : dictionnary (default to `None`)
        Valve characteristics dictionnary. Needed for `catalog` option
    X : 1D array (default to `None`)
        Space array. Needed for `catalog` option

    Returns
    -------
    events_t : nd array
        Event times.
    events_x : nd array
        Event locations, if catalog is True.


    """
    # Call event count
    out = event_count('open', states, time, catalog=catalog, VALVES=VALVES, X=X)

    return out

#------------------------------------------------------------------------------

def close_count(states, time, catalog=False, VALVES=None, X=None):
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
        that `VALVES` and `X` are specified.
    VALVES : dictionnary (default to `None`)
        Valve characteristics dictionnary. Needed for `catalog` option
    X : 1D array (default to `None`)
        Space array. Needed for `catalog` option

    Returns
    -------
    events_t : nd array
        Event times.
    events_x : nd array
        Event locations, if catalog is True.

    """

    # Call event count
    out = event_count('close', states, time, catalog=catalog, VALVES=VALVES, X=X)

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

#-------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------

def crosscorr(sig1, sig2, dt, norm=True, no_bias=True):
    """
    Calculates the cross-correlation of two signal. Signals can be of different
    sizes, but the correlation lag might be strange in this case.

    Parameters
    ----------
    sig1, sig2 : 1D arrays
        Signals to cross-correlate, dimensions N and M.
    dt : float
        Time delta in signals.
    norm : bool (default to `True`)
        Option to normalize the cross-correlation by the product of the signals
        norm.
    no_bias : bool (default to `True`)
        Option to remove bias due to variable number of points at each lag in
        cross-correlation.

    Returns
    -------
    corr : 1D array
        Cross-correlation of the input signals, dimension N + M - 1
    lag : 1D array
        Time lag for each value of the cross-correlation, dimension N + M - 1.
        Centering around 0 is only ensured if sig1 and sig2 are the same size.

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

    # Remove bias from number of points
    if no_bias:
        bias = crosscorr_bias(sig1, sig2)
        corr = corr/bias

    # Compute time lag of autocorrelation
    lag = np.arange(0, len(sig1) + len(sig2) - 1) - (len(sig1) - 1)
    lag = lag.astype(float) * dt

    # Remove both ends of lag and correlation, to get rid of edge effects due
    # to removing the bias
    if no_bias:
        valid = (abs(lag) < 4/5*max(lag))
        lag = lag[valid]
        corr = corr[valid]

    return corr, lag


#------------------------------------------------------------------------------

def crosscorr_bias(sig1, sig2):
    """
    Computes the bias vector due to the number of points of at each lag of the
    cross correlation.

    Parameters
    ----------
    sig1 : 1D array
        First signal to correlate, dimension N.
    sig2 : 1D array
        Second signal to correlate, dimension M.

    Returns
    -------
    bias : 1D array
        The bias vector, corresponding to all values of lag (both negative and
        positive), dimension N+M-1

    """
    N = len(sig1)
    M = len(sig2)

    # Bias for part where signals overlap entirely
    bias_full = [1 for ii in range(max(M, N) - min(M, N) + 1)]

    # Bias for part before full overlap
    bias_bef = [ii/min(N, M) for ii in range(1, min(M, N))]

    # Bias for part after full overlap
    bias_aft = [ii/min(N, M) for ii in range(min(M, N) - 1, 0, -1)]

    # Assemble
    bias = np.concatenate((bias_bef, bias_full, bias_aft))

    return bias

#------------------------------------------------------------------------------


def corr_coeff(sig1, sig2, dt):
    """
    Calculates the correlation coefficient between two signals.

    The correlation coefficient is taken as the maximum of the cross-correlation
    of the two signals, the corresponding lag is also returned.

    Parameters
    ----------
    sig1, sig2 : 1D arrays
        Signals to cross-correlate, dimensions N and M.
    dt : float
        Time delta in signals.

    Returns
    -------
    cc : float
        Correlation coefficient.
    cc_lag : 1D array
        Time lag corresponding to correlation coefficient.

    """

    # Calculates the cross correlation
    corr, corr_lag = crosscorr(sig1, sig2, dt, norm=True, no_bias=False)

    # Get correlation coefficient and time lag
    cc = max(abs(corr))
    cc_lag = corr_lag[np.argmax(abs(corr))]

    return cc, cc_lag

#------------------------------------------------------------------------------


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


#-------------------------------------------------------------------------------

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
            cc, cc_lag = corr_coeff(activities[ix], activities[jx], dt)
            corr_mat[ix, jx] = corr_mat[jx, ix] = cc
            lag_mat[ix, jx] = lag_mat[jx, ix] = cc_lag

    return corr_mat, lag_mat

#-------------------------------------------------------------------------------

def calc_alpha(ev_count, dt, per_max):
    """
    Compute the slope of the event count auto-correlation spectrum.

    Parameters
    ----------
    ev_count : 1D array
        Event count, number of events in regular time bins, dimension N_bins.
    dt : float
        Size of time bin.
    per_max : float
        Maximum period to take into account in the fit.

    Returns
    -------
    sp : 1D array
        Multi-taper spectrum of the event count auto-correlation.
    per : 1D array
        Periods associated at which the amplitude spectrum is computed.
    alpha : float
        The log-log slope of the spectrum of event count auto-correlation,
        along period, not frequency.

    """
    # Compute un-bias autocorrelation
    # -------------------------------
    a_corr, _ = crosscorr(ev_count, ev_count, dt, norm=True, no_bias=True)

    # Compute its spectrum
    # --------------------
    sp, fq = mtspec(a_corr, dt, time_bandwidth=3.5, number_of_tapers=5)
    sp = np.sqrt(sp) * len(sp)  # convert PSD into amplitude

    # --> Get rid of 0 frequency
    sp = sp[fq > 0]
    fq = fq[fq > 0]

    per = 1/fq

    # Compute slope of spectrum
    # -------------------------
    log_per = np.log10(per[per < per_max])
    log_sp = np.log10(sp[per < per_max])

    p = np.polyfit(log_per, log_sp, 1)
    alpha = p[0]

    return sp, per, alpha

#-------------------------------------------------------------------------------

def detect_period(ev_count, dt):
    """
    Detect a periodicity in the activity autocorrelation. The period
    corresponds to the lag of the first peak that falls above the 99.99%
    confidence interval for a white noise.

    Parameters
    ----------
    ev_count : 1D array
        Event count, number of events in regular time bins, dimension N_bins.
    dt : float
        Size of time bin.

    Returns
    -------
    period : float
        Period detected with the autocorrelation.
    validity : float
        Estimation of the validity of the estimated period. It corresponds to a
        correlation coefficient between the count autocorrelation and a the
        autocorrelation of a sine (with the same period as the detected one).
        The lower (closer to 0) the validity is, the more the detected period
        can only be interpreted as a correlation time-scale. The higher (closer
        to 1), the more it corresponds to an actual periodicity.

    """
    # Compute un-bias autocorrelation
    # -------------------------------
    a_corr, lag = crosscorr(ev_count, ev_count, dt, norm=True, no_bias=False)

    # Smooth it
    # ---------
    smooth_len = 0.01
    a_corr = savgol_filter(a_corr, int((smooth_len/dt//2) * 2 + 1), 2)

    # Compute confidence interval
    # ---------------------------
    lvl = 1 - 1e-3
    conf = erfinv(lvl) * np.sqrt(2) / np.sqrt(len(ev_count))

    # Detect period
    # -------------
    a_corr = a_corr[lag >= 0]  # working only with positive lag
    lag = lag[lag >= 0]

    if np.any(a_corr > conf):
    # --> If some non-white noise points (above confidence interval), locate
    # them
        above = a_corr > conf
        bumps_idx = []  # list of the bumps indices above conf. int.
        bump = []  # indices of each bump above conf. int.

        for ii in range(len(above)):
            if above[ii]:  # for this index, value above conf. int.
                bump.append(ii)  # add it to the bump
            else:  # for this index, we are not in a bump (anymore)
                if len(bump) > 0:  # if we were in a bump before...
                    bumps_idx.append(bump)  # ...store it
                bump = []  # reinitialize bump

        bumps_idx = bumps_idx[1:]  # remove the 0 correlation bump

        done = False
        while (len(bumps_idx) > 0) and not done:
            first_bump = bumps_idx[0]
            if len(first_bump) > 1:  # if long bump: that's a period!
                # --> detect the period
                period = lag[first_bump][np.argmax(a_corr[first_bump])]
                # --> estimate validity
                count_time = np.arange(0, len(ev_count)*dt+1, dt)
                sine = np.sin(2*np.pi/period * count_time)
                sine_corr, _ = crosscorr(sine, sine, dt, no_bias=False)
                validity, _ = corr_coeff(sine_corr, a_corr, dt)

                # --> And we arrrre
                done = True

            else:  # if only one point in bump...
                bumps_idx = bumps_idx[1:]   # ... remove it and carry on

        if (len(bumps_idx) == 0) and not done:
        # --> If no long bumps, no significant period
            period = None
            validity = None

    else:
        # --> If none, no significant periodicity
        period = None
        validity = None

    return period, validity


#-------------------------------------------------------------------------------

def poisson_adequation(ev_count, alpha):
    """
    Performs a chi-square test to try to refute the null hypothesis: "the event
    count follows a Poisson distribution".

    Parameters
    ----------
    ev_count : 1D array
        Array containing the binned event count.
    alpha : float, lower than 1
        Risk level taken in the refutation of the null hypothesis. 1 - alpha
        corresponds to the confidence interval with which it can be refuted.

    Returns
    -------
    adequation : bool
        Answer of the test. If True, the null hypothesis cannot be refuted, the
        event count could very well be explained by a Poisson distribution. If
        False, the null hypothesis can be refuted with a level of confidence of
        1 - alpha.
    p_value : float
        p-value of test, corresponds to the probability of obtaining the test
        statistic, under realisation of the null hypothesis. Output of
        scipy.stats.chisquare().

    """
    # Build frequencies of observed count values for observed counts
    # --------------------------------------------------------------
    bins = np.arange(-0.5, max(ev_count)+1, 1)
    f, _ = np.histogram(ev_count, bins=bins)
    f = f / len(ev_count)

    # Build frequencies for a Poisson process
    # ---------------------------------------
    lambd = np.mean(ev_count)
    f_th = poisson.pmf((bins[1:]+bins[:-1])/2, lambd)

    # Perform test
    # ------------
    T, p_value = chisquare(f, f_th, 1)
#    print("chi_square_adequation -- T = {:.2e}, v = {:.2e}".format(T, \
        #                                       chi2.ppf(1 - alpha, (len(f) - 1) - 1)))

    if T > chi2.ppf(1 - alpha, (len(f) - 1) - 1):
        adequation = False
    else:
        adequation = True

    return adequation, p_value

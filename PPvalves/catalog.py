r"""Module used to produce and analyze activity catalogs.
`PPvalves.catalog` is adapted to analyse the results of PPv runs but also of
real catalogs $(t_i, x_i)$.
"""
# >> Imports
import numpy as np
from mtspec import mtspec
from scipy.special import erfinv
from scipy.signal import savgol_filter
from scipy.stats import chisquare, chi2, poisson


# >> Core
#------------------------------------------------------------------------------

def open_ratio(states):
    """
    Computes the ratio of open valves to total number of valves in time.

    Parameters
    ----------
    states: 2D array
        History of valve states, dimension `Ntimes, Nvalves`.  `False` is
        closed, `True` is open.
    Returns
    -------
    ratio : 1D array
        For each time `states` is specified, the proportion of open valves is
        returned.

    Notes
    -----
    Parameter `states` is usually taken from `run_ppv` output: `v_activity[times,0,valve_idx]`, where
    `valve_idx` selects the indices of the valves you want an event count f.
    """
    Nv = np.shape(states)[1]  # Total number of valves
    ratio = np.sum(states, axis=1) / Nv

    return ratio

#------------------------------------------------------------------------------

def event_count(ev_type, states, time, catalog=False, VALVES=None, X=None):
    """Produces a catalog of valve events (openings or closings).

    Produces a catalog of events in time --- and optionally space. The time of
    an event occuring is taken as the time at which the corresponding threshold
    of pressure differential is crossed at the given valve.

    Parameters
    ----------
    ev_type : str
        type of event to count, either `"close"`, or `"open"`.
    states: 2D array
        History of valve states, dimension `Ntimes, Nvalves`.  `False` is
        closed, `True` is open.
    time : 1D array
        Array of physical times, same shape as first dimension of `states`.
    catalog : bool, optional
        If `catalog=True`, the allows `event_count` to return
        the position of events `x_events` in addition to their time `t_event`.
        Then, it recquires that `VALVES` and `X` are specified.
    VALVES : dictionnary, optional
        Valve characteristics dictionnary. Needed for `catalog` option.
    X : 1D array, optional
        Physical sppace array. Needed for `catalog` option.

    Returns
    -------
    events_t : 1D array
        Event times.
    events_x : 1D array
        Event locations, if catalog is True.

    Notes
    -----
    Parameter `states` is usually taken from `run_ppv` output: `v_activity[times,0,valve_idx]`, where
    `valve_idx` selects the indices of the valves you want an event count f.
    """
    # >> Check if catalog option is on
    if catalog & isinstance(VALVES, type(None)) & isinstance(X, type(None)):
        raise ValueError("When catalog option is turned on, the valves'" + \
         'location needs to be specified with the v_x argument.')


    # >> For several valves
    if len(states.shape) > 1:
        n_valves = states.shape[1] # number of valves

        all_events_i = []  # init: time indices of events
        if catalog:
            v_x = X[VALVES['idx']] + VALVES['width']/2 # location of valves
            all_events_x = []  # init: locations of events

        for iv in range(n_valves):
            # -> Run through valves, for each, compute (t, x)
            v_states = states[:, iv].astype(int)  # states for this valve

            if ev_type == 'close':  # get 1 -> 0 events
                events = (v_states[1:]-v_states[:-1]) == -1
                v_events_i = np.where(events)[0]+1  # events idxs

            elif ev_type == 'open':  # get 0 -> 1 events
                events = (v_states[1:]-v_states[:-1]) == 1
                v_events_i = np.where(events)[0]+1  # events idxs

            v_events_i = v_events_i.tolist()
            all_events_i.extend(v_events_i)  # add valve events t to all events

            # -> build event location vector
            if catalog:
                v_events_x = [v_x[iv] for ii in\
            	 range(len(v_events_i))]
                all_events_x.extend(v_events_x) # add valve events t to all events

        # >> Convert event time idx to time
        events_t = time[all_events_i]
        if catalog:
            events_x = np.array(all_events_x)

        # >> Sort events t, x in chronological order and return
        if catalog:
            id_sort = np.argsort(events_t)
            events_t = events_t[id_sort]
            events_x = events_x[id_sort]
            return events_t, events_x

        else:
            events_t = np.sort(events_t)
            return events_t

    # >> For one valve only
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
    """Counts and makes a catalog of valve opening events.

    Parameters
    ----------
    states: 2D array
        History of valve states, dimension `Ntimes, Nvalves`.  `False` is
        closed, `True` is open.
    time : 1D array
        Array of physical times, same shape as first dimension of `states`.
    catalog : bool, optional
        If `catalog=True`, the allows `event_count` to return
        the position of events `x_events` in addition to their time `t_event`.
        Then, it recquires that `VALVES` and `X` are specified.
    VALVES : dictionnary, optional
        Valve characteristics dictionnary. Needed for `catalog` option.
    X : 1D array, optional
        Physical sppace array. Needed for `catalog` option.

    Returns
    -------
    events_t : 1D array
        Event times.
    events_x : 1D array
        Event locations, if catalog is True.

    Notes
    -----
        - This is a shortcut function to use instead of `event_count`.
        - Parameter `states` is usually taken from `run_ppv` output:
          `v_activity[times,0,valve_idx]`, where `valve_idx` selects the
          indices of the valves you want an event count f.

    See also
    --------
    event_count : General event count.
    """
    # >> Call event count
    out = event_count('open', states, time, catalog=catalog, VALVES=VALVES, X=X)

    return out

#------------------------------------------------------------------------------

def close_count(states, time, catalog=False, VALVES=None, X=None):
    """Counts and makes a catalog of valve opening events.

    Parameters
    ----------
    states: 2D array
        History of valve states, dimension `Ntimes, Nvalves`.  `False` is
        closed, `True` is open.
    time : 1D array
        Array of physical times, same shape as first dimension of `states`.
    catalog : bool, optional
        If `catalog=True`, the allows `event_count` to return
        the position of events `x_events` in addition to their time `t_event`.
        Then, it recquires that `VALVES` and `X` are specified.
    VALVES : dictionnary, optional
        Valve characteristics dictionnary. Needed for `catalog` option.
    X : 1D array, optional
        Physical sppace array. Needed for `catalog` option.

    Returns
    -------
    events_t : 1D array
        Event times.
    events_x : 1D array
        Event locations, if catalog is True.

    Notes
    -----
        - This is a shortcut function to use instead of `event_count`.
        - Parameter `states` is usually taken from `run_ppv` output:
          `v_activity[times,0,valve_idx]`, where `valve_idx` selects the
          indices of the valves you want an event count f.

    See also
    --------
    event_count : General event count.
    """
    # >> Call event count
    out = event_count('close', states, time, catalog=catalog, VALVES=VALVES, X=X)

    return out

#---------------------------------------------------------------------------------

def recurrence(event_time):
    """Computes time before next event for a sequence of events.

    Parameters
    ----------
    event_time : 1D array
        Event times.

    Returns
    -------
    time_before_next : 1D array
        Time before next event for the all the events *but the last one*:
        dimension is one less than `event_time`. Same unit as the event times.
    """

    time_before_next = event_time[1:] - event_time[:-1]

    return time_before_next

#-------------------------------------------------------------------------------

def event_count_signal(event_time, dt, t0=0., tn=None):
    """Computes the event count signal.

    The event count signal is a time-binned count of events. It can easily be
    linked to an activity rate, as it is the number of events in a given time
    bin across time.

    Parameters
    ----------
    event_time : 1D array
        Event times.
    dt : float
        Binning length to compute event count, final discretization of the
        event count signal.
    t0 : float, optional
        Beginning time of the first bin. If left blank, it starts at `t0=0`.
    tn : float, optional
        Beginning time of the final bin. Default is time of last event.

    Returns
    -------
    event_count_signal : 1D array
        Evenly spaced count of events in time bins of length `dt`.
    bin_edges : 1D array
        Boundaries of time bins, one element longer than event_count_signal.
    """

    # >> Check if tn is 'max' and if so set it to its value
    if tn is None:
        tn = max(event_time)

    # >> The evenly spaced count is in fact a simple histogram
    ev_count, bin_edges = np.histogram(event_time, bins=int((tn-t0)//dt+2),
                                       range=(t0, tn+dt))

    # /!\ bin_edges length is 1 unit longer than event_count

    return ev_count, bin_edges

#-------------------------------------------------------------------------------

def crosscorr(sig1, sig2, dt, norm=True, no_bias=True):
    """Homemade cross-correlation function.

    Computes the cross-correlation time-series of two signals. Function was
    designed for signals of same length. If signals of different sizes are
    used, correlation lag might be strange, use at your own risk.

    Parameters
    ----------
    sig1, sig2 : 1D arrays
        Signals to cross-correlate, dimensions N and M. Use same discretization
        lenght `dt` for best results.
    dt : float
        Time step used in both signals.
    norm : bool, optional
        By default, normalizes the signals before cross-correlating them. `norm
        = False` turns it off.
    no_bias : bool, optional
        By default, removes the bias due to the variable number of points used
        to compute the correlation at each lag. This option also removes one
        fifth of the cross-correlation at each end, to get rid of edge effects
        due to unbiasing.

    Returns
    -------
    corr : 1D array
        Cross-correlation of the input signals, dimension N + M - 1.
    lag : 1D array
        Time lag for each value of the cross-correlation, dimension N + M - 1.
        Centering around 0 is only ensured if sig1 and sig2 are the same size.
    """

    # >> Normalize signals
    if norm:
        sig1 = (sig1 - np.mean(sig1)) / (np.std(sig1)*len(sig1))
        sig2 = (sig2 - np.mean(sig2)) / np.std(sig2)

    # >> Compute correlation
    corr = np.correlate(sig1.astype(float), sig2.astype(float),'full')

    # >> Normalize
    if norm:
        corr = corr.astype(float) / (np.linalg.norm(sig1)*np.linalg.norm(sig2))

    # >> Remove bias due to number of points varying at each lag
    if no_bias:
        bias = crosscorr_bias(sig1, sig2)
        corr = corr/bias

    # >> Compute time lag of autocorrelation
    lag = np.arange(0, len(sig1) + len(sig2) - 1) - (len(sig1) - 1)
    lag = lag.astype(float) * dt

    # >> Removes both ends of lag and correlation, to get rid of edge effects
    # due to removing the bias
    if no_bias:
        valid = (abs(lag) < 4/5*max(lag))
        lag = lag[valid]
        corr = corr[valid]

    return corr, lag


#------------------------------------------------------------------------------

def crosscorr_bias(sig1, sig2):
    """Computes the bias of at each lag of the cross correlation.

    Parameters
    ----------
    sig1, sig2 : 1D array
        Signals to correlate, dimension N and M.

    Returns
    -------
    bias : 1D array
        The bias vector, corresponding to all values of lag (both negative and
        positive), dimension N + M - 1.

    """
    N = len(sig1)
    M = len(sig2)

    # >> Bias for part where signals overlap entirely
    bias_full = [1 for ii in range(max(M, N) - min(M, N) + 1)]

    # >> Bias for part before full overlap
    bias_bef = [ii/min(N, M) for ii in range(1, min(M, N))]

    # >> Bias for part after full overlap
    bias_aft = [ii/min(N, M) for ii in range(min(M, N) - 1, 0, -1)]

    # >> Concatenate
    bias = np.concatenate((bias_bef, bias_full, bias_aft))

    return bias

#------------------------------------------------------------------------------


def corr_coeff(sig1, sig2, dt):
    """Computes a correlation coefficient between two signals.

    The correlation coefficient is taken as the maximum of the cross-correlation
    of the two signals, the corresponding time lag is also returned.

    Parameters
    ----------
    sig1, sig2 : 1D arrays
        Signals to cross-correlate, dimensions N and M.
    dt : float
        Time step of signals.

    Returns
    -------
    cc : float
        Correlation coefficient.
    cc_lag : 1D array
        Time lag corresponding to correlation coefficient.
    """
    # >> Calculates the cross correlation
    corr, corr_lag = crosscorr(sig1, sig2, dt, norm=True, no_bias=False)

    # >> Get correlation coefficient and time lag
    cc = max(abs(corr))
    cc_lag = corr_lag[np.argmax(abs(corr))]

    return cc, cc_lag

#------------------------------------------------------------------------------


def act_interval_frac(event_t, tau):
    """Computes the fraction of time-interval of various length in which there
    is at least an event.

    Parameters
    ----------
    event_t : 1D array
        Array of event times, dimension `N_ev`.
    tau : 1D array
        Array of time interval lengths to test, dimension `N_tau`. If all
        values of `tau` are not multiples of `min(tau)`, the output `tau` will
        be different than the input one.

    Returns
    -------
    X_tau : 1D array
        Fraction of intervals of corresponding lengths `tau` that contain at
        least an event, dimension `N_tau`
    tau : 1D array
        Input `tau` adjusted to match the actual interval length that is
        tested, which here has to be a multiple of the min of `tau`. Dimension
        `N_tau`.

    """
    tau_min = min(tau)  # lowest tau is taken as sweeping window time increment
    count, _ = event_count_signal(event_t, tau_min, t0=min(event_t))

    X_tau = np.zeros(len(tau))

    for ii, tt in enumerate(tau):
        # >> First: build sweeping window
        len_tau_win = int(np.round(tt/tau_min))
        tau_win = np.ones(len_tau_win)  # sweeping window to count events
        tau[ii] = len_tau_win * tau_min

        # >> Second: correlate to count intervals with activity
        corr = np.correlate(count, tau_win, mode='valid')

        X_tau[ii] = 1 - np.sum(corr == 0) / len(corr)  # compute fraction

    return X_tau, tau


#-------------------------------------------------------------------------------

def correlation_matrix(event_t, event_x, dt, X):
    """Computes the correlation of activity in different zones in space.

    Parameters
    ----------
    event_t : 1D array
        Array of event times, dimension `N_ev`.
    event_x : 1D array
        Array of event positions, dimension `N_ev`.
    dt : float
        Time step to compute the activity of the different zones.
    X : 1D array
        Array of the positions of the activity zones boundaries, dimension
        `N_X`.

    Returns
    -------
    corr_mat : 2D array
        Cross-correlation coefficient matrix for the activity of the different
        regions specified by `X`, dimensions `N_X - 1`, `N_X - 1`.
    lag_mat : 2D array
        Lag time matrix corresponding to the maximum cross-correlation for the
        activity of different regions specified by `X`, dimensions `N_X - 1`, `N_X -
        1`.
    """
    # >> Compute the activity in each region
    activities = []  #  List of regional activities
    for ix in range(len(X)-1):
        region = (event_x >= X[ix]) & (event_x < X[ix + 1])
        regional_act, _ = event_count_signal(event_t[region], dt,
                                             t0=min(event_t))
        activities.append(regional_act)

    # >> Perform the cross-correlation and build the corr and lag matrices
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
    """Computes the slope of the event count auto-correlation spectrum.

    This spectral slope is a good estimator of time-clustering of a
    point-process: if it is close to 0, the point process shows less time
    clustering than if it is significantly higher [1]_.

    Parameters
    ----------
    ev_count : 1D array
        Event count, number of events in regular time bins, dimension `N_bins`.
    dt : float
        Size of time bin.
    per_max : float
        Maximum period to take into account in the fit.

    Returns
    -------
    sp : 1D array
        Multi-taper spectrum of the event count auto-correlation.
    per : 1D array
        Periods at which the amplitude spectrum is computed.
    alpha : float
        The log-log slope of the spectrum of event count auto-correlation,
        along period, not frequency.

    References
    ----------
    .. [1] Lowen, S. B., & Teich, M. C. (2005). Fractal-Based Point Processes
       (Vol. 366). John Wiley and Sons, Inc.
    """
    # >> Compute un-bias autocorrelation
    a_corr, _ = crosscorr(ev_count, ev_count, dt, norm=True, no_bias=True)

    # >> Compute its spectrum
    sp, fq = mtspec(a_corr, dt, time_bandwidth=3.5, number_of_tapers=5)
    sp = np.sqrt(sp) * len(sp)  # convert PSD into amplitude

    # -> Get rid of 0 frequency
    sp = sp[fq > 0]
    fq = fq[fq > 0]

    per = 1/fq

    # >> Compute slope of spectrum
    log_per = np.log10(per[per < per_max])
    log_sp = np.log10(sp[per < per_max])

    p = np.polyfit(log_per, log_sp, 1)
    alpha = p[0]

    return sp, per, alpha

#-------------------------------------------------------------------------------

def detect_period(ev_count, dt):
    """Detect a periodicity in an event count signal (or activity rate time
    series).

    The period is computed using the autocorrelation of the event count signal
    (smoothed over 0.01 scaled units).
    It corresponds to the lag of the first peak (more than 1 ponit) that is above
    the 99.99% confidence interval (null hypothesis: autocorrelation is
    autocorrelation of white noise).

    Parameters
    ----------
    ev_count : 1D array
        Event count, number of events in regular time bins, dimension N_bins.
    dt : float
        Size of time bin.

    Returns
    -------
    period : float
        Period detected with the autocorrelation. If no period is detected,
        returns `None`.
    validity : float
        Homemade estimation of the validity of the detected period. Low
        confidence 0, high confidence 1. See notes for more.

    Notes
    -----
    `validity` corresponds to a correlation coefficient between the count
    autocorrelation and a the autocorrelation of a sine (with the same period
    as the detected one). The lower (closer to 0) the validity is, the more
    the detected period can only be interpreted as a correlation time-scale.
    The higher (closer to 1), the more it corresponds to an actual
    periodicity (there are other bumps).

    Please note that this is not an perfectly robust function.
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
    lvl = 1 - 1e-3  #
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
    """Tests if event count signal can be considered to emerge from a Poisson
    process.

    Performs a chi-square test to try to refute the null hypothesis: "the event
    count follows a Poisson distribution".

    Parameters
    ----------
    ev_count : 1D array
        Array containing the binned event count.
    alpha : float
        Risk level taken in the refutation of the null hypothesis (should be
        lower than 1). `1 - alpha` corresponds to the confidence interval with
        which it can be refuted.

    Returns
    -------
    adequation : bool
        Answer of the test. If `True`, the null hypothesis cannot be refuted, the
        event count could very well be explained by a Poisson distribution. If
        False, the null hypothesis can be refuted with a level of confidence of
        1 - alpha.
    p_value : float
        p-value of test, corresponds to the probability of obtaining the test
        statistic, under realisation of the null hypothesis. Output of
        `scipy.stats.chisquare()`.

    See also
    --------
    scipy.stats.chisquare() : performs a chi-square test.
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

    if T > chi2.ppf(1 - alpha, (len(f) - 1) - 1):
        adequation = False
    else:
        adequation = True

    return adequation, p_value

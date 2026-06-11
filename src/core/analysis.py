import warnings
import copy
import logging

import numpy as np
from scipy.interpolate import CubicSpline as spCubicSpline

from ..utils.tools import interval_selection, piezo_selection


ana_logger = logging.getLogger("ascam.analysis")
debug_logger = logging.getLogger("ascam.debug")


def interpolate(
    signal, time, interpolation_factor
):
    """Interpolate the signal with a cubic spline."""

    spline = spCubicSpline(time, signal)
    interpolation_time = np.arange(
        time[0], time[-1], (time[1] - time[0]) / interpolation_factor
    )
    return spline(interpolation_time), interpolation_time


class Idealizer:
    """Container object for the different idealization functions."""

    @classmethod
    def idealize_episode(
        cls,
        signal,
        time,
        amplitudes,
        thresholds = None,
        resolution = None,
        interpolation_factor = 1,
    ):
        """Get idealization for single episode."""

        if thresholds is None or thresholds.size != amplitudes.size - 1:
            thresholds = (amplitudes[1:] + amplitudes[:-1]) / 2

        if interpolation_factor != 1:
            signal, time = interpolate(signal, time, interpolation_factor)

        idealization = cls.threshold_crossing(signal, amplitudes, thresholds)

        if resolution is not None:
            idealization = cls.apply_resolution(idealization, time, resolution)
        return idealization, time

    @staticmethod
    def threshold_crossing(
        signal,
        amplitudes,
        thresholds = None,
    ):
        """Perform a threshold-crossing idealization on the signal.

        Arguments:
            signal - data to be idealized
            amplitudes - amplitudes to which signal will be idealized
            thresholds - the thresholds above/below which signal is mapped
                to an amplitude"""

        amplitudes = copy.copy(
            np.sort(amplitudes)
        )  # sort amplitudes in descending order
        amplitudes = amplitudes[::-1]

        # if thresholds are not or incorrectly supplied take midpoint between
        # amplitudes as thresholds
        if thresholds is not None and (thresholds.size != amplitudes.size - 1):
            warnings.warn(
                f"Too many or too few thresholds given, there should be "
                f"{amplitudes.size - 1} but there are {thresholds.size}.\n"
                f"Thresholds = {thresholds}."
            )

            thresholds = (amplitudes[1:] + amplitudes[:-1]) / 2

        # for convenience we include the trivial case of only 1 amplitude
        if amplitudes.size == 1:
            idealization = np.ones(signal.size) * amplitudes
        else:
            idealization = np.zeros(len(signal))
            # np.where returns a tuple containing array so we have to get the
            # first element to get the indices
            inds = np.where(signal > thresholds[0])[0]
            idealization[inds] = amplitudes[0]
            for thresh, amp in zip(thresholds, amplitudes[1:]):
                inds = np.where(signal < thresh)[0]
                idealization[inds] = amp

        return idealization

    @staticmethod
    def apply_resolution(
        idealization, time, resolution
    ):
        """Remove from the idealization any events that are too short.

        Args:
            idealization - an idealized current trace
            time - the corresponding time array
            resolution - the minimum duration for an event"""
        ana_logger.debug(f"Apply resolution={resolution}.")

        events = Idealizer.extract_events(idealization, time)

        i = 0
        end_ind = len(events[:, 1])
        while i < end_ind:
            if events[i, 1] < resolution:
                i_start = int(np.where(time == events[i, 2])[0])
                i_end = int(np.where(time == events[i, 3])[0]) + 1
                # add the first but not the last event to the next,
                # otherwise, flip a coin
                if (np.random.binomial(1, 0.5) or i == 0) and i != end_ind - 1:
                    i_end = int(np.where(time == events[i + 1, 3])[0]) + 1
                    idealization[i_start:i_end] = events[i + 1, 0]
                    # set amplitude
                    events[i, 0] = events[i + 1, 0]
                    # add duration
                    events[i, 1] += events[i + 1, 1]
                    # set end_time
                    events[i, 3] = events[i + 1, 3]
                    # delete next event
                    events = np.delete(events, i + 1, axis=0)
                else:  # add to the previous event
                    i_start = int(np.where(time == events[i - 1, 2])[0])
                    idealization[i_start:i_end] = events[i - 1, 0]
                    # add duration
                    events[i - 1, 1] += events[i, 1]
                    # set end_time
                    events[i - 1, 3] = events[i, 3]
                    # delete current event
                    events = np.delete(events, i, axis=0)
                # now one less event to iterate over
                end_ind -= 1
            else:
                i += 1
        if np.any(Idealizer.extract_events(idealization, time)[:, 1] < resolution):
            ana_logger.warning(
                "Filter events below the resolution failed! Some events are still too short."
            )
        return idealization

    @staticmethod
    def extract_events(
        idealization, time
    ):
        """Summarize an idealized trace as a list of events.

        Args:
            idealization [1D numpy array] - an idealized current trace
            time [1D numpy array] - the corresponding time array
        Return:
            event_list [4D numpy array] - an array containing the amplitude of
                the event, its duration, the time it starts and the time it
                end in its columns"""

        events = np.where(idealization[1:] != idealization[:-1])[0]
        # events = events.astype(int)
        # events+1 marks the indices of the last time point of an event
        # starting from 0 to events[0] is the first event, from events[0]+1
        # to events[1] is the second...  and from events[-1]+1 to
        # t_end is the last event, hence
        n_events = events.size + 1
        # init the array that will be final output table, events in rows and
        # amplitude, duration, start and end in columns
        event_list = np.zeros((n_events, 4))
        # fill the array
        if n_events == 1:
            event_list[0][0] = idealization[0]
            event_list[0][2] = time[0]
            event_list[0][3] = time[-1]
        else:
            event_list[0][0] = idealization[0]
            event_list[0][2] = time[0]
            event_list[0][3] = time[int(events[0])]

            event_list[1:, 0] = idealization[events + 1]
            event_list[1:, 2] = time[events + 1]
            event_list[1:-1, 3] = time[events[1:]]

            event_list[-1][0] = idealization[int(events[-1]) + 1]
            event_list[-1][2] = time[(int(events[-1])) + 1]
            event_list[-1][3] = time[-1]
        # get the duration column
        # because the start and end times of events are inclusive bounds
        # ie [a,b] the length is b-a+1, so we need to add to each event the
        # sampling interval
        sampling_interval = time[1] - time[0]
        event_list[:, 1] = event_list[:, 3] - event_list[:, 2] + sampling_interval
        return event_list


def detect_first_activation(
    time, signal, threshold
):
    """Return the time where a signal first crosses below a threshold."""

    return time[np.argmax(signal < threshold)]


def detect_first_events(
        time, signal, threshold, piezo, idealization, states
):
    """Return the first activation time and first event at each state.
    first_activation: float
    first_events: 2xnstates matrix with start time and duration of the first
    event in each state.
    """

    first_activation = time[np.argmax(signal < threshold)]
    piezo_time, _ = piezo_selection(time, piezo, signal)

    events_list = Idealizer.extract_events(idealization, time)
    first_events = -np.ones((2, len(states)))
    exit_time = max(piezo_time[0], first_activation)
    # We skip events before first activation time and before piezo
    events_list = events_list[events_list[:, 2] >= exit_time, :]
    for i, state in enumerate(states):
        event_ids = np.where(events_list[:, 0] == state)[0]
        if len(event_ids) > 0:
            event_id = min(event_ids)
        else:
            continue
        event_start = events_list[event_id, 2]
        event_duration = events_list[event_id, 1]
        first_events[:, i] = [ event_start, event_duration ]
    first_events[first_events == -1] = None
    return first_activation, first_events


def baseline_correction(
    time,
    signal,
    sampling_rate,
    intervals = None,
    degree = 1,
    method = "Polynomial",
    piezo = None,
    selection = "piezo",
    active = False,
    deviation = 0.05,
):
    """Perform polynomial/offset baseline correction on the given signal.

    Parameters:
        time - 1D array containing times of the measurements in signal
               units of `time_unit`
        signal - time series of measurements
        intervals - interval or list of intervals from which to
                   estimate the baseline (in ms)
        sampling_rate - sampling frequency (in Hz)
        time_unit - units of the time vector, 'ms' or 's'
        method - `baseline` can subtract a fitted polynomial of
                 desired degree OR subtract the mean
        degree - if method is 'poly', the degree of the polynomial
    Returns:
        original signal less the fitted baseline"""

    if selection.lower() == "intervals":
        t, s = interval_selection(time, signal, intervals, sampling_rate)
    elif selection.lower() == "piezo":
        t, s = piezo_selection(time, piezo, signal, active, deviation)
    else:
        t = time
        s = signal

    if method.lower() == "offset":
        offset = np.mean(s)
        output = signal - offset
    elif method.lower() == "polynomial":
        coeffs = np.polyfit(t, s, degree)
        baseline = np.zeros_like(time)
        for i in range(degree + 1):
            baseline += coeffs[i] * (time ** (degree - i))
        output = signal - baseline
    return output


def _percentile_baseline_segment(signal, window_samples, percentile, step):
    """Estimate the running-percentile baseline of one contiguous block.

    The percentile is evaluated on a coarse grid of window centres and then
    linearly interpolated back to full resolution; this is smooth and fast (it
    avoids running a rank filter over every one of the (possibly tens of
    millions of) samples in a long recording).

    Parameters:
        signal [1D array] - the block to estimate the baseline of
        window_samples [int] - sliding-window width in samples
        percentile [float] - percentile (0-100) tracking the closed level
        step [int] - spacing of the grid of window centres in samples
    Returns:
        baseline [1D array] - the estimated baseline, same length as signal"""

    n = signal.size
    if n == 0:
        return np.array([], dtype=float)
    half = window_samples // 2
    centres = np.arange(0, n, step)
    if centres[-1] != n - 1:
        centres = np.append(centres, n - 1)
    values = np.empty(centres.size, dtype=float)
    for i, c in enumerate(centres):
        low = max(0, c - half)
        high = min(n, c + half + 1)
        values[i] = np.percentile(signal[low:high], percentile)
    return np.interp(np.arange(n), centres, values)


def running_percentile_baseline(
    time,
    signal,
    sampling_rate,
    window_duration,
    percentile=50,
    segment_boundaries=None,
):
    """Subtract a running-percentile estimate of the closed-channel baseline.

    A percentile of the signal in a sliding window tracks the closed (baseline)
    level even as it drifts, because channel openings are brief and sparse and
    so do not dominate the chosen percentile. This is well suited to continuous
    bilayer recordings whose baseline wanders over time.

    Choosing the percentile: the closed level is the median of the whole
    distribution whenever the channel is open less than half the time, so the
    50th percentile (the default) tracks the baseline without bias for either
    polarity at low open probability. Shift the percentile toward the closed
    side only as the open probability Po grows: the unbiased value is
    ~(50 + 50*Po) for inward (negative-going) openings and ~(50 - 50*Po) for
    outward (positive-going) openings. (A very high/low percentile such as 90/10
    is only appropriate when the channel is open most of the time.)

    A plain sliding percentile smears across a sudden baseline jump (it ramps
    over roughly one window width). If `segment_boundaries` is supplied (e.g.
    from `detect_baseline_jumps`), the baseline is estimated independently within
    each segment so it snaps at the jumps instead of bleeding across them.

    Parameters:
        time [1D array] - time vector (unused for the estimate, kept for a
            signature consistent with `baseline_correction`)
        signal [1D array] - the current trace
        sampling_rate [float] - sampling rate in Hz
        window_duration [float] - sliding-window width in seconds
        percentile [float] - percentile (0-100) tracking the closed level
        segment_boundaries [sequence of int or None] - sample indices at which
            the baseline jumps; the trace is split there and each segment is
            corrected on its own
    Returns:
        the signal with the running-percentile baseline subtracted"""

    signal = np.asarray(signal, dtype=float)
    window_samples = max(1, int(round(window_duration * sampling_rate)))
    step = max(1, window_samples // 4)

    # build the list of segment edges [0, b0, b1, ..., N]
    n = signal.size
    if segment_boundaries is None or len(segment_boundaries) == 0:
        edges = [0, n]
    else:
        interior = sorted({int(b) for b in segment_boundaries if 0 < b < n})
        edges = [0] + interior + [n]

    baseline = np.empty(n, dtype=float)
    for start, end in zip(edges[:-1], edges[1:]):
        baseline[start:end] = _percentile_baseline_segment(
            signal[start:end], window_samples, percentile, step
        )
    return signal - baseline


def detect_baseline_jumps(
    signal,
    sampling_rate,
    percentile=50,
    downsample_hz=100,
    sensitivity=1.0,
    min_duration=0.05,
    min_jump_size=None,
):
    """Detect sudden baseline jumps with the PELT changepoint algorithm.

    To stay fast and to decouple detection from changes in open probability, the
    algorithm does not work on the raw signal. Instead it builds a robust
    closed-level series by taking the given percentile of the signal in
    consecutive blocks (one block per 1/`downsample_hz` seconds) and runs PELT
    on that. A baseline jump shifts the closed level; merely having more or
    longer openings does not, so it is not mistaken for a jump.

    PELT with a piecewise-constant (L2) cost will try to approximate smooth
    drift with a staircase of small steps, so PELT alone over-segments a
    drifting baseline. To separate genuine jumps from drift, PELT is used only
    to propose candidate change points, and a candidate is kept only if the
    closed level actually steps across it by at least a magnitude threshold.
    Drift produces sub-threshold steps and is rejected; a real multi-pA jump is
    kept.

    Parameters:
        signal [1D array] - the current trace (original sampling rate)
        sampling_rate [float] - sampling rate in Hz
        percentile [float] - percentile tracking the closed level (match the one
            used for the baseline correction; see running_percentile_baseline)
        downsample_hz [float] - rate of the closed-level series PELT runs on
        sensitivity [float] - scales the automatic magnitude threshold; larger
            values detect more (smaller) jumps
        min_duration [float] - minimum segment duration in seconds
        min_jump_size [float or None] - minimum step in signal units to count as
            a jump; if None an automatic threshold (~6 robust sigma of the
            closed level, scaled by sensitivity) is used
    Returns:
        jump_indices [1D int array] - jump locations as sample indices in the
            original sampling rate (empty if none are found)"""

    import ruptures

    signal = np.asarray(signal, dtype=float)
    factor = max(1, int(round(sampling_rate / downsample_hz)))
    n_blocks = signal.size // factor
    if n_blocks < 4:
        return np.array([], dtype=int)

    # robust closed-level series: chosen percentile within each block
    blocks = signal[: n_blocks * factor].reshape(n_blocks, factor)
    closed = np.percentile(blocks, percentile, axis=1)

    # robust noise scale of the *raw* signal (MAD of its first difference).
    # A jump worth correcting is several pA, i.e. a few times this noise; the
    # closed-level series itself is far too smooth to set a physical threshold.
    raw_diffs = np.diff(signal)
    sigma_raw = (
        1.4826 * np.median(np.abs(raw_diffs - np.median(raw_diffs))) / np.sqrt(2)
        if raw_diffs.size
        else 0.0
    )
    sigma_raw = max(sigma_raw, 1e-30)

    if min_jump_size is None:
        threshold = 4.0 * sigma_raw / max(sensitivity, 1e-6)
    else:
        threshold = float(min_jump_size)

    # PELT proposes candidates; a low penalty keeps the real jump among them.
    # Penalty is scaled to the closed-level noise so candidates are plentiful.
    min_size = max(1, int(round(min_duration * downsample_hz)))
    closed_diffs = np.diff(closed)
    closed_sigma = (
        1.4826 * np.median(np.abs(closed_diffs - np.median(closed_diffs)))
        if closed_diffs.size
        else 0.0
    )
    penalty = max((closed_sigma ** 2) * np.log(n_blocks), 1e-12 * (np.ptp(closed) ** 2 + 1.0))
    algo = ruptures.Pelt(model="l2", min_size=min_size).fit(closed.reshape(-1, 1))
    candidates = [b for b in algo.predict(pen=penalty) if 0 < b < n_blocks]

    # keep only candidates where the closed level steps by >= threshold,
    # comparing the median just before and just after the candidate
    kept = []
    for b in candidates:
        before = np.median(closed[max(0, b - min_size) : b])
        after = np.median(closed[b : min(n_blocks, b + min_size)])
        magnitude = abs(after - before)
        if magnitude >= threshold:
            kept.append((b, magnitude))

    # merge surviving candidates closer than one minimum segment, keeping the
    # strongest, so a single jump yields a single boundary
    kept.sort()
    merged = []
    for b, magnitude in kept:
        if merged and b - merged[-1][0] < min_size:
            if magnitude > merged[-1][1]:
                merged[-1] = (b, magnitude)
        else:
            merged.append((b, magnitude))

    jumps = sorted(b * factor for b, _ in merged)
    return np.array(jumps, dtype=int)

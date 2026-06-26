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
        level_contribution = 0.1,
        track_mode = "off",
        region = None,
    ):
        """Get idealization for single episode.

        Optional ClampFit-style additions (both default to the original
        behavior):
            level_contribution, track_mode - per-event level tracking, see
                `track_levels`. track_mode is "off", "baseline" or "all".
            region - a (t0, t1) tuple (in the same units as `time`). When given,
                only the samples inside the span are idealized, so the returned
                idealization/time cover the region only.
        """

        # restrict to the cursor-selected detection region, if any
        if region is not None:
            lo, hi = np.searchsorted(time, sorted(region))
            if hi - lo >= 2:  # need at least two samples to idealize
                signal = signal[lo:hi]
                time = time[lo:hi]

        if thresholds is None or thresholds.size != amplitudes.size - 1:
            thresholds = (amplitudes[1:] + amplitudes[:-1]) / 2

        if interpolation_factor != 1:
            signal, time = interpolate(signal, time, interpolation_factor)

        idealization = cls.threshold_crossing(signal, amplitudes, thresholds)

        if track_mode and track_mode != "off":
            idealization = cls.track_levels(
                signal, idealization, amplitudes, level_contribution, track_mode
            )

        if resolution is not None:
            idealization = cls.apply_resolution(idealization, time, resolution)
        # also return the (possibly region-sliced / interpolated) signal so the
        # event amplitudes can be measured against the exact samples used
        return idealization, time, signal

    @staticmethod
    def track_levels(
        signal,
        idealization,
        amplitudes,
        level_contribution=0.1,
        track_mode="baseline",
        n_iter=3,
    ):
        """ClampFit-style per-event level tracking.

        Detection stays half-amplitude, but the level estimates (and therefore
        the half-amplitude thresholds between them) are allowed to follow slow
        drift instead of being fixed. This lets a closed level that drifts toward
        the open level still be detected correctly, rather than crossing a fixed
        threshold and being mistaken for openings.

        The method alternates two cheap, vectorized steps until the idealization
        stops changing (a few iterations):

        1. Walk the current segments (runs of constant amplitude) left to right.
           For each segment, record the half-amplitude thresholds currently in
           force, classify it by nearest running estimate, and update that
           estimate by a per-event exponentially weighted step:

               estimate[level] = (1 - c) * estimate[level] + c * (segment mean)

        2. Re-detect the whole trace using those per-sample (drifting)
           thresholds.

        Updates are per event (one per dwell), matching the ClampFit "level
        contribution" knob. The returned idealization uses the *nominal* user
        amplitudes (not the drifting estimates), so downstream event grouping,
        histograms and amplitude lines keep working unchanged.

        Args:
            signal - the current trace the idealization was computed from
            idealization - the initial threshold-crossing idealization
            amplitudes - the user level amplitudes; sorted descending the most
                positive (index 0) is treated as the closed/baseline level
            level_contribution - c above, the fraction each event contributes to
                its level's running average (typically 0.1-0.2)
            track_mode - "baseline" updates only the closed level; "all" updates
                whichever level each event is assigned to
            n_iter - maximum detect/track iterations (converges in a few)
        Returns:
            the re-detected idealization array (same shape as the input)
        """
        amplitudes = np.sort(amplitudes)[::-1]  # descending; [0] = closed level
        n_levels = amplitudes.size
        if n_levels < 2:
            # only one level: nothing to threshold or track
            return idealization

        ideal = idealization
        for _ in range(n_iter):
            # segment boundaries: start index of each run, plus one-past-the-end
            change_inds = np.where(ideal[1:] != ideal[:-1])[0] + 1
            starts = np.concatenate(([0], change_inds))
            ends = np.concatenate((change_inds, [ideal.size]))  # exclusive
            seg_lengths = ends - starts

            # per-segment means of the signal in one vectorized pass
            cumsum = np.concatenate(([0.0], np.cumsum(signal, dtype=float)))
            seg_means = (cumsum[ends] - cumsum[starts]) / seg_lengths

            # causal per-event pass: evolve the level estimates and record the
            # thresholds in force during each segment (loop is O(events))
            est = amplitudes.astype(float).copy()
            seg_thresholds = np.empty((starts.size, n_levels - 1))
            for i, mean in enumerate(seg_means):
                seg_thresholds[i] = (est[1:] + est[:-1]) / 2  # half-amplitude
                level = int(np.argmin(np.abs(est - mean)))
                if track_mode == "all" or level == 0:
                    est[level] = (
                        1 - level_contribution
                    ) * est[level] + level_contribution * mean

            # re-detect with the (drifting) per-sample thresholds, vectorized
            thr_per_sample = np.repeat(seg_thresholds, seg_lengths, axis=0)
            new_ideal = np.full(signal.size, amplitudes[0], dtype=float)
            for k in range(n_levels - 1):
                new_ideal[signal < thr_per_sample[:, k]] = amplitudes[k + 1]

            if np.array_equal(new_ideal, ideal):
                ideal = new_ideal
                break
            ideal = new_ideal
        return ideal

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
                # Deterministic dead-time imposition (Colquhoun & Sigworth):
                # a sub-resolution interval is concatenated with the PRECEDING
                # resolvable interval. The first interval has no predecessor, so
                # it is concatenated with the following one instead (the only
                # deterministic choice at the start of the trace).
                if i == 0:
                    if end_ind == 1:
                        # A single event spans the whole trace; nothing to merge.
                        break
                    # merge the (too-short) first event into the next one
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
                else:  # add to the preceding event
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
        idealization, time, signal=None
    ):
        """Summarize an idealized trace as a list of events.

        Args:
            idealization [1D numpy array] - an idealized current trace
            time [1D numpy array] - the corresponding time array
            signal [1D numpy array, optional] - the current trace the
                idealization was measured from. When given, the mean current of
                this signal over each event is appended as a fifth column.
        Return:
            event_list [2D numpy array] - one row per event with columns
                [amplitude, duration, t_start, t_end] (the idealized amplitude),
                plus a fifth [measured_amplitude] column when `signal` is given."""

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

        if signal is None:
            return event_list

        # append the mean current of `signal` over each event. The event sample
        # spans are [start, end] inclusive: event 0 is [0, events[0]], event j is
        # [events[j-1]+1, events[j]], and the last is [events[-1]+1, N-1].
        if n_events == 1:
            starts = np.array([0])
            ends = np.array([idealization.size - 1])
        else:
            starts = np.concatenate(([0], events + 1))
            ends = np.concatenate((events, [idealization.size - 1]))
        cumsum = np.concatenate(([0.0], np.cumsum(signal, dtype=float)))
        measured = (cumsum[ends + 1] - cumsum[starts]) / (ends + 1 - starts)
        return np.column_stack((event_list, measured))


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

import pytest
import numpy as np

from src.core.idealization import Idealizer


# (trace, events)
test_traces = [
    (
        np.array([1, 1, 2, 1, 1, 1], dtype=float),
        np.array([[1, 2, 0, 1], 
                  [2, 1, 2, 2], 
                  [1, 3, 3, 5]], dtype=float),
    ),
    (
        np.array([1, 1, 1, 2, 2, 3], dtype=float),
        np.array([[1, 3, 0, 2], 
                  [2, 2, 3, 4], 
                  [3, 1, 5, 5]], dtype=float),
    ),
    (
        np.array([2, 1, 1, 2, 2, 3], dtype=float),
        np.array([[2, 1, 0, 0], 
                  [1, 2, 1, 2], 
                  [2, 2, 3, 4], 
                  [3, 1, 5, 5]], dtype=float),
    ),
    (
        np.array([2, 1, 1, 2, 2, 3, 3], dtype=float),
        np.array([[2, 1, 0, 0], 
                  [1, 2, 1, 2], 
                  [2, 2, 3, 4], 
                  [3, 2, 5, 6]], dtype=float),
    ),
]

resolution_test_event_series = [
    (
        2,
        np.array([1, 1, 1, 2, 2, 3]),
    ),
    (
        2,
        np.array([2, 1, 1, 2, 2, 3]),
    ),
    (
        4,
        np.array([2,2,2,2,1,2,3,3,3,4,6,6,6,6,6,6,1,1,1,1 , 1, 2, 2, 3, 2,2,2,2,5,5,5,5,5,5,3]),
    ),
    (
        4,
        np.array([2, 1, 1, 2, 2, 3, 3,3,3,3,3,3,3, 5,5,5,5,5, 2,2,2,2,2,2,2, 1,1,1,1,1,1,1]),
    ),
    (
        4,
        np.array([2, 1, 1, 2, 2, 3, 3]),
    ),
]


@pytest.mark.parametrize("resolution, trace", resolution_test_event_series)
def test_extract_events_with_resolution(resolution, trace):
    idealization = Idealizer.apply_resolution(trace, np.arange(len(trace)), resolution)
    out = Idealizer.extract_events(idealization, np.arange(len(idealization)))
    assert np.all(out[:, 1] >= resolution)

@pytest.mark.parametrize("trace, events", test_traces)
def test_extract_events(trace, events):
    out = Idealizer.extract_events(trace, np.arange(len(trace)))
    print(out)
    print(events)
    assert np.all(out == events)


# --- Deterministic dead-time imposition (Colquhoun & Sigworth) ---------------
# apply_resolution mutates its input, so every call below is given a fresh copy.


def test_resolution_two_state_brief_opening_merges_into_closed():
    """A brief opening flanked by the closed level disappears into it."""
    trace = np.array([1, 1, 1, 2, 1, 1, 1], dtype=float)
    time = np.arange(len(trace), dtype=float)
    out = Idealizer.apply_resolution(trace.copy(), time, resolution=2)
    # the whole trace collapses to the closed level
    assert np.all(out == 1)


def test_resolution_merges_into_preceding_interval():
    """A sub-resolution interval between two *different* levels is concatenated
    with the PRECEDING interval, not the following one."""
    trace = np.array([1, 1, 1, 2, 3, 3, 3], dtype=float)
    time = np.arange(len(trace), dtype=float)
    out = Idealizer.apply_resolution(trace.copy(), time, resolution=2)
    events = Idealizer.extract_events(out, time)
    # brief "2" (one sample) is absorbed by the preceding "1", giving
    # [1 (dur 4), 3 (dur 3)] -- the merged sample takes amplitude 1, not 3.
    assert np.array_equal(events[:, 0], np.array([1, 3]))
    assert np.array_equal(events[:, 1], np.array([4, 3]))


def test_resolution_is_deterministic_across_calls():
    """The old coin-flip merge made multi-level idealizations vary run to run;
    the deterministic rule must give identical results every time."""
    trace = np.array([1, 1, 1, 2, 3, 3, 3, 1, 4, 4, 4], dtype=float)
    time = np.arange(len(trace), dtype=float)
    results = [
        Idealizer.apply_resolution(trace.copy(), time, resolution=2)
        for _ in range(8)
    ]
    for r in results[1:]:
        assert np.array_equal(r, results[0])


def test_resolution_first_event_merges_forward():
    """The first interval has no predecessor, so a too-short first interval is
    concatenated with the following one (taking its amplitude)."""
    trace = np.array([2, 1, 1, 1], dtype=float)
    time = np.arange(len(trace), dtype=float)
    out = Idealizer.apply_resolution(trace.copy(), time, resolution=2)
    assert np.all(out == 1)


def test_resolution_single_event_left_untouched():
    """A single event spanning the whole trace cannot be merged; leave it."""
    trace = np.array([2, 2], dtype=float)
    time = np.arange(len(trace), dtype=float)
    out = Idealizer.apply_resolution(trace.copy(), time, resolution=5)
    assert np.array_equal(out, np.array([2, 2], dtype=float))


# --- ClampFit-style level tracking + detection region ------------------------

# A drifting closed level interleaved with clean -2 openings. The closed level
# drifts 0 -> -0.4 -> -0.8 -> -1.2; the last closed segment (samples 60:70) has
# drifted past the static half-amplitude threshold (-1) and would be mistaken
# for an opening without level tracking.
_DRIFT_SEGMENTS = [0.0, -2.0, -0.4, -2.0, -0.8, -2.0, -1.2, -2.0]
_DRIFT_AMPS = np.array([0.0, -2.0])


def _drift_trace():
    signal = np.repeat(_DRIFT_SEGMENTS, 10).astype(float)
    time = np.arange(signal.size, dtype=float)
    return signal, time


def test_idealize_episode_backward_compatible():
    """With tracking off and no region, the pipeline matches plain
    threshold crossing."""
    signal, time = _drift_trace()
    out, out_time, _ = Idealizer.idealize_episode(signal, time, _DRIFT_AMPS)
    expected = Idealizer.threshold_crossing(
        signal, _DRIFT_AMPS, np.array([-1.0])
    )
    assert np.array_equal(out, expected)
    assert np.array_equal(out_time, time)


def test_static_detection_mislabels_drifted_closed_level():
    """Sanity: without tracking the drifted closed segment is called open."""
    signal, time = _drift_trace()
    out, _, _ = Idealizer.idealize_episode(signal, time, _DRIFT_AMPS)
    assert np.all(out[60:70] == -2.0)  # wrongly detected as an opening


def test_baseline_tracking_corrects_drift():
    """Baseline tracking with enough contribution recovers the drifted closed
    segment as closed, while genuine openings stay open."""
    signal, time = _drift_trace()
    out, _, _ = Idealizer.idealize_episode(
        signal, time, _DRIFT_AMPS, track_mode="baseline", level_contribution=0.5
    )
    assert np.all(out[60:70] == 0.0)   # corrected to closed
    assert np.all(out[10:20] == -2.0)  # real opening untouched
    assert np.all(out[0:10] == 0.0)


def test_low_level_contribution_lags_behind_drift():
    """A small contribution updates the baseline too slowly to keep up, so the
    drifted segment is still mislabeled (demonstrates the knob matters)."""
    signal, time = _drift_trace()
    out, _, _ = Idealizer.idealize_episode(
        signal, time, _DRIFT_AMPS, track_mode="baseline", level_contribution=0.1
    )
    assert np.all(out[60:70] == -2.0)


def test_track_all_mode_smoke():
    """'all' mode runs and still classifies clear closed/open stretches."""
    signal, time = _drift_trace()
    out, _, _ = Idealizer.idealize_episode(
        signal, time, _DRIFT_AMPS, track_mode="all", level_contribution=0.5
    )
    assert out.shape == signal.shape
    assert np.all(out[0:10] == 0.0)    # clearly closed
    assert np.all(out[10:20] == -2.0)  # clearly open


def test_idealize_region_restricts_to_span():
    """A detection region slices the idealization to the cursor span."""
    signal = np.zeros(100, dtype=float)
    signal[40:60] = -2.0
    time = np.arange(100, dtype=float)
    out, out_time, _ = Idealizer.idealize_episode(
        signal, time, _DRIFT_AMPS, region=(20, 50)
    )
    assert out.size == 30
    assert out_time[0] == 20 and out_time[-1] == 49
    assert np.all(out[:20] == 0.0)     # original samples 20:40, closed
    assert np.all(out[20:] == -2.0)    # original samples 40:50, open


# --- Measured (mean) amplitude per event -------------------------------------


def test_extract_events_without_signal_unchanged():
    """Without a signal, extract_events still returns the 4-column table."""
    idealization = np.array([1, 1, 2, 2], dtype=float)
    time = np.arange(idealization.size, dtype=float)
    out = Idealizer.extract_events(idealization, time)
    assert out.shape == (2, 4)


def test_extract_events_appends_measured_mean_amplitude():
    """With a signal, a fifth column holds the mean current of each event."""
    idealization = np.array([1, 1, 2, 2], dtype=float)
    signal = np.array([1.0, 1.2, 2.0, 2.4], dtype=float)
    time = np.arange(idealization.size, dtype=float)
    out = Idealizer.extract_events(idealization, time, signal)
    assert out.shape == (2, 5)
    # first four columns match the no-signal call
    base = Idealizer.extract_events(idealization, time)
    assert np.array_equal(out[:, :4], base)
    # measured column = per-event means
    assert np.allclose(out[:, 4], [1.1, 2.2])


def test_measured_amplitude_three_events():
    """Means are computed over the correct (inclusive) sample spans."""
    idealization = np.array([0, 0, 0, 5, 5, 0], dtype=float)
    signal = np.array([0.0, 1.0, 2.0, 4.0, 6.0, -3.0], dtype=float)
    time = np.arange(idealization.size, dtype=float)
    out = Idealizer.extract_events(idealization, time, signal)
    # events: [0:3] mean (0+1+2)/3=1.0 ; [3:5] mean (4+6)/2=5.0 ; [5] mean -3.0
    assert np.allclose(out[:, 4], [1.0, 5.0, -3.0])

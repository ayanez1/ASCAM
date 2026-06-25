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

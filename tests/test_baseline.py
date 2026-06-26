import numpy as np
import pytest

from src.core.analysis import _piecewise_offset_baseline, baseline_correction_jumps


def test_piecewise_offset_flattens_a_known_step():
    """A two-level signal split at the known boundary collapses to ~0 in both
    segments (the per-segment offset is removed)."""
    signal = np.concatenate([np.zeros(100), np.full(100, 5.0)])
    out = _piecewise_offset_baseline(signal, boundaries=[100], percentile=50)
    assert np.allclose(out[:100], 0.0)
    assert np.allclose(out[100:], 0.0)


def test_piecewise_offset_preserves_within_segment_amplitudes():
    """Offsets are removed but the shape within each segment is preserved
    (open and closed shift together)."""
    seg1 = np.array([0, 0, 0, 2, 0, 0], dtype=float)          # baseline 0, one opening
    seg2 = np.array([5, 5, 5, 7, 5, 5], dtype=float)          # baseline 5, one opening
    signal = np.concatenate([seg1, seg2])
    out = _piecewise_offset_baseline(signal, boundaries=[len(seg1)], percentile=50)
    # each segment's closed level -> 0, opening retains its +2 amplitude
    assert np.allclose(out[:len(seg1)], [0, 0, 0, 2, 0, 0])
    assert np.allclose(out[len(seg1):], [0, 0, 0, 2, 0, 0])


def test_piecewise_offset_no_boundaries_is_single_segment():
    """With no boundaries the whole trace is one segment (its offset removed)."""
    signal = np.full(50, 3.0)
    out = _piecewise_offset_baseline(signal, boundaries=[], percentile=50)
    assert np.allclose(out, 0.0)


def test_baseline_correction_jumps_detects_and_flattens_step():
    """End-to-end: PELT finds one step in a noisy trace and the two segments
    collapse to a common baseline. Requires the optional `ruptures` package."""
    pytest.importorskip("ruptures")
    rng = np.random.default_rng(0)
    sampling_rate = 20_000.0
    n = 40_000  # 2 s
    signal = rng.normal(0.0, 0.2, n)
    signal[n // 2:] += 5.0  # a clean +5 baseline step halfway through
    corrected, boundaries = baseline_correction_jumps(
        signal, sampling_rate, percentile=50, sensitivity=1.0
    )
    assert len(boundaries) == 1
    # both halves now sit near zero
    assert abs(np.median(corrected[: n // 2])) < 0.1
    assert abs(np.median(corrected[n // 2:])) < 0.1

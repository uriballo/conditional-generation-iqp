import numpy as np
import pytest

from iqp_cond.interval_encoding import (
    baseline_angle,
    build_inference_ensemble,
    build_interval_encoding,
)


def test_zero_width_equals_baseline():
    conditions = np.array([0.0, 0.2, 1.0])
    angles = build_interval_encoding(conditions, 4, half_width=0.0, seed=7)
    expected = np.broadcast_to(baseline_angle(conditions)[:, None], angles.shape)
    assert np.array_equal(angles, expected)


def test_bounds_and_reproducibility():
    conditions = np.array([0.1, 0.5, 0.9])
    widths = np.array([0.01, 0.2, 0.05])
    first = build_interval_encoding(conditions, 6, widths, seed=11)
    second = build_interval_encoding(conditions, 6, widths, seed=11)
    assert np.array_equal(first, second)
    center = baseline_angle(conditions)[:, None]
    assert np.all(first >= center - widths[:, None])
    assert np.all(first <= center + widths[:, None])


def test_shape_and_validation():
    assert build_interval_encoding([0.1, 0.2], 3).shape == (2, 3)
    with pytest.raises(ValueError):
        build_interval_encoding([], 3)
    with pytest.raises(ValueError):
        build_interval_encoding([1.1], 3)
    with pytest.raises(ValueError):
        build_interval_encoding([0.1], 0)
    with pytest.raises(ValueError):
        build_interval_encoding([0.1, 0.2], 2, [-1.0, 0.0])
    with pytest.raises(ValueError):
        build_interval_encoding([0.1, 0.2], 2, [0.1])


def test_ensemble_shape_and_reproducibility():
    ensemble = build_inference_ensemble([0.1, 0.4], 5, 7, 0.1, seed=3)
    repeat = build_inference_ensemble([0.1, 0.4], 5, 7, 0.1, seed=3)
    assert ensemble.shape == (7, 2, 5)
    assert np.array_equal(ensemble, repeat)

"""Tests for exact-distribution metrics used by the Gaussian studies."""

from __future__ import annotations

import numpy as np

from experiments.common.metrics import (
    bootstrap_mean_interval,
    complement_tvd,
    hamming_profile,
    paired_bootstrap_interval,
    within_shell_residual,
)


def test_hamming_profile_collapses_state_probabilities_by_weight() -> None:
    probabilities = np.asarray([0.1, 0.2, 0.3, 0.4])
    assert np.allclose(hamming_profile(probabilities, 2), [0.1, 0.5, 0.4])


def test_within_shell_residual_known_cases() -> None:
    uniform = np.full(4, 0.25)
    skewed = np.asarray([0.4, 0.1, 0.25, 0.25])
    assert within_shell_residual(uniform, 2) == 0.0
    assert np.isclose(within_shell_residual(skewed, 2), 0.075)


def test_complement_consistency_tvd_uses_bitwise_complement() -> None:
    at_condition = np.asarray([0.1, 0.2, 0.3, 0.4])
    at_complement = at_condition[::-1]
    assert complement_tvd(at_condition, at_complement) == 0.0

    at_complement = np.asarray([0.35, 0.3, 0.2, 0.15])
    assert np.isclose(
        complement_tvd(at_condition, at_complement),
        0.05,
    )


def test_complement_consistency_supports_batch_axes() -> None:
    at_condition = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    at_complement = at_condition[:, ::-1]
    assert np.allclose(
        complement_tvd(at_condition, at_complement),
        [0.0, 0.0],
    )


def test_bootstrap_intervals_are_deterministic_and_paired() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert bootstrap_mean_interval(values, seed=7, n_resamples=2000) == (
        bootstrap_mean_interval(values, seed=7, n_resamples=2000)
    )
    low, high = paired_bootstrap_interval(
        values + 0.5,
        values,
        seed=11,
        n_resamples=2000,
    )
    assert np.isclose(low, 0.5)
    assert np.isclose(high, 0.5)

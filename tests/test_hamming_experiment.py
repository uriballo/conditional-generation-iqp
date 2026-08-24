from math import pi, sqrt

import numpy as np
import pytest

from iqp_cond.exact import enumerate_bitstrings
from iqp_cond.hamming_balls import HammingBallsTarget
from iqp_cond.hamming_experiment import (
    N_QUBITS,
    TRAIN_CENTER_COUNTS,
    BallMixtureTarget,
    ball_columns,
    ball_fit_analysis,
    center_angle_for_count,
    count_modes,
    heldout_angles,
    sample_datasets,
)

FLIP = 0.08


def test_center_angle_path_maps_classes_to_expected_angles():
    assert center_angle_for_count(1) == pytest.approx(0.0)
    assert center_angle_for_count(2) == pytest.approx(pi / 12)
    assert center_angle_for_count(4) == pytest.approx(pi / 6)
    assert center_angle_for_count(8) == pytest.approx(pi / 4)


@pytest.mark.parametrize("center_count", [0, -1, 9, 16])
def test_center_angle_path_rejects_out_of_range(center_count):
    with pytest.raises(ValueError, match="between 1 and 8"):
        center_angle_for_count(center_count)


def test_heldout_angles_are_midpoints_between_training_angles():
    angles = heldout_angles()
    train = np.asarray([center_angle_for_count(k) for k in TRAIN_CENTER_COUNTS])

    assert angles.shape == (3,)
    assert angles == pytest.approx(np.asarray([pi / 24, pi / 8, 5 * pi / 24]))
    assert np.all(angles > 0) and np.all(angles < pi / 4)
    assert np.allclose(angles, 0.5 * (train[:-1] + train[1:]))


def test_mixture_target_is_normalized_and_convex():
    left = HammingBallsTarget.create(10, 1, FLIP, pattern_seed=1)
    right = HammingBallsTarget.create(10, 2, FLIP, pattern_seed=1)
    mixture = BallMixtureTarget(left, right, 0.5)
    bitstrings = enumerate_bitstrings(10)
    probabilities = mixture.bitstring_probabilities(bitstrings)

    assert probabilities.sum() == pytest.approx(1.0)
    assert np.all(probabilities > 0)
    assert np.allclose(
        probabilities,
        0.5 * left.bitstring_probabilities(bitstrings)
        + 0.5 * right.bitstring_probabilities(bitstrings),
    )


def test_mixture_sampling_matches_exact_probabilities():
    left = HammingBallsTarget.create(10, 1, FLIP, pattern_seed=2)
    right = HammingBallsTarget.create(10, 2, FLIP, pattern_seed=2)
    mixture = BallMixtureTarget(left, right, 0.5)
    samples = mixture.sample(np.random.default_rng(0), 500_000)
    bitstrings = enumerate_bitstrings(10)
    exact = mixture.bitstring_probabilities(bitstrings)
    shifts = np.arange(10, dtype=np.int64)
    indices = (samples.astype(np.int64) * (1 << shifts)).sum(axis=1)
    empirical = np.bincount(indices, minlength=1024) / samples.shape[0]
    tvd = 0.5 * np.abs(empirical - exact).sum()

    assert tvd < 0.02


def test_ball_fit_recovers_planted_mixture_weights():
    centers = HammingBallsTarget.create(10, 4, FLIP, pattern_seed=3).centers
    bitstrings = enumerate_bitstrings(10)
    columns = ball_columns(bitstrings, centers, FLIP)
    planted = 0.5 * columns[:, 0] + 0.25 * columns[:, 1] + 0.25 * columns[:, 2]
    fit = ball_fit_analysis(planted, columns)

    assert fit["effective_center_count"] == 3
    assert fit["ball_weights"][:3] == pytest.approx([0.5, 0.25, 0.25], abs=1e-6)
    assert fit["ball_weights"][3] == pytest.approx(0.0, abs=1e-9)
    assert fit["background_weight"] == pytest.approx(0.0, abs=1e-9)
    assert fit["relative_l2_residual"] < 1e-9


def test_ball_fit_absorbs_uniform_background():
    centers = HammingBallsTarget.create(10, 2, FLIP, pattern_seed=4).centers
    bitstrings = enumerate_bitstrings(10)
    columns = ball_columns(bitstrings, centers, FLIP)
    uniform = np.full(bitstrings.shape[0], 1.0 / bitstrings.shape[0])
    noisy = 0.9 * columns[:, 0] + 0.1 * uniform
    fit = ball_fit_analysis(noisy, columns)

    assert fit["effective_center_count"] == 1
    assert fit["ball_weights"][0] == pytest.approx(0.9, abs=1e-6)
    assert fit["background_weight"] == pytest.approx(0.1, abs=1e-6)


def test_count_modes_identifies_planted_centers():
    centers = HammingBallsTarget.create(10, 2, FLIP, pattern_seed=5).centers
    bitstrings = enumerate_bitstrings(10)
    columns = ball_columns(bitstrings, centers, FLIP)
    mixture = 0.5 * (columns[:, 0] + columns[:, 1])

    assert count_modes(mixture, 10) == 2


def test_count_modes_returns_zero_for_uniform_distribution():
    uniform = np.full(2**10, 1.0 / 2**10)

    assert count_modes(uniform, 10) == 0


def test_dataset_shapes_and_alignment():
    training, validation, heldout = sample_datasets(seed=0)

    assert training.shape == (4, 10_000, N_QUBITS)
    assert validation.shape == (4, 2_000, N_QUBITS)
    assert heldout.shape == (3, 2_000, N_QUBITS)
    assert np.all((training == 0) | (training == 1))


def test_small_system_ball_fit_has_tiny_relative_residual():
    target = HammingBallsTarget.create(19, 8, FLIP, pattern_seed=0)
    bitstrings = enumerate_bitstrings(19)
    columns = ball_columns(bitstrings, target.centers, FLIP)
    probabilities = target.bitstring_probabilities(bitstrings)
    fit = ball_fit_analysis(probabilities, columns)

    assert fit["effective_center_count"] == 8
    assert fit["relative_l2_residual"] < 1e-6


def test_planted_class_effective_counts_follow_center_counts():
    family = HammingBallsTarget.nested_family()
    bitstrings = enumerate_bitstrings(N_QUBITS)
    columns = ball_columns(bitstrings, family[-1].centers, FLIP)

    for target in family:
        fit = ball_fit_analysis(target.bitstring_probabilities(bitstrings), columns)
        assert fit["effective_center_count"] == target.n_centers
        assert sqrt(fit["relative_l2_residual"]) < 1e-3

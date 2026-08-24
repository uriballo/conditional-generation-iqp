import numpy as np
import pytest

from iqp_cond.exact import enumerate_bitstrings
from iqp_cond.multidimensional import (
    HELDOUT_CONDITIONS,
    TRAIN_CONDITIONS,
    condition_angles,
    distribution_metrics,
    gaussian_target,
    oracle_lookup,
    target_parameters,
)


def test_condition_split_is_a_corner_training_grid_and_cross_holdout():
    assert TRAIN_CONDITIONS.shape == (4, 2)
    assert HELDOUT_CONDITIONS.shape == (5, 2)
    assert not set(map(tuple, TRAIN_CONDITIONS)) & set(
        map(tuple, HELDOUT_CONDITIONS),
    )
    assert np.any(np.all(HELDOUT_CONDITIONS == [0.5, 0.5], axis=1))
    assert np.all(np.isin(HELDOUT_CONDITIONS[:, 0], [0.0, 0.5, 1.0]))
    assert np.all(np.isin(HELDOUT_CONDITIONS[:, 1], [0.0, 0.5, 1.0]))


def test_vector_product_encoding_separates_common_and_contrast_coordinates():
    angles = condition_angles(np.asarray([[0.25, 0.0], [0.25, 1.0]]), 8)

    np.testing.assert_allclose(angles[0, :4] + angles[0, 4:], 2 * np.pi / 6 * 0.25)
    np.testing.assert_allclose(angles[1, :4] + angles[1, 4:], 2 * np.pi / 6 * 0.25)
    np.testing.assert_allclose(angles[0, :4] - angles[0, 4:], -np.pi / 8)
    np.testing.assert_allclose(angles[1, :4] - angles[1, 4:], np.pi / 8)


def test_target_parameters_and_calibrated_target_are_independent_by_coordinate():
    mean_a, width_a = target_parameters([0.0, 0.0], 8)
    mean_b, width_b = target_parameters([1.0, 0.0], 8)
    mean_c, width_c = target_parameters([0.0, 1.0], 8)

    assert (mean_a, width_a) == pytest.approx((3.5, 0.75))
    assert mean_b == pytest.approx(mean_a - 1.5)
    assert width_b == pytest.approx(width_a)
    assert mean_c == pytest.approx(mean_a)
    assert width_c == pytest.approx(1.5)

    target = gaussian_target([0.25, 0.75], 8)
    assert target.actual_mean == pytest.approx(3.125, abs=1e-10)
    assert target.actual_std == pytest.approx(1.2977467210, abs=1e-9)
    assert target.shell_probabilities.sum() == pytest.approx(1.0)


def test_oracle_lookup_chooses_the_best_stored_distribution():
    target = np.asarray([0.2, 0.8])
    stored = np.asarray(
        [
            [0.2, 0.8],
            [0.9, 0.1],
        ],
    )

    result = oracle_lookup(stored, target, np.asarray([[0.0, 0.0], [1.0, 1.0]]))

    assert result["lookup_index"] == 0
    assert result["bitstring_tvd"] == pytest.approx(0.0)


def test_exact_metrics_are_zero_for_the_exact_target_distribution():
    target = gaussian_target([0.5, 0.5], 8)
    bitstrings = enumerate_bitstrings(8)
    probabilities = target.bitstring_probabilities(bitstrings)

    metrics = distribution_metrics(target, probabilities, bitstrings)

    assert metrics["bitstring_tvd"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["shell_tvd"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["mean_error"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["std_error"] == pytest.approx(0.0, abs=1e-12)


def test_repeating_the_center_encoding_is_a_valid_no_condition_control():
    angles = condition_angles(np.repeat([[0.5, 0.5]], 4, axis=0), 8)

    np.testing.assert_allclose(angles, np.repeat(angles[0][None, :], 4, axis=0))

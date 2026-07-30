from math import pi, sqrt

import numpy as np
import pytest

from iqp_cond.interpolation import (
    HELDOUT_ANGLE_FRACTIONS,
    TRAIN_ANGLE_FRACTIONS,
    broadcast_angles,
    interpolation_targets,
    sample_angle_targets,
    target_mean_for_angle,
)


def test_target_mean_mapping_preserves_original_endpoints():
    assert target_mean_for_angle(19, 0.0) == pytest.approx(19 / 2)
    assert target_mean_for_angle(19, pi / 4) == pytest.approx(19 / 4)
    assert target_mean_for_angle(19, pi / 8) == pytest.approx(3 * 19 / 8)


def test_heldout_angles_interleave_training_angles():
    combined = np.sort(
        np.concatenate([TRAIN_ANGLE_FRACTIONS, HELDOUT_ANGLE_FRACTIONS]),
    )
    assert combined == pytest.approx(np.arange(9) / 32)
    assert set(TRAIN_ANGLE_FRACTIONS).isdisjoint(HELDOUT_ANGLE_FRACTIONS)


def test_interpolation_targets_have_requested_means_and_fixed_width():
    angles = pi * np.asarray([0.0, 1 / 8, 1 / 4])
    targets = interpolation_targets(angles)

    for angle, target in zip(angles, targets, strict=True):
        assert target.actual_mean == pytest.approx(
            target_mean_for_angle(19, angle),
            abs=1e-11,
        )
        assert target.width == pytest.approx(sqrt(19) / 2)


def test_angle_dataset_and_broadcast_shapes():
    angles = pi * TRAIN_ANGLE_FRACTIONS
    samples = sample_angle_targets(angles, seed=7, n_samples=11)
    encoded = broadcast_angles(angles)

    assert samples.shape == (5, 11, 19)
    assert encoded.shape == (5, 19)
    assert encoded[:, 0] == pytest.approx(angles)
    assert np.all(encoded == encoded[:, :1])


@pytest.mark.parametrize("angle", [-0.1, pi / 4 + 0.1])
def test_target_mean_rejects_extrapolation(angle):
    with pytest.raises(ValueError, match="between 0 and pi/4"):
        target_mean_for_angle(19, angle)

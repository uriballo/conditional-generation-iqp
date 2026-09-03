import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.stochastic_interval import (
    sample_interval_angles,
    train_with_resampled_angles,
)
from iqp_cond.training import TrainingConfig


def test_interval_sampling_is_deterministic_and_bounded():
    baseline = np.full((2, 3), 0.5)
    first = sample_interval_angles(baseline, 0.2, 17)
    np.testing.assert_array_equal(first, sample_interval_angles(baseline, 0.2, 17))
    assert np.all(first >= 0.3) and np.all(first <= 0.7)


def test_zero_width_sampling_is_baseline():
    baseline = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_array_equal(sample_interval_angles(baseline, 0.0, 9), baseline)


def test_stochastic_training_returns_finite_history():
    ansatz = complete_ansatz(2)
    centers = np.asarray([[0.1, 0.1], [0.2, 0.2]])
    targets = np.zeros((2, 2, 2), dtype=float)
    config = TrainingConfig(
        steps=1,
        batch_size=2,
        validation_interval=1,
        n_ops=2,
        n_iqp_samples=8,
        validation_n_ops=2,
        validation_n_iqp_samples=8,
    )
    result = train_with_resampled_angles(
        "tiny",
        ansatz,
        centers,
        0.05,
        targets,
        targets,
        seed=3,
        config=config,
        validation_members=2,
    )
    assert result.condition_angles.shape == centers.shape
    assert result.train_losses.shape == (1,)
    assert np.isfinite(result.train_losses).all()
    assert np.isfinite(result.validation_losses).all()

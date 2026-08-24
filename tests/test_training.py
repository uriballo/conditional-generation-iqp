import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import exact_iqp_probabilities_many
from iqp_cond.mmd import exact_target_fouriers
from iqp_cond.training import (
    TrainingConfig,
    combine_condition_gradients,
    train_exact,
)


def test_mean_gradient_strategy_preserves_the_standard_update():
    gradients = np.asarray([[1.0, 3.0], [5.0, 7.0]])

    combined = combine_condition_gradients(gradients, "mean")

    np.testing.assert_allclose(combined, [3.0, 5.0])


def test_normalized_gradient_strategy_equalizes_condition_scales():
    gradients = np.asarray([[3.0, 0.0], [0.0, 4.0]])

    combined = combine_condition_gradients(gradients, "normalized")

    np.testing.assert_allclose(combined, [0.5, 0.5])


def test_pcgrad_removes_a_two_condition_negative_projection():
    gradients = np.asarray([[1.0, 0.0], [-1.0, 1.0]])

    combined = np.asarray(combine_condition_gradients(gradients, "pcgrad"))
    normalized = gradients / np.linalg.norm(gradients, axis=1, keepdims=True)

    assert combined @ normalized[0] >= -1e-12
    assert combined @ normalized[1] >= -1e-12


def test_zero_condition_gradient_remains_finite():
    gradients = np.asarray([[0.0, 0.0], [1.0, 0.0]])

    for strategy in ("normalized", "pcgrad"):
        assert np.isfinite(combine_condition_gradients(gradients, strategy)).all()


def test_exact_training_accepts_pcgrad_for_multiple_conditions():
    ansatz = complete_ansatz(2)
    angles = np.asarray([[0.1, 0.1], [0.2, 0.2]])
    targets = exact_iqp_probabilities_many(
        np.zeros(ansatz.n_parameters),
        ansatz.generators,
        angles,
    )
    config = TrainingConfig(
        steps=1,
        validation_interval=1,
        exact_gradient_strategy="pcgrad",
    )

    result = train_exact(
        "pcgrad_test",
        ansatz,
        angles,
        exact_target_fouriers(targets),
        seed=0,
        config=config,
    )

    assert result.train_steps.tolist() == [1]
    assert np.isfinite(result.train_losses).all()

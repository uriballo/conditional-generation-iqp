import numpy as np
import pytest

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.training import TrainingConfig, train


def test_initial_params_are_used_by_train_at_step_zero():
    ansatz = complete_ansatz(2)
    angles = np.asarray([[0.1, 0.1]])
    initial = np.arange(ansatz.n_parameters, dtype=float) / 10.0
    config = TrainingConfig(
        steps=1,
        validation_interval=2,
        n_ops=2,
        n_iqp_samples=8,
        validation_n_ops=2,
        validation_n_iqp_samples=8,
    )
    targets = np.ones((1, 2, 2), dtype=float) / 2.0
    result = train(
        ansatz,
        angles,
        targets,
        targets,
        seed=0,
        config=config,
        initial_params=initial,
    )
    np.testing.assert_array_equal(result.best_params, initial)


@pytest.mark.parametrize("initial", [np.zeros(2), np.asarray([0.0, np.nan, 1.0])])
def test_training_rejects_bad_initial_params(initial):
    ansatz = complete_ansatz(2)
    angles = np.asarray([[0.1, 0.1]])
    config = TrainingConfig(steps=1, n_ops=2, n_iqp_samples=8)
    targets = np.ones((1, 2, 2), dtype=float) / 2.0
    with pytest.raises(ValueError):
        train(
            ansatz,
            angles,
            targets,
            targets,
            seed=0,
            config=config,
            initial_params=initial,
        )

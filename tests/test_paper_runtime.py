"""Protocol and numerical parity checks against the published library."""

import hashlib
import json

import iqp_cond as ciqp
import numpy as np
import pytest

from paper_experiments.runtime import build_job
from paper_experiments.studies import make_targets
from paper_experiments.training import make_trainer


def test_reported_settings_and_serialization():
    j = build_job("scaling", 100, size=64)
    assert json.loads(json.dumps(j)) == j
    assert j["config"]["n_ops"] == 512 and j["config"]["mc_samples"] == 1024
    assert j["config"]["bodyness_targets"] == [2.0, 4.0]
    with pytest.raises(ValueError):
        build_job("shared", 5400)


def test_adam_matches_public_trainer_and_checkpoints_do_not_change_it():
    encoder = ciqp.AngleEncoder(np.zeros((2, 1, 3)), axes=("YYY",))
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(3), encoder)
    targets = np.zeros((2, 5, 3))
    targets[:, :, 0] = 1
    config = ciqp.TrainingConfig(
        steps=6,
        batch_size=2,
        n_ops=4,
        mc_samples=4,
        validation_interval=2,
        validation_n_ops=4,
        validation_mc_samples=4,
        sigmas=(1.0, 2.0),
        learning_rate=0.02,
        final_learning_rate=0.01,
        sample_chunk=2,
    )
    initial = np.linspace(-0.2, 0.3, 6)
    expected_seen = []
    actual_seen = []

    def selection(theta):
        expected_seen.append(np.asarray(theta).copy())
        return float(np.square(theta).sum())

    expected = ciqp.train(
        model,
        targets,
        targets,
        seed=7,
        config=config,
        initial_params=initial,
        selection_metric=selection,
    )
    actual = make_trainer(
        dict(name="adam", learning_rate=0.02, b1=0.9, final_learning_rate_ratio=0.5)
    )(
        model,
        targets,
        targets,
        seed=7,
        config=config,
        initial_params=initial,
        selection_metric=lambda theta: float(np.square(theta).sum()),
        checkpoint_callback=lambda step, theta: actual_seen.append((step, theta)),
    )
    assert [step for step, _ in actual_seen] == list(expected.validation_steps)
    np.testing.assert_allclose(
        expected_seen, [theta for _, theta in actual_seen], rtol=1e-7, atol=1e-9
    )
    for key in (
        "final_params",
        "best_params",
        "train_losses",
        "validation_losses",
        "selection_scores",
    ):
        np.testing.assert_allclose(
            getattr(actual, key), getattr(expected, key), rtol=1e-7, atol=1e-9
        )
    assert actual.best_step == expected.best_step
    actual_seen[0][1][:] = 999
    assert not np.any(actual.best_params == 999)


def test_curie_target_pool_matches_saved_run():
    # Seed and pool hash of the final ordinary-input beta=1.6 case.
    pool = make_targets(np.asarray([1.6]), 16, 10000, np.random.default_rng(5404))
    assert (
        hashlib.sha256(pool.tobytes()).hexdigest()
        == "bd5c9c723a30aace13a1d353fb0979f8b0d2b54eba0e91a2b872a62b18643dce"
    )

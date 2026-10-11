"""Independent checks of the adopted scaling selection and restart protocol."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from paper_experiments import engine
from paper_experiments.runtime import build_job, config_for
from paper_experiments.scaling import (
    make_validation_selector,
    normalized_validation_score,
)
from paper_experiments.studies import build_study
from paper_experiments.training import make_trainer


def _small_case():
    job = build_job("scaling", 100, size=64)
    config = replace(
        config_for(job),
        n_visible=4,
        steps=2,
        restarts=2,
        n_train=8,
        n_validation=8,
        n_test=8,
        batch_size=4,
        n_ops=8,
        mc_samples=16,
        validation_n_ops=8,
        validation_mc_samples=16,
        validation_interval=1,
        sample_chunk=8,
        bodyness_targets=(0.5, 1.0),
        validation_scales=(1.0, 2.0, 3.0, 4.0),
    )
    study = build_study(job, config)
    model = engine._model(study, config, study.training_conditions)
    rng = np.random.default_rng(100)
    training = study.make_targets(study.training_conditions, 4, 8, rng)
    validation = study.make_targets(study.training_conditions, 4, 8, rng)
    return job, config, study, model, training, validation


def _explicit_selector(model, validation, config, seed):
    tc = config.training_config()

    def evaluate(theta, key):
        return model.mmd(
            theta,
            jnp.asarray(validation),
            key,
            sigmas=tc.sigmas,
            n_ops=tc.validation_n_ops,
            n_samples=tc.validation_mc_samples,
            shared_samples=True,
            gemm_dtype=tc.gemm_dtype,
            estimator_dtype=tc.estimator_dtype,
            sample_chunk=tc.sample_chunk,
        ).per_condition

    evaluate = jax.jit(evaluate)

    def select(theta):
        banks = np.stack(
            [
                np.asarray(evaluate(theta, jax.random.PRNGKey(seed + 4_000_000 + rep)))
                for rep in range(4)
            ]
        )
        return float(np.mean(banks.mean(axis=0) / np.asarray(config.validation_scales)))

    return select


def test_normalization_retains_negative_estimates_and_rejects_invalid_scales():
    assert normalized_validation_score([-2.0, 1.0], [2.0, 4.0]) == -0.375
    with pytest.raises(ValueError):
        normalized_validation_score([1.0], [0.0])
    with pytest.raises(ValueError):
        normalized_validation_score([float("nan")], [1.0])


def test_four_bank_selector_matches_explicit_independent_estimates():
    _, config, _, model, _, validation = _small_case()
    theta = np.linspace(-0.2, 0.3, model.ansatz.n_parameters)
    actual = make_validation_selector(model, validation, config, 100)
    expected = _explicit_selector(model, validation, config, 100)
    np.testing.assert_allclose(actual(theta), expected(theta), rtol=1e-12, atol=1e-12)
    assert actual(theta) == actual(theta)


def test_large_runner_matches_frozen_restart_and_selection_recipe(tmp_path):
    job, config, study, model, training, validation = _small_case()
    settings = {
        key: value
        for key, value in job["optimizer"].items()
        if key not in {"encoder_scale", "init_scale"}
    }
    settings["compute_validation_mmd"] = False
    trainer = make_trainer(settings)
    selector = _explicit_selector(model, validation, config, 100)
    expected = []
    for restart in range(2):
        initial = np.random.default_rng(100 + 101 + restart).normal(
            scale=config.init_scale, size=model.ansatz.n_parameters
        )
        expected.append(
            trainer(
                model,
                training,
                validation,
                seed=100,
                config=config.training_config(),
                initial_params=initial,
                selection_metric=selector,
            )
        )
    captured = {}

    def assess(model, snapshots, conditions, seed, best_step):
        captured.update(seed=seed, steps=sorted(snapshots))
        return {}

    actual = engine.run_large(
        config, study, 100, assess, output_root=tmp_path, training_fn=trainer
    )
    selected = int(np.argmin([r.best_selection_score for r in expected]))
    assert actual["selected_restart"] == selected
    assert actual["best_step"] == expected[selected].best_step
    assert actual["best_validation_loss"] is None
    assert actual["validation_losses"] == [None, None, None]
    assert actual["selection_metric"] == "normalized_fixed_bank_validation_mmd"
    assert captured == {"seed": 100 + 2_000_003, "steps": [0, 1, 2]}
    with np.load(tmp_path / actual["study"] / actual["run_id"] / "restarts.npz") as raw:
        np.testing.assert_allclose(
            raw["best_params"], [r.best_params for r in expected]
        )
        np.testing.assert_allclose(
            raw["train_losses"], [r.train_losses for r in expected]
        )
        np.testing.assert_allclose(
            raw["selection_scores"], [r.selection_scores for r in expected]
        )

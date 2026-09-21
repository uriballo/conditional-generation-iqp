"""Fast synthetic tests for optimization-diagnostic extraction."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from experiments.common.config import RunConfig
from experiments.scalar_gaussian_translation import campaign, diagnostics


def _synthetic_arrays() -> dict[str, np.ndarray]:
    return {
        "restart_seeds": np.asarray([11, 12]),
        "train_steps": np.asarray([[0, 2, 4], [0, 2, 4]]),
        "train_losses": np.asarray([[0.5, 0.2, 0.08], [0.8, 0.4, 0.3]]),
        "validation_steps": np.asarray([[0, 2, 4], [0, 2, 4]]),
        "validation_losses": np.asarray([[0.6, 0.3, 0.1], [0.9, 0.5, 0.4]]),
        "selection_scores": np.asarray([[0.4, 0.2, 0.09], [0.8, 0.3, 0.25]]),
        "best_steps": np.asarray([4, 2]),
        "best_selection_scores": np.asarray([0.09, 0.3]),
        "wall_seconds": np.asarray([10.0, 20.0]),
    }


def test_restart_curves_and_first_threshold_crossings() -> None:
    rows = diagnostics.restart_curve_rows(
        _synthetic_arrays(),
        seed=301,
        encoder="broadcast_affine",
        loss_levels=(0.1,),
        tvd_levels=(0.1, 0.2),
    )
    assert len(rows) == 2
    first = rows[0]
    assert first["train_curve"] == {
        "steps": [0, 2, 4],
        "loss": [0.5, 0.2, 0.08],
    }
    assert first["validation_curve"]["steps"] == [0, 2, 4]
    assert first["selection_curve"]["metric"] == "mean_exact_training_condition_tvd"
    assert first["best_step"] == 4
    assert first["best_selection_score"] == 0.09
    assert first["final_selection_score"] == 0.09
    assert first["threshold_times"]["training_loss"]["0.1"] == {
        "updates": 4,
        "estimated_seconds": 10.0,
    }
    assert first["threshold_times"]["selection_tvd"]["0.2"]["updates"] == 2
    assert rows[1]["threshold_times"]["selection_tvd"]["0.1"]["updates"] is None


def test_controlled_subset_requires_all_encoder_seed_pairs() -> None:
    entries = []
    for encoder in diagnostics.ENCODERS:
        for seed in (301, 302, 303):
            entries.append(
                {
                    "job": campaign.CampaignJob(
                        "validation",
                        "validation",
                        encoder,
                        "interpolation",
                        (0.0, 0.5, 1.0),
                        3,
                        "even",
                        None,
                        seed,
                        "fixed_total",
                        "selected",
                        {"updates": 3},
                    )
                }
            )
    selected = diagnostics._controlled_entries(entries, "validation")
    assert len(selected) == 9
    assert [item["job"].encoder for item in selected[:3]] == [
        "anti_block",
        "anti_block",
        "anti_block",
    ]
    with np.testing.assert_raises_regex(ValueError, "full controlled"):
        diagnostics._controlled_entries(entries[:-1], "validation")


def test_mmd_objective_gradient_wrapper() -> None:
    class TinyModel:
        def mmd(self, params, targets, key, **kwargs):
            del targets, key, kwargs
            return SimpleNamespace(loss=jnp.sum(params**2))

    config = RunConfig(study="scalar_gaussian_translation", n_visible=8, n_ancillas=0)

    def objective(params):
        return diagnostics._mmd_objective(
            params,
            model=TinyModel(),
            targets=jnp.zeros((1, 2, 8)),
            key=jax.random.PRNGKey(2),
            config=config,
        )

    value, gradient = jax.value_and_grad(objective)(jnp.asarray([1.0, 2.0]))
    assert float(value) == 5.0
    assert np.asarray(gradient).tolist() == [2.0, 4.0]


def test_matched_gradient_diagnostics_use_public_model_mmd(tmp_path: Path) -> None:
    config = RunConfig(
        study="scalar_gaussian_translation",
        encoder="broadcast_affine",
        condition_protocol="interpolation",
        n_visible=8,
        n_ancillas=0,
        n_train=4,
        n_validation=4,
        n_test=4,
        steps=1,
        batch_size=2,
        n_ops=2,
        mc_samples=4,
        validation_n_ops=2,
        validation_mc_samples=4,
        restarts=1,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(config.as_dict()))
    np.savez(run_dir / "restarts.npz", restart_seeds=np.asarray([402]))
    job = campaign.CampaignJob(
        "validation",
        "validation",
        "broadcast_affine",
        "interpolation",
        (0.0, 0.5, 1.0),
        3,
        "even",
        None,
        301,
        "fixed_total",
        "selected",
        {"updates": 1},
    )
    result = diagnostics._gradient_diagnostics(
        [{"job": job, "checkpoint": run_dir / "checkpoint.npz"}]
    )
    assert len(result["rows"]) == 3
    assert {row["encoder"] for row in result["rows"]} == set(diagnostics.ENCODERS)
    assert len({row["initialization_sha256"] for row in result["rows"]}) == 1
    assert all(np.isfinite(row["gradient_norm"]) for row in result["rows"])

"""The run harness: a tiny end-to-end run writes a complete run directory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.common import runner
from experiments.common.config import RunConfig
from experiments.gaussian.study import build_study


def test_smoke_run_writes_metrics_and_checkpoint(tmp_path: Path) -> None:
    config = RunConfig(
        study="gaussian",
        encoder="anti_block",
        condition_protocol="full5",
        n_visible=3,
        n_ancillas=1,
        n_train=8,
        n_validation=8,
        n_test=8,
        steps=2,
        batch_size=4,
        n_ops=8,
        mc_samples=16,
        validation_interval=2,
        restarts=2,
        seeds=(0,),
    )
    study = build_study()
    record = runner.run(config, study, 0, output_root=tmp_path)

    directory = Path(record["directory"])
    assert directory.is_dir()
    for name in (
        "config.json",
        "run.json",
        "metrics.json",
        "checkpoint.npz",
        "restarts.npz",
    ):
        assert (directory / name).exists()

    metrics = json.loads((directory / "metrics.json").read_text())
    assert np.isfinite(metrics["train_tvd_mean"])
    assert np.isfinite(metrics["held_out_tvd_mean"])
    assert 0.0 <= metrics["train_tvd_mean"] <= 1.0
    assert record["run_id"].startswith(
        "gaussian-conditional-full5-fixed_total-n3a1-anti_block-s0-",
    )
    assert metrics["selected_restart"] in (0, 1)
    assert metrics["selection_metric"] == "mean_exact_training_condition_tvd"
    assert np.isfinite(metrics["best_selection_score"])
    assert np.isfinite(metrics["held_out_sample_mmd"])
    assert metrics["resources"]["optimization_restarts"] == 2

    checkpoint = np.load(directory / "checkpoint.npz")
    n_parameters = config.n_total + config.n_total * (config.n_total - 1) // 2
    assert checkpoint["params"].shape == (n_parameters,)
    assert checkpoint["train_probabilities"].shape == (5, 2**3)
    assert checkpoint["held_out_probabilities"].shape == (16, 2**3)

    restarts = np.load(directory / "restarts.npz")
    assert restarts["best_params"].shape == (2, n_parameters)
    assert restarts["selection_scores"].shape == (2, 2)


def test_encoder_only_run_skips_optimization(tmp_path: Path) -> None:
    config = RunConfig(
        study="gaussian",
        encoder="target_mean",
        condition_protocol="full5",
        model_kind="encoder_only",
        n_visible=3,
        n_train=8,
        n_validation=8,
        n_test=8,
        steps=2,
        batch_size=4,
        n_ops=8,
        mc_samples=16,
        validation_interval=2,
        restarts=1,
        seeds=(0,),
    )
    record = runner.run(config, build_study(), 0, output_root=tmp_path)

    directory = Path(record["directory"])
    assert record["selected_restart"] is None
    assert record["resources"]["optimization_restarts"] == 0
    assert not (directory / "restarts.npz").exists()

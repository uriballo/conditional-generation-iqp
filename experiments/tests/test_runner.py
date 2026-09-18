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
        encoder="broadcast",
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
        seeds=(0,),
    )
    study = build_study()
    record = runner.run(config, study, 0, output_root=tmp_path)

    directory = Path(record["directory"])
    assert directory.is_dir()
    for name in ("config.json", "run.json", "metrics.json", "checkpoint.npz"):
        assert (directory / name).exists()

    metrics = json.loads((directory / "metrics.json").read_text())
    assert np.isfinite(metrics["train_tvd_mean"])
    assert np.isfinite(metrics["held_out_tvd_mean"])
    assert 0.0 <= metrics["train_tvd_mean"] <= 1.0
    assert record["run_id"].startswith("gaussian-n3a1-broadcast-s0-")

    checkpoint = np.load(directory / "checkpoint.npz")
    n_parameters = config.n_total + config.n_total * (config.n_total - 1) // 2
    assert checkpoint["params"].shape == (n_parameters,)
    assert checkpoint["train_probabilities"].shape == (5, 2**3)

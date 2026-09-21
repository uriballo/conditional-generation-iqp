"""Manifest and execution safety for the scalar Gaussian campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.common.config import RunConfig
from experiments.scalar_gaussian_translation import campaign
from experiments.scalar_gaussian_translation.study import build_study


def _selected(tmp_path: Path) -> Path:
    path = tmp_path / "selected_config.json"
    path.write_text(
        json.dumps(
            {
                "hyperparameter_id": "frozen",
                "hyperparameters": {
                    "learning_rate": 0.01,
                    "updates": 300,
                    "restarts": 2,
                    "kernel_bodyness": [2, 4],
                },
            }
        )
    )
    return path


def test_source_snapshot_hashes_actual_library_runtime() -> None:
    files, digest = campaign._source_snapshot()
    assert len(digest) == 64
    assert "iqp-cond/src/iqp_cond/training.py" in files
    assert "experiments/scalar_gaussian_translation/analysis.py" not in files


def test_stage_job_counts_and_content_addressing() -> None:
    first = campaign.build_jobs("stage1")
    assert len(first) == 6
    assert {j.encoder for j in first} == set(campaign.STAGE1_ENCODERS)
    assert {j.anchor_count for j in first} == {3}
    assert all(j.placement == "even" and j.layout_seed is None for j in first)
    stage2 = campaign.build_jobs("stage2", viable_encoders=("broadcast_affine",))
    assert len(stage2) == 24
    assert len({j.job_id for j in stage2}) == len(stage2)
    assert len({j.hyperparameter_id for j in stage2}) == 8


def test_main_and_mirror_job_counts_keep_seed_roles_separate(tmp_path: Path) -> None:
    selected = _selected(tmp_path)
    jobs = campaign.build_jobs("validation", "validation", selected_config=selected)
    assert len(jobs) == 936
    main = [
        j
        for j in jobs
        if j.protocol in campaign.ORIGINAL_PROTOCOLS and j.encoder != "condition_blind"
    ]
    assert len(main) == 864
    assert {j.layout_seed for j in main if j.placement == "random"} == {
        1101,
        1102,
        1103,
    }
    assert {j.optimizer_seed for j in main} == {301, 302, 303}
    assert all(
        j.layout_seed not in {j.optimizer_seed for j in main}
        for j in main
        if j.layout_seed is not None
    )
    assert all(len(j.anchors) == j.anchor_count for j in jobs)
    run_keys = {
        (campaign._run_config(job).digest(), job.optimizer_seed) for job in jobs
    }
    assert len(run_keys) == len(jobs)


def test_budgets_endpoints_n8_and_matched_exposure(tmp_path: Path) -> None:
    selected = _selected(tmp_path)
    jobs = campaign.build_jobs("validation", "validation", selected_config=selected)
    assert all(
        j.protocol not in {"near", "far", "interpolation"}
        or (j.anchors[0], j.anchors[-1])
        == campaign.protocol.protocol_interval(j.protocol)
        for j in jobs
    )
    job = next(
        j
        for j in jobs
        if j.encoder == "broadcast_affine"
        and j.protocol == "near"
        and j.anchor_count == 4
        and j.placement == "even"
        and j.budget_mode == "matched_exposure"
    )
    config = campaign._run_config(job)
    assert config.condition_protocol == "near-k4-even"
    assert config.steps == 400
    assert config.n_visible == 8 and config.n_ancillas == 0
    assert config.n_train == config.n_validation == config.n_test == 512
    assert config.n_ops == config.validation_n_ops == 64
    assert config.mc_samples == config.validation_mc_samples == 256
    assert config.gemm_dtype == "float32" and config.sample_chunk == 128
    assert config.selection_mode == "exact_training_tvd"
    assert config.init_scale == 0.5
    with pytest.raises(ValueError):
        campaign._run_config(
            replace(job, hyperparameters={**job.hyperparameters, "updates": 301})
        )
    with pytest.raises(ValueError, match="n_visible=8"):
        campaign._validate_config(
            RunConfig(study="scalar_gaussian_translation", n_visible=12)
        )


def test_protocol_hash_and_production_disabled(tmp_path: Path) -> None:
    manifest = tmp_path / "production.json"
    payload = campaign.write_manifest(
        manifest, "production", "dev", selected_config=_selected(tmp_path)
    )
    assert payload["enabled"] is False and payload["production"] is True
    assert payload["production_enabled"] is False
    assert payload["job_count"] == 3120
    assert {job["optimizer_seed"] for job in payload["jobs"]} == set(range(501, 511))
    assert (
        payload["protocol_sha256"]
        == hashlib.sha256(campaign.PROTOCOL_PATH.read_bytes()).hexdigest()
    )
    loaded, _ = campaign._load_manifest(manifest)
    with pytest.raises(RuntimeError, match="production"):
        campaign._assert_runnable(loaded, manifest)


def test_manifest_rejects_changed_source_snapshot(tmp_path: Path) -> None:
    selected = _selected(tmp_path)
    manifest = tmp_path / "validation.json"
    campaign.write_manifest(
        manifest, "validation", "validation", selected_config=selected
    )
    payload = json.loads(manifest.read_text())
    payload["source_file_sha256"]["pyproject.toml"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="source snapshot"):
        campaign._load_manifest(manifest)


def test_audit_manifest_locks_existing_gate_file(tmp_path: Path) -> None:
    selected = _selected(tmp_path)
    gates = tmp_path / "frozen_gates.json"
    gates.write_text('{"frozen":true}\n')
    manifest = tmp_path / "audit.json"
    payload = campaign.write_manifest(
        manifest, "audit", "audit", selected_config=selected, audit_gate_path=gates
    )
    assert payload["enabled"] is True
    assert (
        payload["frozen_gates_sha256"] == hashlib.sha256(gates.read_bytes()).hexdigest()
    )
    gates.write_text('{"frozen":false}\n')
    with pytest.raises(RuntimeError, match="gates"):
        campaign._assert_runnable(payload, manifest)


def test_direct_audit_requires_gate_lock(tmp_path: Path) -> None:
    selected = _selected(tmp_path)
    with pytest.raises(FileNotFoundError, match="audit-split"):
        campaign.write_manifest(
            tmp_path / "direct.json", "direct", "audit", selected_config=selected
        )


def test_job_execution_writes_record_and_resume_is_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    jobs = campaign.build_jobs("stage1")
    job = jobs[0]
    run_dir = tmp_path / "scalar_gaussian_translation" / "run-x"
    run_dir.mkdir(parents=True)
    fake_record = {
        "run_id": "run-x",
        "directory": str(run_dir),
        "study": "scalar_gaussian_translation",
    }
    calls = []
    monkeypatch.setattr(
        campaign.runner, "run", lambda *a, **k: calls.append(1) or fake_record
    )
    monkeypatch.setattr(campaign.tracking, "append_index", lambda *a, **k: None)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "phase": "stage1",
                "enabled": True,
                "protocol_sha256": campaign._sha256(campaign.PROTOCOL_PATH),
                "jobs": [campaign.asdict(job)],
            }
        )
    )
    campaign._execute(job, tmp_path, manifest)
    record_path = (
        tmp_path / "scalar_gaussian_translation" / "job_records" / f"{job.job_id}.json"
    )
    assert json.loads(record_path.read_text())["job"]["job_id"] == job.job_id
    campaign._execute(job, tmp_path, manifest)
    assert len(calls) == 2


def test_real_smoke_run_writes_complete_artifacts(tmp_path: Path) -> None:
    job = campaign.build_jobs("stage1")[0]
    config = replace(
        campaign._run_config(job),
        n_train=16,
        n_validation=16,
        n_test=16,
        steps=1,
        batch_size=4,
        n_ops=4,
        mc_samples=8,
        validation_interval=1,
        validation_n_ops=4,
        validation_mc_samples=8,
        sample_chunk=8,
        restarts=1,
    )
    study = build_study(
        job.protocol,
        job.anchor_count,
        job.placement,
        job.layout_seed or 1101,
        job.encoder,
        job.split,
    )
    record = campaign.runner.run(
        config, study, job.optimizer_seed, output_root=tmp_path
    )
    run_dir = Path(record["directory"])
    assert {
        "checkpoint.npz",
        "config.json",
        "metrics.json",
        "restarts.npz",
        "run.json",
    } <= {path.name for path in run_dir.iterdir()}
    assert record["n_visible"] == 8 and record["n_ancillas"] == 0

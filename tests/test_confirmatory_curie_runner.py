from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from iqp_cond.artifacts import read_immutable_json
from iqp_cond.confirmatory_curie import (
    CurieConfirmatoryArm,
    CurieConfirmatoryManifest,
    plan_curie_confirmatory_runs,
)
from iqp_cond.confirmatory_curie_runner import (
    arm_ansatz,
    arm_encoding,
    condition_angle,
    condition_angles,
    execute_run,
    target_samples,
)
from iqp_cond.encoding import ANGLE_ENCODING, SPIN_FLIP_BRANCH_ENCODING

REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY / "configs" / "curie_weiss_confirmatory_v1.json"


def _run(arm: CurieConfirmatoryArm):
    manifest = CurieConfirmatoryManifest.from_json(MANIFEST)
    return manifest, next(
        run
        for run in plan_curie_confirmatory_runs(manifest)
        if run.n_qubits == 8 and run.seed == 100 and run.arm is arm
    )


def test_arm_materialization_and_fixed_angles() -> None:
    manifest, full = _run(CurieConfirmatoryArm.FULL_JOINT_R1)
    assert arm_ansatz(full).n_qubits == 9
    assert arm_encoding(full) == SPIN_FLIP_BRANCH_ENCODING
    angles = condition_angles(manifest, full, manifest.training_betas)
    assert angles.shape == (6, 9)
    assert np.all(angles == angles[:, :1])
    assert condition_angle(1.0, manifest) == pytest.approx(0.0)

    _, product = _run(CurieConfirmatoryArm.M0_PRODUCT)
    assert arm_encoding(product) == ANGLE_ENCODING
    _, no_condition = _run(CurieConfirmatoryArm.NO_CONDITION_M0)
    no_condition_angles = condition_angles(
        manifest, no_condition, manifest.training_betas
    )
    assert np.allclose(no_condition_angles, 0.0)


def test_deterministic_role_separated_target_samples() -> None:
    manifest, run = _run(CurieConfirmatoryArm.M0_SPIN_FLIP)
    first = target_samples(run, manifest.training_betas, role="training", count=8)
    second = target_samples(run, manifest.training_betas, role="training", count=8)
    validation = target_samples(
        run, manifest.training_betas, role="validation", count=8
    )
    assert np.array_equal(first, second)
    assert first.shape == (6, 8, 8)
    assert not np.array_equal(first, validation)


def test_cli_dry_run_selects_one_declared_identity() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts" / "run_confirmatory_curie.py"),
            "--manifest",
            str(MANIFEST),
            "--arm",
            "m0_product",
            "--n-qubits",
            "8",
            "--seed",
            "100",
            "--dry-run",
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["count"] == 1
    assert payload["runs"][0]["arm"] == "m0_product"


def test_one_update_smoke_writes_replayable_immutable_result(tmp_path: Path) -> None:
    manifest, run = _run(CurieConfirmatoryArm.M0_PRODUCT)
    payload = execute_run(
        REPOSITORY,
        manifest,
        run,
        tmp_path,
        invocation=("pytest",),
        updates=1,
        lightweight=True,
    )
    assert len(payload["metrics"]["evaluation_rows"]) == 11
    assert payload["metrics"]["encoder_parameters_trainable"] is False
    assert all(
        np.isfinite(row["bitstring_tvd"])
        for row in payload["metrics"]["evaluation_rows"]
    )
    record = tmp_path / "runs" / f"{run.run_id}.json"
    assert read_immutable_json(record) == payload
    assert (
        execute_run(
            REPOSITORY,
            manifest,
            run,
            tmp_path,
            invocation=("different",),
            updates=2,
            lightweight=True,
        )
        == payload
    )

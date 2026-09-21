"""Focused checks for production post-run attestation helpers."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.scalar_gaussian_translation import production_attestation


def test_validate_npz_accepts_finite_arrays(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.npz"
    np.savez(path, weights=np.array([1.0, 2.0]))
    production_attestation._validate_npz(path)


def test_validate_npz_rejects_non_finite_arrays(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.npz"
    np.savez(path, weights=np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="non-finite"):
        production_attestation._validate_npz(path)


def test_gate_status_requires_exact_native_true_counts() -> None:
    payload = {
        "encoders": {
            "broadcast_affine": {
                "checks": {str(i): True for i in range(28)},
                "pass": True,
            },
            "qubit_affine": {"checks": {str(i): True for i in range(28)}, "pass": True},
            "anti_block": {"checks": {str(i): True for i in range(12)}, "pass": None},
        }
    }
    result = production_attestation._gate_status(payload)
    assert result["broadcast_affine"]["check_count"] == 28
    assert result["anti_block"]["reported_pass"] is None
    payload["encoders"]["qubit_affine"]["checks"]["0"] = False
    with pytest.raises(ValueError, match="qubit_affine"):
        production_attestation._gate_status(payload)


def test_artifact_records_hash_and_size(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"ok": True}) + "\n")
    result = production_attestation._artifact(path)
    assert result["path"] == str(path)
    assert len(result["sha256"]) == 64
    assert result["bytes"] == path.stat().st_size

"""Execute one frozen Curie--Weiss temperature/ancilla confirmatory run."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .ancilla import exact_ancilla_marginal_probabilities_from_full_angles
from .ansatz import Ansatz, complete_ansatz
from .artifacts import (
    DuplicateArtifactError,
    read_immutable_json,
    sha256_file,
    write_immutable_json,
)
from .confirmatory_curie import (
    CurieConfirmatoryArm,
    CurieConfirmatoryManifest,
    CurieConfirmatoryRunIdentity,
    curie_weiss_probabilities,
    disconnected_ansatz,
    visible_matched_ansatz,
)
from .curie_weiss_data import exact_curie_weiss_target
from .curie_weiss_statistics import thermodynamic_metrics
from .encoding import ANGLE_ENCODING, SPIN_FLIP_BRANCH_ENCODING
from .exact import enumerate_bitstrings, shell_probabilities
from .gaussian_data import write_immutable_npz
from .gaussian_runner import exact_mmd_squared
from .training import TrainingConfig, TrainingResult, train


def condition_angle(beta: float, manifest: CurieConfirmatoryManifest) -> float:
    """Evaluate the frozen piecewise-linear map, in radians."""
    if not math.isfinite(beta):
        raise ValueError("beta must be finite")
    for piece in manifest.raw["condition_map"]["pieces"]:
        if float(piece["start"]) <= beta <= float(piece["end"]):
            return math.pi * (
                float(piece["anchor"])
                + float(piece["slope"]) * (beta - float(piece["start"]))
            )
    raise ValueError("beta is outside the frozen condition map")


def _is_no_condition(arm: CurieConfirmatoryArm) -> bool:
    return arm in {
        CurieConfirmatoryArm.NO_CONDITION_M0,
        CurieConfirmatoryArm.NO_CONDITION_M5,
    }


def arm_ansatz(run: CurieConfirmatoryRunIdentity) -> Ansatz:
    """Materialize the declared generator bank for one arm."""
    if run.arm.value.startswith("visible_matched_r"):
        return visible_matched_ansatz(run.n_qubits, run.ancillas)
    if run.arm.value.startswith("disconnected_r"):
        return disconnected_ansatz(run.n_qubits, run.ancillas)
    return complete_ansatz(run.n_qubits + run.ancillas)


def arm_encoding(run: CurieConfirmatoryRunIdentity) -> str:
    """Return the fixed encoder family for one declared arm."""
    if run.arm is CurieConfirmatoryArm.M0_PRODUCT:
        return ANGLE_ENCODING
    return SPIN_FLIP_BRANCH_ENCODING


def condition_angles(
    manifest: CurieConfirmatoryManifest,
    run: CurieConfirmatoryRunIdentity,
    betas: tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    """Return fixed, non-trainable encoder angles for training or evaluation."""
    ansatz = arm_ansatz(run)
    values = np.asarray(tuple(float(beta) for beta in betas), dtype=np.float64)
    if _is_no_condition(run.arm):
        mapped = np.full(values.shape, condition_angle(1.0, manifest))
    else:
        mapped = np.asarray(
            [condition_angle(float(beta), manifest) for beta in values],
            dtype=np.float64,
        )
    return np.repeat(mapped[:, None], ansatz.n_qubits, axis=1)


def _split_seed(run: CurieConfirmatoryRunIdentity, beta: float, role: str) -> int:
    raw = f"{run.run_id}|{beta:.17g}|{role}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & (2**63 - 1)


def target_samples(
    run: CurieConfirmatoryRunIdentity,
    betas: tuple[float, ...],
    *,
    role: str,
    count: int,
) -> np.ndarray:
    """Draw a deterministic finite target set for the Section 3 estimator."""
    if role not in {"training", "validation"} or count < 2:
        raise ValueError("target sample role/count is invalid")
    bitstrings = enumerate_bitstrings(run.n_qubits)
    batches = []
    for beta in betas:
        probabilities = curie_weiss_probabilities(run.n_qubits, beta)
        rng = np.random.default_rng(_split_seed(run, beta, role))
        indices = rng.choice(probabilities.size, size=count, p=probabilities)
        batches.append(bitstrings[indices])
    return np.asarray(batches, dtype=np.int8)


def training_config(
    manifest: CurieConfirmatoryManifest,
    *,
    updates: int | None = None,
    lightweight: bool = False,
) -> TrainingConfig:
    settings = manifest.raw["training"]
    steps = int(settings["updates"] if updates is None else updates)
    if steps < 1:
        raise ValueError("updates must be positive")
    conditions = len(manifest.training_betas)
    total_minibatch = int(settings["total_target_minibatch"])
    if total_minibatch % conditions:
        raise ValueError("target minibatch must divide over training conditions")
    return TrainingConfig(
        steps=steps,
        batch_size=total_minibatch // conditions,
        learning_rate=float(settings["initial_learning_rate"]),
        final_learning_rate=float(settings["final_learning_rate"]),
        validation_interval=min(int(settings["validation_interval"]), steps),
        sigmas=tuple(float(value) for value in settings["bandwidths"]),
        n_ops=4 if lightweight else int(settings["visible_character_masks"]),
        n_iqp_samples=16 if lightweight else int(settings["training_model_samples"]),
        validation_n_ops=4 if lightweight else int(settings["visible_character_masks"]),
        validation_n_iqp_samples=16
        if lightweight
        else int(settings["validation_model_samples"]),
    )


def train_run(
    manifest: CurieConfirmatoryManifest,
    run: CurieConfirmatoryRunIdentity,
    *,
    updates: int | None = None,
    lightweight: bool = False,
) -> TrainingResult:
    """Fit only the IQP phases; all condition-encoder values remain fixed."""
    config = training_config(manifest, updates=updates, lightweight=lightweight)
    sample_count = (
        max(config.batch_size, 2)
        if lightweight
        else int(manifest.raw["training"]["reference_minibatch"])
    )
    training = target_samples(
        run,
        manifest.training_betas,
        role="training",
        count=sample_count,
    )
    validation = target_samples(
        run,
        manifest.training_betas,
        role="validation",
        count=sample_count,
    )
    return train(
        f"curie_confirmatory_{run.run_id[:12]}",
        arm_ansatz(run),
        condition_angles(manifest, run, manifest.training_betas),
        training,
        validation,
        run.seed,
        config,
        arm_encoding(run),
    )


def evaluate_run(
    manifest: CurieConfirmatoryManifest,
    run: CurieConfirmatoryRunIdentity,
    result: TrainingResult,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Exactly evaluate every training and held-out temperature."""
    betas = np.asarray(
        (*manifest.training_betas, *manifest.held_out_betas), dtype=np.float64
    )
    angles = condition_angles(manifest, run, betas)
    effective_ancillas = (
        0 if run.arm.value.startswith("visible_matched_r") else run.ancillas
    )
    probabilities = exact_ancilla_marginal_probabilities_from_full_angles(
        result.best_params,
        result.generators,
        angles,
        n_data_qubits=run.n_qubits,
        n_ancillas=effective_ancillas,
        encoding=arm_encoding(run),
    )
    bitstrings = enumerate_bitstrings(run.n_qubits)
    rows: list[dict[str, Any]] = []
    targets = []
    for beta, model in zip(betas, probabilities, strict=True):
        target = exact_curie_weiss_target(run.n_qubits, float(beta))
        targets.append(target.probabilities)
        row = thermodynamic_metrics(target, model).to_dict()
        row.update(
            {
                "beta": float(beta),
                "role": "training"
                if float(beta) in manifest.training_betas
                else "held_out",
                "critical_region": float(beta)
                in tuple(float(v) for v in manifest.raw["temperatures"]["critical"]),
                "exact_mmd": exact_mmd_squared(model, target.probabilities),
                "target_profile": target.profile.tolist(),
                "model_profile": shell_probabilities(model, bitstrings).tolist(),
            }
        )
        rows.append(row)
    metrics: dict[str, Any] = {
        "evaluation_rows": rows,
        "best_validation_conditional_mmd": result.best_validation_loss,
        "best_step": result.best_step,
        "parameter_count": int(result.generators.shape[0]),
        "encoder": arm_encoding(run),
        "encoder_parameters_trainable": False,
        "performance_thresholds_applied": False,
    }
    arrays = {
        "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
        "run_id": np.asarray(run.run_id, dtype="S64"),
        "betas": betas,
        "condition_angles": angles,
        "generators": np.asarray(result.generators, dtype=np.int8),
        "best_params": np.asarray(result.best_params, dtype=np.float64),
        "final_params": np.asarray(result.final_params, dtype=np.float64),
        "model_probabilities": np.asarray(probabilities, dtype=np.float64),
        "target_probabilities": np.asarray(targets, dtype=np.float64),
        "train_steps": result.train_steps,
        "train_losses": result.train_losses,
        "validation_steps": result.validation_steps,
        "validation_losses": result.validation_losses,
    }
    return metrics, arrays


def _source_digest(repository: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "configs/curie_weiss_confirmatory_v1.json",
        "src/iqp_cond/ancilla.py",
        "src/iqp_cond/ansatz.py",
        "src/iqp_cond/confirmatory_curie.py",
        "src/iqp_cond/confirmatory_curie_runner.py",
        "src/iqp_cond/curie_weiss_data.py",
        "src/iqp_cond/curie_weiss_statistics.py",
        "src/iqp_cond/encoding.py",
        "src/iqp_cond/gaussian_runner.py",
        "src/iqp_cond/training.py",
        "src/iqp_cond/mmd.py",
        "src/iqp_cond/simulator.py",
        "src/iqp_cond/exact.py",
    ):
        path = repository / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _portable_path(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.resolve()))
    except ValueError:
        return str(path.resolve())


def execute_run(
    repository: Path,
    manifest: CurieConfirmatoryManifest,
    run: CurieConfirmatoryRunIdentity,
    output_directory: Path,
    *,
    invocation: tuple[str, ...],
    updates: int | None = None,
    lightweight: bool = False,
) -> dict[str, Any]:
    """Train, evaluate, and atomically record one immutable run identity."""
    repository = repository.resolve()
    output_directory = output_directory.resolve()
    record_path = output_directory / "runs" / f"{run.run_id}.json"
    arrays_path = output_directory / "arrays" / f"{run.run_id}.npz"
    if record_path.exists():
        existing = read_immutable_json(record_path)
        if existing.get("run_identity") != run.to_dict():
            raise RuntimeError("result path contains a conflicting run identity")
        if not arrays_path.exists() or sha256_file(arrays_path) != existing.get(
            "arrays_sha256"
        ):
            raise RuntimeError("completed result has missing or conflicting arrays")
        return existing
    if arrays_path.exists():
        raise RuntimeError("orphaned arrays conflict with an incomplete run identity")

    started = datetime.now(UTC)
    timer = time.perf_counter()
    result = train_run(
        manifest,
        run,
        updates=updates,
        lightweight=lightweight,
    )
    metrics, arrays = evaluate_run(manifest, run, result)
    arrays_sha256 = write_immutable_npz(arrays_path, arrays)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "run_identity": run.to_dict(),
        "run_id": run.run_id,
        "code_revision": _git_revision(repository),
        "source_tree_digest": _source_digest(repository),
        "invocation": list(invocation),
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.perf_counter() - timer,
        "peak_memory_bytes": _peak_rss_bytes(),
        "platform": platform.platform(),
        "optimizer_updates": int(updates or manifest.raw["training"]["updates"]),
        "lightweight": lightweight,
        "arrays_path": _portable_path(arrays_path, repository),
        "arrays_sha256": arrays_sha256,
        "metrics": metrics,
    }
    try:
        write_immutable_json(record_path, payload)
    except DuplicateArtifactError:
        existing = read_immutable_json(record_path)
        if existing != payload:
            raise RuntimeError(
                "concurrent writer produced a conflicting result"
            ) from None
    return payload


def describe_run(run: CurieConfirmatoryRunIdentity) -> str:
    return json.dumps(
        {
            "run_id": run.run_id,
            "n_qubits": run.n_qubits,
            "arm": run.arm.value,
            "ancillas": run.ancillas,
            "seed": run.seed,
        },
        sort_keys=True,
    )

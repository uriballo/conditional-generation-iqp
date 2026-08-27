"""Train and exactly evaluate the frozen single-temperature references."""

from __future__ import annotations

import hashlib
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
from .ansatz import complete_ansatz
from .artifacts import (
    DuplicateArtifactError,
    read_immutable_json,
    sha256_file,
    write_immutable_json,
)
from .confirmatory_curie import CurieConfirmatoryManifest
from .curie_single_temperature import (
    ANCILLAS,
    SingleTemperatureManifest,
    SingleTemperatureRunIdentity,
    neutral_angles,
    split_seed,
    target_probabilities,
)
from .curie_weiss_data import exact_curie_weiss_target
from .curie_weiss_statistics import thermodynamic_metrics
from .encoding import SPIN_FLIP_BRANCH_ENCODING
from .exact import enumerate_bitstrings, shell_probabilities
from .gaussian_data import write_immutable_npz
from .gaussian_runner import exact_mmd_squared
from .training import TrainingConfig, TrainingResult, train


def training_config(
    manifest: SingleTemperatureManifest,
    *,
    updates: int | None = None,
    lightweight: bool = False,
) -> TrainingConfig:
    """Return the frozen per-condition training configuration."""
    settings = manifest.raw["training"]
    steps = int(settings["updates"] if updates is None else updates)
    if steps < 1:
        raise ValueError("updates must be positive")
    return TrainingConfig(
        steps=steps,
        batch_size=2 if lightweight else int(settings["batch_size"]),
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


def target_samples(
    run: SingleTemperatureRunIdentity,
    *,
    role: str,
    count: int,
) -> np.ndarray:
    """Draw one deterministic target set with role-separated randomness."""
    if count < 2:
        raise ValueError("target sample count must be at least two")
    bitstrings = enumerate_bitstrings(run.n_qubits)
    probabilities = target_probabilities(run.n_qubits, run.beta)
    rng = np.random.default_rng(split_seed(run, role))
    indices = rng.choice(probabilities.size, size=count, p=probabilities)
    return np.asarray(bitstrings[indices], dtype=np.int8)


def train_run(
    repository: Path,
    manifest: SingleTemperatureManifest,
    run: SingleTemperatureRunIdentity,
    *,
    updates: int | None = None,
    lightweight: bool = False,
) -> TrainingResult:
    """Fit one neutral-input $r=5$ circuit to one target temperature."""
    repository = repository.resolve()
    base = CurieConfirmatoryManifest.from_json(
        manifest.resolve_base_manifest(repository)
    )
    if base.digest != manifest.raw["base_manifest_digest"]:
        raise ValueError("base manifest digest differs")
    config = training_config(manifest, updates=updates, lightweight=lightweight)
    count = (
        max(config.batch_size, 2)
        if lightweight
        else int(manifest.raw["training"]["reference_minibatch"])
    )
    training = target_samples(run, role="training", count=count)
    validation = target_samples(run, role="validation", count=count)
    angles = neutral_angles(run.n_qubits)[None, :]
    return train(
        f"curie_single_{run.run_id[:12]}",
        complete_ansatz(run.n_qubits + ANCILLAS),
        angles,
        training[None, ...],
        validation[None, ...],
        run.seed,
        config,
        SPIN_FLIP_BRANCH_ENCODING,
    )


def evaluate_run(
    run: SingleTemperatureRunIdentity,
    result: TrainingResult,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Exactly evaluate the selected checkpoint at its fitted temperature."""
    angles = neutral_angles(run.n_qubits)[None, :]
    probabilities = exact_ancilla_marginal_probabilities_from_full_angles(
        result.best_params,
        result.generators,
        angles,
        n_data_qubits=run.n_qubits,
        n_ancillas=ANCILLAS,
        encoding=SPIN_FLIP_BRANCH_ENCODING,
    )[0]
    target = exact_curie_weiss_target(run.n_qubits, run.beta)
    bitstrings = enumerate_bitstrings(run.n_qubits)
    row = thermodynamic_metrics(target, probabilities).to_dict()
    row.update(
        {
            "beta": run.beta,
            "role": run.role,
            "exact_mmd": exact_mmd_squared(probabilities, target.probabilities),
            "target_profile": target.profile.tolist(),
            "model_profile": shell_probabilities(probabilities, bitstrings).tolist(),
        }
    )
    metrics = {
        "evaluation": row,
        "best_step": result.best_step,
        "best_validation_conditional_mmd": result.best_validation_loss,
        "parameter_count": int(result.generators.shape[0]),
        "encoder": SPIN_FLIP_BRANCH_ENCODING,
        "encoder_parameters_trainable": False,
        "performance_thresholds_applied": False,
    }
    arrays = {
        "manifest_digest": np.asarray(run.configuration_digest, dtype="S64"),
        "run_id": np.asarray(run.run_id, dtype="S64"),
        "beta": np.asarray(run.beta, dtype=np.float64),
        "condition_angles": angles,
        "generators": np.asarray(result.generators, dtype=np.int8),
        "best_params": np.asarray(result.best_params, dtype=np.float64),
        "final_params": np.asarray(result.final_params, dtype=np.float64),
        "model_probabilities": np.asarray(probabilities, dtype=np.float64),
        "target_probabilities": np.asarray(target.probabilities, dtype=np.float64),
        "train_steps": result.train_steps,
        "train_losses": result.train_losses,
        "validation_steps": result.validation_steps,
        "validation_losses": result.validation_losses,
    }
    return metrics, arrays


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_digest(repository: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "configs/curie_weiss_confirmatory_v1.json",
        "configs/curie_weiss_single_temperature_v1.json",
        "src/iqp_cond/ancilla.py",
        "src/iqp_cond/ansatz.py",
        "src/iqp_cond/artifacts.py",
        "src/iqp_cond/confirmatory_curie.py",
        "src/iqp_cond/curie_single_temperature.py",
        "src/iqp_cond/curie_single_temperature_runner.py",
        "src/iqp_cond/curie_weiss_data.py",
        "src/iqp_cond/curie_weiss_statistics.py",
        "src/iqp_cond/encoding.py",
        "src/iqp_cond/exact.py",
        "src/iqp_cond/gaussian_data.py",
        "src/iqp_cond/gaussian_runner.py",
        "src/iqp_cond/mmd.py",
        "src/iqp_cond/simulator.py",
        "src/iqp_cond/training.py",
        "scripts/run_curie_single_temperature.py",
    ):
        path = repository / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
    manifest: SingleTemperatureManifest,
    run: SingleTemperatureRunIdentity,
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
    expected_updates = int(
        manifest.raw["training"]["updates"] if updates is None else updates
    )
    if expected_updates < 1:
        raise ValueError("updates must be positive")

    if record_path.exists():
        existing = read_immutable_json(record_path)
        if existing.get("run_identity") != run.to_dict():
            raise RuntimeError("result path contains a conflicting run identity")
        if existing.get("manifest_digest") != manifest.digest:
            raise RuntimeError("result path contains a conflicting manifest")
        if existing.get("optimizer_updates") != expected_updates:
            raise RuntimeError("result path contains a conflicting optimizer budget")
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
        repository,
        manifest,
        run,
        updates=updates,
        lightweight=lightweight,
    )
    metrics, arrays = evaluate_run(run, result)
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
        "optimizer_updates": expected_updates,
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


def describe_run(run: SingleTemperatureRunIdentity) -> str:
    """Return a stable one-line JSON description for logs and dry runs."""
    import json

    return json.dumps(
        {
            "run_id": run.run_id,
            "n_qubits": run.n_qubits,
            "beta": run.beta,
            "role": run.role,
            "seed": run.seed,
            "parameter_count": run.parameter_count,
        },
        sort_keys=True,
    )

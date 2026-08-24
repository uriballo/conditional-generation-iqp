"""Train and exactly evaluate one declared Curie--Weiss model."""

from __future__ import annotations

import hashlib
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .ansatz import complete_ansatz
from .curie_weiss_campaign import CurieWeissCampaignStore, CurieWeissResultBundle
from .curie_weiss_data import (
    exact_curie_weiss_target,
    load_dataset_for_run,
    write_immutable_npz,
)
from .curie_weiss_protocol import (
    CurieWeissCampaignManifest,
    CurieWeissFamily,
    CurieWeissRunIdentity,
    conditioned_angle,
    plan_curie_weiss_runs,
)
from .curie_weiss_statistics import thermodynamic_metrics
from .curie_weiss_training import train_curie_weiss
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    shell_probabilities,
)
from .gaussian_runner import exact_mmd_squared
from .ising_training import runtime_environment
from .training import TrainingConfig


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_digest(repository: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "configs/curie_weiss_foundations_v1.json",
        "src/iqp_cond/curie_weiss_protocol.py",
        "src/iqp_cond/curie_weiss_data.py",
        "src/iqp_cond/curie_weiss_campaign.py",
        "src/iqp_cond/curie_weiss_training.py",
        "src/iqp_cond/curie_weiss_statistics.py",
        "src/iqp_cond/curie_weiss_runner.py",
        "src/iqp_cond/mmd.py",
        "src/iqp_cond/exact.py",
    ):
        digest.update(relative.encode())
        digest.update((repository / relative).read_bytes())
    return digest.hexdigest()


def training_seed(run: CurieWeissRunIdentity) -> int:
    """Return the literal matched optimizer seed frozen in the run identity."""
    return run.optimizer_seed


def training_config(
    manifest: CurieWeissCampaignManifest,
    run: CurieWeissRunIdentity,
    *,
    updates: int | None = None,
    lightweight: bool = False,
) -> TrainingConfig:
    settings = manifest.raw["training"]
    conditions = 1 if run.family is CurieWeissFamily.TEMPERATURE_REFERENCE else 6
    batch = 450 if conditions == 1 else 75
    return TrainingConfig(
        steps=int(settings["updates"] if updates is None else updates),
        batch_size=batch,
        learning_rate=float(settings["initial_learning_rate"]),
        final_learning_rate=float(settings["final_learning_rate"]),
        validation_interval=min(int(settings["validation_interval"]), updates or 2_000),
        sigmas=tuple(float(v) for v in settings["bandwidths"]),
        n_ops=2 if lightweight else int(settings["training_operators"]),
        n_iqp_samples=16 if lightweight else int(settings["training_model_samples"]),
        validation_n_ops=2 if lightweight else int(settings["validation_operators"]),
        validation_n_iqp_samples=16
        if lightweight
        else int(settings["validation_model_samples"]),
    )


def evaluation_angles(
    manifest: CurieWeissCampaignManifest,
    run: CurieWeissRunIdentity,
) -> np.ndarray:
    if run.family is CurieWeissFamily.NO_CONDITION:
        beta_inputs = [1.0] * 11
    elif run.family is CurieWeissFamily.TEMPERATURE_REFERENCE:
        beta_inputs = [float(run.reference_beta)] * 11
    else:
        beta_inputs = list(manifest.evaluation_betas)
    return np.asarray([conditioned_angle(beta, run.n_qubits) for beta in beta_inputs])


def evaluate_run(
    manifest: CurieWeissCampaignManifest,
    run: CurieWeissRunIdentity,
    params: np.ndarray,
    generators: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    betas = np.asarray(manifest.evaluation_betas)
    angles = evaluation_angles(manifest, run)
    probabilities = exact_iqp_probabilities_many(params, generators, angles)
    bitstrings = enumerate_bitstrings(run.n_qubits)
    rows = []
    target_probabilities = []
    for beta, model in zip(betas, probabilities, strict=True):
        target = exact_curie_weiss_target(run.n_qubits, float(beta))
        metrics = thermodynamic_metrics(target, model).to_dict()
        metrics.update(
            {
                "beta": float(beta),
                "role": "training"
                if float(beta) in manifest.training_betas
                else "held_out",
                "exact_conditional_mmd": exact_mmd_squared(model, target.probabilities),
                "target_profile": target.profile.tolist(),
                "model_profile": shell_probabilities(model, bitstrings).tolist(),
            }
        )
        rows.append(metrics)
        target_probabilities.append(target.probabilities)
    return (
        {"evaluation_rows": rows, "performance_thresholds_applied": False},
        {
            "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
            "run_id": np.asarray(run.run_id, dtype="S64"),
            "betas": betas,
            "roles": np.asarray([row["role"] for row in rows], dtype="S8"),
            "condition_angles": angles,
            "model_probabilities": probabilities,
            "target_probabilities": np.asarray(target_probabilities),
        },
    )


def _portable(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.resolve()))
    except ValueError:
        return str(path.resolve())


def execute_curie_weiss_run(
    repository: Path,
    manifest: CurieWeissCampaignManifest,
    store: CurieWeissCampaignStore,
    dataset_directory: Path,
    run: CurieWeissRunIdentity,
    *,
    invocation: tuple[str, ...],
    updates: int | None = None,
    require_preflight: bool = True,
    lightweight: bool = False,
) -> CurieWeissResultBundle:
    repository = repository.resolve()
    store.validate_declared(run)
    if store.has_complete(run):
        return store.load_result(run)
    if require_preflight:
        evidence = store.require_preflight()
        if evidence.get("source_tree_digest") != source_digest(repository):
            raise ValueError("preflight does not bind the current source")
    store.reserve(run, require_preflight=require_preflight)
    started, timer = datetime.now(UTC), time.perf_counter()
    data = load_dataset_for_run(manifest, dataset_directory, run)
    config = training_config(manifest, run, updates=updates, lightweight=lightweight)
    result = train_curie_weiss(
        complete_ansatz(run.n_qubits),
        run.family,
        data.condition_angles,
        data.training,
        data.validation,
        training_seed(run),
        config,
    )
    metrics, probability_arrays = evaluate_run(
        manifest,
        run,
        result.best_params,
        result.generators,
    )
    metrics.update(
        {
            "best_validation_conditional_mmd": result.best_validation_loss,
            "best_step": result.best_step,
            "optimizer_updates": config.steps,
            "source_tree_digest": source_digest(repository),
            "learned_coefficient_count": int(result.expansion.shape[1]),
            "compiled_generator_count": int(result.expansion.shape[0]),
        }
    )
    checkpoint = store.checkpoint_path(run)
    checkpoint_sha = write_immutable_npz(
        checkpoint,
        {
            "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
            "run_id": np.asarray(run.run_id, dtype="S64"),
            "generators": result.generators.astype(np.int8),
            "expansion": result.expansion,
            "best_coefficients": result.best_coefficients,
            "final_coefficients": result.final_coefficients,
            "best_params": result.best_params,
            "final_params": result.final_params,
            "best_step": np.asarray(result.best_step),
            "best_validation_loss": np.asarray(result.best_validation_loss),
            "train_steps": result.train_steps,
            "train_losses": result.train_losses,
            "validation_steps": result.validation_steps,
            "validation_losses": result.validation_losses,
        },
    )
    probability_path = store.probabilities_path(run)
    probability_sha = write_immutable_npz(probability_path, probability_arrays)
    completed = datetime.now(UTC)
    source = source_digest(repository)
    bundle = CurieWeissResultBundle(
        run,
        manifest.digest,
        _git_revision(repository),
        {
            **runtime_environment(repository / "uv.lock"),
            "platform": platform.platform(),
            "source_tree_digest": source,
        },
        invocation,
        started.isoformat(),
        completed.isoformat(),
        time.perf_counter() - timer,
        _peak_rss_bytes(),
        _portable(data.source_path, repository),
        data.source_sha256,
        _portable(checkpoint, repository),
        checkpoint_sha,
        _portable(probability_path, repository),
        probability_sha,
        {
            "optimizer_seed": training_seed(run),
            "data_seed": run.data_seed,
            "training_estimator_seed": training_seed(run),
            "validation_estimator_seed": training_seed(run) + 1_000_000,
        },
        metrics,
    )
    store.record_result(bundle)
    return bundle


def runtime_preflight(
    repository: Path,
    manifest: CurieWeissCampaignManifest,
    dataset_directory: Path,
) -> dict[str, object]:
    run = next(
        r
        for r in plan_curie_weiss_runs(manifest)
        if r.n_qubits == 16
        and r.optimizer_seed == 0
        and r.family is CurieWeissFamily.CONDITIONED
    )
    data = load_dataset_for_run(manifest, dataset_directory, run)
    config = training_config(manifest, run, updates=100)
    start = time.perf_counter()
    result = train_curie_weiss(
        complete_ansatz(16),
        run.family,
        data.condition_angles,
        data.training,
        data.validation,
        0,
        config,
    )
    elapsed = time.perf_counter() - start
    peak = _peak_rss_bytes()
    projected = elapsed * 20
    limits = manifest.raw["runtime_preflight"]
    finite = bool(
        np.isfinite(result.train_losses).all()
        and np.isfinite(result.validation_losses).all()
    )
    passed = (
        finite
        and projected <= limits["max_projected_seconds_per_model"]
        and peak <= limits["max_peak_rss_bytes"]
    )
    return {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "updates": 100,
        "elapsed_seconds": elapsed,
        "projected_seconds_per_model": projected,
        "peak_rss_bytes": peak,
        "finite": finite,
        "passed": passed,
        "performance_gate": False,
        "source_tree_digest": source_digest(repository.resolve()),
    }

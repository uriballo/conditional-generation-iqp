"""Training and exact evaluation for one declared Gaussian campaign run."""

from __future__ import annotations

import hashlib
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from math import exp
from pathlib import Path

import numpy as np

from .ansatz import complete_ansatz
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    fwht,
    shell_probabilities,
    total_variation_distance,
)
from .gaussian_campaign import (
    GaussianCampaignStore,
    GaussianResultBundle,
)
from .gaussian_data import (
    condition_angle_matrix,
    gaussian_target,
    load_dataset_for_run,
    master_grid,
    write_immutable_npz,
)
from .gaussian_protocol import (
    GaussianCampaignManifest,
    GaussianFamily,
    GaussianRunIdentity,
    plan_gaussian_runs,
)
from .ising_training import runtime_environment
from .training import TrainingConfig, train


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


def _source_digest(repository: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "configs/gaussian_foundations_v1.json",
        "src/iqp_cond/gaussian_protocol.py",
        "src/iqp_cond/gaussian_data.py",
        "src/iqp_cond/gaussian_campaign.py",
        "src/iqp_cond/gaussian_runner.py",
        "src/iqp_cond/training.py",
        "src/iqp_cond/mmd.py",
        "src/iqp_cond/exact.py",
    ):
        path = repository / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _portable_path(path: Path, repository: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.resolve()))
    except ValueError:
        return str(path.resolve())


def training_seed(run: GaussianRunIdentity) -> int:
    if run.family is not GaussianFamily.SINGLE_PEAK:
        return run.optimizer_seed
    return run.optimizer_seed + 10_000 * (int(run.bank_peak_index) + 1)


def training_config(
    manifest: GaussianCampaignManifest,
    run: GaussianRunIdentity,
    *,
    updates: int | None = None,
) -> TrainingConfig:
    n_conditions = 1 if run.family is GaussianFamily.SINGLE_PEAK else run.peak_count
    if 450 % n_conditions:
        raise ValueError("total target minibatch must divide evenly by conditions")
    settings = manifest.raw["training"]
    return TrainingConfig(
        steps=int(settings["updates"] if updates is None else updates),
        batch_size=450 // n_conditions,
        learning_rate=float(settings["initial_learning_rate"]),
        final_learning_rate=float(settings["final_learning_rate"]),
        validation_interval=min(int(settings["validation_interval"]), updates or 2_000),
        sigmas=tuple(float(value) for value in settings["bandwidths"]),
        n_ops=int(settings["training_operators"]),
        n_iqp_samples=int(settings["training_model_samples"]),
        validation_n_ops=int(settings["validation_operators"]),
        validation_n_iqp_samples=int(settings["validation_model_samples"]),
    )


def exact_mmd_squared(
    model_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    sigmas: tuple[float, ...] = (1.0, 2.0, 4.0),
) -> float:
    model = np.asarray(fwht(model_probabilities).real, dtype=np.float64)
    target = np.asarray(fwht(target_probabilities).real, dtype=np.float64)
    n_qubits = round(np.log2(model.size))
    weights = enumerate_bitstrings(n_qubits).sum(axis=1)
    losses = []
    for sigma in sigmas:
        probability = (1.0 - exp(-1.0 / (2.0 * sigma**2))) / 2.0
        mask_law = probability**weights * (1.0 - probability) ** (n_qubits - weights)
        losses.append(float(np.sum(mask_law * (model - target) ** 2)))
    return float(np.mean(losses))


def _evaluate(
    manifest: GaussianCampaignManifest,
    run: GaussianRunIdentity,
    params: np.ndarray,
    generators: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if run.family is GaussianFamily.SINGLE_PEAK:
        conditions = master_grid()[[int(run.bank_peak_index)]]
        roles = np.asarray(["training"], dtype="S8")
    else:
        conditions = np.asarray(
            [*manifest.training_grid(run.peak_count), *manifest.held_out_grid],
            dtype=np.float64,
        )
        roles = np.asarray(
            ["training"] * run.peak_count + ["held_out"] * 8,
            dtype="S8",
        )
    angles = (
        np.zeros((conditions.size, run.n_qubits), dtype=np.float64)
        if run.family is GaussianFamily.NO_CONDITION
        else condition_angle_matrix(conditions, run.n_qubits)
    )
    model_probabilities = exact_iqp_probabilities_many(params, generators, angles)
    bitstrings = enumerate_bitstrings(run.n_qubits)
    weights = bitstrings.sum(axis=1).astype(np.float64)
    rows = []
    for index, condition in enumerate(conditions):
        target = gaussian_target(run.n_qubits, run.width_factor, float(condition))
        target_probabilities = target.bitstring_probabilities(bitstrings)
        model = model_probabilities[index]
        model_shells = shell_probabilities(model, bitstrings)
        model_mean = float(weights @ model)
        model_std = float(
            np.sqrt(max(float(((weights - model_mean) ** 2) @ model), 0.0))
        )
        rows.append(
            {
                "role": roles[index].decode(),
                "condition": float(condition),
                "target_mean": target.actual_mean,
                "model_mean": model_mean,
                "mean_absolute_error": abs(model_mean - target.actual_mean),
                "target_std": target.actual_std,
                "model_std": model_std,
                "std_absolute_error": abs(model_std - target.actual_std),
                "profile_tvd": total_variation_distance(
                    target.shell_probabilities, model_shells
                ),
                "bitstring_tvd": total_variation_distance(target_probabilities, model),
                "exact_conditional_mmd": exact_mmd_squared(model, target_probabilities),
                "target_shell_probabilities": target.shell_probabilities.tolist(),
                "model_shell_probabilities": model_shells.tolist(),
            }
        )
    return (
        {
            "evaluation_rows": rows,
            "best_validation_conditional_mmd": None,
            "source_tree_digest": "filled_by_executor",
            "performance_thresholds_applied": False,
        },
        {
            "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
            "run_id": np.asarray(run.run_id, dtype="S64"),
            "conditions": conditions,
            "roles": roles,
            "condition_angles": angles,
            "model_probabilities": model_probabilities,
        },
    )


def execute_gaussian_run(
    repository: Path,
    manifest: GaussianCampaignManifest,
    store: GaussianCampaignStore,
    dataset_directory: Path,
    run: GaussianRunIdentity,
    *,
    invocation: tuple[str, ...],
    updates: int | None = None,
    require_preflight: bool = True,
) -> GaussianResultBundle:
    repository = repository.resolve()
    store.validate_declared(run)
    if store.has_complete(run):
        return store.load_result(run)
    if require_preflight:
        preflight = store.require_preflight()
        if preflight.get("source_tree_digest") != _source_digest(repository):
            raise ValueError(
                "runtime preflight does not bind the current training source"
            )
    store.reserve(run, require_preflight=require_preflight)
    started = datetime.now(UTC)
    start_time = time.perf_counter()
    data = load_dataset_for_run(manifest, dataset_directory, run)
    config = training_config(manifest, run, updates=updates)
    ansatz = complete_ansatz(run.n_qubits)
    result = train(
        f"gaussian_{run.run_id[:12]}",
        ansatz,
        data.condition_angles,
        data.training,
        data.validation,
        training_seed(run),
        config,
    )
    metrics, probability_arrays = _evaluate(
        manifest, run, result.best_params, np.asarray(result.generators)
    )
    source_digest = _source_digest(repository)
    metrics["best_validation_conditional_mmd"] = result.best_validation_loss
    metrics["source_tree_digest"] = source_digest
    metrics["best_step"] = result.best_step
    metrics["training_count_per_peak"] = data.training_count_per_peak
    metrics["validation_count_per_peak"] = data.validation_count_per_peak
    metrics["optimizer_updates"] = config.steps
    checkpoint = store.checkpoint_path(run)
    checkpoint_sha = write_immutable_npz(
        checkpoint,
        {
            "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
            "run_id": np.asarray(run.run_id, dtype="S64"),
            "generators": np.asarray(result.generators, dtype=np.int8),
            "condition_angles": result.condition_angles,
            "best_params": result.best_params,
            "final_params": result.final_params,
            "best_step": np.asarray(result.best_step, dtype=np.int64),
            "best_validation_loss": np.asarray(
                result.best_validation_loss, dtype=np.float64
            ),
            "train_steps": result.train_steps,
            "train_losses": result.train_losses,
            "validation_steps": result.validation_steps,
            "validation_losses": result.validation_losses,
        },
    )
    probabilities = store.probabilities_path(run)
    probabilities_sha = write_immutable_npz(probabilities, probability_arrays)
    completed = datetime.now(UTC)
    bundle = GaussianResultBundle(
        run=run,
        manifest_digest=manifest.digest,
        code_revision=_git_revision(repository),
        environment={
            **runtime_environment(repository / "uv.lock"),
            "platform": platform.platform(),
            "source_tree_digest": source_digest,
        },
        invocation=invocation,
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        runtime_seconds=time.perf_counter() - start_time,
        peak_memory_bytes=_peak_rss_bytes(),
        dataset_path=_portable_path(data.source_path, repository),
        dataset_sha256=data.source_sha256,
        checkpoint_path=_portable_path(checkpoint, repository),
        checkpoint_sha256=checkpoint_sha,
        probabilities_path=_portable_path(probabilities, repository),
        probabilities_sha256=probabilities_sha,
        estimator_randomness={
            "optimizer_seed": training_seed(run),
            "data_seed": run.data_seed,
            "training_estimator_seed": training_seed(run),
            "validation_estimator_seed": training_seed(run) + 1_000_000,
        },
        metrics=metrics,
    )
    store.record_result(bundle)
    return bundle


def runtime_preflight(
    repository: Path,
    manifest: GaussianCampaignManifest,
    dataset_directory: Path,
) -> dict[str, object]:
    # The preflight uses a synthetic declared-shape run and never enters the
    # immutable 1,020-result ledger.
    run = next(
        run
        for run in plan_gaussian_runs(manifest)
        if run.n_qubits == 19
        and run.width_factor == 0.5
        and run.peak_count == 9
        and run.family is GaussianFamily.CONDITIONED
        and run.optimizer_seed == 0
    )
    data = load_dataset_for_run(manifest, dataset_directory, run)
    config = training_config(manifest, run, updates=100)
    started = time.perf_counter()
    result = train(
        "gaussian_runtime_preflight",
        complete_ansatz(19),
        data.condition_angles,
        data.training,
        data.validation,
        training_seed(run),
        config,
    )
    elapsed = time.perf_counter() - started
    projected = elapsed * 20.0
    peak = _peak_rss_bytes()
    finite = bool(
        np.isfinite(result.train_losses).all()
        and np.isfinite(result.validation_losses).all()
    )
    limits = manifest.raw["runtime_preflight"]
    passed = (
        finite
        and projected <= float(limits["max_projected_seconds_per_model"])
        and peak <= int(limits["max_peak_rss_bytes"])
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
        "source_tree_digest": _source_digest(repository.resolve()),
    }

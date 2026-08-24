"""Matched-capacity visible-only control for the Curie-Weiss v2 campaign."""

from __future__ import annotations

import hashlib
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .artifacts import read_immutable_json, sha256_file, write_immutable_json
from .curie_weiss_ancilla_protocol import (
    CurieWeissAncillaFamily,
    CurieWeissAncillaManifest,
    CurieWeissAncillaRunIdentity,
    broadcast_angle,
)
from .curie_weiss_ancilla_runner import (
    _full_mask_mmd,
    _generator_identity_initialization,
    load_reused_dataset,
    training_config,
)
from .curie_weiss_data import exact_curie_weiss_target
from .curie_weiss_statistics import thermodynamic_metrics
from .exact import enumerate_bitstrings, exact_iqp_probabilities_many
from .gaussian_data import write_immutable_npz
from .gaussian_statistics import paired_bootstrap_interval
from .mmd import mmd_mask_weights
from .training import _batch_schedule

jax.config.update("jax_enable_x64", True)

CONTROL_VERSION = "curie_weiss_visible_123local_v2"
VISIBLE_QUBITS = 8
PARAMETER_COUNT = 92
CRITICAL_BETAS = (0.9, 1.0, 1.1)
COLD_BETAS = (1.3, 1.4, 1.5)


def visible_123local_generators(n_qubits: int = VISIBLE_QUBITS) -> np.ndarray:
    """Return every one-, two-, and three-body Z generator."""
    rows: list[np.ndarray] = []
    for locality in (1, 2, 3):
        for support in itertools.combinations(range(n_qubits), locality):
            row = np.zeros(n_qubits, dtype=np.int8)
            row[list(support)] = 1
            rows.append(row)
    generators = np.asarray(rows, dtype=np.int8)
    expected = sum(
        len(tuple(itertools.combinations(range(n_qubits), locality)))
        for locality in (1, 2, 3)
    )
    if generators.shape != (expected, n_qubits):
        raise AssertionError("visible generator enumeration is incomplete")
    return generators


def conditioned_run(
    manifest: CurieWeissAncillaManifest, seed: int
) -> CurieWeissAncillaRunIdentity:
    return CurieWeissAncillaRunIdentity(
        manifest.campaign_version,
        manifest.visible_qubits,
        manifest.headline_ancillas,
        CurieWeissAncillaFamily.CONDITIONED,
        None,
        seed,
        manifest.digest,
    )


def source_digest(repository: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "configs/curie_weiss_m5_foundations_v2.json",
        "src/iqp_cond/curie_weiss_capacity_control.py",
        "src/iqp_cond/curie_weiss_ancilla_protocol.py",
        "src/iqp_cond/curie_weiss_ancilla_runner.py",
        "src/iqp_cond/mmd.py",
        "src/iqp_cond/simulator.py",
        "src/iqp_cond/exact.py",
    ):
        path = repository / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ControlTrainingResult:
    best_params: np.ndarray
    final_params: np.ndarray
    best_step: int
    best_validation_loss: float
    validation_steps: np.ndarray
    validation_losses: np.ndarray


def train_control(
    repository: Path,
    manifest: CurieWeissAncillaManifest,
    seed: int,
) -> ControlTrainingResult:
    run = conditioned_run(manifest, seed)
    dataset = load_reused_dataset(repository, manifest, run)
    config = training_config(manifest, run)
    generators_np = visible_123local_generators()
    generators = jnp.asarray(generators_np, dtype=jnp.float64)
    masks = jnp.asarray(enumerate_bitstrings(VISIBLE_QUBITS), dtype=jnp.float64)
    weights = mmd_mask_weights(VISIBLE_QUBITS)
    angles = jnp.asarray(
        [
            broadcast_angle(manifest, beta, VISIBLE_QUBITS)
            for beta in manifest.training_betas
        ]
    )
    params = _generator_identity_initialization(seed, generators_np)
    schedule = optax.cosine_decay_schedule(
        config.learning_rate,
        config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    state = optimizer.init(params)
    batches = _batch_schedule(
        dataset.training.shape[0], dataset.training.shape[1], config, seed
    )

    def loss(current, targets, key, samples):
        return _full_mask_mmd(
            current,
            generators,
            angles,
            targets,
            config.sigmas,
            samples,
            key,
            masks,
            masks,
            weights,
        )

    value_gradient = jax.jit(jax.value_and_grad(loss), static_argnums=(3,))
    validate = jax.jit(
        lambda current: loss(
            current,
            jnp.asarray(dataset.validation),
            jax.random.PRNGKey(seed + 1_000_000),
            config.validation_n_iqp_samples,
        )
    )
    best = final = params
    best_step = 0
    best_validation = float(validate(params))
    validation_steps = [0]
    validation_losses = [best_validation]
    base_key = jax.random.PRNGKey(seed)
    for step in range(config.steps):
        indices = batches[step]
        batch = np.stack(
            [
                dataset.training[index, indices[index]]
                for index in range(dataset.training.shape[0])
            ]
        )
        _, gradient = value_gradient(
            final,
            jnp.asarray(batch),
            jax.random.fold_in(base_key, step),
            config.n_iqp_samples,
        )
        updates, state = optimizer.update(gradient, state, final)
        final = optax.apply_updates(final, updates)
        completed = step + 1
        if completed % config.validation_interval == 0:
            current = float(validate(final))
            validation_steps.append(completed)
            validation_losses.append(current)
            if current < best_validation:
                best = final
                best_step = completed
                best_validation = current
    return ControlTrainingResult(
        np.asarray(best),
        np.asarray(final),
        best_step,
        best_validation,
        np.asarray(validation_steps),
        np.asarray(validation_losses),
    )


def evaluate_control(
    manifest: CurieWeissAncillaManifest, params: np.ndarray
) -> tuple[list[dict[str, Any]], np.ndarray]:
    generators = visible_123local_generators()
    angles = np.asarray(
        [
            broadcast_angle(manifest, beta, VISIBLE_QUBITS)
            for beta in manifest.evaluation_betas
        ]
    )
    probabilities = exact_iqp_probabilities_many(params, generators, angles)
    rows: list[dict[str, Any]] = []
    for beta, model in zip(manifest.evaluation_betas, probabilities, strict=True):
        target = exact_curie_weiss_target(VISIBLE_QUBITS, beta)
        metrics = thermodynamic_metrics(target, model).to_dict()
        metrics.update(
            {
                "beta": beta,
                "role": "training"
                if beta in manifest.training_betas
                else "held_out",
            }
        )
        rows.append(metrics)
    return rows, probabilities


def run_seed(
    repository: Path,
    manifest: CurieWeissAncillaManifest,
    output: Path,
    seed: int,
) -> Path:
    result_path = output / "results" / f"seed-{seed}.json"
    if result_path.exists():
        return result_path
    output.joinpath("results").mkdir(parents=True, exist_ok=True)
    output.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
    output.joinpath("probabilities").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = train_control(repository, manifest, seed)
    rows, probabilities = evaluate_control(manifest, result.best_params)
    generators = visible_123local_generators()
    checkpoint_path = output / "checkpoints" / f"seed-{seed}.npz"
    checkpoint_sha = write_immutable_npz(
        checkpoint_path,
        {
            "best_params": result.best_params,
            "final_params": result.final_params,
            "generators": generators,
            "best_step": np.asarray(result.best_step),
            "best_validation_loss": np.asarray(result.best_validation_loss),
            "validation_steps": result.validation_steps,
            "validation_losses": result.validation_losses,
        },
    )
    probability_path = output / "probabilities" / f"seed-{seed}.npz"
    probability_sha = write_immutable_npz(
        probability_path,
        {
            "betas": np.asarray(manifest.evaluation_betas),
            "model_probabilities": probabilities,
        },
    )
    source = manifest.source_for_seed(seed)
    payload = {
        "schema_version": 1,
        "control_version": CONTROL_VERSION,
        "source_manifest_digest": manifest.digest,
        "source_tree_digest": source_digest(repository),
        "seed": seed,
        "visible_qubits": VISIBLE_QUBITS,
        "generator_localities": [1, 2, 3],
        "learned_parameter_count": len(generators),
        "optimizer_updates": int(manifest.raw["training"]["updates"]),
        "dataset_path": source.path,
        "dataset_sha256": source.sha256,
        "estimator_randomness": {
            "training_estimator_seed": seed,
            "validation_estimator_seed": seed + 1_000_000,
        },
        "checkpoint_path": str(checkpoint_path.relative_to(repository)),
        "checkpoint_sha256": checkpoint_sha,
        "probabilities_path": str(probability_path.relative_to(repository)),
        "probabilities_sha256": probability_sha,
        "best_step": result.best_step,
        "best_validation_conditional_mmd": result.best_validation_loss,
        "runtime_seconds": time.perf_counter() - started,
        "evaluation_rows": rows,
    }
    write_immutable_json(result_path, payload)
    return result_path


def _ancilla_rows(report_path: Path) -> dict[tuple[int, float], dict[str, Any]]:
    report = read_immutable_json(report_path)
    rows = report["response_rows"]
    return {
        (int(row["seed"]), float(row["beta"])): row["conditioned"] for row in rows
    }


def _mean_for_betas(
    rows: dict[float, dict[str, Any]], metric: str, betas: tuple[float, ...]
) -> float:
    return float(np.mean([rows[beta][metric] for beta in betas]))


def summarize(
    repository: Path,
    manifest: CurieWeissAncillaManifest,
    output: Path,
    ancilla_report: Path,
) -> dict[str, Any]:
    control: dict[int, dict[float, dict[str, Any]]] = {}
    for seed in manifest.optimizer_seeds:
        payload = read_immutable_json(output / "results" / f"seed-{seed}.json")
        if (
            payload["source_manifest_digest"] != manifest.digest
            or payload["source_tree_digest"] != source_digest(repository)
        ):
            raise ValueError("control result provenance differs from current source")
        control[seed] = {
            float(row["beta"]): row for row in payload["evaluation_rows"]
        }
    ancilla = _ancilla_rows(ancilla_report)
    regions = {
        "training": manifest.training_betas,
        "held_out": manifest.held_out_betas,
        "critical": CRITICAL_BETAS,
        "cold": COLD_BETAS,
    }
    comparisons: list[dict[str, Any]] = []
    for metric in ("profile_tvd", "bitstring_tvd"):
        for region, betas in regions.items():
            effects: dict[int, float] = {}
            control_means: dict[int, float] = {}
            ancilla_means: dict[int, float] = {}
            for seed in manifest.optimizer_seeds:
                control_mean = _mean_for_betas(control[seed], metric, betas)
                ancilla_mean = float(
                    np.mean([ancilla[(seed, beta)][metric] for beta in betas])
                )
                control_means[seed] = control_mean
                ancilla_means[seed] = ancilla_mean
                effects[seed] = control_mean - ancilla_mean
            interval = paired_bootstrap_interval(
                effects, replicates=10_000, seed=20_260_815
            )
            comparisons.append(
                {
                    "metric": metric,
                    "region": region,
                    "betas": list(betas),
                    "direction": "positive means lower TVD for the ancilla model",
                    "ancilla_mean": float(np.mean(list(ancilla_means.values()))),
                    "visible_control_mean": float(
                        np.mean(list(control_means.values()))
                    ),
                    "interval": interval.to_dict(),
                }
            )
    summary = {
        "schema_version": 1,
        "control_version": CONTROL_VERSION,
        "status": "complete",
        "source_manifest_digest": manifest.digest,
        "source_tree_digest": source_digest(repository),
        "ancilla_report_sha256": sha256_file(ancilla_report),
        "control_model": {
            "visible_qubits": VISIBLE_QUBITS,
            "generator_localities": [1, 2, 3],
            "learned_parameter_count": PARAMETER_COUNT,
        },
        "ancilla_model": {
            "visible_qubits": VISIBLE_QUBITS,
            "ancillas": 5,
            "learned_parameter_count": 91,
        },
        "paired_seeds": list(manifest.optimizer_seeds),
        "bootstrap_replicates": 10_000,
        "comparisons": comparisons,
    }
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary_path.unlink()
    write_immutable_json(summary_path, summary)
    return summary


def load_manifest(path: Path) -> CurieWeissAncillaManifest:
    return CurieWeissAncillaManifest.from_json(path)


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Matched-capacity Curie-Weiss control",
        "",
        "Positive paired effects mean that the five-ancilla model has lower TVD.",
        "Conditions are averaged within each optimizer seed before bootstrapping.",
        "",
        "| Metric | Region | Ancilla | Visible 1/2/3-local | Effect [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["comparisons"]:
        interval = row["interval"]
        lines.append(
            f"| {row['metric']} | {row['region']} | {row['ancilla_mean']:.3f} | "
            f"{row['visible_control_mean']:.3f} | {interval['estimate']:+.3f} "
            f"[{interval['lower']:+.3f}, {interval['upper']:+.3f}] |"
        )
    return "\n".join(lines) + "\n"

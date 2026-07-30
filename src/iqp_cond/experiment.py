"""Experiment matrix construction, exact evaluation, and result persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .ansatz import Ansatz, complete_ansatz, heavy_hex_ansatz, lattice_ansatz
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities,
    shell_probabilities,
    total_variation_distance,
)
from .mmd import conditional_mmd
from .targets import GaussianHammingTarget
from .training import TrainingConfig, TrainingResult, train


@dataclass(frozen=True)
class ModelSpec:
    """One row of the agreed experiment matrix."""

    name: str
    ansatz: Ansatz
    condition_angles: np.ndarray
    target_indices: tuple[int, ...]


def targets(n_qubits: int = 19) -> tuple[GaussianHammingTarget, ...]:
    """Return the two calibrated Gaussian Hamming-weight targets."""
    width = sqrt(n_qubits) / 2.0
    return (
        GaussianHammingTarget.create(n_qubits, n_qubits / 2.0, width),
        GaussianHammingTarget.create(n_qubits, n_qubits / 4.0, width),
    )


def model_specs(n_qubits: int = 19) -> dict[str, ModelSpec]:
    """Return all primary models and baselines."""
    if n_qubits != 19:
        raise ValueError("the agreed experiment matrix uses 19 qubits")
    primary_angles = np.stack(
        [
            np.zeros(n_qubits),
            np.full(n_qubits, np.pi / 4.0),
        ],
    )
    specs = {}
    for ansatz in (heavy_hex_ansatz(), lattice_ansatz(), complete_ansatz()):
        name = f"angle_{ansatz.name}"
        specs[name] = ModelSpec(name, ansatz, primary_angles, (0, 1))

    complete = complete_ansatz()
    specs["no_condition_complete"] = ModelSpec(
        "no_condition_complete",
        complete,
        np.full((2, n_qubits), np.pi / 8.0),
        (0, 1),
    )
    specs["basis_condition_complete"] = ModelSpec(
        "basis_condition_complete",
        complete,
        np.stack([np.zeros(n_qubits), np.full(n_qubits, np.pi)]),
        (0, 1),
    )
    specs["separate_a_complete"] = ModelSpec(
        "separate_a_complete",
        complete,
        primary_angles[[0]],
        (0,),
    )
    specs["separate_b_complete"] = ModelSpec(
        "separate_b_complete",
        complete,
        primary_angles[[1]],
        (1,),
    )
    return specs


def datasets(
    seed: int,
    n_training: int = 10_000,
    n_validation: int = 2_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate matched, disjoint condition-balanced datasets."""
    rng = np.random.default_rng(seed + 100_000)
    target_pair = targets()
    training = np.stack([target.sample(rng, n_training) for target in target_pair])
    validation = np.stack(
        [target.sample(rng, n_validation) for target in target_pair],
    )
    return training, validation


def _weight_moments(shells: np.ndarray) -> tuple[float, float]:
    weights = np.arange(shells.size, dtype=np.float64)
    mean = float(weights @ shells)
    variance = float(((weights - mean) ** 2) @ shells)
    return mean, sqrt(max(variance, 0.0))


def evaluate(
    result: TrainingResult,
    target_indices: tuple[int, ...],
    validation_targets: np.ndarray,
    config: TrainingConfig,
) -> dict[str, object]:
    """Evaluate a trained model exactly and with held-out Conditional MMD."""
    bitstrings = enumerate_bitstrings(result.generators.shape[1])
    target_pair = targets(result.generators.shape[1])
    conditions = []
    for local_index, target_index in enumerate(target_indices):
        target = target_pair[target_index]
        model_probs = exact_iqp_probabilities(
            result.best_params,
            result.generators,
            result.condition_angles[local_index],
        )
        target_probs = target.bitstring_probabilities(bitstrings)
        model_shells = shell_probabilities(model_probs, bitstrings)
        model_mean, model_std = _weight_moments(model_shells)
        conditions.append(
            {
                "condition": "A" if target_index == 0 else "B",
                "bitstring_tvd": total_variation_distance(
                    target_probs,
                    model_probs,
                ),
                "shell_tvd": total_variation_distance(
                    target.shell_probabilities,
                    model_shells,
                ),
                "target_mean": target.actual_mean,
                "model_mean": model_mean,
                "mean_absolute_error": abs(model_mean - target.actual_mean),
                "target_std": target.actual_std,
                "model_std": model_std,
                "std_absolute_error": abs(model_std - target.actual_std),
                "target_shell_probabilities": target.shell_probabilities.tolist(),
                "model_shell_probabilities": model_shells.tolist(),
            },
        )

    heldout = conditional_mmd(
        jnp.asarray(result.best_params),
        jnp.asarray(result.generators),
        jnp.asarray(result.condition_angles),
        jnp.asarray(validation_targets[list(target_indices)]),
        config.sigmas,
        config.validation_n_ops,
        config.validation_n_iqp_samples,
        jax.random.PRNGKey(result.seed + 2_000_000),
    )
    return {
        "model_name": result.model_name,
        "topology": result.topology,
        "seed": result.seed,
        "n_parameters": int(result.generators.shape[0]),
        "best_step": result.best_step,
        "best_validation_loss": result.best_validation_loss,
        "heldout_mmd": float(heldout.loss),
        "heldout_mmd_per_condition": np.asarray(
            heldout.per_condition,
        ).tolist(),
        "conditions": conditions,
    }


def run_one(
    spec: ModelSpec,
    seed: int,
    config: TrainingConfig,
) -> tuple[TrainingResult, dict[str, object]]:
    """Train and evaluate one model/seed configuration."""
    training_data, validation_data = datasets(seed)
    result = train(
        spec.name,
        spec.ansatz,
        spec.condition_angles,
        training_data[list(spec.target_indices)],
        validation_data[list(spec.target_indices)],
        seed,
        config,
    )
    metrics = evaluate(result, spec.target_indices, validation_data, config)
    return result, metrics


def save_result(
    result: TrainingResult,
    metrics: dict[str, object],
    config: TrainingConfig,
    directory: Path,
) -> None:
    """Persist parameters, histories, metrics, and the training configuration."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{result.model_name}_seed{result.seed}"
    np.savez_compressed(
        directory / f"{stem}.npz",
        generators=result.generators,
        condition_angles=result.condition_angles,
        best_params=result.best_params,
        final_params=result.final_params,
        train_steps=result.train_steps,
        train_losses=result.train_losses,
        validation_steps=result.validation_steps,
        validation_losses=result.validation_losses,
    )
    payload = dict(metrics)
    payload["training_config"] = asdict(config)
    (directory / f"{stem}.json").write_text(
        json.dumps(payload, indent=2) + "\n",
    )

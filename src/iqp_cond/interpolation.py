"""Held-out angle interpolation experiment for the conditional IQP model."""

from __future__ import annotations

import json
from dataclasses import asdict
from math import pi, sqrt
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .ansatz import complete_ansatz
from .encoding import ANGLE_ENCODING
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    shell_probabilities,
    total_variation_distance,
)
from .mmd import conditional_mmd
from .targets import GaussianHammingTarget
from .training import TrainingConfig, TrainingResult, train

TRAIN_ANGLE_FRACTIONS = np.asarray([0, 2, 4, 6, 8], dtype=np.float64) / 32.0
HELDOUT_ANGLE_FRACTIONS = np.asarray([1, 3, 5, 7], dtype=np.float64) / 32.0


def target_mean_for_angle(n_qubits: int, angle: float) -> float:
    """Map an encoding angle in ``[0, pi/4]`` to its target Hamming mean."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    if not 0.0 <= angle <= pi / 4.0:
        raise ValueError("angle must lie between 0 and pi/4")
    return n_qubits / 2.0 - n_qubits * angle / pi


def interpolation_targets(
    scalar_angles: np.ndarray,
    n_qubits: int = 19,
    width_scale: float = 1.0,
) -> tuple[GaussianHammingTarget, ...]:
    """Construct the Gaussian targets assigned to scalar encoding angles.

    ``width_scale`` shrinks the peak width (default ``1.0`` keeps the
    canonical width ``sqrt(n_qubits)/2``); smaller scales spread the peaks
    apart relative to their width.
    """
    if width_scale <= 0.0:
        raise ValueError("width_scale must be positive")
    width = width_scale * sqrt(n_qubits) / 2.0
    return tuple(
        GaussianHammingTarget.create(
            n_qubits,
            target_mean_for_angle(n_qubits, float(angle)),
            width,
        )
        for angle in np.asarray(scalar_angles)
    )


def broadcast_angles(
    scalar_angles: np.ndarray,
    n_qubits: int = 19,
) -> np.ndarray:
    """Broadcast one scalar encoding angle over all qubits."""
    scalar_angles = np.asarray(scalar_angles, dtype=np.float64)
    if scalar_angles.ndim != 1:
        raise ValueError("scalar_angles must be one-dimensional")
    return np.repeat(scalar_angles[:, None], n_qubits, axis=1)


def sample_angle_targets(
    scalar_angles: np.ndarray,
    seed: int,
    n_samples: int,
    n_qubits: int = 19,
    width_scale: float = 1.0,
) -> np.ndarray:
    """Sample an aligned target dataset for each supplied angle."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            target.sample(rng, n_samples)
            for target in interpolation_targets(
                scalar_angles,
                n_qubits,
                width_scale,
            )
        ],
    )


def _weight_moments(shells: np.ndarray) -> tuple[float, float]:
    weights = np.arange(shells.size, dtype=np.float64)
    mean = float(weights @ shells)
    variance = float(((weights - mean) ** 2) @ shells)
    return mean, sqrt(max(variance, 0.0))


def point_metrics(
    angle: float,
    role: str,
    target: GaussianHammingTarget,
    target_probabilities: np.ndarray,
    model_probabilities: np.ndarray,
    bitstrings: np.ndarray,
) -> dict[str, object]:
    model_shells = shell_probabilities(model_probabilities, bitstrings)
    model_mean, model_std = _weight_moments(model_shells)
    return {
        "role": role,
        "angle": angle,
        "angle_fraction_pi": angle / pi,
        "target_mean": target.actual_mean,
        "model_mean": model_mean,
        "mean_absolute_error": abs(model_mean - target.actual_mean),
        "target_std": target.actual_std,
        "model_std": model_std,
        "std_absolute_error": abs(model_std - target.actual_std),
        "bitstring_tvd": total_variation_distance(
            target_probabilities,
            model_probabilities,
        ),
        "shell_tvd": total_variation_distance(
            target.shell_probabilities,
            model_shells,
        ),
        "target_shell_probabilities": target.shell_probabilities.tolist(),
        "model_shell_probabilities": model_shells.tolist(),
    }


def run_interpolation(
    seed: int,
    config: TrainingConfig,
    encoding: str = ANGLE_ENCODING,
) -> tuple[TrainingResult, dict[str, object]]:
    """Train on five angles and evaluate four interleaved held-out angles."""
    n_qubits = 19
    train_angles = pi * TRAIN_ANGLE_FRACTIONS
    heldout_angles = pi * HELDOUT_ANGLE_FRACTIONS
    ansatz = complete_ansatz()
    training_data = sample_angle_targets(
        train_angles,
        seed + 300_000,
        10_000,
        n_qubits,
    )
    validation_data = sample_angle_targets(
        train_angles,
        seed + 400_000,
        2_000,
        n_qubits,
    )
    heldout_data = sample_angle_targets(
        heldout_angles,
        seed + 500_000,
        2_000,
        n_qubits,
    )
    model_name = (
        "interpolation_complete"
        if encoding == ANGLE_ENCODING
        else f"interpolation_{encoding}_complete"
    )
    result = train(
        model_name,
        ansatz,
        broadcast_angles(train_angles, n_qubits),
        training_data,
        validation_data,
        seed,
        config,
        encoding,
    )

    all_angles = np.concatenate([train_angles, heldout_angles])
    all_roles = ["train"] * len(train_angles) + ["heldout"] * len(heldout_angles)
    all_targets = interpolation_targets(all_angles, n_qubits)
    bitstrings = enumerate_bitstrings(n_qubits)
    model_probabilities = exact_iqp_probabilities_many(
        result.best_params,
        result.generators,
        broadcast_angles(all_angles, n_qubits),
        encoding,
    )
    points = []
    for angle, role, target, model_probs in zip(
        all_angles,
        all_roles,
        all_targets,
        model_probabilities,
        strict=True,
    ):
        target_probs = target.bitstring_probabilities(bitstrings)
        point = point_metrics(
            float(angle),
            role,
            target,
            target_probs,
            model_probs,
            bitstrings,
        )
        if role == "heldout":
            nearest_index = int(np.argmin(np.abs(train_angles - angle)))
            nearest_probs = model_probabilities[nearest_index]
            nearest_shells = shell_probabilities(nearest_probs, bitstrings)
            point.update(
                {
                    "nearest_train_angle": float(train_angles[nearest_index]),
                    "nearest_train_angle_fraction_pi": float(
                        TRAIN_ANGLE_FRACTIONS[nearest_index],
                    ),
                    "nearest_bitstring_tvd": total_variation_distance(
                        target_probs,
                        nearest_probs,
                    ),
                    "nearest_shell_tvd": total_variation_distance(
                        target.shell_probabilities,
                        nearest_shells,
                    ),
                },
            )
        points.append(point)

    heldout = conditional_mmd(
        jnp.asarray(result.best_params),
        jnp.asarray(result.generators),
        jnp.asarray(broadcast_angles(heldout_angles, n_qubits)),
        jnp.asarray(heldout_data),
        config.sigmas,
        config.validation_n_ops,
        config.validation_n_iqp_samples,
        jax.random.PRNGKey(seed + 3_000_000),
        encoding,
    )
    metrics = {
        "model_name": result.model_name,
        "encoding": encoding,
        "topology": result.topology,
        "seed": seed,
        "n_parameters": int(result.generators.shape[0]),
        "best_step": result.best_step,
        "best_validation_loss": result.best_validation_loss,
        "heldout_mmd": float(heldout.loss),
        "heldout_mmd_per_angle": np.asarray(heldout.per_condition).tolist(),
        "target_mapping": "mean(alpha) = n/2 - n*alpha/pi",
        "train_angle_fractions_pi": TRAIN_ANGLE_FRACTIONS.tolist(),
        "heldout_angle_fractions_pi": HELDOUT_ANGLE_FRACTIONS.tolist(),
        "points": points,
    }
    return result, metrics


def save_interpolation_result(
    result: TrainingResult,
    metrics: dict[str, object],
    config: TrainingConfig,
    directory: Path,
) -> None:
    """Persist one interpolation seed and its complete evaluation."""
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
        encoding=np.asarray(result.encoding),
    )
    payload = dict(metrics)
    payload["training_config"] = asdict(config)
    (directory / f"{stem}.json").write_text(
        json.dumps(payload, indent=2) + "\n",
    )

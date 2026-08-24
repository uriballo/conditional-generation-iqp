"""Sparse-grid held-out interpolation for the ancilla-assisted model."""

from __future__ import annotations

import json
from dataclasses import asdict
from math import pi
from pathlib import Path

import numpy as np

from .ancilla import (
    ancilla_ansatz,
    ancilla_condition_angles,
    exact_ancilla_marginal_probabilities,
)
from .encoding import ANGLE_ENCODING
from .exact import (
    enumerate_bitstrings,
    shell_probabilities,
    total_variation_distance,
)
from .interpolation import (
    broadcast_angles,
    interpolation_targets,
    point_metrics,
    sample_angle_targets,
)
from .targets import GaussianHammingTarget
from .training import TrainingConfig, TrainingResult, train

SPARSE2_TRAIN_FRACTIONS = np.asarray([0, 8], dtype=np.float64) / 32.0
SPARSE2_HELDOUT_FRACTIONS = np.asarray([2, 4, 6], dtype=np.float64) / 32.0
SPARSE3_TRAIN_FRACTIONS = np.asarray([0, 4, 8], dtype=np.float64) / 32.0
SPARSE3_HELDOUT_FRACTIONS = np.asarray([2, 6], dtype=np.float64) / 32.0

GRID_FRACTIONS = {
    2: (SPARSE2_TRAIN_FRACTIONS, SPARSE2_HELDOUT_FRACTIONS),
    3: (SPARSE3_TRAIN_FRACTIONS, SPARSE3_HELDOUT_FRACTIONS),
}


def grid_fractions(
    n_peaks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (training, held-out) angle fractions for a sparse grid."""
    if n_peaks not in GRID_FRACTIONS:
        raise ValueError(
            f"unknown sparse grid with {n_peaks} training peaks; "
            f"expected one of {sorted(GRID_FRACTIONS)}",
        )
    train_fractions, heldout_fractions = GRID_FRACTIONS[n_peaks]
    return train_fractions.copy(), heldout_fractions.copy()


def oracle_adjacent_lookup(
    train_probabilities: np.ndarray,
    train_fractions: np.ndarray,
    heldout_fractions: np.ndarray,
    targets: tuple[GaussianHammingTarget, ...],
    bitstrings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower TVD over the two training angles adjacent to each held-out angle.

    Returns ``(shell_tvd, bitstring_tvd)`` arrays aligned with
    ``heldout_fractions``. The oracle is a memorization diagnostic: beating it
    requires doing better than either stored neighboring output.
    """
    train_fractions = np.asarray(train_fractions, dtype=np.float64)
    heldout_fractions = np.asarray(heldout_fractions, dtype=np.float64)
    train_probabilities = np.asarray(train_probabilities, dtype=np.float64)
    if train_probabilities.ndim != 2:
        raise ValueError("train_probabilities must be rank two")
    if train_probabilities.shape[0] != train_fractions.size:
        raise ValueError("one probability row per training angle")
    if len(targets) != heldout_fractions.size:
        raise ValueError("one target per held-out angle")

    shell_tvds = np.empty(heldout_fractions.size, dtype=np.float64)
    bitstring_tvds = np.empty(heldout_fractions.size, dtype=np.float64)
    for heldout_index, (fraction, target) in enumerate(
        zip(heldout_fractions, targets, strict=True),
    ):
        distances = np.abs(train_fractions - fraction)
        adjacent_indices = np.argsort(distances, kind="stable")[:2]
        target_probs = target.bitstring_probabilities(bitstrings)
        adjacent_shell = []
        adjacent_bitstring = []
        for index in adjacent_indices:
            stored_probs = train_probabilities[index]
            adjacent_shell.append(
                total_variation_distance(
                    target.shell_probabilities,
                    shell_probabilities(stored_probs, bitstrings),
                ),
            )
            adjacent_bitstring.append(
                total_variation_distance(target_probs, stored_probs),
            )
        shell_tvds[heldout_index] = min(adjacent_shell)
        bitstring_tvds[heldout_index] = min(adjacent_bitstring)
    return shell_tvds, bitstring_tvds


def run_sparse_ancilla(
    n_peaks: int,
    n_ancillas: int,
    seed: int,
    config: TrainingConfig,
    encoding: str = ANGLE_ENCODING,
    n_data_qubits: int = 19,
    scheme: str = "broadcast_all",
    width_scale: float = 1.0,
) -> tuple[TrainingResult, dict[str, object]]:
    """Train on a sparse grid of peak positions and evaluate held-out peaks."""
    train_fractions, heldout_fractions = grid_fractions(n_peaks)
    train_angles = pi * train_fractions
    heldout_angles = pi * heldout_fractions
    ansatz = ancilla_ansatz(n_data_qubits, n_ancillas)

    training_data = sample_angle_targets(
        train_angles,
        seed + 300_000,
        10_000,
        n_data_qubits,
        width_scale,
    )
    validation_data = sample_angle_targets(
        train_angles,
        seed + 400_000,
        2_000,
        n_data_qubits,
        width_scale,
    )

    data_condition_angles = broadcast_angles(train_angles, n_data_qubits)
    full_condition_angles = ancilla_condition_angles(
        data_condition_angles,
        n_ancillas,
        scheme,
    )
    model_name = f"sparse{n_peaks}_ancilla_{n_ancillas}_{scheme}_complete"
    if width_scale != 1.0:
        model_name += f"_w{width_scale:g}"
    result = train(
        model_name,
        ansatz,
        full_condition_angles,
        training_data,
        validation_data,
        seed,
        config,
        encoding,
    )

    all_angles = np.concatenate([train_angles, heldout_angles])
    all_roles = ["train"] * len(train_angles) + ["heldout"] * len(heldout_angles)
    all_targets = interpolation_targets(all_angles, n_data_qubits, width_scale)
    bitstrings = enumerate_bitstrings(n_data_qubits)
    marginal_probabilities = exact_ancilla_marginal_probabilities(
        result.best_params,
        result.generators,
        broadcast_angles(all_angles, n_data_qubits),
        n_data_qubits=n_data_qubits,
        n_ancillas=n_ancillas,
        scheme=scheme,
        encoding=encoding,
    )

    points = []
    for angle, role, target, model_probs in zip(
        all_angles,
        all_roles,
        all_targets,
        marginal_probabilities,
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
            point.update(
                {
                    "nearest_train_angle_fraction_pi": float(
                        train_fractions[nearest_index],
                    ),
                },
            )
        points.append(point)

    heldout_points = [p for p in points if p["role"] == "heldout"]
    metrics = {
        "model_name": result.model_name,
        "n_peaks": n_peaks,
        "n_ancillas": n_ancillas,
        "total_qubits": n_data_qubits + n_ancillas,
        "n_parameters": ansatz.n_parameters,
        "scheme": scheme,
        "encoding": encoding,
        "seed": seed,
        "best_step": result.best_step,
        "best_validation_loss": result.best_validation_loss,
        "target_mapping": "mean(alpha) = n/2 - n*alpha/pi",
        "target_width_scale": width_scale,
        "train_angle_fractions_pi": train_fractions.tolist(),
        "heldout_angle_fractions_pi": heldout_fractions.tolist(),
        "heldout_shell_tvd": float(
            np.mean([p["shell_tvd"] for p in heldout_points]),
        ),
        "heldout_bitstring_tvd": float(
            np.mean([p["bitstring_tvd"] for p in heldout_points]),
        ),
        "points": points,
    }
    return result, metrics


def save_sparse_result(
    result: TrainingResult,
    metrics: dict[str, object],
    config: TrainingConfig,
    directory: Path,
) -> None:
    """Persist one sparse-grid seed and its complete evaluation."""
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

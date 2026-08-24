"""Exact feasibility utilities for two-coordinate conditional generation."""

from __future__ import annotations

from math import pi

import numpy as np

from .exact import (
    enumerate_bitstrings,
    shell_probabilities,
    total_variation_distance,
)
from .targets import GaussianHammingTarget

N_FEASIBILITY_QUBITS = 8
TRAIN_CONDITIONS = np.asarray(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    dtype=np.float64,
)
HELDOUT_CONDITIONS = np.asarray(
    [
        [0.5, 0.0],
        [0.5, 0.5],
        [0.5, 1.0],
        [0.0, 0.5],
        [1.0, 0.5],
    ],
    dtype=np.float64,
)


def _validate_conditions(conditions: np.ndarray) -> np.ndarray:
    conditions = np.asarray(conditions, dtype=np.float64)
    if conditions.ndim == 1:
        conditions = conditions[None, :]
    if conditions.ndim != 2 or conditions.shape[1] != 2:
        raise ValueError("conditions must have shape (n_conditions, 2)")
    if not np.all((0.0 <= conditions) & (conditions <= 1.0)):
        raise ValueError("condition coordinates must lie in [0, 1]")
    return conditions


def target_parameters(
    condition: np.ndarray | list[float],
    n_qubits: int = N_FEASIBILITY_QUBITS,
) -> tuple[float, float]:
    """Return the frozen target mean and shell width for one Condition."""
    condition = _validate_conditions(np.asarray(condition))[0]
    if n_qubits < 6:
        raise ValueError("n_qubits must be at least six")
    mean = n_qubits / 2.0 - 0.5 - 1.5 * condition[0]
    width = 0.75 + 0.75 * condition[1]
    return float(mean), float(width)


def gaussian_target(
    condition: np.ndarray | list[float],
    n_qubits: int = N_FEASIBILITY_QUBITS,
) -> GaussianHammingTarget:
    """Construct the exact Target assigned to one two-coordinate Condition."""
    mean, width = target_parameters(condition, n_qubits)
    return GaussianHammingTarget.create(n_qubits, mean, width)


def targets_for_conditions(
    conditions: np.ndarray,
    n_qubits: int = N_FEASIBILITY_QUBITS,
) -> tuple[GaussianHammingTarget, ...]:
    """Construct exact Targets in the supplied Condition order."""
    conditions = _validate_conditions(conditions)
    return tuple(gaussian_target(condition, n_qubits) for condition in conditions)


def condition_angles(
    conditions: np.ndarray,
    n_qubits: int = N_FEASIBILITY_QUBITS,
) -> np.ndarray:
    """Map two coordinates to a fixed block-contrast product-state encoding."""
    conditions = _validate_conditions(conditions)
    if n_qubits < 4 or n_qubits % 2:
        raise ValueError("n_qubits must be an even number of at least four")
    common = (pi / 6.0) * conditions[:, 0]
    contrast = (pi / 16.0) * (2.0 * conditions[:, 1] - 1.0)
    half = n_qubits // 2
    return np.concatenate(
        [
            np.repeat((common + contrast)[:, None], half, axis=1),
            np.repeat((common - contrast)[:, None], half, axis=1),
        ],
        axis=1,
    )


def target_probability_matrix(
    conditions: np.ndarray,
    n_qubits: int = N_FEASIBILITY_QUBITS,
) -> np.ndarray:
    """Enumerate exact bitstring probabilities for a Condition batch."""
    bitstrings = enumerate_bitstrings(n_qubits)
    return np.stack(
        [
            target.bitstring_probabilities(bitstrings)
            for target in targets_for_conditions(conditions, n_qubits)
        ],
    )


def distribution_metrics(
    target: GaussianHammingTarget,
    model_probabilities: np.ndarray,
    bitstrings: np.ndarray | None = None,
) -> dict[str, float]:
    """Return exact full-distribution and structural diagnostics."""
    model_probabilities = np.asarray(model_probabilities, dtype=np.float64)
    if bitstrings is None:
        bitstrings = enumerate_bitstrings(target.n_qubits)
    bitstrings = np.asarray(bitstrings)
    if model_probabilities.shape != (2**target.n_qubits,):
        raise ValueError("model_probabilities must contain every bitstring")
    if not np.isfinite(model_probabilities).all() or model_probabilities.min() < 0.0:
        raise ValueError("model_probabilities must be finite and non-negative")
    if not np.isclose(model_probabilities.sum(), 1.0, atol=1e-8):
        raise ValueError("model_probabilities must sum to one")

    model_shells = shell_probabilities(model_probabilities, bitstrings)
    weights = bitstrings.sum(axis=1).astype(np.float64)
    model_mean = float(weights @ model_probabilities)
    model_variance = float(((weights - model_mean) ** 2) @ model_probabilities)
    target_mean = target.actual_mean
    target_std = target.actual_std
    return {
        "bitstring_tvd": total_variation_distance(
            target.bitstring_probabilities(bitstrings),
            model_probabilities,
        ),
        "shell_tvd": total_variation_distance(
            target.shell_probabilities,
            model_shells,
        ),
        "mean_error": abs(model_mean - target_mean),
        "std_error": abs(np.sqrt(max(model_variance, 0.0)) - target_std),
        "target_mean": target_mean,
        "model_mean": model_mean,
        "target_std": target_std,
        "model_std": np.sqrt(max(model_variance, 0.0)),
    }


def oracle_lookup(
    stored_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    training_conditions: np.ndarray,
) -> dict[str, object]:
    """Return the best stored output for one target by exact bitstring TVD."""
    stored_probabilities = np.asarray(stored_probabilities, dtype=np.float64)
    target_probabilities = np.asarray(target_probabilities, dtype=np.float64)
    training_conditions = _validate_conditions(training_conditions)
    if stored_probabilities.ndim != 2:
        raise ValueError(
            "stored_probabilities must have shape (n_conditions, n_states)",
        )
    if stored_probabilities.shape[0] != training_conditions.shape[0]:
        raise ValueError("stored probabilities and training Conditions must align")
    if (
        target_probabilities.ndim != 1
        or target_probabilities.shape[0] != stored_probabilities.shape[1]
    ):
        raise ValueError("target_probabilities must align with stored probabilities")
    tvds = np.asarray(
        [
            total_variation_distance(target_probabilities, candidate)
            for candidate in stored_probabilities
        ],
    )
    best = int(np.argmin(tvds))
    return {
        "lookup_index": best,
        "lookup_condition": training_conditions[best].tolist(),
        "bitstring_tvd": float(tvds[best]),
    }

"""Evaluation metrics for the Gaussian studies.

All metrics consume exact visible distributions from
:func:`iqp_cond.simulation.probabilities`, which are little-endian: index bit
``q`` belongs to qubit ``q``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bitstring_tvd",
    "hamming_profile",
    "mean_weight",
    "moment_errors",
    "profile_tvd",
]


def bitstring_tvd(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """Total-variation distance between two ``(..., 2**n)`` distributions."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("distributions must have the same shape")
    return 0.5 * np.abs(left - right).sum(axis=-1)


def hamming_profile(probabilities: np.ndarray, n_qubits: int) -> np.ndarray:
    """Collapse a ``(..., 2**n)`` distribution to its weight-shell profile."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape[-1] != 2**n_qubits:
        raise ValueError("the last axis must have size 2**n_qubits")
    weights = np.bitwise_count(np.arange(2**n_qubits)).astype(np.int64)
    profile = np.zeros((*values.shape[:-1], n_qubits + 1), dtype=np.float64)
    for weight in range(n_qubits + 1):
        profile[..., weight] = values[..., weights == weight].sum(axis=-1)
    return profile


def profile_tvd(
    first: np.ndarray,
    second: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """Total-variation distance between Hamming-weight profiles."""
    return bitstring_tvd(
        hamming_profile(first, n_qubits),
        hamming_profile(second, n_qubits),
    )


def mean_weight(probabilities: np.ndarray, n_qubits: int) -> np.ndarray:
    """Expected Hamming weight of a ``(..., 2**n)`` distribution."""
    profile = hamming_profile(probabilities, n_qubits)
    weights = np.arange(n_qubits + 1, dtype=np.float64)
    return profile @ weights


def moment_errors(
    model: np.ndarray,
    target: np.ndarray,
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean error / n, sd error / sqrt(n))`` per condition."""
    model_profile = hamming_profile(model, n_qubits)
    target_profile = hamming_profile(target, n_qubits)
    weights = np.arange(n_qubits + 1, dtype=np.float64)

    def moments(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = profile @ weights
        deviations = weights - mean[..., None]
        variance = (profile * deviations**2).sum(axis=-1)
        return mean, np.sqrt(np.maximum(variance, 0.0))

    model_mean, model_std = moments(model_profile)
    target_mean, target_std = moments(target_profile)
    return (
        (model_mean - target_mean) / n_qubits,
        (model_std - target_std) / np.sqrt(n_qubits),
    )

"""Evaluation metrics for the Gaussian studies.

All metrics consume exact visible distributions from
:func:`iqp_cond.simulation.probabilities`, which are little-endian: index bit
``q`` belongs to qubit ``q``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bitstring_tvd",
    "bootstrap_mean_interval",
    "complement_tvd",
    "hamming_profile",
    "mean_weight",
    "moment_errors",
    "paired_bootstrap_interval",
    "profile_tvd",
    "within_shell_residual",
]


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int = 0,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a deterministic percentile-bootstrap interval for a mean."""
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 2 or not np.all(np.isfinite(samples)):
        raise ValueError("values must be a finite one-dimensional sample")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, samples.size, size=(n_resamples, samples.size))
    means = samples[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def paired_bootstrap_interval(
    first: np.ndarray,
    second: np.ndarray,
    *,
    seed: int = 0,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap the seed-paired mean difference ``first - second``."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired samples must have the same shape")
    return bootstrap_mean_interval(
        left - right,
        seed=seed,
        n_resamples=n_resamples,
        confidence=confidence,
    )


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


def within_shell_residual(probabilities: np.ndarray, n_qubits: int) -> np.ndarray:
    """TV distance from the uniform distribution within each weight shell.

    The residual is the sum of each shell's probability mass times the
    conditional TV distance from uniform over bitstrings in that shell. It is
    zero exactly when the distribution is exchangeable under wire
    permutations within each Hamming-weight shell.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape[-1] != 2**n_qubits:
        raise ValueError("the last axis must have size 2**n_qubits")
    weights = np.bitwise_count(np.arange(2**n_qubits)).astype(np.int64)
    result = np.zeros(values.shape[:-1], dtype=np.float64)
    for weight in range(n_qubits + 1):
        shell = weights == weight
        shell_values = values[..., shell]
        mass = shell_values.sum(axis=-1)
        uniform_probability = mass[..., None] / int(shell.sum())
        result += 0.5 * np.abs(shell_values - uniform_probability).sum(axis=-1)
    return result


def complement_tvd(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compare ``second`` with the bitwise complement of ``first``.

    The last axis must represent a full ``2**n`` state distribution. For
    little-endian indexing, complementing all bits reverses that axis.
    """
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("distributions must have the same shape")
    if left.ndim == 0:
        raise ValueError("distributions must have a state axis")
    size = left.shape[-1]
    if size < 1 or size & (size - 1):
        raise ValueError("the last axis must have a power-of-two size")
    return bitstring_tvd(left[..., ::-1], right)


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

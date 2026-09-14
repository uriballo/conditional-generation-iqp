"""Exact finite-size Curie--Weiss target distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exact import enumerate_bitstrings


@dataclass(frozen=True, slots=True)
class CurieWeissTarget:
    n_qubits: int
    beta: float
    probabilities: np.ndarray
    magnetization: np.ndarray
    energy: np.ndarray

    @property
    def profile(self) -> np.ndarray:
        bits = enumerate_bitstrings(self.n_qubits)
        return np.bincount(
            bits.sum(axis=1),
            weights=self.probabilities,
            minlength=self.n_qubits + 1,
        )


def exact_curie_weiss_target(n_qubits: int, beta: float) -> CurieWeissTarget:
    if n_qubits not in {8, 12, 16} or not np.isfinite(beta) or beta <= 0:
        raise ValueError("invalid Curie--Weiss target coordinates")
    bits = enumerate_bitstrings(n_qubits)
    magnetization_sum = (2.0 * bits - 1.0).sum(axis=1)
    energy = -(magnetization_sum**2) / (2.0 * n_qubits)
    log_weights = -beta * energy
    probabilities = np.exp(log_weights - np.max(log_weights))
    probabilities /= probabilities.sum()
    return CurieWeissTarget(
        n_qubits,
        float(beta),
        probabilities,
        magnetization_sum / n_qubits,
        energy,
    )

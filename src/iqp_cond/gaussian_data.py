"""Gaussian Hamming-profile targets and the broadcast condition angle map."""

from __future__ import annotations

from math import pi, sqrt

import numpy as np

from .targets import GaussianHammingTarget


def target_mean(n_qubits: int, condition: float) -> float:
    if n_qubits < 2 or not 0.0 <= condition <= 1.0:
        raise ValueError("target coordinates are out of range")
    return n_qubits * (0.15 + 0.70 * condition)


def target_width(n_qubits: int, width_factor: float) -> float:
    if n_qubits < 2 or width_factor <= 0.0:
        raise ValueError("target width coordinates are out of range")
    return width_factor * sqrt(n_qubits)


def gaussian_target(
    n_qubits: int,
    width_factor: float,
    condition: float,
) -> GaussianHammingTarget:
    return GaussianHammingTarget.create(
        n_qubits,
        target_mean(n_qubits, condition),
        target_width(n_qubits, width_factor),
    )


def condition_angle(condition: float) -> float:
    if not 0.0 <= condition <= 1.0:
        raise ValueError("condition must lie in [0, 1]")
    return pi * (0.35 - 0.70 * condition)


def condition_angle_matrix(conditions: np.ndarray, n_qubits: int) -> np.ndarray:
    conditions = np.asarray(conditions, dtype=np.float64)
    if conditions.ndim != 1 or not np.all((0.0 <= conditions) & (conditions <= 1.0)):
        raise ValueError("conditions must be a one-dimensional array in [0, 1]")
    return np.repeat(
        np.asarray([condition_angle(float(value)) for value in conditions])[:, None],
        n_qubits,
        axis=1,
    )

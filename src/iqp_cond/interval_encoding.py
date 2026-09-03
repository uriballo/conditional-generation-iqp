"""Exploratory interval-based Gaussian condition encoding.

This module turns scalar conditions into per-qubit angle matrices around the
baseline ``alpha(c) = pi * (0.35 - 0.70 * c)``.  The resulting variation is a
classical ensemble of encodings: averaging output probabilities from several
draws is an ensemble average, not the output of one product-state circuit.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ArrayLike = float | list[float] | tuple[float, ...] | np.ndarray
WidthSpec = ArrayLike | Callable[[np.ndarray], ArrayLike]


def baseline_angle(conditions: ArrayLike) -> np.ndarray:
    """Return the affine baseline angle for each condition."""
    values = _conditions(conditions)
    return np.pi * (0.35 - 0.70 * values)


def build_interval_encoding(
    conditions: ArrayLike,
    n_qubits: int,
    half_width: WidthSpec = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Build a deterministic per-condition, per-qubit interval encoding.

    Each entry is sampled uniformly from ``[alpha(c)-w(c), alpha(c)+w(c)]``.
    A zero width returns the baseline exactly, independent of ``seed``.
    """
    values = _conditions(conditions)
    _validate_n_qubits(n_qubits)
    widths = _widths(values, half_width)
    center = baseline_angle(values)[:, None]
    if np.all(widths == 0.0):
        return np.broadcast_to(center, (values.size, n_qubits)).copy()
    rng = np.random.default_rng(seed)
    draws = rng.uniform(
        -widths[:, None], widths[:, None], (values.size, n_qubits)
    )
    return center + draws


def build_inference_ensemble(
    conditions: ArrayLike,
    n_qubits: int,
    n_members: int,
    half_width: WidthSpec = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Build ``n_members`` independently drawn interval-encoding matrices.

    The result has shape ``(n_members, n_conditions, n_qubits)`` and is
    reproducible for a fixed seed.
    """
    if isinstance(n_members, bool) or not isinstance(n_members, (int, np.integer)):
        raise ValueError("n_members must be a positive integer")
    if n_members <= 0:
        raise ValueError("n_members must be a positive integer")
    values = _conditions(conditions)
    _validate_n_qubits(n_qubits)
    widths = _widths(values, half_width)
    center = baseline_angle(values)[None, :, None]
    if np.all(widths == 0.0):
        return np.broadcast_to(center, (n_members, values.size, n_qubits)).copy()
    rng = np.random.default_rng(seed)
    draws = rng.uniform(
        -widths[None, :, None], widths[None, :, None],
        (n_members, values.size, n_qubits),
    )
    return center + draws


def _conditions(conditions: ArrayLike) -> np.ndarray:
    values = np.asarray(conditions, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ValueError(
            "conditions must be a non-empty one-dimensional array in [0, 1]"
        )
    return values


def _validate_n_qubits(n_qubits: int) -> None:
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, (int, np.integer)):
        raise ValueError("n_qubits must be a positive integer")
    if n_qubits <= 0:
        raise ValueError("n_qubits must be a positive integer")


def _widths(conditions: np.ndarray, spec: WidthSpec) -> np.ndarray:
    raw = spec(conditions) if callable(spec) else spec
    widths = np.asarray(raw, dtype=float)
    if widths.ndim == 0:
        widths = np.full(conditions.shape, widths.item())
    if widths.shape != conditions.shape or not np.all(np.isfinite(widths)):
        raise ValueError(
            "half_width must be a finite scalar or one value per condition"
        )
    if np.any(widths < 0):
        raise ValueError("half_width must be non-negative")
    return widths

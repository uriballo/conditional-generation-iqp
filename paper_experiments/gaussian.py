"""Reviewed discrete Gaussian shell family."""

from __future__ import annotations

import iqp_cond as ciqp
import numpy as np

from .config import RunConfig

N_VISIBLE = 16
_SHELL_INDICES = {}


def _states(n_visible: int) -> np.ndarray:
    values = np.arange(2**n_visible, dtype=np.int64)
    return ((values[:, None] >> np.arange(n_visible)) & 1).astype(np.float64)


def _shell_profile(conditions: np.ndarray) -> np.ndarray:
    weights = np.arange(N_VISIBLE + 1, dtype=np.float64)
    means = 5.0 + 6.0 * np.asarray(conditions, dtype=np.float64).reshape(-1, 1)
    profile = np.exp(-((weights[None, :] - means) ** 2) / 8.0)
    return profile / profile.sum(axis=1, keepdims=True)


def _shell_indices(n_visible: int) -> tuple[np.ndarray, ...]:
    if n_visible not in _SHELL_INDICES:
        states = _states(n_visible)
        weights = states.sum(axis=1).astype(int)
        _SHELL_INDICES[n_visible] = tuple(
            np.flatnonzero(weights == k) for k in range(n_visible + 1)
        )
    return _SHELL_INDICES[n_visible]


def target_probabilities(conditions: np.ndarray, n_visible: int) -> np.ndarray:
    """Return exact shell-uniform target probabilities."""
    if n_visible != N_VISIBLE:
        raise ValueError("the frozen isotropic benchmark requires 16 visible qubits")
    weights = _states(n_visible).sum(axis=1).astype(int)
    counts = np.asarray([len(index) for index in _shell_indices(n_visible)])
    profiles = _shell_profile(conditions)
    return np.stack(
        [profiles[i, weights] / counts[weights] for i in range(len(profiles))]
    )


def make_targets(
    conditions: np.ndarray,
    n_visible: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample target bitstrings by drawing a shell, then a uniform bitstring."""
    profiles = _shell_profile(conditions)
    states = _states(n_visible).astype(np.float64)
    shell_indices = _shell_indices(n_visible)
    samples = []
    for profile in profiles:
        shell = rng.choice(n_visible + 1, size=n_samples, p=profile)
        samples.append(np.stack([states[rng.choice(shell_indices[k])] for k in shell]))
    return np.stack(samples)


def make_encoder(conditions: np.ndarray, config: RunConfig) -> ciqp.Encoder:
    """Broadcast ``RY(1.50)`` followed by ``RZ(c-0.5)`` on every qubit."""
    values = np.asarray(conditions, dtype=np.float64).reshape(-1)
    if config.n_total != N_VISIBLE:
        raise ValueError("the frozen encoder requires a 16-qubit register")
    ry = np.full((len(values), N_VISIBLE), 1.50)
    rz = np.broadcast_to((values - 0.5)[:, None], (len(values), N_VISIBLE))
    return ciqp.AngleEncoder(
        np.stack((ry, rz), axis=1), axes=("Y" * N_VISIBLE, "Z" * N_VISIBLE)
    )

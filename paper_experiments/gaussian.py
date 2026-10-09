"""Reviewed discrete Gaussian shell family."""

from __future__ import annotations

import hashlib
import json
from math import comb
from pathlib import Path

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


def export_midpoint_profiles(checkpoints: dict[int, Path], destination: Path) -> dict:
    """Export exact development profiles from the already selected circuits.

    The additional simulation is read-only: it neither trains nor reselects
    checkpoints. An existing reporting condition verifies the simulator against
    the probabilities saved with each checkpoint.
    """
    if not checkpoints:
        raise ValueError("midpoint profiles require selected checkpoints")
    fitting = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    conditions = (fitting[:-1] + fitting[1:]) / 2
    weights = _states(N_VISIBLE).sum(axis=1).astype(int)
    records = []
    for seed, checkpoint in sorted(checkpoints.items()):
        checkpoint = Path(checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with np.load(checkpoint) as saved:
            np.testing.assert_allclose(saved["training_conditions"], fitting)
            params = saved["params"].copy()
            generators = saved["generators"].copy()
            reference_condition = float(saved["evaluation_conditions"][0])
            reference_probabilities = saved["held_out_probabilities"][0].copy()
        queried = np.r_[conditions, reference_condition]
        angles = np.stack(
            (
                np.full((len(queried), N_VISIBLE), 1.5),
                np.repeat((queried - 0.5)[:, None], N_VISIBLE, axis=1),
            ),
            axis=1,
        )
        encoder = ciqp.AngleEncoder(angles, axes=("Y" * N_VISIBLE, "Z" * N_VISIBLE))
        model = ciqp.ConditionalModel(ciqp.Ansatz(generators), encoder)
        probabilities = np.asarray(model.probabilities(params))
        if not np.isfinite(probabilities).all() or probabilities.min() < 0:
            raise ValueError("midpoint simulation returned invalid probabilities")
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-12, rtol=0)
        np.testing.assert_allclose(
            probabilities[-1], reference_probabilities, atol=2e-12, rtol=0
        )
        profiles = np.array(
            [np.bincount(weights, weights=q, minlength=17) for q in probabilities[:4]]
        )
        records.append(
            dict(
                seed=int(seed),
                checkpoint_sha256=digest,
                profiles=profiles.tolist(),
                probability_crosscheck_max_error=float(
                    np.max(np.abs(probabilities[-1] - reference_probabilities))
                ),
            )
        )
    chance = np.sin(1.5 / 2) ** 2
    encoder_profile = [
        comb(N_VISIBLE, k) * chance**k * (1 - chance) ** (N_VISIBLE - k)
        for k in range(N_VISIBLE + 1)
    ]
    payload = dict(
        conditions=conditions.tolist(),
        training_conditions=fitting.tolist(),
        condition_role=(
            "development: excluded from fitting, used for checkpoint selection"
        ),
        seeds=[record["seed"] for record in records],
        records=records,
        target_profiles=_shell_profile(conditions).tolist(),
        encoder_profile=encoder_profile,
        aggregation="Mean exact profile probabilities across selected runs",
        evaluation="Exact public ConditionalModel.probabilities, without retraining",
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return payload

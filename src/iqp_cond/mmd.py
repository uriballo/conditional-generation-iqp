"""Unbiased conditional MMD estimation in the Pauli Fourier representation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .simulator import _estimate_pauli_z_unchecked


@dataclass(frozen=True)
class ConditionalMMDResult:
    """Conditional MMD estimate with condition- and bandwidth-level diagnostics."""

    loss: jnp.ndarray
    per_condition: jnp.ndarray
    per_bandwidth_condition: jnp.ndarray
    model_expectations: tuple[jnp.ndarray, ...]
    target_expectations: tuple[jnp.ndarray, ...]
    operators: tuple[jnp.ndarray, ...]


def hamming_rbf(x: jnp.ndarray, y: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Evaluate ``exp(-d_H(x,y)/(2 sigma^2))``."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return jnp.exp(-jnp.sum(jnp.not_equal(x, y), axis=-1) / (2.0 * sigma**2))


def mmd_operator_probability(sigma: float | jnp.ndarray) -> jnp.ndarray:
    """Return the Bernoulli probability for Fourier observable masks."""
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    return (1.0 - jnp.exp(-1.0 / (2.0 * sigma**2))) / 2.0


def sample_mmd_operators(
    key: jax.Array,
    n_ops: int,
    n_qubits: int,
    sigma: float,
) -> jnp.ndarray:
    """Sample Pauli-Z masks for the Hamming-RBF Fourier representation."""
    if n_ops < 1 or n_qubits < 1:
        raise ValueError("n_ops and n_qubits must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return jax.random.bernoulli(
        key,
        p=mmd_operator_probability(sigma),
        shape=(n_ops, n_qubits),
    ).astype(jnp.float64)


def _target_terms(
    targets: jnp.ndarray,
    ops: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    target_characters = 1.0 - 2.0 * jnp.mod(
        jnp.einsum("btn,rn->btr", targets, ops),
        2.0,
    )
    n_targets = targets.shape[1]
    means = jnp.mean(target_characters, axis=1)
    if n_targets < 2:
        raise ValueError("each condition needs at least two target samples")
    sums = jnp.sum(target_characters, axis=1)
    unbiased_squares = (sums**2 - n_targets) / (n_targets * (n_targets - 1))
    return means, unbiased_squares


def conditional_mmd(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    condition_angles: jnp.ndarray,
    targets: jnp.ndarray,
    sigmas: tuple[float, ...],
    n_ops: int,
    n_iqp_samples: int,
    key: jax.Array,
) -> ConditionalMMDResult:
    """Estimate equally weighted, multiscale, unbiased squared Conditional MMD."""
    targets = jnp.asarray(targets, dtype=jnp.float64)
    condition_angles = jnp.asarray(condition_angles, dtype=jnp.float64)
    if targets.ndim != 3:
        raise ValueError(
            "targets must have shape (n_conditions, n_targets, n_qubits)",
        )
    if condition_angles.shape != (targets.shape[0], targets.shape[2]):
        raise ValueError(
            "condition_angles must have shape (n_conditions, n_qubits)",
        )
    if not sigmas:
        raise ValueError("sigmas must not be empty")

    bandwidth_keys = jax.random.split(key, len(sigmas))
    bandwidth_losses = []
    model_expectations = []
    target_expectations = []
    all_ops = []
    for sigma, sigma_key in zip(sigmas, bandwidth_keys, strict=True):
        op_key, model_key = jax.random.split(sigma_key)
        ops = sample_mmd_operators(
            op_key,
            n_ops,
            targets.shape[2],
            float(sigma),
        )
        model = _estimate_pauli_z_unchecked(
            params,
            generators,
            ops,
            condition_angles,
            model_key,
            n_iqp_samples,
        )
        target_means, target_squares = _target_terms(targets, ops)
        model_sums = jnp.sum(model.sample_values, axis=1)
        model_square_sums = jnp.sum(model.sample_values**2, axis=1)
        model_squares = (
            model_sums**2 - model_square_sums
        ) / (n_iqp_samples * (n_iqp_samples - 1))
        per_op = (
            model_squares
            - 2.0 * model.values * target_means
            + target_squares
        )
        bandwidth_losses.append(jnp.mean(per_op, axis=-1))
        model_expectations.append(model.values)
        target_expectations.append(target_means)
        all_ops.append(ops)

    per_bandwidth_condition = jnp.stack(bandwidth_losses, axis=0)
    per_condition = jnp.mean(per_bandwidth_condition, axis=0)
    return ConditionalMMDResult(
        loss=jnp.mean(per_condition),
        per_condition=per_condition,
        per_bandwidth_condition=per_bandwidth_condition,
        model_expectations=tuple(model_expectations),
        target_expectations=tuple(target_expectations),
        operators=tuple(all_ops),
    )

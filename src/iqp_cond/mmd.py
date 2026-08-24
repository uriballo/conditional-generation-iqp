"""Unbiased conditional MMD estimation in the Pauli Fourier representation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .encoding import ANGLE_ENCODING
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many_jax,
    fwht,
    fwht_jax,
)
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


@dataclass(frozen=True)
class ExactConditionalMMDResult:
    """Zero-variance multiscale Conditional MMD against exact targets."""

    loss: jnp.ndarray
    per_condition: jnp.ndarray
    per_bandwidth_condition: jnp.ndarray


def hamming_rbf(x: jnp.ndarray, y: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Evaluate ``exp(-d_H(x,y)/(2 sigma^2))``."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return jnp.exp(-jnp.sum(jnp.not_equal(x, y), axis=-1) / (2.0 * sigma**2))


def mmd_operator_probability(sigma: float | jnp.ndarray) -> jnp.ndarray:
    """Return the Bernoulli probability for Fourier observable masks."""
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    return (1.0 - jnp.exp(-1.0 / (2.0 * sigma**2))) / 2.0


def mmd_mask_weights(n_qubits: int) -> jnp.ndarray:
    """Return the Hamming weight of every little-endian Fourier mask."""
    bitstrings = enumerate_bitstrings(n_qubits)
    return jnp.asarray(bitstrings.sum(axis=1), dtype=jnp.float64)


def exact_target_fouriers(
    target_probabilities: jnp.ndarray | np.ndarray,
) -> jnp.ndarray:
    """Return the full Fourier coefficients of each exact target law."""
    target_probabilities = jnp.asarray(target_probabilities, dtype=jnp.float64)
    if target_probabilities.ndim != 2:
        raise ValueError("target_probabilities must have shape (n_targets, 2**n)")
    return jnp.stack(
        [jnp.asarray(fwht(row).real) for row in target_probabilities],
    )


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
    encoding: str = ANGLE_ENCODING,
) -> ConditionalMMDResult:
    """Estimate equally weighted, multiscale, unbiased squared Conditional MMD."""
    targets = jnp.asarray(targets, dtype=jnp.float64)
    condition_angles = jnp.asarray(condition_angles, dtype=jnp.float64)
    if targets.ndim != 3:
        raise ValueError(
            "targets must have shape (n_conditions, n_targets, n_qubits)",
        )
    n_conditions, _, n_data_qubits = targets.shape
    if (
        condition_angles.shape[0] != n_conditions
        or condition_angles.shape[1] < n_data_qubits
    ):
        raise ValueError(
            f"condition_angles must have shape (n_conditions, >= {n_data_qubits}); "
            f"got {condition_angles.shape}",
        )
    if not sigmas:
        raise ValueError("sigmas must not be empty")

    n_ancillas = condition_angles.shape[1] - n_data_qubits

    bandwidth_keys = jax.random.split(key, len(sigmas))
    bandwidth_losses = []
    model_expectations = []
    target_expectations = []
    all_ops = []
    for sigma, sigma_key in zip(sigmas, bandwidth_keys, strict=True):
        op_key, model_key = jax.random.split(sigma_key)
        ops_data = sample_mmd_operators(
            op_key,
            n_ops,
            n_data_qubits,
            float(sigma),
        )
        if n_ancillas > 0:
            zeros = jnp.zeros((ops_data.shape[0], n_ancillas), dtype=ops_data.dtype)
            ops_model = jnp.concatenate([ops_data, zeros], axis=1)
        else:
            ops_model = ops_data

        model = _estimate_pauli_z_unchecked(
            params,
            generators,
            ops_model,
            condition_angles,
            model_key,
            n_iqp_samples,
            encoding,
        )
        target_means, target_squares = _target_terms(targets, ops_data)
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
        all_ops.append(ops_data)

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


def exact_conditional_mmd(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    condition_angles: jnp.ndarray,
    target_fouriers: jnp.ndarray,
    sigmas: tuple[float, ...],
    encoding: str = ANGLE_ENCODING,
    characters: jnp.ndarray | None = None,
) -> ExactConditionalMMDResult:
    """Evaluate the zero-variance multiscale Conditional MMD exactly.

    With population kernels, MMD^2 = sum_m q_sigma(m) (F_model[m] -
    F_target[m])^2 in the Fourier representation, where q_sigma is the
    product Bernoulli mask weight and F = FWHT(p) are the full Fourier
    coefficients of the model and target laws.  ``target_fouriers`` must
    have shape (n_conditions, 2**n).  Callers that JIT must precompute the
    character matrix with ``generator_characters`` and pass it in as a
    closure constant.
    """
    condition_angles = jnp.asarray(condition_angles, dtype=jnp.float64)
    target_fouriers = jnp.asarray(target_fouriers, dtype=jnp.float64)
    if target_fouriers.ndim != 2:
        raise ValueError("target_fouriers must have shape (n_conditions, 2**n)")
    n_conditions, n_states = target_fouriers.shape
    if condition_angles.shape[0] != n_conditions:
        raise ValueError("condition_angles and target_fouriers must align")
    if not sigmas:
        raise ValueError("sigmas must not be empty")
    n_data_qubits = round(np.log2(n_states))
    n_model_qubits = condition_angles.shape[1]
    if 2**n_data_qubits != n_states:
        raise ValueError("target_fouriers width must be a power of two")
    if n_model_qubits < n_data_qubits:
        raise ValueError(
            "condition_angles must include at least the target data qubits",
        )
    mask_weights = mmd_mask_weights(n_data_qubits)

    model_probabilities = exact_iqp_probabilities_many_jax(
        params,
        generators,
        condition_angles,
        encoding,
        characters,
    )
    if n_model_qubits > n_data_qubits:
        n_ancillas = n_model_qubits - n_data_qubits
        model_probabilities = model_probabilities.reshape(
            n_conditions,
            2**n_ancillas,
            n_states,
        ).sum(axis=1)
    model_fouriers = fwht_jax(model_probabilities)

    per_bandwidth_condition = []
    for sigma in sigmas:
        p = mmd_operator_probability(sigma)
        q = (p**mask_weights) * (
            (1.0 - p) ** (n_data_qubits - mask_weights)
        )
        squared = jnp.sum(
            q * (model_fouriers - target_fouriers) ** 2,
            axis=-1,
        )
        per_bandwidth_condition.append(squared)
    per_bandwidth_condition = jnp.stack(per_bandwidth_condition, axis=0)
    per_condition = jnp.mean(per_bandwidth_condition, axis=0)
    return ExactConditionalMMDResult(
        loss=jnp.mean(per_condition),
        per_condition=per_condition,
        per_bandwidth_condition=per_bandwidth_condition,
    )

"""Monte Carlo Pauli-Z estimation for product-state conditioned IQP circuits."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ExpectationEstimate:
    """Pauli-Z estimates and uncertainty for every condition and observable."""

    values: jnp.ndarray
    standard_errors: jnp.ndarray
    sample_values: jnp.ndarray


def post_hadamard_amplitudes(angles: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return local amplitudes of ``H RY(angle)|0>`` for outcomes zero and one."""
    angles = jnp.asarray(angles, dtype=jnp.float64)
    cosine = jnp.cos(angles / 2.0)
    sine = jnp.sin(angles / 2.0)
    scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    return (cosine + sine) / scale, (cosine - sine) / scale


def _as_condition_batch(angles: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
    angles = jnp.asarray(angles, dtype=jnp.float64)
    if angles.ndim == 1:
        angles = angles[None, :]
    if angles.ndim != 2 or angles.shape[1] != n_qubits:
        raise ValueError("angles must have shape (n_conditions, n_qubits)")
    return angles


def _validate_binary(name: str, values: jnp.ndarray) -> None:
    if not bool(jnp.all((values == 0) | (values == 1))):
        raise ValueError(f"{name} must contain only zeros and ones")


def _safe_ratio(numerator: jnp.ndarray, denominator: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(denominator != 0.0, numerator / denominator, 0.0)


def _sample_reference(
    key: jax.Array,
    amp_one: jnp.ndarray,
    n_samples: int,
) -> jnp.ndarray:
    probabilities_one = jnp.square(amp_one)
    return jax.random.bernoulli(
        key,
        p=probabilities_one[:, None, :],
        shape=(amp_one.shape[0], n_samples, amp_one.shape[1]),
    ).astype(jnp.float64)


def _amplitude_ratios(
    samples: jnp.ndarray,
    ops: jnp.ndarray,
    amp_zero: jnp.ndarray,
    amp_one: jnp.ndarray,
) -> jnp.ndarray:
    zero_to_one = _safe_ratio(amp_one, amp_zero)[:, None, None, :]
    one_to_zero = _safe_ratio(amp_zero, amp_one)[:, None, None, :]
    local_flip_ratios = jnp.where(
        samples[:, :, None, :] == 0.0,
        zero_to_one,
        one_to_zero,
    )
    selected = jnp.where(ops[None, None, :, :] == 1.0, local_flip_ratios, 1.0)
    return jnp.prod(selected, axis=-1)


def _phase_differences(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    ops: jnp.ndarray,
    samples: jnp.ndarray,
) -> jnp.ndarray:
    sample_parities = jnp.einsum("bsn,mn->bsm", samples, generators)
    sample_characters = 1.0 - 2.0 * jnp.mod(sample_parities, 2.0)
    op_parities = jnp.mod(jnp.einsum("rn,mn->rm", ops, generators), 2.0)
    weighted_flips = 2.0 * op_parities * params[None, :]
    return jnp.einsum("bsm,rm->bsr", sample_characters, weighted_flips)


def estimate_pauli_z(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    ops: jnp.ndarray,
    angles: jnp.ndarray,
    key: jax.Array,
    n_samples: int,
) -> ExpectationEstimate:
    """Estimate Pauli-Z expectations for one or more product-state conditions."""
    params = jnp.asarray(params, dtype=jnp.float64)
    generators = jnp.asarray(generators, dtype=jnp.float64)
    ops = jnp.asarray(ops, dtype=jnp.float64)
    if generators.ndim != 2:
        raise ValueError("generators must have shape (n_generators, n_qubits)")
    if params.shape != (generators.shape[0],):
        raise ValueError("params must have shape (n_generators,)")
    if ops.ndim == 1:
        ops = ops[None, :]
    if ops.ndim != 2 or ops.shape[1] != generators.shape[1]:
        raise ValueError("ops must have shape (n_ops, n_qubits)")
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    _validate_binary("generators", generators)
    _validate_binary("ops", ops)

    angles = _as_condition_batch(angles, int(generators.shape[1]))
    return _estimate_pauli_z_unchecked(
        params,
        generators,
        ops,
        angles,
        key,
        n_samples,
    )


def _estimate_pauli_z_unchecked(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    ops: jnp.ndarray,
    angles: jnp.ndarray,
    key: jax.Array,
    n_samples: int,
) -> ExpectationEstimate:
    amp_zero, amp_one = post_hadamard_amplitudes(angles)
    samples = _sample_reference(key, amp_one, n_samples)
    ratios = _amplitude_ratios(samples, ops, amp_zero, amp_one)
    phases = _phase_differences(params, generators, ops, samples)
    sample_values = jnp.real(ratios * jnp.exp(1j * phases))
    values = jnp.mean(sample_values, axis=1)
    standard_errors = jnp.std(sample_values, axis=1, ddof=1) / jnp.sqrt(n_samples)
    return ExpectationEstimate(values, standard_errors, sample_values)

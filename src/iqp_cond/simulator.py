"""Monte Carlo Pauli-Z estimation for product-state conditioned IQP circuits."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import jax
import jax.numpy as jnp

from .bond_reference import BOND_ENCODING, sample_two_rings, two_ring_ratios
from .encoding import (
    ANGLE_ENCODING,
    block_post_hadamard_amplitudes,
    is_block_encoding,
    is_radial_encoding,
    is_radial_shell_encoding,
    is_two_block_encoding,
    post_hadamard_amplitudes,
    radial_encoding_amplitudes,
    radial_shell_amplitudes,
    two_block_shell_amplitudes,
)


@dataclass(frozen=True)
class ExpectationEstimate:
    """Pauli-Z estimates and uncertainty for every condition and observable."""

    values: jnp.ndarray
    standard_errors: jnp.ndarray
    sample_values: jnp.ndarray


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


def _sample_reference_single(
    key: jax.Array,
    amp_one: jnp.ndarray,
    n_samples: int,
) -> jnp.ndarray:
    probabilities_one = jnp.abs(amp_one) ** 2
    return jax.random.bernoulli(
        key,
        p=probabilities_one[:, None, :],
        shape=(amp_one.shape[0], n_samples, amp_one.shape[1]),
    ).astype(jnp.float64)


def _sample_reference_blocks(
    key: jax.Array,
    block_amps: jnp.ndarray,
    singleton: tuple[jnp.ndarray, jnp.ndarray] | None,
    n_samples: int,
) -> jnp.ndarray:
    n_conditions, n_pairs, _ = block_amps.shape
    p_blocks = jnp.abs(block_amps) ** 2  # (n_conditions, n_pairs, 4)
    logits = jnp.log(jnp.maximum(p_blocks, 1e-30))  # (n_conditions, n_pairs, 4)

    # Sample categorical choices: shape (n_conditions, n_pairs, n_samples)
    pair_key, single_key = jax.random.split(key)
    # Broadcast logits to (n_conditions, n_samples, n_pairs, 4)
    logits_expanded = jnp.broadcast_to(
        logits[:, None, :, :],
        (n_conditions, n_samples, n_pairs, 4),
    )
    cat_samples = jax.random.categorical(
        pair_key, logits_expanded, axis=-1
    )  # (n_cond, n_samp, n_pairs)

    bit0 = (cat_samples % 2).astype(jnp.float64)
    bit1 = (cat_samples // 2).astype(jnp.float64)

    # Interleave bit0 and bit1 into (n_conditions, n_samples, 2 * n_pairs)
    paired_bits = jnp.stack([bit0, bit1], axis=-1).reshape(
        n_conditions, n_samples, 2 * n_pairs
    )

    if singleton is not None:
        _, single_amp_one = singleton
        p_single = jnp.abs(single_amp_one) ** 2  # (n_conditions,)
        single_samples = jax.random.bernoulli(
            single_key,
            p=p_single[:, None, None],
            shape=(n_conditions, n_samples, 1),
        ).astype(jnp.float64)
        return jnp.concatenate([paired_bits, single_samples], axis=-1)

    return paired_bits


def _amplitude_ratios_single(
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


def _amplitude_ratios_blocks(
    samples: jnp.ndarray,
    ops: jnp.ndarray,
    block_amps: jnp.ndarray,
    singleton: tuple[jnp.ndarray, jnp.ndarray] | None,
) -> jnp.ndarray:
    n_conditions, n_pairs, _ = block_amps.shape
    if n_pairs > 0:
        z0 = samples[:, :, None, 0 : 2 * n_pairs : 2].astype(jnp.int32)
        z1 = samples[:, :, None, 1 : 2 * n_pairs : 2].astype(jnp.int32)
        j_sample = z0 + 2 * z1  # (n_conditions, n_samples, 1, n_pairs)

        a0 = ops[None, None, :, 0 : 2 * n_pairs : 2].astype(jnp.int32)
        a1 = ops[None, None, :, 1 : 2 * n_pairs : 2].astype(jnp.int32)
        m_mask = a0 + 2 * a1  # (1, 1, n_ops, n_pairs)

        j_flip = j_sample ^ m_mask  # (n_conditions, n_samples, n_ops, n_pairs)

        # block_amps shape: (n_conditions, n_pairs, 4)
        # We index block_amps along condition (axis 0), pair (axis 1), and basis state
        amps_expanded = block_amps[:, None, None, :, :]  # (n_cond, 1, 1, n_pairs, 4)

        # Gather sample and flip amplitudes
        num = jnp.take_along_axis(amps_expanded, j_flip[..., None], axis=-1)[..., 0]
        denom = jnp.take_along_axis(amps_expanded, j_sample[..., None], axis=-1)[..., 0]
        pair_ratios = _safe_ratio(num, denom)  # (n_cond, n_samples, n_ops, n_pairs)
        total_ratio = jnp.prod(pair_ratios, axis=-1)  # (n_cond, n_samples, n_ops)
    else:
        n_conditions, n_samples, _ = samples.shape
        n_ops = ops.shape[0]
        total_ratio = jnp.ones((n_conditions, n_samples, n_ops), dtype=jnp.complex128)

    if singleton is not None:
        single_zero, single_one = singleton
        single_samples = samples[:, :, -1:]
        single_ops = ops[:, -1:]
        single_ratio = _amplitude_ratios_single(
            single_samples,
            single_ops,
            single_zero[:, None],
            single_one[:, None],
        )
        total_ratio = total_ratio * single_ratio

    return total_ratio


def _sample_reference_radial(
    key: jax.Array,
    shell_amp: jnp.ndarray,
    n_samples: int,
) -> jnp.ndarray:
    """Sample bitstrings from a radial reference ``|phi(z)|^2 = |phi_{|z|}|^2``.

    Draws a weight ``k`` from ``q(k) = C(n,k) |phi_k|^2`` and then a uniformly
    random bitstring of that weight via a random permutation threshold.
    """
    n_conditions, n_shell = shell_amp.shape
    n_qubits = n_shell - 1
    binom = jnp.asarray(
        [comb(n_qubits, k) for k in range(n_shell)],
        dtype=jnp.float64,
    )
    p = binom[None, :] * jnp.abs(shell_amp) ** 2
    p = p / jnp.sum(p, axis=-1, keepdims=True)
    logits = jnp.log(jnp.maximum(p, 1e-30))
    logits = jnp.broadcast_to(
        logits[:, None, :],
        (n_conditions, n_samples, n_shell),
    )
    cat_key, bit_key = jax.random.split(key)
    weights = jax.random.categorical(cat_key, logits, axis=-1).astype(jnp.float64)
    flat = weights.reshape(-1)  # (n_conditions * n_samples,)
    u = jax.random.uniform(bit_key, (flat.size, n_qubits))
    order = jnp.argsort(u, axis=-1)
    rank = jnp.argsort(order, axis=-1)
    bits = (rank < flat[:, None]).astype(jnp.float64)
    return bits.reshape(n_conditions, n_samples, n_qubits)


def _amplitude_ratios_radial(
    samples: jnp.ndarray,
    ops: jnp.ndarray,
    shell_amp: jnp.ndarray,
) -> jnp.ndarray:
    """Radial amplitude ratios ``phi_{|z xor a|}/phi_{|z|}`` for sampled strings."""
    n_conditions, _n_samples, n_qubits = samples.shape
    k = samples.sum(axis=-1)  # (n_conditions, n_samples)
    w = ops.sum(axis=-1)  # (n_ops,)
    overlap = jnp.einsum("csn,on->cso", samples, ops)
    k_prime = k[..., None] + w[None, None, :] - 2.0 * overlap
    k_idx = k.astype(jnp.int32)
    kp_idx = jnp.clip(k_prime.astype(jnp.int32), 0, n_qubits)
    c_axis = jnp.arange(n_conditions)
    denom = shell_amp[c_axis[:, None], k_idx]  # (n_conditions, n_samples)
    num = shell_amp[c_axis[:, None, None], kp_idx]
    return _safe_ratio(num, denom[..., None])


def _sample_reference_two_block(
    key: jax.Array,
    shell_a: jnp.ndarray,
    shell_b: jnp.ndarray,
    n_samples: int,
) -> jnp.ndarray:
    """Sample bitstrings from a product of two block-radial references."""
    key_a, key_b = jax.random.split(key)
    samples_a = _sample_reference_radial(key_a, shell_a, n_samples)
    samples_b = _sample_reference_radial(key_b, shell_b, n_samples)
    return jnp.concatenate([samples_a, samples_b], axis=-1)


def _amplitude_ratios_two_block(
    samples: jnp.ndarray,
    ops: jnp.ndarray,
    shell_a: jnp.ndarray,
    shell_b: jnp.ndarray,
) -> jnp.ndarray:
    """Product of the per-block radial amplitude ratios."""
    m_a = shell_a.shape[-1] - 1
    ratio_a = _amplitude_ratios_radial(samples[:, :, :m_a], ops[:, :m_a], shell_a)
    ratio_b = _amplitude_ratios_radial(samples[:, :, m_a:], ops[:, m_a:], shell_b)
    return ratio_a * ratio_b


def _phase_differences(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    ops: jnp.ndarray,
    samples: jnp.ndarray,
) -> jnp.ndarray:
    sample_parities = jnp.einsum("bsn,mn->bsm", samples, generators)
    sample_characters = 1.0 - 2.0 * jnp.mod(sample_parities, 2.0)
    op_parities = jnp.mod(jnp.einsum("rn,mn->rm", ops, generators), 2.0)
    weighted_flips = -2.0 * op_parities * params[None, :]
    return jnp.einsum("bsm,rm->bsr", sample_characters, weighted_flips)


def estimate_pauli_z(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    ops: jnp.ndarray,
    angles: jnp.ndarray,
    key: jax.Array,
    n_samples: int,
    encoding: str = ANGLE_ENCODING,
) -> ExpectationEstimate:
    """Estimate Pauli-Z expectations for one or more product or block conditions."""
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
        encoding,
    )


def _estimate_pauli_z_unchecked(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    ops: jnp.ndarray,
    angles: jnp.ndarray,
    key: jax.Array,
    n_samples: int,
    encoding: str = ANGLE_ENCODING,
) -> ExpectationEstimate:
    if encoding == BOND_ENCODING:
        n_qubits = int(generators.shape[1])
        couplings = angles[:, jnp.asarray([0, n_qubits // 2])]
        keys = jax.random.split(key, angles.shape[0])
        samples = jax.vmap(lambda k, c: sample_two_rings(k, c, n_qubits, n_samples))(
            keys, couplings
        )

        ratios = jax.vmap(lambda z, c: two_ring_ratios(z, ops, c))(samples, couplings)

    elif is_radial_shell_encoding(encoding):
        shell_amp = radial_shell_amplitudes(angles, int(generators.shape[1]))
        samples = _sample_reference_radial(key, shell_amp, n_samples)
        ratios = _amplitude_ratios_radial(samples, ops, shell_amp)
    elif is_two_block_encoding(encoding):
        n_qubits = int(generators.shape[1])
        m_a = n_qubits // 2
        m_b = n_qubits - m_a
        shell_a, shell_b = two_block_shell_amplitudes(angles, m_a, m_b)
        samples = _sample_reference_two_block(key, shell_a, shell_b, n_samples)
        ratios = _amplitude_ratios_two_block(samples, ops, shell_a, shell_b)
    elif is_radial_encoding(encoding):
        shell_amp = radial_encoding_amplitudes(
            angles,
            int(generators.shape[1]),
            encoding,
        )
        samples = _sample_reference_radial(key, shell_amp, n_samples)
        ratios = _amplitude_ratios_radial(samples, ops, shell_amp)
    elif is_block_encoding(encoding):
        block_amps, singleton = block_post_hadamard_amplitudes(angles, encoding)
        samples = _sample_reference_blocks(key, block_amps, singleton, n_samples)
        ratios = _amplitude_ratios_blocks(samples, ops, block_amps, singleton)
    else:
        amp_zero, amp_one = post_hadamard_amplitudes(angles, encoding)
        samples = _sample_reference_single(key, amp_one, n_samples)
        ratios = _amplitude_ratios_single(samples, ops, amp_zero, amp_one)

    phases = _phase_differences(params, generators, ops, samples)
    sample_values = jnp.real(ratios * jnp.exp(1j * phases))
    values = jnp.mean(sample_values, axis=1)
    standard_errors = jnp.std(sample_values, axis=1, ddof=1) / jnp.sqrt(n_samples)
    return ExpectationEstimate(values, standard_errors, sample_values)

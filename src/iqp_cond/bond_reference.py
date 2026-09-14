"""Transfer-matrix reference states for the bond-aware encoder.

The reference is a one-dimensional Ising amplitude state followed by a
fixed nearest-neighbour phase.  All routines operate on transfer matrices;
they therefore avoid constructing a ``2**m`` table and remain JIT friendly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

BOND_ENCODING = "two_ring_bond"
DEFAULT_ETA = jnp.pi / 8.0


def _transfer(g: jnp.ndarray, eta: jnp.ndarray) -> jnp.ndarray:
    spins = jnp.asarray((1, -1), dtype=jnp.float64)
    return jnp.exp((g / 2.0 + 1j * eta) * spins[:, None] * spins[None, :])


def _log_partition(g: jnp.ndarray, m: int) -> jnp.ndarray:
    # The supported range is numerically benign, but this form also remains
    # stable if the helper is used for larger rings.
    x = jnp.asarray(g)
    a = m * jnp.log(2.0 * jnp.cosh(x))
    b = m * jnp.log(2.0 * jnp.abs(jnp.sinh(x)))
    return jnp.logaddexp(a, b)


def _step_matrix(t: jnp.ndarray, bit: jnp.ndarray) -> jnp.ndarray:
    diagonal = jnp.stack(
        (jnp.ones_like(bit, dtype=t.real.dtype), jnp.where(bit == 0, 1.0, -1.0)),
        axis=-1,
    )
    return diagonal[..., :, None] * t


def ring_amplitudes(
    bits: jnp.ndarray, g: jnp.ndarray, eta: jnp.ndarray = DEFAULT_ETA
) -> jnp.ndarray:
    """Return post-Hadamard amplitudes for a phased Ising ring.

    ``bits`` has shape ``(..., m)`` and the output has the same leading shape.
    Odd-parity strings are exactly zero by the global spin-flip symmetry.
    """
    bits = jnp.asarray(bits, dtype=jnp.int32)
    if bits.ndim < 1:
        raise ValueError("bits must have a final ring axis")
    m = bits.shape[-1]
    if m < 3:
        raise ValueError("ring size must be at least three")
    g_arr = jnp.asarray(g)
    if g_arr.shape != ():
        raise ValueError("g must be scalar")
    t = _transfer(g_arr, jnp.asarray(eta))
    valid = jnp.isfinite(g_arr) & (g_arr >= 0.1) & (g_arr <= 0.7)
    t = jnp.where(valid, t, jnp.nan + 0j)
    matrix = jnp.broadcast_to(jnp.eye(2, dtype=t.dtype), (*bits.shape[:-1], 2, 2))
    log_scale = jnp.zeros(bits.shape[:-1], dtype=jnp.float64)
    for k in range(m):
        step = _step_matrix(t, bits[..., k])
        matrix = (
            matrix[..., :, 0, None] * step[..., None, 0, :]
            + matrix[..., :, 1, None] * step[..., None, 1, :]
        )
        scale = jnp.maximum(jnp.max(jnp.abs(matrix), axis=(-2, -1)), 1e-300)
        matrix = matrix / scale[..., None, None]
        log_scale = log_scale + jnp.log(scale)
    numerator = jnp.trace(matrix, axis1=-2, axis2=-1)
    log_norm = 0.5 * _log_partition(g_arr, m) + 0.5 * m * jnp.log(2.0)
    amp = numerator * jnp.exp(log_scale - log_norm)
    amp = jnp.where(valid, amp, jnp.nan + 0j)
    parity = jnp.mod(jnp.sum(bits, axis=-1), 2)
    return jnp.where(valid, jnp.where(parity == 0, amp, 0j), jnp.nan + 0j)


def _suffix(t: jnp.ndarray, m: int) -> jnp.ndarray:
    out = [jnp.eye(4, dtype=t.dtype)]
    s = _step_matrix(t, jnp.asarray(0))
    s = jnp.kron(s, s.conj())
    z = _step_matrix(t, jnp.asarray(1))
    z = jnp.kron(z, z.conj())
    total = s + z
    for _ in range(m):
        nxt = total @ out[-1]
        scale = jnp.maximum(jnp.max(jnp.abs(nxt)), 1e-300)
        out.append(nxt / scale)
    return jnp.stack(out[::-1])


def sample_ring(
    key: jax.Array,
    g: jnp.ndarray,
    m: int,
    n_samples: int,
    eta: jnp.ndarray = DEFAULT_ETA,
) -> jnp.ndarray:
    """Draw strings without enumeration; g must be in [.1,.7].

    Invalid numerical weights or couplings produce NaN strings.
    Ring size and sample count must be static when JIT compiled.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if m < 3:
        raise ValueError("ring size must be at least three")
    g_arr = jnp.asarray(g)
    if g_arr.shape != ():
        raise ValueError("g must be scalar")
    t = _transfer(g_arr, jnp.asarray(eta))
    valid = jnp.isfinite(g_arr) & (g_arr >= 0.1) & (g_arr <= 0.7)
    t = jnp.where(valid, t, jnp.nan + 0j)
    suffix = _suffix(t, m)
    e = []
    for bit in (0, 1):
        a = _step_matrix(t, jnp.asarray(bit))
        e.append(jnp.kron(a, a.conj()))
    e0, e1 = e
    prefix0 = jnp.broadcast_to(jnp.eye(4, dtype=t.dtype), (n_samples, 4, 4))

    def body(carry, k):
        prefix, key = carry
        key, sub = jax.random.split(key)
        # suffix[k+1] is common to all samples, so arbitrary rescaling cancels.
        tail = suffix[k + 1]
        w0 = jnp.real(jnp.einsum("nij,jk,ki->n", prefix, e0, tail))
        w1 = jnp.real(jnp.einsum("nij,jk,ki->n", prefix, e1, tail))
        total = w0 + w1
        bad = (
            (~jnp.isfinite(total))
            | (total <= 0)
            | (w0 < -1e-10 * jnp.abs(total))
            | (w1 < -1e-10 * jnp.abs(total))
        )
        p1 = jnp.where(bad, jnp.nan, jnp.clip(w1 / total, 0.0, 1.0))
        bit = jnp.where(
            bad,
            jnp.nan,
            (jax.random.uniform(sub, (n_samples,)) < p1).astype(jnp.float64),
        )
        chosen = jnp.where(bit[:, None, None] == 0, e0, e1)
        nxt = sum(prefix[:, :, j, None] * chosen[:, None, j, :] for j in range(4))
        nxt = jnp.where(bad[:, None, None], jnp.nan + 0j, nxt)
        scale = jnp.maximum(jnp.max(jnp.abs(nxt), axis=(-2, -1)), 1e-300)
        return (nxt / scale[:, None, None], key), bit

    (_, _), bits = jax.lax.scan(body, (prefix0, key), jnp.arange(m))
    return bits.T


def two_ring_amplitudes(bits: jnp.ndarray, couplings: jnp.ndarray) -> jnp.ndarray:
    """Product amplitude for two equal-size rings, ordered ``A then B``."""
    bits = jnp.asarray(bits, dtype=jnp.int32)
    if bits.ndim < 1 or bits.shape[-1] < 6 or bits.shape[-1] % 2:
        raise ValueError(
            "two-ring bits require an even final axis of size at least six"
        )
    m = bits.shape[-1] // 2
    c = jnp.asarray(couplings)
    if c.shape != (2,):
        raise ValueError("couplings must have shape (2,)")
    return ring_amplitudes(bits[..., :m], c[0]) * ring_amplitudes(bits[..., m:], c[1])


def sample_two_rings(
    key: jax.Array, couplings: jnp.ndarray, n_qubits: int, n_samples: int
) -> jnp.ndarray:
    """Draw from the product of two equal-size bond-aware rings."""
    if n_qubits < 6 or n_qubits % 2:
        raise ValueError("n_qubits must be even and at least six")
    couplings = jnp.asarray(couplings)
    if couplings.shape != (2,):
        raise ValueError("couplings must have shape (2,)")
    k0, k1 = jax.random.split(key)
    m = n_qubits // 2
    return jnp.concatenate(
        (
            sample_ring(k0, couplings[0], m, n_samples),
            sample_ring(k1, couplings[1], m, n_samples),
        ),
        axis=1,
    )


def _ring_ratio_trace(bits, g):
    """Gauge-scaled transfer trace; its common normalization cancels in ratios.

    The transfer matrix has absolute row sum one. This avoids repeated per-query
    logarithms while keeping products bounded; tiny denominators still require
    the same finite-precision caution as normalized amplitude queries.
    """
    t = _transfer(g, DEFAULT_ETA) / (2 * jnp.cosh(g / 2))
    matrix = jnp.broadcast_to(jnp.eye(2, dtype=t.dtype), (*bits.shape[:-1], 2, 2))
    for k in range(bits.shape[-1]):
        step = _step_matrix(t, bits[..., k])
        matrix = (
            matrix[..., :, 0, None] * step[..., None, 0, :]
            + matrix[..., :, 1, None] * step[..., None, 1, :]
        )
    trace = jnp.trace(matrix, axis1=-2, axis2=-1)
    return jnp.where(jnp.sum(bits, axis=-1) % 2 == 0, trace, 0j)


def two_ring_ratios(samples, ops, couplings):
    """Amplitude ratios for reference draws and masks, without normalizers.

    Inputs have shapes (samples, qubits), (masks, qubits), and (2,).
    Supported numerical regime is the same [.1,.7] as reference sampling.
    """
    m = samples.shape[-1] // 2
    if samples.shape[-1] < 6 or samples.shape[-1] % 2:
        raise ValueError("two rings require an even number of at least six qubits")
    if ops.ndim != 2 or ops.shape[1] != samples.shape[1]:
        raise ValueError("ops must have shape (masks, qubits)")
    result = jnp.ones((samples.shape[0], ops.shape[0]), dtype=jnp.complex128)
    for layer in range(2):
        z = samples[:, layer * m : (layer + 1) * m].astype(jnp.int32)
        a = ops[:, layer * m : (layer + 1) * m].astype(jnp.int32)
        numerator = _ring_ratio_trace(z[:, None, :] ^ a[None, :, :], couplings[layer])
        denominator = _ring_ratio_trace(z, couplings[layer])
        result *= numerator / denominator[:, None]
    valid = jnp.all(jnp.isfinite(samples)) & jnp.all(
        jnp.isfinite(couplings) & (couplings >= 0.1) & (couplings <= 0.7)
    )
    return jnp.where(valid, result, jnp.nan + 0j)

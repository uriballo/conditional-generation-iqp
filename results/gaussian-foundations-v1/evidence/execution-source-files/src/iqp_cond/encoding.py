"""Classically tractable product encodings for conditional IQP circuits."""

from __future__ import annotations

import jax.numpy as jnp

ANGLE_ENCODING = "angle"
SYMMETRIC_REUPLOAD_ENCODING = "symmetric_reupload"
ENCODINGS = (ANGLE_ENCODING, SYMMETRIC_REUPLOAD_ENCODING)


def _ry(
    zero: jnp.ndarray,
    one: jnp.ndarray,
    angles: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    cosine = jnp.cos(angles / 2.0)
    sine = jnp.sin(angles / 2.0)
    return cosine * zero - sine * one, sine * zero + cosine * one


def product_state_amplitudes(
    angles: jnp.ndarray,
    encoding: str = ANGLE_ENCODING,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return local amplitudes immediately before the first Hadamards.

    ``symmetric_reupload`` uses
    ``RY(a/2) RZ(pi/2 sin(4a)) RY(a/2)|0>``.  It agrees with plain angle
    encoding at ``a=0`` and ``a=pi/4``, while tracing a noncommuting,
    complex-valued path between those training endpoints.
    """
    angles = jnp.asarray(angles, dtype=jnp.float64)
    zero = jnp.ones_like(angles, dtype=jnp.complex128)
    one = jnp.zeros_like(angles, dtype=jnp.complex128)
    if encoding == ANGLE_ENCODING:
        return _ry(zero, one, angles)
    if encoding == SYMMETRIC_REUPLOAD_ENCODING:
        zero, one = _ry(zero, one, angles / 2.0)
        phase = (jnp.pi / 2.0) * jnp.sin(4.0 * angles)
        zero = zero * jnp.exp(-0.5j * phase)
        one = one * jnp.exp(0.5j * phase)
        return _ry(zero, one, angles / 2.0)
    raise ValueError(f"unknown encoding: {encoding!r}; expected one of {ENCODINGS}")


def post_hadamard_amplitudes(
    angles: jnp.ndarray,
    encoding: str = ANGLE_ENCODING,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return local amplitudes after the first Hadamards."""
    zero, one = product_state_amplitudes(angles, encoding)
    scale = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    return (zero + one) / scale, (zero - one) / scale

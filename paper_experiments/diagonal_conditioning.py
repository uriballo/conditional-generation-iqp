"""Diagonal-layer conditioning with a fixed GHZ-type input for Curie--Weiss.

One IQP circuit whose phases are a Chebyshev series in the inverse temperature,
theta(beta) = sum_k T_k(beta - 1) w_k, so beta in [0, 2] maps to x in [-1, 1].
The input is the parity encoder at t = pi/2 for every condition, whose reference
is uniform on even-parity strings; the input carries no condition.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .cw_followup_fit import PROBABILITY_TOLERANCE, _exact_kernel, _target_arrays
from .parity_encoder import ParityEncoder

N_QUBITS = 16
N_PHASES = N_QUBITS * (N_QUBITS + 1) // 2
GHZ_SCALE = np.pi / 2


def ghz_encoder(n_conditions, n=N_QUBITS):
    """Condition-independent parity input at t = pi/2."""
    return ParityEncoder.from_betas([1.0] * n_conditions, n, scale=GHZ_SCALE)


def chebyshev(beta, terms):
    """T_0..T_{terms-1} at x = beta - 1, stacked on the last axis."""
    x = jnp.asarray(beta, dtype=jnp.float64) - 1.0
    values = [jnp.ones_like(x), x][:terms]
    while len(values) < terms:
        values.append(2 * x * values[-1] - values[-2])
    return jnp.stack(values, axis=-1)


def theta_of(weights, beta, terms):
    """Phases at beta from flattened weights of shape (terms * phases,)."""
    w = jnp.reshape(weights, (terms, -1))
    return chebyshev(beta, terms) @ w


def initial_weights(normal_draws, init_scale, terms):
    """w_0 from the paired constant-theta initialization; w_k = 0 for k >= 1."""
    w = np.zeros((terms, len(normal_draws)))
    w[0] = np.asarray(normal_draws) * init_scale
    return w.reshape(-1)


def exact_probabilities(betas, weights, terms, n=N_QUBITS):
    """Exact model laws with per-temperature phases and the GHZ reference."""
    kernel = _exact_kernel(n)
    reference = ghz_encoder(1, n).post_hadamard_amplitudes()
    rows = []
    for beta in betas:
        value = np.asarray(kernel(theta_of(weights, beta, terms), reference))
        if (
            not np.isfinite(value).all()
            or np.any(value < -PROBABILITY_TOLERANCE)
            or not np.allclose(value.sum(1), 1, atol=PROBABILITY_TOLERANCE, rtol=0)
        ):
            raise FloatingPointError("invalid exact model probabilities")
        rows.append(value)
    return np.concatenate(rows)


def evaluate(betas, weights, terms, n=N_QUBITS):
    """The audited ``cw_followup_fit.evaluate`` report for diagonal conditioning."""
    betas = tuple(float(b) for b in betas)
    q = exact_probabilities(betas, weights, terms, n)
    p, k, target_shells = _target_arrays(betas, n)
    tvd = np.abs(q - p).sum(1) / 2
    shells = np.stack([np.bincount(k, weights=row, minlength=n + 1) for row in q])
    absolute_m = np.abs((2 * np.arange(n + 1) - n) / n)
    magnetization = shells @ absolute_m
    # H=-n m^2/2, so C=beta^2 Var(H)/n.  Compute both model and exact target.
    m2 = ((2 * np.arange(n + 1) - n) / n) ** 2
    energies = -n * m2 / 2
    heat = (
        np.asarray(betas) ** 2 * (shells @ (energies**2) - (shells @ energies) ** 2) / n
    )
    target_heat = (
        np.asarray(betas) ** 2
        * (target_shells @ (energies**2) - (target_shells @ energies) ** 2)
        / n
    )
    return dict(
        beta=np.asarray(betas),
        tvd=tvd,
        shells=shells,
        target_shells=target_shells,
        magnetization=magnetization,
        magnetization_error=abs(magnetization - target_shells @ absolute_m),
        heat=heat,
        target_heat=target_heat,
        heat_error=abs(heat - target_heat),
        signed_magnetization=shells @ ((2 * np.arange(n + 1) - n) / n),
        flip_defect=np.abs(q - q[:, ::-1]).sum(1) / 2,
    )

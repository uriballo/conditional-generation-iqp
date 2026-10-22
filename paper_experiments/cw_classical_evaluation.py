"""Accelerated exact finite-state evaluation for classical CW models."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from . import cw_classical_transformer as transformer
from . import cw_classical_vae as vae


@cache
def _states(n):
    return ((jnp.arange(1 << n)[:, None] >> jnp.arange(n)) & 1).astype(jnp.float64)


@jax.jit
def _rbm(params, beta, states):
    a = params["a0"] + beta * params["abeta"]
    b = params["b0"] + beta * params["bbeta"]
    raw = states @ a + jnp.sum(jax.nn.softplus(states @ params["W"] + b), axis=-1)
    q = jnp.exp(raw - logsumexp(raw))
    return (q + q[::-1]) / 2


@jax.jit
def _transformer(params, blocks, beta):
    lp = jax.lax.map(
        lambda xx: transformer._base_log_prob(params, xx, beta), blocks
    ).reshape((-1,))
    return jnp.exp(jnp.logaddexp(lp, lp[::-1]) - jnp.log(2.0))


@jax.jit
def _vae(params, blocks, beta, zs, weights):
    inputs = jnp.stack([zs, jnp.full(zs.shape, beta)], -1)
    h = jnp.tanh(vae._linear(inputs, params["dec1"]))
    logits = vae._linear(h, params["dec2"])
    positive = jax.nn.log_sigmoid(logits)
    negative = jax.nn.log_sigmoid(-logits)
    logweights = jnp.log(weights)

    def one(xx):
        raw = xx @ positive.T + (1 - xx) @ negative.T
        return jnp.exp(logsumexp(raw + logweights[None, :], axis=1))

    q = jax.lax.map(one, blocks).reshape((-1,))
    return (q + q[::-1]) / 2


def optimized_probabilities(family, params, beta, n, vae_settings):
    """Evaluate a normalized exact law."""
    states = _states(int(n))
    batch_size = min(8192, states.shape[0])
    blocks = states.reshape((-1, batch_size, states.shape[1]))
    if family == "rbm":
        return _rbm(params, beta, states), {"kind": "exact_finite_state_law"}
    if family == "transformer":
        q = _transformer(params, blocks, beta)
        return q, {"kind": "exact_finite_state_law"}
    if family != "vae":
        raise ValueError(f"unknown family: {family!r}")
    initial = vae_settings["initial_nodes"]
    maximum = vae_settings["max_nodes"]
    tolerance = vae_settings["tvd_tolerance"]
    if initial < 2 or initial & (initial - 1):
        raise ValueError("initial_nodes must be a power of two >= 2")
    if maximum < 4 * initial or maximum > 512 or maximum & (maximum - 1):
        raise ValueError(
            "max_nodes must be a power of two, <=512, and allow two refinements"
        )
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    errors = {}
    previous_error = np.inf
    prev = None
    nodes = initial
    while nodes <= maximum:
        gx, gw = vae._normal_quadrature(nodes)
        cur = _vae(params, blocks, beta, jnp.asarray(gx), jnp.asarray(gw))
        if not bool(jnp.all(jnp.isfinite(cur))):
            raise FloatingPointError("quadrature returned nonfinite probabilities")
        total = float(jnp.sum(cur))
        if not np.isfinite(total) or abs(total - 1.0) > 1e-4:
            raise FloatingPointError(
                f"quadrature probabilities are not normalized: {total}"
            )
        if prev is not None:
            err = float(0.5 * jnp.sum(jnp.abs(cur - prev)))
            errors[nodes] = err
            if err <= tolerance and previous_error <= tolerance:
                return cur, {"nodes": nodes, "errors": errors, "tolerance": tolerance}
            previous_error = err
        prev = cur
        nodes *= 2
    raise FloatingPointError(f"quadrature did not converge: {errors}")

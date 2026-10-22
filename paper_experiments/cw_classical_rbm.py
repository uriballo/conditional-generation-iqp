"""Small conditional restricted Boltzmann machines for the Curie--Weiss benchmark."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

family = "rbm"


def architecture(n: int = 16, capacity: str = "small") -> dict:
    """Return the RBM dimensions and parameter count for a capacity tier."""
    hidden = {"small": 6, "large": 53}.get(capacity)
    if hidden is None:
        raise ValueError(f"unknown capacity: {capacity!r}")
    count = n * hidden + 2 * n + 2 * hidden
    return {"n": n, "hidden": hidden, "capacity": capacity, "parameter_count": count}


def init(key, n: int = 16, capacity: str = "small", init_scale: float = 1.0):
    """Initialize affine-in-beta RBM parameters."""
    spec = architecture(n, capacity)
    k_w, _ = jax.random.split(key)
    h = spec["hidden"]
    # Xavier scale keeps the initial free energies well behaved as dimensions vary.
    scale = init_scale / jnp.sqrt(float(n + h))
    return {
        "W": scale * jax.random.normal(k_w, (n, h)),
        "a0": jnp.zeros((n,)),
        "abeta": jnp.zeros((n,)),
        "b0": jnp.zeros((h,)),
        "bbeta": jnp.zeros((h,)),
    }


def parameter_count(params) -> int:
    return int(sum(x.size for x in params.values()))


def _log_partition(params, a, b):
    n = params["a0"].shape[0]
    h = params["W"].shape[1]
    if h <= 6:
        hidden = ((jnp.arange(1 << h)[:, None] >> jnp.arange(h)) & 1).astype(a.dtype)
        hidden_scores = hidden @ b + jnp.sum(
            jax.nn.softplus(a + hidden @ params["W"].T), axis=-1
        )
        log_z = logsumexp(hidden_scores)
    else:
        visible = ((jnp.arange(1 << n)[:, None] >> jnp.arange(n)) & 1).astype(a.dtype)
        scores = visible @ a + jnp.sum(
            jax.nn.softplus(visible @ params["W"] + b), axis=-1
        )
        log_z = logsumexp(scores)
    return log_z


def log_prob(params, x, beta):
    """Log probability of a batch of binary visible states after flip symmetrization."""
    x = jnp.asarray(x, dtype=params["W"].dtype)
    a = params["a0"] + beta * params["abeta"]
    b = params["b0"] + beta * params["bbeta"]
    raw = x @ a + jnp.sum(jax.nn.softplus(x @ params["W"] + b), axis=-1)
    complemented = (1 - x) @ a + jnp.sum(
        jax.nn.softplus((1 - x) @ params["W"] + b), axis=-1
    )
    return (
        jnp.logaddexp(raw, complemented) - jnp.log(2.0) - _log_partition(params, a, b)
    )


def loss(params, x, beta, key=None):
    del key
    return -jnp.mean(log_prob(params, x, beta))


def sample(params, key, beta, draws):
    """Sample hidden mixtures for small RBMs, visible laws for large RBMs."""
    n = params["a0"].shape[0]
    a = params["a0"] + beta * params["abeta"]
    b = params["b0"] + beta * params["bbeta"]
    sample_key, flip_key = jax.random.split(key)
    h = params["W"].shape[1]
    if h <= 6:
        hidden = ((jnp.arange(1 << h)[:, None] >> jnp.arange(h)) & 1).astype(a.dtype)
        hs = hidden @ b + jnp.sum(jax.nn.softplus(a + hidden @ params["W"].T), axis=-1)
        cdf = jnp.cumsum(jnp.exp(hs - logsumexp(hs))).at[-1].set(1.0)
        u = jax.random.uniform(sample_key, (draws,))
        hi = jnp.searchsorted(cdf, u, side="right")
        probs = jax.nn.sigmoid(a + hidden[hi] @ params["W"].T)
        raw = jax.random.bernoulli(flip_key, probs).astype(a.dtype)
        flip_key = jax.random.fold_in(flip_key, 1)
    else:
        states = ((jnp.arange(1 << n)[:, None] >> jnp.arange(n)) & 1).astype(a.dtype)
        scores = states @ a + jnp.sum(
            jax.nn.softplus(states @ params["W"] + b), axis=-1
        )
        cdf = jnp.cumsum(jnp.exp(scores - logsumexp(scores))).at[-1].set(1.0)
        u = jax.random.uniform(sample_key, (draws,))
        raw = states[jnp.searchsorted(cdf, u, side="right")]
    flips = jax.random.bernoulli(flip_key, 0.5, (draws, 1))
    return jnp.where(flips, 1 - raw, raw)

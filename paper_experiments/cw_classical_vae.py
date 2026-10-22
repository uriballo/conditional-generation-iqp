"""Small conditional VAE baselines for the Curie--Weiss benchmark."""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

family = "vae"


@cache
def _normal_quadrature(nodes):
    jacobi = np.zeros((nodes, nodes))
    off = np.sqrt(np.arange(1, nodes, dtype=float))
    jacobi[np.arange(nodes - 1), np.arange(1, nodes)] = off
    jacobi[np.arange(1, nodes), np.arange(nodes - 1)] = off
    values, vectors = np.linalg.eigh(jacobi)
    weights = np.maximum(vectors[0, :] ** 2, 0.0)
    return values, weights / weights.sum()


def architecture(n=16, capacity="small"):
    ew, dw = {"small": (2, 4), "large": (20, 28)}.get(capacity, (None, None))
    if ew is None:
        raise ValueError("capacity must be 'small' or 'large'")
    enc = (n + 1) * ew + ew + 2 * (ew + 1)
    dec = (1 + 1) * dw + dw + dw * n + n
    return {
        "n": n,
        "capacity": capacity,
        "encoder_width": ew,
        "decoder_width": dw,
        "latent_dim": 1,
        "generation_parameter_count": dec,
        "encoder_parameter_count": enc,
        "training_parameter_count": enc + dec,
        "parameter_count": enc + dec,
    }


def init(key, n=16, capacity="small", init_scale=1.0):
    s = architecture(n, capacity)
    ew, dw = s["encoder_width"], s["decoder_width"]
    ks = iter(jax.random.split(key, 5))

    def lin(k, a, b):
        lim = init_scale * jnp.sqrt(6.0 / (a + b))
        return {
            "w": jax.random.uniform(k, (a, b), minval=-lim, maxval=lim),
            "b": jnp.zeros(b),
        }

    return {
        "enc1": lin(next(ks), n + 1, ew),
        "mean": lin(next(ks), ew, 1),
        "logvar": lin(next(ks), ew, 1),
        "dec1": lin(next(ks), 2, dw),
        "dec2": lin(next(ks), dw, n),
    }


def parameter_count(params):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def _linear(x, p):
    return x @ p["w"] + p["b"]


def _enc(params, x, beta):
    h = jnp.tanh(
        _linear(
            jnp.concatenate(
                [x, jnp.broadcast_to(jnp.asarray(beta), (*x.shape[:-1], 1))], -1
            ),
            params["enc1"],
        )
    )
    return _linear(h, params["mean"])[..., 0], _linear(h, params["logvar"])[
        ..., 0
    ].clip(-12.0, 12.0)


def decoder_log_prob(params, x, beta, z):
    x = jnp.asarray(x)
    z = jnp.asarray(z)
    if x.ndim == 1:
        x = x[None, :]
    if z.ndim == 0:
        z = z[None]
    z = jnp.broadcast_to(z, (x.shape[0],))
    b = jnp.broadcast_to(jnp.asarray(beta), (x.shape[0],))
    h = jnp.tanh(_linear(jnp.stack([z, b], -1), params["dec1"]))
    logits = _linear(h, params["dec2"])
    raw = jnp.sum(
        x * jax.nn.log_sigmoid(logits) + (1 - x) * jax.nn.log_sigmoid(-logits), -1
    )
    flip = jnp.sum(
        (1 - x) * jax.nn.log_sigmoid(logits) + x * jax.nn.log_sigmoid(-logits), -1
    )
    return jnp.logaddexp(raw, flip) - jnp.log(2.0)


def loss(params, x, beta, key):
    x = jnp.asarray(x)
    mean, lv = _enc(params, x, beta)
    z = mean + jnp.exp(0.5 * lv) * jax.random.normal(key, mean.shape)
    recon = decoder_log_prob(params, x, beta, z)
    kl = 0.5 * (jnp.exp(lv) + mean**2 - 1 - lv)
    return -jnp.mean(recon - kl)


def sample(params, key, beta, draws):
    k1, k2 = jax.random.split(key)
    z = jax.random.normal(k1, (draws,))
    h = jnp.tanh(_linear(jnp.stack([z, jnp.full((draws,), beta)], -1), params["dec1"]))
    p = jax.nn.sigmoid(_linear(h, params["dec2"]))
    x = jax.random.bernoulli(k2, p).astype(jnp.int32)
    return jnp.where(
        jax.random.bernoulli(jax.random.fold_in(k2, 1), 0.5, (draws, 1)), 1 - x, x
    )


def probabilities(params, beta, nodes=32, batch_size=2048):
    if nodes < 2 or nodes & (nodes - 1):
        raise ValueError("nodes must be a power of two >= 2")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    n = params["dec2"]["w"].shape[1]
    states = ((jnp.arange(1 << n)[:, None] >> jnp.arange(n)) & 1).astype(jnp.float64)
    ghx, ghw = _normal_quadrature(nodes)
    zs = jnp.asarray(ghx)
    ws = jnp.asarray(ghw)
    chunks = []
    for start in range(0, 1 << n, batch_size):
        xx = states[start : start + batch_size]
        h = jnp.tanh(
            _linear(jnp.stack([zs, jnp.full((nodes,), beta)], -1), params["dec1"])
        )
        logits = _linear(h, params["dec2"])
        raw = (
            xx @ jax.nn.log_sigmoid(logits).T + (1 - xx) @ jax.nn.log_sigmoid(-logits).T
        )
        flip = (1 - xx) @ jax.nn.log_sigmoid(logits).T + xx @ jax.nn.log_sigmoid(
            -logits
        ).T
        lp = jnp.logaddexp(raw, flip) - jnp.log(2.0)
        chunks.append(jnp.exp(logsumexp(lp + jnp.log(ws)[None, :], axis=1)))
    return jnp.concatenate(chunks)


def log_prob(params, x, beta, nodes=32):
    x = jnp.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    gx, gw = _normal_quadrature(nodes)
    h = jnp.tanh(
        _linear(
            jnp.stack([jnp.asarray(gx), jnp.full((nodes,), beta)], -1), params["dec1"]
        )
    )
    logits = _linear(h, params["dec2"])
    raw = x @ jax.nn.log_sigmoid(logits).T + (1 - x) @ jax.nn.log_sigmoid(-logits).T
    flip = (1 - x) @ jax.nn.log_sigmoid(logits).T + x @ jax.nn.log_sigmoid(-logits).T
    return logsumexp(
        jnp.logaddexp(raw, flip) - jnp.log(2.0) + jnp.log(jnp.asarray(gw))[None, :],
        axis=1,
    )


def checked_probabilities(
    params, beta, initial_nodes=32, max_nodes=512, tolerance=1e-5, batch_size=2048
):
    if initial_nodes < 2 or initial_nodes & (initial_nodes - 1):
        raise ValueError("initial_nodes must be a power of two >= 2")
    if max_nodes < 4 * initial_nodes or max_nodes > 512 or max_nodes & (max_nodes - 1):
        raise ValueError(
            "max_nodes must be a power of two, <=512, and allow two refinements"
        )
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    prev = probabilities(params, beta, initial_nodes, batch_size)
    errors = {}
    nodes = initial_nodes
    consecutive = 0
    while nodes < max_nodes:
        nodes *= 2
        cur = probabilities(params, beta, nodes, batch_size)
        if not bool(jnp.all(jnp.isfinite(cur))):
            raise FloatingPointError("quadrature returned nonfinite probabilities")
        total = float(jnp.sum(cur))
        if not np.isfinite(total) or abs(total - 1.0) > 1e-4:
            raise FloatingPointError(
                f"quadrature probabilities are not normalized: {total}"
            )
        err = float(0.5 * jnp.sum(jnp.abs(cur - prev)))
        errors[nodes] = err
        if err <= tolerance:
            consecutive += 1
            if consecutive >= 2:
                return cur, {"nodes": nodes, "errors": errors, "tolerance": tolerance}
        else:
            consecutive = 0
        prev = cur
    raise FloatingPointError(f"quadrature did not converge: {errors}")

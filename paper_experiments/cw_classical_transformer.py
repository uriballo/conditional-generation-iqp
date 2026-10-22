"""Compact conditional binary autoregressive Transformer for Curie--Weiss."""

from __future__ import annotations

import jax
import jax.nn as jnn
import jax.numpy as jnp


def architecture(n: int = 16, capacity: str = "small") -> dict:
    if capacity not in {"small", "large"}:
        raise ValueError("capacity must be 'small' or 'large'")
    d = 3 if capacity == "small" else 8
    f = 7 if capacity == "small" else 34
    count = 3 * d + 2 * d + (3 * d * d + 3 * d) + (d * d + d)
    count += d * f + f + f * d + d + 6 * d + d + 1
    return {"n": n, "d_model": d, "ff_dim": f, "heads": 1, "parameter_count": count}


def _linear(key, fan_in, fan_out, scale):
    k1, _ = jax.random.split(key)
    lim = jnp.sqrt(6.0 / (fan_in + fan_out)) * scale
    return {
        "w": jax.random.uniform(k1, (fan_in, fan_out), minval=-lim, maxval=lim),
        "b": jnp.zeros((fan_out,)),
    }


def init(key, n: int = 16, capacity: str = "small", init_scale: float = 1.0):
    spec = architecture(n, capacity)
    d, f = spec["d_model"], spec["ff_dim"]
    keys = iter(jax.random.split(key, 7))
    return {
        "token": jax.random.normal(next(keys), (3, d)) * init_scale,
        "_n": jnp.empty((n, 0), dtype=jnp.float32),
        "beta": _linear(next(keys), 1, d, init_scale),
        "qkv": _linear(next(keys), d, 3 * d, init_scale),
        "out": _linear(next(keys), d, d, init_scale),
        "ff1": _linear(next(keys), d, f, init_scale),
        "ff2": _linear(next(keys), f, d, init_scale),
        "ln1": {"scale": jnp.ones(d), "bias": jnp.zeros(d)},
        "ln2": {"scale": jnp.ones(d), "bias": jnp.zeros(d)},
        "lnf": {"scale": jnp.ones(d), "bias": jnp.zeros(d)},
        "head": _linear(next(keys), d, 1, init_scale),
    }


def parameter_count(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def _ln(x, p):
    centered = x - x.mean(-1, keepdims=True)
    return centered / jnp.sqrt(x.var(-1, keepdims=True) + 1e-5) * p["scale"] + p["bias"]


def _positions(n, d):
    pos = jnp.arange(n + 1)[:, None]
    i = jnp.arange(d)[None, :]
    angles = pos / jnp.power(10000.0, (2 * (i // 2)) / d)
    return jnp.where((i % 2) == 0, jnp.sin(angles), jnp.cos(angles))


def _logits(params, x, beta):
    x = jnp.asarray(x, dtype=jnp.int32)
    if x.ndim == 1:
        x = x[None, :]
    b = jnp.asarray(beta).reshape(-1)
    if b.size == 1:
        b = jnp.broadcast_to(b, (x.shape[0],))
    n, d = x.shape[1], params["token"].shape[1]
    shifted = jnp.concatenate([jnp.full((x.shape[0], 1), 2), x[:, :-1]], axis=1)
    h = params["token"][shifted] + _positions(n, d)[1:][None, :, :]
    h = h + b[:, None, None] * params["beta"]["w"][0] + params["beta"]["b"]
    mask = jnp.tril(jnp.ones((n, n)))
    attn_in = _ln(h, params["ln1"])
    qkv = attn_in @ params["qkv"]["w"] + params["qkv"]["b"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    att = jnp.einsum("bid,bjd->bij", q, k) / jnp.sqrt(d)
    att = jnn.softmax(jnp.where(mask, att, -1e30), axis=-1)
    h = h + jnp.einsum("bij,bjd->bid", att, v) @ params["out"]["w"] + params["out"]["b"]
    ff = _ln(h, params["ln2"]) @ params["ff1"]["w"] + params["ff1"]["b"]
    h = h + jnn.gelu(ff) @ params["ff2"]["w"] + params["ff2"]["b"]
    h = _ln(h, params["lnf"])
    logits = h @ params["head"]["w"] + params["head"]["b"]
    logits = logits[..., 0]
    return logits


def _base_log_prob(params, x, beta):
    logits = _logits(params, x, beta)
    x = jnp.asarray(x, dtype=jnp.int32)
    terms = x * jnn.log_sigmoid(logits) + (1 - x) * jnn.log_sigmoid(-logits)
    return jnp.sum(terms, axis=1)


def log_prob(params, x, beta):
    x = jnp.asarray(x, dtype=jnp.int32)
    return jnp.logaddexp(
        _base_log_prob(params, x, beta), _base_log_prob(params, 1 - x, beta)
    ) - jnp.log(2.0)


def loss(params, x, beta, key=None):
    del key
    return -jnp.mean(log_prob(params, x, beta))


def sample(params, key, beta, draws):
    n = params["_n"].shape[0]
    # Infer the configured output length from beta-independent sequential decoding.
    out = jnp.zeros((draws, n), dtype=jnp.int32)
    keys = jax.random.split(key, n + 1)
    for i in range(n):
        p = jax.nn.sigmoid(_logits(params, out, beta)[:, i])
        out = out.at[:, i].set(jax.random.bernoulli(keys[i], p).astype(jnp.int32))
    return jnp.where(jax.random.bernoulli(keys[-1], 0.5, (draws, 1)), 1 - out, out)

import jax
import jax.numpy as jnp
import numpy as np

from paper_experiments.cw_classical_vae import (
    architecture,
    checked_probabilities,
    init,
    log_prob,
    loss,
    parameter_count,
    probabilities,
    sample,
)


def test_parameter_counts():
    assert architecture(capacity="small")["parameter_count"] == 134
    assert architecture(capacity="large")["parameter_count"] == 950
    assert parameter_count(init(jax.random.key(0))) == 134


def test_quadrature_normalizes_and_is_symmetric():
    p = init(jax.random.key(1), n=4)
    q = np.asarray(probabilities(p, 0.7, nodes=16))
    states = np.asarray((jnp.arange(16)[:, None] >> jnp.arange(4)) & 1)
    assert np.isclose(q.sum(), 1.0, atol=1e-6)
    assert np.allclose(q, q[::-1])
    assert np.allclose(q, np.exp(np.asarray(log_prob(p, states, 0.7))), atol=5e-6)


def test_loss_gradient_and_sampler():
    p = init(jax.random.key(2), n=4)
    x = ((jnp.arange(8)[:, None] >> jnp.arange(4)) & 1).astype(jnp.float32)
    g = jax.grad(loss)(p, x, 0.4, jax.random.key(3))
    assert all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(g))
    draws = np.asarray(sample(p, jax.random.key(4), 0.8, 10000))
    states = np.asarray((jnp.arange(16)[:, None] >> jnp.arange(4)) & 1)
    expected = np.asarray(probabilities(p, 0.8, nodes=32))
    observed = np.array([np.all(draws == state, axis=1).mean() for state in states])
    assert np.max(np.abs(observed - expected)) < 0.04


def test_checked_metadata():
    p = init(jax.random.key(5), n=4)
    q, meta = checked_probabilities(p, 0.5, initial_nodes=8, max_nodes=64)
    assert np.isclose(np.asarray(q).sum(), 1.0, atol=1e-6)
    assert meta["nodes"] in (32, 64)


def test_fixed_key_elbo_and_marginal_gradients():
    p = init(jax.random.key(6), n=4)
    x = jnp.zeros((3, 4))
    key = jax.random.key(7)
    analytic = jax.grad(loss)(p, x, 0.5, key)
    eps = 1e-3
    leaf = "dec2"
    i = (0, 0)
    pp = {**p, leaf: {**p[leaf], "w": p[leaf]["w"].at[i].add(eps)}}
    pm = {**p, leaf: {**p[leaf], "w": p[leaf]["w"].at[i].add(-eps)}}
    fd = (float(loss(pp, x, 0.5, key)) - float(loss(pm, x, 0.5, key))) / (2 * eps)
    assert np.isclose(fd, float(analytic[leaf]["w"][i]), rtol=2e-2, atol=2e-3)
    state = jnp.zeros((1, 4))
    ap = jax.grad(lambda q: log_prob(q, state, 0.5).sum())(p)
    pp = {**p, leaf: {**p[leaf], "w": p[leaf]["w"].at[i].add(eps)}}
    pm = {**p, leaf: {**p[leaf], "w": p[leaf]["w"].at[i].add(-eps)}}
    fd = (
        float(log_prob(pp, state, 0.5).sum()) - float(log_prob(pm, state, 0.5).sum())
    ) / (2 * eps)
    assert np.isclose(fd, float(ap[leaf]["w"][i]), rtol=3e-2, atol=3e-3)


def test_checked_probabilities_rejects_nonconvergence():
    p = init(jax.random.key(8), n=4)
    with np.testing.assert_raises(FloatingPointError):
        checked_probabilities(p, 0.5, initial_nodes=2, max_nodes=8, tolerance=1e-20)

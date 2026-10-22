import jax
import jax.numpy as jnp
import numpy as np

from paper_experiments.cw_classical_rbm import (
    architecture,
    init,
    log_prob,
    loss,
    parameter_count,
    sample,
)


def test_parameter_counts():
    assert architecture(capacity="small")["parameter_count"] == 140
    assert architecture(capacity="large")["parameter_count"] == 986
    assert parameter_count(init(jax.random.key(0))) == 140


def test_log_prob_matches_bruteforce_and_normalizes():
    params = init(jax.random.key(1), n=4)
    params = dict(
        params,
        abeta=jnp.arange(4, dtype=jnp.float64) * 0.03,
        bbeta=jnp.arange(6, dtype=jnp.float64)[:6] * 0.02,
    )
    states = ((jnp.arange(16)[:, None] >> jnp.arange(4)) & 1).astype(jnp.float64)
    lp = np.asarray(log_prob(params, states, 0.7))
    # Independent visible-hidden enumeration of the unsymmetrized RBM.
    hstates = ((jnp.arange(64)[:, None] >> jnp.arange(6)) & 1).astype(jnp.float64)
    a = params["a0"] + 0.7 * params["abeta"]
    b = params["b0"] + 0.7 * params["bbeta"]
    visible_hidden = (states @ a)[:, None] + (hstates @ b)[None, :]
    visible_hidden += jnp.einsum("xi,yh,ih->xy", states, hstates, params["W"])
    brute = np.array(jax.scipy.special.logsumexp(visible_hidden, axis=1))
    brute -= np.logaddexp.reduce(brute)
    brute_sym = np.logaddexp(brute, brute[::-1]) - np.log(2.0)
    assert np.allclose(lp, brute_sym, atol=1e-10)
    assert np.isclose(np.logaddexp.reduce(lp), 0.0, atol=1e-6)
    assert np.allclose(lp, np.asarray(log_prob(params, 1 - states, 0.7)))


def test_gradient_is_finite():
    params = init(jax.random.key(2), n=4)
    x = ((jnp.arange(8)[:, None] >> jnp.arange(4)) & 1).astype(jnp.float64)
    grads = jax.grad(loss)(params, x, 0.4)
    assert all(np.all(np.isfinite(np.asarray(g))) for g in grads.values())


def test_gradient_matches_finite_difference():
    params = init(jax.random.key(5), n=4)
    params = dict(params, abeta=jnp.ones(4) * 0.07)
    x = jnp.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    analytic = jax.grad(loss)(params, x, 0.6)
    eps = 1e-5
    for name, index in (("W", (0, 1)), ("abeta", (2,))):
        plus = dict(params, **{name: params[name].at[index].add(eps)})
        minus = dict(params, **{name: params[name].at[index].add(-eps)})
        numeric = (float(loss(plus, x, 0.6)) - float(loss(minus, x, 0.6))) / (2 * eps)
        assert np.isclose(numeric, float(analytic[name][index]), rtol=2e-4, atol=2e-5)


def test_sampler_matches_exact_distribution():
    params = init(jax.random.key(3), n=4)
    draws = np.asarray(sample(params, jax.random.key(4), 0.8, 20000))
    states = ((jnp.arange(16)[:, None] >> jnp.arange(4)) & 1).astype(jnp.int32)
    expected = np.exp(np.asarray(log_prob(params, states, 0.8)))
    observed = np.array(
        [np.all(draws == state, axis=1).mean() for state in np.asarray(states)]
    )
    assert np.max(np.abs(observed - expected)) < 0.025


def test_large_visible_sampler_is_normalized():
    params = init(jax.random.key(6), n=4, capacity="large")
    states = ((jnp.arange(16)[:, None] >> jnp.arange(4)) & 1).astype(jnp.float64)
    assert np.isclose(np.exp(np.asarray(log_prob(params, states, 0.2))).sum(), 1.0)
    draws = np.asarray(sample(params, jax.random.key(7), 0.2, 5000))
    assert draws.shape == (5000, 4)

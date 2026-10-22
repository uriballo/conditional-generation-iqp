import itertools

import jax
import jax.numpy as jnp
import numpy as np

from paper_experiments.cw_classical_transformer import (
    _logits,
    init,
    log_prob,
    loss,
    parameter_count,
    sample,
)


def test_parameter_counts():
    assert parameter_count(init(jax.random.key(0), capacity="small")) == 137
    assert parameter_count(init(jax.random.key(0), capacity="large")) == 971


def test_normalization_and_flip_symmetry():
    p = init(jax.random.key(1), n=4)
    x = jnp.asarray(list(itertools.product((0, 1), repeat=4)))
    lp = log_prob(p, x, 0.7)
    assert np.isclose(float(jax.scipy.special.logsumexp(lp)), 0.0, atol=1e-5)
    np.testing.assert_allclose(lp, log_prob(p, 1 - x, 0.7), atol=1e-6)


def test_causal_logits_do_not_see_future_tokens():
    p = init(jax.random.key(2), n=4)
    x = jnp.zeros((1, 4), dtype=jnp.int32)
    y = x.at[0, 3].set(1)
    np.testing.assert_allclose(_logits(p, x, 0.4)[:, :3], _logits(p, y, 0.4)[:, :3])


def test_gradient_is_finite_and_sampler_has_expected_shape():
    p = init(jax.random.key(3), n=4)
    value, grad = jax.value_and_grad(loss)(p, jnp.zeros((3, 4), dtype=jnp.int32), 0.5)
    assert np.isfinite(float(value))
    assert all(np.all(np.isfinite(a)) for a in jax.tree_util.tree_leaves(grad))
    draws = sample(p, jax.random.key(4), 0.5, 500)
    assert draws.shape == (500, 4)
    assert 0.1 < float(draws.mean()) < 0.9


def test_finite_difference_covers_feedforward_and_final_norm_biases():
    p = init(jax.random.key(5), n=4)
    x = jnp.asarray([[0, 1, 0, 1], [1, 0, 1, 0]])
    for path in (("ff1", "b"), ("ff2", "b"), ("lnf", "bias")):
        i = 0
        eps = 1e-3
        plus = p[path[0]][path[1]].at[i].add(eps)
        minus = p[path[0]][path[1]].at[i].add(-eps)
        pp = {**p, path[0]: {**p[path[0]], path[1]: plus}}
        pm = {**p, path[0]: {**p[path[0]], path[1]: minus}}
        fd = (loss(pp, x, 0.4) - loss(pm, x, 0.4)) / (2 * eps)
        analytic = jax.grad(loss)(p, x, 0.4)[path[0]][path[1]][i]
        assert np.isfinite(float(fd)) and np.isclose(fd, analytic, rtol=2e-2, atol=2e-4)


def test_sampler_matches_full_symmetrized_law_for_four_bits():
    p = init(jax.random.key(6), n=4)
    states = jnp.asarray(list(itertools.product((0, 1), repeat=4)))
    raw_target = np.exp(np.asarray(log_prob(p, states, 0.3)))
    target = np.zeros(16)
    target[states @ (2 ** np.arange(4))] = raw_target
    draws = np.asarray(sample(p, jax.random.key(7), 0.3, 20000))
    ids = draws @ (2 ** np.arange(4))
    empirical = np.bincount(ids, minlength=16) / len(ids)
    assert np.max(np.abs(empirical - target)) < 0.03

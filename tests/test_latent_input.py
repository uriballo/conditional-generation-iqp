"""Tests for ancilla-purified input-state conditioning."""

import jax.numpy as jnp
import numpy as np
import pytest

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import exact_iqp_probabilities_many_jax
from iqp_cond.latent_input import (
    bernoulli_branch_weights,
    exact_latent_input_probabilities,
    latent_input_angles,
)


def test_bernoulli_weights_are_normalized_and_use_no_target_information():
    weights = np.asarray(bernoulli_branch_weights(jnp.asarray([0.25, 0.75]), 2))
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.allclose(weights[0], [0.5625, 0.1875, 0.1875, 0.0625])
    assert np.allclose(weights[1], weights[0, ::-1])


def test_overlap_margin_keeps_every_branch_supported():
    weights = np.asarray(
        bernoulli_branch_weights(jnp.asarray([0.0, 1.0]), 3, margin=0.25)
    )
    assert np.all(weights > 0.0)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_zero_input_couplings_reduce_to_one_shared_visible_circuit():
    ansatz = complete_ansatz(3)
    phases = jnp.linspace(-0.2, 0.3, ansatz.n_parameters)
    generators = jnp.asarray(ansatz.generators)
    base_angles = jnp.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    ordinary = exact_iqp_probabilities_many_jax(
        phases,
        generators,
        base_angles,
    )
    latent = exact_latent_input_probabilities(
        phases,
        generators,
        base_angles,
        jnp.asarray([0.2, 0.8]),
        jnp.zeros((3, 4)),
    )
    assert np.allclose(latent, ordinary, atol=1e-12)


def test_visible_marginal_is_manual_input_mixture_with_shared_phases():
    ansatz = complete_ansatz(2)
    phases = jnp.asarray([0.13, -0.21, 0.08])
    generators = jnp.asarray(ansatz.generators)
    base_angles = jnp.asarray([[0.2, -0.4], [0.7, 0.1]])
    condition = jnp.asarray([0.3, 0.6])
    couplings = jnp.asarray([[0.15, -0.05], [-0.08, 0.11]])

    angles = latent_input_angles(base_angles, couplings)
    branches = jnp.stack(
        [
            exact_iqp_probabilities_many_jax(phases, generators, branch)
            for branch in angles
        ]
    )
    weights = bernoulli_branch_weights(condition, 2)
    manual = jnp.einsum("cb,bcp->cp", weights, branches)
    evaluated = exact_latent_input_probabilities(
        phases,
        generators,
        base_angles,
        condition,
        couplings,
    )
    assert np.allclose(evaluated, manual, atol=1e-12)


def test_invalid_margin_is_rejected():
    with pytest.raises(ValueError, match="margin"):
        bernoulli_branch_weights(jnp.asarray([0.5]), 1, margin=0.5)

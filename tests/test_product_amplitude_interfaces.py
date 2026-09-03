import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.encoding import product_state_amplitudes
from iqp_cond.exact import (
    exact_iqp_visible_probabilities,
    exact_iqp_visible_probabilities_product_amplitudes,
)
from iqp_cond.simulator import estimate_pauli_z_product_amplitudes


def test_explicit_amplitudes_match_angle_exact_path():
    generators = np.asarray(complete_ansatz(3).generators)
    params = np.linspace(-0.2, 0.3, len(generators))
    angles = np.array([[0.2, -0.4, 0.7], [-0.1, 0.5, 0.9]])
    zero, one = product_state_amplitudes(jnp.asarray(angles))
    amplitudes = np.stack([np.asarray(zero), np.asarray(one)], axis=-1)
    expected = exact_iqp_visible_probabilities(params, generators, angles, 2, 1)
    actual = exact_iqp_visible_probabilities_product_amplitudes(
        params, generators, amplitudes, 2, 1
    )
    assert np.allclose(actual, expected, atol=1e-12)
    assert np.allclose(actual.sum(axis=1), 1.0)


def test_product_amplitude_estimator_is_finite():
    generators = jnp.asarray(complete_ansatz(3).generators)
    params = jnp.linspace(-0.2, 0.2, generators.shape[0])
    amplitudes = jnp.asarray(
        [[[1.0, 0.0], [2**-0.5, 2**-0.5], [0.6, 0.8]]],
        dtype=jnp.complex128,
    )
    estimate = estimate_pauli_z_product_amplitudes(
        params,
        generators,
        jnp.asarray([[1, 0, 0], [1, 1, 0]]),
        amplitudes,
        jax.random.PRNGKey(2),
        128,
    )
    assert estimate.values.shape == (1, 2)
    assert np.all(np.isfinite(np.asarray(estimate.values)))

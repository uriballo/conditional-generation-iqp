"""Independent checks of the bond reference inside the IQP estimator."""

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.bond_reference import BOND_ENCODING
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    exact_iqp_probabilities_many_jax,
    fwht,
    generator_characters,
)
from iqp_cond.simulator import estimate_pauli_z


def test_bond_exact_moments_gradients_and_symmetry():
    bits = enumerate_bitstrings(8)
    generators = np.asarray(complete_ansatz(8).generators)
    params = np.random.default_rng(31).normal(0, 0.15, len(generators))
    angles = np.array([[0.3] * 4 + [0.7] * 4])
    chars = generator_characters(generators)
    q = exact_iqp_probabilities_many(params, generators, angles, BOND_ENCODING)[0]
    qjax = exact_iqp_probabilities_many_jax(
        params, generators, angles, BOND_ENCODING, chars
    )[0]
    np.testing.assert_allclose(qjax, q, atol=1e-13)
    for flip in (15, 240):
        np.testing.assert_allclose(q, q[np.arange(256) ^ flip], atol=1e-13)
    # Independent physical input, including the fixed ZZ phase.
    s = 1 - 2 * bits
    energy = 0.3 * (s[:, :4] * np.roll(s[:, :4], -1, axis=1)).sum(1)
    energy += 0.7 * (s[:, 4:] * np.roll(s[:, 4:], -1, axis=1)).sum(1)
    bonds = (s[:, :4] * np.roll(s[:, :4], -1, axis=1)).sum(1)
    bonds += (s[:, 4:] * np.roll(s[:, 4:], -1, axis=1)).sum(1)
    physical = np.exp(energy / 2 + 1j * np.pi / 8 * bonds)
    physical /= np.linalg.norm(physical)
    reference = fwht(physical) / 16
    expected = np.abs(fwht(reference * np.exp(1j * (chars @ params))) / 16) ** 2
    np.testing.assert_allclose(q, expected, atol=1e-13)
    # Integrate the ratio identity over every support state: no sampling error.
    support = np.flatnonzero(np.abs(reference) > 1e-12)
    masks = np.array([0, 3, 17, 51, 255])
    z = support[:, None]
    flipped = z ^ masks[None, :]
    ratio = reference[flipped] / reference[z]
    delta = chars[flipped] - chars[z]
    factors = ratio * np.exp(1j * np.einsum("smg,g->sm", delta, params))
    weights = np.abs(reference[support]) ** 2
    moments = weights @ factors.real
    np.testing.assert_allclose(moments, fwht(q).real[masks], atol=1e-12)
    grad = np.einsum("s,sm,smg->mg", weights, (1j * factors).real, delta)

    def exact_moments(p):
        prob = exact_iqp_probabilities_many_jax(
            p, generators, angles, BOND_ENCODING, chars
        )[0]
        return jnp.asarray(1 - 2 * ((bits @ bits[masks].T) % 2)).T @ prob

    np.testing.assert_allclose(
        grad, jax.jacrev(exact_moments)(jnp.asarray(params)), atol=1e-12
    )
    # Deduplicating identical characters preserves the phase on support.
    _, indices, inverse = np.unique(
        chars[support].T, axis=0, return_index=True, return_inverse=True
    )
    assert len(indices) == 30
    collapsed = np.zeros(30)
    np.add.at(collapsed, inverse, params)
    qshort = exact_iqp_probabilities_many(
        collapsed, generators[indices], angles, BOND_ENCODING
    )[0]
    np.testing.assert_allclose(qshort, q, atol=1e-13)


def test_bond_sampled_moments_and_phase_gradient():
    bits = enumerate_bitstrings(6)
    g = np.asarray(complete_ansatz(6).generators)
    p = jnp.asarray(np.random.default_rng(44).normal(0, 0.2, len(g)))
    angles = jnp.asarray([[0.1] * 3 + [0.7] * 3])
    ops = jnp.asarray(bits[[0, 3, 5, 9, 27, 63]])
    key = jax.random.PRNGKey(912)

    def sampled(p):
        return estimate_pauli_z(p, g, ops, angles, key, 12000, BOND_ENCODING).values[0]

    def exact(p):
        q = exact_iqp_probabilities_many_jax(
            p, g, angles, BOND_ENCODING, generator_characters(g)
        )[0]
        return (1 - 2 * ((jnp.asarray(bits) @ ops.T) % 2)).T @ q

    np.testing.assert_allclose(sampled(p), exact(p), atol=0.035)
    np.testing.assert_allclose(jax.jacrev(sampled)(p), jax.jacrev(exact)(p), atol=0.065)

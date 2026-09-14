"""Six-site-layer estimator checks; no coupled held-out target laws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.bond_reference import BOND_ENCODING, two_ring_amplitudes
from iqp_cond.encoding import TWO_BLOCK_BRANCH_ENCODING
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    exact_iqp_probabilities_many_jax,
    fwht,
    generator_characters,
)


@pytest.mark.parametrize("a,b", [(0.1, 0.7), (0.7, 0.1), (0.38, 0.1)])
def test_six_site_bond_physical_reference_and_integrated_gradient(a, b):
    bits = enumerate_bitstrings(12)
    g = np.asarray(complete_ansatz(12).generators)
    chars = generator_characters(g)
    support = (bits[:, :6].sum(1) % 2 == 0) & (bits[:, 6:].sum(1) % 2 == 0)
    assert len(g) == 78 and len(np.unique(chars[support].T, axis=0)) == 78
    s = 1 - 2 * bits
    ea = (s[:, :6] * np.roll(s[:, :6], -1, axis=1)).sum(1)
    eb = (s[:, 6:] * np.roll(s[:, 6:], -1, axis=1)).sum(1)
    physical = np.exp((a * ea + b * eb) / 2 + 1j * np.pi / 8 * (ea + eb))
    physical /= np.linalg.norm(physical)
    reference = fwht(physical) / 64
    actual = np.asarray(two_ring_amplitudes(jnp.asarray(bits), jnp.asarray([a, b])))
    np.testing.assert_allclose(actual, reference, atol=1e-13)
    assert np.all(actual[~support] == 0)
    params = np.random.default_rng(3700).normal(0, 0.15, 78)
    angles = np.array([[a] * 6 + [b] * 6])
    q = exact_iqp_probabilities_many(params, g, angles, BOND_ENCODING)[0]
    expected = np.abs(fwht(reference * np.exp(1j * (chars @ params))) / 64) ** 2
    np.testing.assert_allclose(q, expected, atol=1e-13)
    masks = np.array([0, 3, 65, 195, 4095])
    ids = np.flatnonzero(np.abs(reference) > 1e-12)
    z = ids[:, None]
    flipped = z ^ masks[None, :]
    delta = chars[flipped] - chars[z]
    factor = (
        reference[flipped]
        / reference[z]
        * np.exp(1j * np.einsum("smg,g->sm", delta, params))
    )
    weight = np.abs(reference[ids]) ** 2
    np.testing.assert_allclose(weight @ factor.real, fwht(q).real[masks], atol=1e-12)
    gradient = np.einsum("s,sm,smg->mg", weight, (1j * factor).real, delta)

    def moments(p):
        prob = exact_iqp_probabilities_many_jax(p, g, angles, BOND_ENCODING, chars)[0]
        return jnp.asarray(1 - 2 * ((bits @ bits[masks].T) % 2)).T @ prob

    np.testing.assert_allclose(
        gradient, jax.jacrev(moments)(jnp.asarray(params)), atol=1e-12
    )
    np.testing.assert_allclose(q.sum(), 1, atol=1e-13)
    for flip in (63, 4032):
        np.testing.assert_allclose(q, q[np.arange(4096) ^ flip], atol=1e-13)


def test_six_site_cat_numpy_jax_and_both_flips():
    g = np.asarray(complete_ansatz(12).generators)
    params = np.random.default_rng(3700).normal(0, 0.15, 78)
    angles = (
        0.225 * np.pi - 0.20 * np.pi * (np.array([[0.1] * 6 + [0.7] * 6]) - 0.1) / 0.6
    )
    q = exact_iqp_probabilities_many(params, g, angles, TWO_BLOCK_BRANCH_ENCODING)[0]
    got = exact_iqp_probabilities_many_jax(
        params, g, angles, TWO_BLOCK_BRANCH_ENCODING, generator_characters(g)
    )[0]
    np.testing.assert_allclose(q, got, atol=1e-13)
    np.testing.assert_allclose(q.sum(), 1, atol=1e-13)
    for flip in (63, 4032):
        np.testing.assert_allclose(q, q[np.arange(4096) ^ flip], atol=1e-13)

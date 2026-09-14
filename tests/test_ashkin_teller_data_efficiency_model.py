import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import jax
import numpy as np
from ashkin_teller_data_efficiency_model import (
    exact_predictions,
    generators,
    sample_values,
)


def test_generator_shapes_and_ancilla_couplings():
    for r in (0, 2, 4):
        g = generators(r)
        assert g.shape == (78 + 12 * r, 12 + r)
        if r:
            assert np.all(g[78:, :12].sum(1) == 1)
            assert np.all(g[78:, 12:].sum(1) == 1)


def test_ancilla_marginal_is_normalized_and_finite():
    g = generators(2)
    params = np.linspace(0.01, 0.06, len(g))
    p = exact_predictions(params, g, np.array([[0.38, 0.38]]), "cat")[0]
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(), 1.0, rtol=0, atol=1e-12)


def test_sample_values_have_expected_shape():
    g = generators(4)
    params = np.linspace(0.01, 0.06, len(g))
    values = sample_values(
        params, g, g[:5], np.array([0.38, 0.38]), "bond", jax.random.key(4), 32
    )
    assert values.shape == (32, 5)
    assert np.isfinite(values).all()


def test_zero_cross_couplings_embed_original_visible_models():
    from iqp_cond.bond_reference import BOND_ENCODING
    from iqp_cond.encoding import TWO_BLOCK_BRANCH_ENCODING
    from iqp_cond.exact import exact_iqp_probabilities_many

    rng = np.random.default_rng(512)
    params = rng.normal(0, 0.13, 78)
    conditions = np.array([[0.22, 0.54], [0.54, 0.22]])
    for encoder in ("cat", "bond"):
        angles = np.repeat(conditions, 6, axis=1)
        enc = BOND_ENCODING
        if encoder == "cat":
            angles = 0.225 * np.pi - 0.20 * np.pi * (angles - 0.1) / 0.6
            enc = TWO_BLOCK_BRANCH_ENCODING
        expected = exact_iqp_probabilities_many(params, generators(0), angles, enc)
        for r in (0, 2, 4):
            q = exact_predictions(
                np.pad(params, (0, 12 * r)), generators(r), conditions, encoder
            )
            np.testing.assert_allclose(q, expected, atol=1e-12, rtol=0)


def test_physical_input_oracle_and_nontrivial_symmetric_marginal():
    from iqp_cond.exact import enumerate_bitstrings, fwht

    rng = np.random.default_rng(513)
    bits = enumerate_bitstrings(12)
    spin = 1 - 2 * bits
    weights = [bits[:, :6].sum(1), bits[:, 6:].sum(1)]
    bonds = [
        np.sum(spin[:, :6] * np.roll(spin[:, :6], -1, axis=1), axis=1),
        np.sum(spin[:, 6:] * np.roll(spin[:, 6:], -1, axis=1), axis=1),
    ]
    c = np.array([0.22, 0.54])
    g = generators(4)
    params = rng.normal(0, 0.12, len(g))
    fullbits = enumerate_bitstrings(16)
    phase = (1 - 2 * ((fullbits.astype(np.int32) @ g.T) % 2)) @ params
    for encoder in ("cat", "bond"):
        physical = np.ones(4096, dtype=complex)
        for layer in (0, 1):
            if encoder == "cat":
                alpha = 0.225 * np.pi - 0.20 * np.pi * (c[layer] - 0.1) / 0.6
                a, b = np.cos(alpha / 2), np.sin(alpha / 2)
                w = weights[layer]
                physical *= a ** (6 - w) * b**w + b ** (6 - w) * a**w
            else:
                physical *= np.exp((c[layer] / 2 + 1j * np.pi / 8) * bonds[layer])
        physical /= np.linalg.norm(physical)
        initial = np.zeros(65536, dtype=complex)
        initial[:4096] = physical
        reference = fwht(initial) / 256
        evolved = fwht(reference * np.exp(1j * phase)) / 256
        expected = (np.abs(evolved) ** 2).reshape(16, 4096).sum(0)
        q = exact_predictions(params, g, c[None, :], encoder)[0]
        np.testing.assert_allclose(q, expected, atol=1e-12, rtol=0)
        for flip in (63, 4032):
            np.testing.assert_allclose(q, q[np.arange(4096) ^ flip], atol=1e-12, rtol=0)
        zero_cross = params.copy()
        zero_cross[78:] = 0
        base = exact_predictions(zero_cross, g, c[None, :], encoder)[0]
        assert np.abs(q - base).sum() > 1e-3


def test_sampled_moment_and_cross_gradient_against_exact_oracle():
    import jax.numpy as jnp

    from iqp_cond.exact import enumerate_bitstrings

    g = generators(2)
    params = np.random.default_rng(514).normal(0, 0.12, len(g))
    ops = np.zeros((3, 14))
    ops[0, :2] = 1
    ops[2, 0] = 1
    bits = enumerate_bitstrings(12)
    chars = 1 - 2 * ((bits @ ops[:, :12].T) % 2)
    c = np.array([0.38, 0.38])
    tangent = np.zeros(len(g))
    tangent[78] = 1
    for encoder in ("cat", "bond"):
        fn = jax.jit(
            lambda p, encoder=encoder: sample_values(
                p, g, ops, c, encoder, jax.random.key(515), 32768
            )
        )
        values, derivatives = jax.jvp(
            fn, (jnp.asarray(params),), (jnp.asarray(tangent),)
        )
        values, derivatives = np.asarray(values), np.asarray(derivatives)
        exact = exact_predictions(params, g, c[None, :], encoder)[0] @ chars
        eps = 1e-5
        grad = (
            (
                exact_predictions(params + eps * tangent, g, c[None, :], encoder)[0]
                - exact_predictions(params - eps * tangent, g, c[None, :], encoder)[0]
            )
            @ chars
        ) / (2 * eps)
        assert abs(grad[0]) > 1e-5
        assert np.all(
            np.abs(values.mean(0) - exact)
            < 6 * values.std(0) / np.sqrt(len(values)) + 1e-8
        )
        assert np.all(
            np.abs(derivatives.mean(0) - grad)
            < 6 * derivatives.std(0) / np.sqrt(len(values)) + 1e-7
        )
        np.testing.assert_allclose(values[:, 1], 1, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(values[:, 2], 0)


def test_loss_builds_ragged_mask_banks_and_both_scalar_gradients():
    from ashkin_teller_data_efficiency_model import sampled_losses

    rng = np.random.default_rng(516)
    validation = rng.integers(0, 2, (16, 20, 12), dtype=np.int8)
    targets = validation[:, :10]
    g = generators(2)
    for encoder in ("cat", "bond"):
        loss, val, bank = sampled_losses(
            g, validation, 516, encoder, n_samples=32, n_ops=16
        )
        assert bank["masks"].shape[:2] == bank["weights"].shape
        value, grad = jax.jit(jax.value_and_grad(loss))(
            np.full(len(g), 0.04), targets, jax.random.key(517)
        )
        assert (
            np.isfinite(value) and np.isfinite(grad).all() and np.linalg.norm(grad) > 0
        )
        assert np.isfinite(val(np.full(len(g), 0.04)))

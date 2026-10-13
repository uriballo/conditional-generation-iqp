import iqp_cond as ciqp
import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest
from iqp_cond.operators import operator_bank
from scipy.linalg import expm

from paper_experiments.parity_encoder import ParityEncoder
from paper_experiments.squeezed_parity_encoder import (
    SqueezedParityEncoder,
    _input_table,
    _shells,
    angles,
)


def _hadamard(n):
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    matrix = h
    for _ in range(n - 1):
        matrix = np.kron(matrix, h)
    return matrix


def _site(op, i, n):
    return np.kron(np.kron(np.eye(1 << (n - 1 - i)), op), np.eye(1 << i))


def _dense_input(beta, n, scale, kappa):
    """Independent dense construction; little-endian, root is qubit n-1."""
    t, r = angles(beta, scale, kappa)
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1.0, -1.0])
    jy = sum(_site(y, i, n) for i in range(n - 1)) / 2
    jz = sum(_site(z, i, n) for i in range(n - 1)) / 2
    star = sum(_site(z, i, n) @ _site(z, n - 1, n) for i in range(n - 1))
    state = np.ones(1 << n) / 2 ** (n / 2)
    state = expm(1j * r * (jy @ jz + jz @ jy) / (n - 1)) @ state
    return expm(-1j * t / 2 * star) @ state


def test_reference_matches_dense_construction():
    n, betas = 5, [0.0, 0.7, 1.9]
    encoder = SqueezedParityEncoder.from_betas(betas, n, scale=0.5, kappa=0.6)
    reference = np.asarray(encoder.post_hadamard_amplitudes())
    for c, beta in enumerate(betas):
        expected = _hadamard(n) @ _dense_input(beta, n, 0.5, 0.6)
        np.testing.assert_allclose(reference[c], expected, atol=1e-12)


def test_unsqueezed_linear_angles_recover_parity_encoder():
    n, betas = 5, np.array([0.3, 1.2, 2.0])
    shells, _ = _shells(_input_table(0.7 * betas, 0 * betas, n), n)
    index = np.arange(1 << n)
    root, weight = (
        (index >> (n - 1)) & 1,
        np.array([int(x).bit_count() for x in index]) - ((index >> (n - 1)) & 1),
    )
    a = shells[:, root, weight]
    b = np.asarray(
        ParityEncoder.from_betas(betas, n, scale=0.7).post_hadamard_amplitudes()
    )
    np.testing.assert_allclose(np.abs(np.sum(a.conj() * b, axis=1)), 1, atol=1e-12)


def test_preparation_prepares_the_input_state():
    n = 4
    encoder = SqueezedParityEncoder.from_betas([1.1], n, scale=0.5, kappa=0.8)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit():
        encoder.preparation(0)()
        return qml.state()

    state = np.asarray(circuit())
    state = state[[int(f"{i:0{n}b}"[::-1], 2) for i in range(1 << n)]]
    np.testing.assert_allclose(state, _dense_input(1.1, n, 0.5, 0.8), atol=1e-12)


def test_squeeze_makes_input_probabilities_non_uniform():
    n = 6
    squeezed = SqueezedParityEncoder.from_betas([1.0], n, scale=0.5, kappa=0.4)
    assert np.ptp(np.abs(squeezed.input_amplitudes()[0]) ** 2) > 1e-3
    flat = SqueezedParityEncoder.from_betas([1.0], n, scale=0.5, kappa=0.0)
    np.testing.assert_allclose(np.abs(flat.input_amplitudes()[0]) ** 2, 2.0**-n)


def test_ratios_match_explicit_reference_states():
    n = 4
    encoder = SqueezedParityEncoder.from_betas([0.4, 1.5], n, scale=0.6, kappa=0.7)
    masks = np.arange(1 << n)[:, None] >> np.arange(n) & 1
    refs = np.asarray(encoder.post_hadamard_amplitudes())
    support = [i for i in range(1 << n) if min(abs(refs[:, i])) > 1e-12]
    samples = (np.array(support)[:, None] >> np.arange(n)) & 1
    samples = np.broadcast_to(samples, (2, len(support), n)).astype(float)
    actual = np.asarray(encoder.amplitude_ratios(samples, masks))
    expected = np.array(
        [
            [[refs[c, z ^ m] / refs[c, z] for m in range(1 << n)] for z in support]
            for c in range(2)
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_sampling_matches_reference_law():
    n = 4
    encoder = SqueezedParityEncoder.from_betas([1.4], n, scale=0.5, kappa=0.8)
    samples = np.asarray(encoder.sample(jax.random.key(3), 40000))[0].astype(int)
    index = samples @ (1 << np.arange(n))
    empirical = np.bincount(index, minlength=1 << n) / len(index)
    exact = np.abs(np.asarray(encoder.post_hadamard_amplitudes())[0]) ** 2
    np.testing.assert_allclose(empirical, exact, atol=0.01)
    assert np.all(samples.sum(-1) % 2 == 0)


def test_arbitrary_iqp_phases_preserve_global_flip():
    n = 4
    encoder = SqueezedParityEncoder.from_betas([0.3, 1.2, 2.0], n, 0.5, 0.7)
    bits = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    generators = np.asarray(ciqp.Ansatz.complete(n).generators)
    characters = 1 - 2 * (bits @ generators.T % 2)
    params = np.random.default_rng(57).normal(0, 0.9, len(generators))
    reference = np.asarray(encoder.post_hadamard_amplitudes())
    q = np.abs((reference * np.exp(1j * (characters @ params))) @ _hadamard(n)) ** 2
    np.testing.assert_allclose(q, q[:, ::-1], rtol=0, atol=1e-12)


def test_conformance_jit_and_mmd_gradient():
    encoder = SqueezedParityEncoder.from_betas([0.3, 1.2], 4, scale=0.5, kappa=0.6)
    ciqp.check_encoder(encoder)
    selected = jax.jit(lambda i: encoder.select(i).sample(jax.random.key(2), 4))(
        jnp.array(1)
    )
    assert selected.shape == (1, 4, 4)
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(4), encoder)
    params = jnp.zeros(model.ansatz.n_parameters)
    targets = encoder.sample(jax.random.key(5), 8)
    bank = operator_bank((0.5,), 3, 2, 4, seed=3)

    def loss(p):
        return model.mmd(p, targets, jax.random.key(4), bank=bank, n_samples=8).loss

    value, gradient = jax.value_and_grad(loss)(params)
    assert np.isfinite(np.asarray(value)).all()
    assert np.isfinite(np.asarray(gradient)).all()


@pytest.mark.parametrize(
    "betas,n,scale,kappa",
    [([-1.0], 4, 1, 1), ([np.nan], 4, 1, 1), ([1.0], 1, 1, 1), ([1.0], 4, 0, 1)],
)
def test_validation(betas, n, scale, kappa):
    with pytest.raises(ValueError):
        SqueezedParityEncoder.from_betas(betas, n, scale, kappa)

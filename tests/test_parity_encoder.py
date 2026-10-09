import iqp_cond as ciqp
import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest
from iqp_cond.operators import operator_bank

from paper_experiments.parity_encoder import ParityEncoder


def _circuit_state(t, n, hadamard=False):
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit():
        for wire in range(n - 1):
            qml.RX(t, wires=wire)
            qml.CNOT(wires=[wire, n - 1])
        if hadamard:
            for wire in range(n):
                qml.Hadamard(wires=wire)
        return qml.state()

    state = np.asarray(circuit())
    # Library/reference convention is little-endian; PennyLane is big-endian.
    return state[[int(f"{i:0{n}b}"[::-1], 2) for i in range(1 << n)]]


def test_reference_matches_independent_rx_cnot_state():
    encoder = ParityEncoder.from_betas([1.2], 3, scale=0.5)
    expected = _circuit_state(0.6, 3)
    np.testing.assert_allclose(
        np.asarray(encoder.post_hadamard_amplitudes()[0]), expected
    )


def test_physical_global_h_matches_manual_h_transform():
    encoder = ParityEncoder.from_betas([1.2], 3, scale=0.5)
    before = np.asarray(encoder.post_hadamard_amplitudes()[0])
    after = _circuit_state(0.6, 3, hadamard=True)
    # Explicit tensor-product Hadamard, with PennyLane's little-endian basis.
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    matrix = h
    for _ in range(2):
        matrix = np.kron(h, matrix)
    np.testing.assert_allclose(after, matrix @ before)


def test_sampling_has_even_total_parity_and_ratio_phase():
    encoder = ParityEncoder.from_betas([1.0], 4, scale=0.5)
    samples = np.asarray(encoder.sample(__import__("jax").random.key(4), 500))
    assert np.all(samples.sum(axis=-1) % 2 == 0)
    masks = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
    ratios = np.asarray(encoder.amplitude_ratios(samples, masks))[0]
    t = 0.5
    overlap = samples[:, :, :2].sum(axis=-1)
    expected = (-1j * np.tan(t / 2)) ** (2 - 2 * overlap[0])
    np.testing.assert_allclose(ratios[:, 1], expected)
    assert np.all(ratios[:, 2] == 0)


def test_zero_angle_ratios_are_finite_and_global_flip_invariant():
    encoder = ParityEncoder.from_betas([0.0], 3)
    samples = np.zeros((1, 2, 3))
    masks = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 0]])
    ratios = np.asarray(encoder.amplitude_ratios(samples, masks))
    np.testing.assert_array_equal(ratios[0, 0, :2], [1, 0])
    assert np.all(np.isfinite(ratios))
    before = np.asarray(encoder.post_hadamard_amplitudes()[0])
    flipped = before * np.array([(-1) ** i.bit_count() for i in range(8)])
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    matrix = np.kron(np.kron(h, h), h)
    q = np.abs(matrix @ before) ** 2
    q_flip = np.abs(matrix @ flipped) ** 2
    np.testing.assert_allclose(q, q_flip[::-1], atol=1e-12)


@pytest.mark.parametrize("bad", [[-1], [np.nan], [np.inf]])
def test_validation(bad):
    with pytest.raises(ValueError):
        ParityEncoder.from_betas(bad, 2)


def test_ratios_match_all_explicit_even_reference_states():
    encoder = ParityEncoder.from_betas([0.3, 1.2, 2.0], 4, scale=1.5)
    masks = np.arange(16)[:, None] >> np.arange(4) & 1
    samples = np.array([i for i in range(16) if i.bit_count() % 2 == 0])[:, None]
    samples = (samples >> np.arange(4)) & 1
    samples = np.broadcast_to(samples, (3, samples.shape[0], 4))
    actual = np.asarray(encoder.amplitude_ratios(samples, masks))
    refs = np.asarray(encoder.post_hadamard_amplitudes())
    expected = np.empty_like(actual)
    for c in range(3):
        for s, bits in enumerate(samples[c]):
            z = int(sum(int(bit) << i for i, bit in enumerate(bits)))
            for m, mask in enumerate(masks):
                k = int(sum(int(bit) << i for i, bit in enumerate(mask)))
                expected[c, s, m] = refs[c, z ^ k] / refs[c, z] if refs[c, z] else 0
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_jit_select_sampling_and_mmd_gradient():
    encoder = ParityEncoder.from_betas([0.3, 1.2], 4)
    selected = jax.jit(lambda i: encoder.select(i).sample(jax.random.key(2), 4))(
        jnp.array(1)
    )
    assert selected.shape == (1, 4, 4)
    masks = jnp.asarray(np.arange(16)[:, None] >> np.arange(4) & 1)
    ratios = jax.jit(lambda i: encoder.select(i).amplitude_ratios(selected, masks))(
        jnp.array(1)
    )
    assert np.isfinite(np.asarray(ratios)).all()
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(4), encoder)
    params = jnp.zeros(model.ansatz.n_parameters)
    targets = encoder.sample(jax.random.key(5), 8)
    bank = operator_bank((0.5,), 3, 2, 4, seed=3)

    def loss(p):
        return model.mmd(p, targets, jax.random.key(4), bank=bank, n_samples=8).loss

    value, gradient = jax.value_and_grad(loss)(params)
    assert np.isfinite(np.asarray(value)).all()
    assert np.isfinite(np.asarray(gradient)).all()


def test_arbitrary_iqp_phases_preserve_global_flip():
    n = 4
    encoder = ParityEncoder.from_betas([0.3, 1.2, 2.0], n, scale=1.5)
    bits = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    generators = np.asarray(ciqp.Ansatz.complete(n).generators)
    characters = 1 - 2 * (bits @ generators.T % 2)
    params = np.random.default_rng(57).normal(0, 0.9, len(generators))
    reference = np.asarray(encoder.post_hadamard_amplitudes())
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    matrix = np.kron(np.kron(np.kron(h, h), h), h)
    q = np.abs((reference * np.exp(1j * (characters @ params))) @ matrix) ** 2
    np.testing.assert_allclose(q, q[:, ::-1], rtol=0, atol=1e-12)

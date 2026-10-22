import iqp_cond as ciqp
import jax
import numpy as np
import pennylane as qml
import pytest

from paper_experiments.parity_encoder import ParityEncoder
from paper_experiments.squeezed_parity_encoder import SqueezedParityEncoder
from paper_experiments.twisted_parity_encoder import TwistedParityEncoder, angles


def _gate_level_input(beta, n, scale, kappa):
    """Dense statevector from the gates; little-endian, root is qubit n-1."""
    t, mu = angles(beta, scale, kappa)
    index = np.arange(1 << n)
    spin = 1 - 2 * ((index[:, None] >> np.arange(n)) & 1)
    state = np.full(1 << n, 2 ** (-n / 2), complex)
    leaves = n - 1
    pairs = (spin[:, :leaves].sum(1) ** 2 - leaves) / 2
    state = state * np.exp(-1j * mu / leaves * pairs)  # IsingZZ(2 mu/15)
    c, s = np.cos(np.pi / 8), -1j * np.sin(-np.pi / 8)  # RX(-pi/4)
    for q in range(leaves):
        state = c * state + s * state[index ^ (1 << q)]
    star = spin[:, :leaves].sum(1) * spin[:, -1]
    return state * np.exp(-1j * t / 2 * star)  # IsingZZ(t) leaf--root


def _hadamard_transform(state):
    n = int(np.log2(len(state)))
    out = np.array(state, complex)
    for _ in range(n):
        out = out.reshape(-1, 2, 1 << _)
        out = np.stack([out[:, 0] + out[:, 1], out[:, 0] - out[:, 1]], 1)
        out = out.reshape(-1)
    return out / 2 ** (n / 2)


def _dense_table(encoder):
    n = encoder.n_qubits
    index = np.arange(1 << n)
    root = (index >> (n - 1)) & 1
    weight = np.array([int(x).bit_count() for x in index]) - root
    return np.asarray(encoder.input_amplitudes())[:, root, weight]


@pytest.mark.parametrize(
    "scale,kappa,betas",
    [(0.475, 0.4, [0.0, 0.29, 1.0]), (0.3, 0.4, [1.25, 2.0]), (0.7, 0.9, [0.6])],
)
def test_tables_match_gate_level_statevector(scale, kappa, betas):
    n = 16
    encoder = TwistedParityEncoder.from_betas(betas, n, scale, kappa)
    inputs = _dense_table(encoder)
    reference = np.asarray(encoder.post_hadamard_amplitudes())
    for c, beta in enumerate(betas):
        expected = _gate_level_input(beta, n, scale, kappa)
        np.testing.assert_allclose(inputs[c], expected, atol=1e-12)
        np.testing.assert_allclose(
            reference[c], _hadamard_transform(expected), atol=1e-12
        )


def test_global_flip_and_leaf_permutation_invariance():
    n = 16
    state = _gate_level_input(1.1, n, 0.475, 0.4)
    index = np.arange(1 << n)
    np.testing.assert_allclose(state[index ^ ((1 << n) - 1)], state, atol=1e-12)
    bits = (index[:, None] >> np.arange(n)) & 1
    order = np.r_[np.random.default_rng(5).permutation(n - 1), n - 1]
    permuted = (bits[:, order] << np.arange(n)).sum(1)
    np.testing.assert_allclose(state[permuted], state, atol=1e-12)
    encoder = TwistedParityEncoder.from_betas([1.1], n, 0.475, 0.4)
    reference = np.asarray(encoder.post_hadamard_amplitudes())[0]
    odd = np.array([int(x).bit_count() % 2 for x in index]) == 1
    assert np.max(abs(reference[odd])) == 0


def test_preparation_prepares_the_input_state():
    n = 5
    encoder = TwistedParityEncoder.from_betas([1.1], n, scale=0.5, kappa=0.8)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit():
        encoder.preparation(0)()
        return qml.state()

    state = np.asarray(circuit())
    state = state[[int(f"{i:0{n}b}"[::-1], 2) for i in range(1 << n)]]
    np.testing.assert_allclose(state, _gate_level_input(1.1, n, 0.5, 0.8), atol=1e-12)


def test_ratios_match_explicit_reference_states():
    n = 5
    encoder = TwistedParityEncoder.from_betas([0.4, 1.5], n, scale=0.6, kappa=0.7)
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
    n = 5
    encoder = TwistedParityEncoder.from_betas([1.4], n, scale=0.5, kappa=0.8)
    samples = np.asarray(encoder.sample(jax.random.key(3), 40000))[0].astype(int)
    index = samples @ (1 << np.arange(n))
    empirical = np.bincount(index, minlength=1 << n) / len(index)
    exact = np.abs(np.asarray(encoder.post_hadamard_amplitudes())[0]) ** 2
    np.testing.assert_allclose(empirical, exact, atol=0.01)
    assert np.all(samples.sum(-1) % 2 == 0)


def test_library_check_encoder_passes():
    for n in (4, 16):
        encoder = TwistedParityEncoder.from_betas([0.29, 1.25], n, 0.475, 0.4)
        check = ciqp.check_encoder(encoder, n_samples=20_000, probes_per_condition=512)
        assert check.ok, check.failures
        assert {"Hadamard", "IsingZZ", "RX"} >= set(check.preparation_gates)


def test_zero_twist_is_the_parity_encoder_with_the_tanh_schedule():
    n, scale, betas = 16, 0.475, np.array([0.29, 1.0, 1.25, 2.0])
    twisted = TwistedParityEncoder.from_betas(betas, n, scale, kappa=0.0)
    control = SqueezedParityEncoder.from_betas(betas, n, scale, kappa=0.0)
    a = np.asarray(twisted.post_hadamard_amplitudes())
    np.testing.assert_allclose(
        np.abs(np.sum(a.conj() * np.asarray(control.post_hadamard_amplitudes()), 1)),
        1,
        atol=1e-12,
    )
    t, _ = angles(betas, scale, 0.0)
    for c, beta in enumerate(betas):
        parity = ParityEncoder.from_betas([beta], n, scale=t[c] / beta)
        b = np.asarray(parity.post_hadamard_amplitudes())[0]
        assert abs(abs(np.vdot(a[c], b)) - 1) < 1e-12
    np.testing.assert_allclose(np.abs(_dense_table(twisted)) ** 2, 2.0**-n)


def test_twist_makes_input_probabilities_non_uniform():
    n = 16
    encoder = TwistedParityEncoder.from_betas([1.0], n, scale=0.475, kappa=0.4)
    probabilities = np.abs(_dense_table(encoder)[0]) ** 2
    assert 0.5 * np.abs(probabilities - 2.0**-n).sum() > 0.05


@pytest.mark.parametrize(
    "betas,n,scale,kappa",
    [([-1.0], 4, 1, 1), ([np.nan], 4, 1, 1), ([1.0], 1, 1, 1), ([1.0], 4, 0, 1)],
)
def test_validation(betas, n, scale, kappa):
    with pytest.raises(ValueError):
        TwistedParityEncoder.from_betas(betas, n, scale, kappa)

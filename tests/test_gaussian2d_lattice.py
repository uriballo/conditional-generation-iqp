"""Independent circuit and exact-center checks for the revised family."""

import iqp_cond as ciqp
import jax
import numpy as np
import pytest

from paper_experiments.ansatz import rectangular_lattice_ansatz
from paper_experiments.gaussian2d_lattice import (
    CollectivePhaseEncoder,
    basis,
    probabilities,
    target_probabilities,
    target_profiles,
)


def test_exact_center_is_uniform_for_any_lattice_phases():
    for n in (4, 16):
        theta = np.random.default_rng(n).normal(
            size=rectangular_lattice_ansatz(n).n_parameters
        )
        center = [[0.0, 0.0]]
        np.testing.assert_allclose(probabilities(theta, center, n), 2.0**-n, atol=1e-14)
        np.testing.assert_array_equal(
            target_probabilities(center, n), np.full((1, 1 << n), 2.0**-n)
        )
        encoder = CollectivePhaseEncoder.from_conditions(center, n)
        draws = np.asarray(encoder.sample(jax.random.PRNGKey(1), 128))
        assert np.count_nonzero(draws) == 0


def test_full_probabilities_agree_with_public_circuit_oracle():
    pairs = np.array([[0.2, 0.5], [-0.7, -0.4], [0, 0]])
    ansatz = rectangular_lattice_ansatz(4)
    theta = np.random.default_rng(38).normal(size=ansatz.n_parameters)
    encoder = CollectivePhaseEncoder.from_conditions(pairs, 4)
    oracle = ciqp.ConditionalModel(ansatz, encoder).probabilities(theta)
    np.testing.assert_allclose(probabilities(theta, pairs, 4), oracle, atol=1e-13)
    check = ciqp.check_encoder(
        encoder, n_samples=8000, probes_per_condition=200, n_masks=8, seed=33
    )
    assert check.ok, check.failures


def test_targets_are_centered_symmetric_and_broaden_with_v():
    pairs = np.array([(u, v) for u in [-1.0, 0.0, 1.0] for v in [-1.0, 0.0, 1.0]])
    profile = target_profiles(pairs)
    np.testing.assert_allclose(profile.sum(1), 1)
    np.testing.assert_allclose(profile[:3], profile[-3:, ::-1])
    k = np.arange(17)
    mean = profile @ k
    sd = np.sqrt(profile @ k**2 - mean**2)
    assert np.all(mean[:3] < 8) and np.all(mean[-3:] > 8)
    for start in [0, 3, 6]:
        assert np.all(np.diff(sd[start : start + 3]) > 0)
    for row in profile:
        assert np.all(row[1:-1] ** 2 > row[:-2] * row[2:])


def test_ratio_access_agrees_exhaustively_with_dense_reference():
    e = CollectivePhaseEncoder.from_conditions([[0.4, -0.6]], n=4)
    bits, _ = basis(4)
    psi = np.asarray(e.post_hadamard_amplitudes())[0]
    ratios = np.asarray(e.amplitude_ratios(bits[None], bits))[0]
    indices = np.arange(16)
    direct = psi[np.bitwise_xor(indices[:, None], indices[None])] / psi[:, None]
    np.testing.assert_allclose(ratios, direct, atol=1e-12)


def test_invalid_condition_batch_rejected():
    with pytest.raises(ValueError):
        CollectivePhaseEncoder.from_conditions([[2.0, 0.0]])

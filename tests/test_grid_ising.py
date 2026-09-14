from math import comb

import numpy as np

from iqp_cond.exact_tvd_campaign import expected_bodyness_sigmas
from iqp_cond.grid_ising import (
    character_moments,
    enumerate_spins,
    exact_probabilities,
    lattice_edges,
    sample_exact,
    spectrum_diagnostic,
    sufficient_statistics,
)


def test_edges_are_unique_and_probabilities_match_boltzmann_ratios():
    shape = (2, 4)
    edge_list = lattice_edges(shape)
    assert len(edge_list) == 12
    assert len(set(edge_list)) == len(edge_list)
    assert all(left < right for left, right in edge_list)

    beta, field = 0.4, 0.1
    probability = exact_probabilities(shape, beta, field)
    spins = enumerate_spins(shape)
    interaction, magnetization = sufficient_statistics(spins, shape)
    assert np.isclose(probability.sum(), 1.0)
    left, right = 3, 177
    expected_ratio = np.exp(
        beta * (interaction[left] - interaction[right])
        + field * (magnetization[left] - magnetization[right])
    )
    assert np.isclose(probability[left] / probability[right], expected_ratio)


def test_sampling_is_reproducible_and_matches_exact_mean():
    shape = (2, 2)
    first = sample_exact(shape, 0.35, 0.1, 30_000, seed=3)
    second = sample_exact(shape, 0.35, 0.1, 30_000, seed=3)
    assert np.array_equal(first, second)
    exact = exact_probabilities(shape, 0.35, 0.1)
    _, exact_magnetization = sufficient_statistics(enumerate_spins(shape), shape)
    assert abs(first.sum(axis=1).mean() - exact @ exact_magnetization) < 0.04


def test_fwht_matches_direct_character_moments():
    shape = (2, 2)
    probability = exact_probabilities(shape, 0.3, -0.2)
    spins = enumerate_spins(shape)
    bits = ((spins + 1) // 2).astype(np.uint8)
    moments = character_moments(probability)
    for mask in range(1 << 4):
        mask_bits = ((mask >> np.arange(4)) & 1).astype(np.uint8)
        parity = ((bits @ mask_bits) % 2).astype(np.int8)
        direct = probability @ (1 - 2 * parity)
        assert np.isclose(moments[mask], direct)


def test_spectrum_matches_direct_mmd_and_sampling_count():
    shape = (2, 4)
    beta, field = 0.4, 0.0
    result = spectrum_diagnostic(shape, beta, field)
    probability = exact_probabilities(shape, beta, field)
    uniform = np.full_like(probability, 1 / probability.size)
    delta = character_moments(probability - uniform)
    n_spins = 8
    mask_weights = np.asarray([mask.bit_count() for mask in range(1 << n_spins)])
    for sigma, band in zip(
        expected_bodyness_sigmas(n_spins), result["bands"], strict=True
    ):
        inclusion_probability = (1.0 - np.exp(-1.0 / (2.0 * sigma**2))) / 2.0
        per_mask = inclusion_probability**mask_weights * (
            1.0 - inclusion_probability
        ) ** (n_spins - mask_weights)
        assert np.isclose(
            band["total_mmd_squared_vs_uniform"], np.sum(per_mask * delta**2)
        )
        probability_two = (
            comb(n_spins, 2)
            * inclusion_probability**2
            * (1 - inclusion_probability) ** (n_spins - 2)
            / (1 - (1 - inclusion_probability) ** n_spins)
        )
        expected_edges = 2_000 * probability_two * 12 / comb(n_spins, 2)
        assert np.isclose(band["expected_edge_masks_per_operator_bank"], expected_edges)

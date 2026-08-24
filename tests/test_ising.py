from hashlib import sha256
from itertools import pairwise

import numpy as np
import pytest

from iqp_cond.exact import total_variation_distance
from iqp_cond.ising import (
    BetaGrid,
    SamplerThresholds,
    audit_sampler,
    beta_to_broadcast_angles,
    exact_ising_target,
    ising_energies,
    metropolis_chains,
    periodic_lattice_couplings,
)


def test_pennylane_coupling_construction_is_periodic_symmetric_and_reproducible():
    first = periodic_lattice_couplings(666)
    second = periodic_lattice_couplings(666)

    assert first.shape == (16, 16)
    assert np.array_equal(first, second)
    assert np.array_equal(first, first.T)
    assert np.count_nonzero(first) == 64
    assert np.all(first[first > 0.0] <= 2.0)
    assert sha256(first.tobytes()).hexdigest() == (
        "58cd62927e3be95a827e98efb7282af5a62d2232e15953da6362599017989194"
    )
    assert np.count_nonzero(periodic_lattice_couplings(667) != first) > 0


def test_exact_target_is_normalized_little_endian_and_beta_temperature_aligned():
    target = exact_ising_target(666, 1.0 / 3.0)

    assert target.bitstrings[1].tolist() == [1] + [0] * 15
    assert target.probabilities.sum() == pytest.approx(1.0)
    assert np.all(target.probabilities > 0.0)
    assert target.temperature == pytest.approx(3.0)
    assert np.array_equal(
        target.energies,
        ising_energies(target.bitstrings, target.couplings),
    )


def test_beta_grid_requires_anchor_and_interleaved_midpoints():
    grid = BetaGrid.from_interval(0.2, 0.5)
    assert grid.training[2] == pytest.approx(1.0 / 3.0)
    assert grid.heldout == pytest.approx(
        tuple((a + b) / 2.0 for a, b in pairwise(grid.training)),
    )

    with pytest.raises(ValueError, match="midpoints"):
        BetaGrid(grid.training, (0.1, 0.2, 0.3, 0.4))


def test_broadcast_map_spans_zero_to_pi_over_four_without_per_qubit_features():
    grid = BetaGrid.from_interval(0.2, 0.5)
    angles = beta_to_broadcast_angles(np.asarray(grid.training), grid)

    assert angles.shape == (5, 16)
    assert np.all(angles[:, 0, None] == angles)
    assert angles[0, 0] == pytest.approx(0.0)
    assert angles[-1, 0] == pytest.approx(np.pi / 4.0)


def test_higher_beta_changes_the_exact_target():
    low = exact_ising_target(101, 0.2)
    high = exact_ising_target(101, 0.5)

    assert total_variation_distance(low.probabilities, high.probabilities) > 0.1


def test_metropolis_chains_are_reproducible_and_binary():
    target = exact_ising_target(101, 0.3, shape=(2, 2))
    kwargs = dict(
        n_chains=2,
        samples_per_chain=30,
        burn_in=20,
        thinning=2,
        seed=17,
    )
    first = metropolis_chains(target.couplings, target.beta, **kwargs)
    second = metropolis_chains(target.couplings, target.beta, **kwargs)

    assert first.shape == (2, 30, 4)
    assert np.array_equal(first, second)
    assert np.all((first == 0) | (first == 1))


def test_sampler_audit_accepts_exact_iid_and_rejects_stuck_chains():
    target = exact_ising_target(101, 0.25)
    rng = np.random.default_rng(9)
    indices = rng.choice(
        len(target.probabilities),
        size=(4, 2_000),
        p=target.probabilities,
    )
    iid = target.bitstrings[indices]
    thresholds = SamplerThresholds(min_ess_per_chain=200.0)

    accepted = audit_sampler(iid, target, thresholds)
    stuck = np.broadcast_to(iid[0, 0], iid.shape).copy()
    rejected = audit_sampler(stuck, target, thresholds)

    assert accepted.passed, accepted.failed_checks
    assert not rejected.passed
    assert "effective_sample_size" in rejected.failed_checks

from itertools import pairwise
from math import pi, sqrt

import numpy as np
import pytest

from iqp_cond.hopfield import (
    BETA_MAX,
    BETA_MIN,
    HELDOUT_BETAS,
    PATTERNS_19,
    TRAIN_BETAS,
    HopfieldTarget,
    angle_for_beta,
    broadcast_angles,
    hopfield_targets,
    sample_beta_targets,
)


def test_frozen_patterns_have_documented_overlap():
    overlap = PATTERNS_19[0] @ PATTERNS_19[1] / PATTERNS_19.shape[1]
    assert overlap == pytest.approx(1 / 19)
    assert PATTERNS_19.shape == (2, 19)
    assert np.all(np.abs(PATTERNS_19) == 1.0)


def test_angle_mapping_endpoints_and_monotonicity():
    assert angle_for_beta(BETA_MIN) == pytest.approx(0.0)
    assert angle_for_beta(BETA_MAX) == pytest.approx(pi / 4)
    assert angle_for_beta(sqrt(BETA_MIN * BETA_MAX)) == pytest.approx(pi / 8)
    betas = np.logspace(np.log10(BETA_MIN), np.log10(BETA_MAX), 41)
    angles = [angle_for_beta(beta) for beta in betas]
    assert np.all(np.diff(angles) > 0)


@pytest.mark.parametrize("beta", [-1.0, 0.0, 0.1, 1.5])
def test_angle_mapping_rejects_out_of_range(beta):
    with pytest.raises(ValueError, match="beta must lie between"):
        angle_for_beta(beta)


def test_target_normalized_and_flip_symmetric():
    for beta in (0.5, 0.75, 1.0, 1.25):
        target = HopfieldTarget.create(beta)
        assert target.probabilities.sum() == pytest.approx(1.0, abs=1e-12)
        assert target.probabilities.min() > 0.0
        assert target.shell_probabilities.sum() == pytest.approx(1.0)
        weights = np.arange(20, dtype=np.float64)
        assert weights @ target.shell_probabilities == pytest.approx(9.5, abs=1e-9)


def test_pattern_mass_grows_with_beta():
    masses = [
        HopfieldTarget.create(beta).pattern_mass
        for beta in (0.5, 0.75, 1.0, 1.25)
    ]
    assert masses[0] == pytest.approx(0.0005, abs=5e-4)
    assert masses[-1] == pytest.approx(0.0627, abs=5e-4)
    assert all(later > earlier for earlier, later in pairwise(masses))


def test_pattern_states_have_equal_maximum_probability():
    target = HopfieldTarget.create(4.0)
    assert target.probabilities[target.pattern_indices] == pytest.approx(
        target.probabilities.max(),
    )


def test_bitstring_probabilities_match_enumeration_order():
    target = HopfieldTarget.create(1.0)
    bitstrings = np.asarray(
        [[0] * 19, [1] + [0] * 18, [0, 1] + [0] * 17],
        dtype=np.int8,
    )
    direct = target.bitstring_probabilities(bitstrings)
    assert direct[0] == pytest.approx(target.probabilities[0])
    assert direct[1] == pytest.approx(target.probabilities[1])
    assert direct[2] == pytest.approx(target.probabilities[2])
    assert direct.sum() == pytest.approx(target.probabilities[:3].sum())


def test_sampling_is_consistent_with_exact_target():
    rng = np.random.default_rng(42)
    target = HopfieldTarget.create(2.0)
    samples = target.sample(rng, 20_000)
    assert samples.shape == (20_000, 19)
    assert set(np.unique(samples)) <= {0, 1}
    empirical = np.mean(samples, axis=0)
    expected = 1.0 - np.asarray(
        [
            target.probabilities @ ((np.arange(2**19) >> q) & 1)
            for q in range(19)
        ],
    )
    assert np.max(np.abs(empirical - expected)) < 0.02


def test_heldout_betas_interleave_training_betas_in_log_space():
    combined = np.sort(np.concatenate([TRAIN_BETAS, HELDOUT_BETAS]))
    expected = np.asarray(
        [
            0.5,
            np.sqrt(0.5 * 0.75),
            0.75,
            np.sqrt(0.75 * 1.0),
            1.0,
            np.sqrt(1.0 * 1.25),
            1.25,
        ],
    )
    assert np.allclose(combined, expected)
    assert set(TRAIN_BETAS).isdisjoint(HELDOUT_BETAS)


def test_beta_dataset_and_broadcast_shapes():
    samples = sample_beta_targets(TRAIN_BETAS, seed=7, n_samples=11)
    encoded = broadcast_angles(TRAIN_BETAS)

    assert samples.shape == (4, 11, 19)
    assert encoded.shape == (4, 19)
    assert encoded[:, 0] == pytest.approx(
        [angle_for_beta(beta) for beta in TRAIN_BETAS],
    )
    assert np.all(encoded == encoded[:, :1])


def test_targets_are_exact_and_repeatable():
    first = hopfield_targets(TRAIN_BETAS)
    second = hopfield_targets(TRAIN_BETAS)
    for target_a, target_b in zip(first, second, strict=True):
        assert target_a.probabilities == pytest.approx(target_b.probabilities)


def test_exact_forward_jax_matches_numpy():
    import numpy as np

    from iqp_cond.ansatz import complete_ansatz
    from iqp_cond.exact import (
        exact_iqp_probabilities_many,
        exact_iqp_probabilities_many_jax,
    )

    ansatz = complete_ansatz()
    rng = np.random.default_rng(5)
    params = rng.normal(0.0, 0.1, ansatz.n_parameters)
    angles = np.repeat(rng.uniform(0.0, np.pi / 4, (3, 1)), 19, axis=1)
    numpy_probs = exact_iqp_probabilities_many(params, ansatz.generators, angles)
    jax_probs = exact_iqp_probabilities_many_jax(
        np.asarray(params),
        ansatz.generators,
        angles,
    )
    np.testing.assert_allclose(np.asarray(jax_probs), numpy_probs, atol=1e-12)


def test_exact_mmd_vanishes_on_the_zero_parameter_model():
    import numpy as np

    from iqp_cond.ansatz import complete_ansatz
    from iqp_cond.exact import enumerate_bitstrings
    from iqp_cond.hopfield import angle_for_beta
    from iqp_cond.mmd import exact_conditional_mmd, exact_target_fouriers

    angle = angle_for_beta(0.75)
    bitstrings = enumerate_bitstrings(19)
    weights = bitstrings.sum(axis=1).astype(np.float64)
    cosine = np.cos(angle / 2.0)
    sine = np.sin(angle / 2.0)
    target = cosine ** (2.0 * (19.0 - weights)) * sine ** (2.0 * weights)
    target = target / target.sum()
    ansatz = complete_ansatz()
    angles = np.repeat(np.asarray([[angle]]), 19, axis=1)
    result = exact_conditional_mmd(
        np.zeros(ansatz.n_parameters),
        ansatz.generators,
        angles,
        exact_target_fouriers(target[None, :]),
        (1.0, 2.0, 4.0),
    )
    assert float(result.loss) == pytest.approx(0.0, abs=1e-12)
    assert result.per_condition.shape == (1,)


def test_exact_mmd_distinguishes_different_targets():
    from iqp_cond.ansatz import complete_ansatz
    from iqp_cond.hopfield import broadcast_angles
    from iqp_cond.mmd import exact_conditional_mmd, exact_target_fouriers

    target_a = HopfieldTarget.create(0.5)
    target_b = HopfieldTarget.create(1.25)
    ansatz = complete_ansatz()
    result = exact_conditional_mmd(
        np.zeros(ansatz.n_parameters),
        ansatz.generators,
        broadcast_angles(np.asarray([0.5, 1.25])),
        exact_target_fouriers(
            np.stack([target_a.probabilities, target_b.probabilities]),
        ),
        (1.0, 2.0, 4.0),
    )
    assert float(result.loss) > 0.0
    assert result.per_condition.shape == (2,)

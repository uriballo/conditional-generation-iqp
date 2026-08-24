import numpy as np
import pytest

from iqp_cond import HammingBallsTarget
from iqp_cond.exact import enumerate_bitstrings

N_QUBITS = 19
CENTER_WEIGHT = N_QUBITS // 2
FLIP = 0.08


def _bitstring_indices(bitstrings):
    shifts = np.arange(bitstrings.shape[1], dtype=np.int64)
    return (bitstrings.astype(np.int64) * (1 << shifts)).sum(axis=1)


def test_create_is_deterministic_in_pattern_seed():
    first = HammingBallsTarget.create(N_QUBITS, 4, FLIP, pattern_seed=7)
    second = HammingBallsTarget.create(N_QUBITS, 4, FLIP, pattern_seed=7)
    other = HammingBallsTarget.create(N_QUBITS, 4, FLIP, pattern_seed=8)

    assert np.array_equal(first.centers, second.centers)
    assert not np.array_equal(first.centers, other.centers)


def test_centers_are_binary_with_fixed_weight():
    target = HammingBallsTarget.create(N_QUBITS, 8, FLIP)

    assert target.centers.shape == (8, N_QUBITS)
    assert np.all((target.centers == 0) | (target.centers == 1))
    assert np.all(target.centers.sum(axis=1) == CENTER_WEIGHT)


def test_nested_family_uses_prefixes():
    family = HammingBallsTarget.nested_family()

    assert [target.n_centers for target in family] == [1, 2, 4, 8]
    for larger, smaller in zip(family[1:], family[:-1], strict=True):
        assert np.array_equal(larger.centers[: smaller.n_centers], smaller.centers)


def test_nested_family_matches_create_at_full_count():
    family = HammingBallsTarget.nested_family()
    direct = HammingBallsTarget.create(N_QUBITS, 8, FLIP, pattern_seed=0)

    assert np.array_equal(family[-1].centers, direct.centers)


def test_bitstring_probabilities_normalized_and_positive():
    for target in HammingBallsTarget.nested_family():
        probabilities = target.bitstring_probabilities(enumerate_bitstrings(N_QUBITS))

        assert probabilities.sum() == pytest.approx(1.0)
        assert np.all(probabilities > 0)


def test_single_ball_matches_closed_form():
    target = HammingBallsTarget.nested_family()[0]
    center = target.centers[0]
    distances = np.count_nonzero(enumerate_bitstrings(N_QUBITS) != center, axis=1)
    expected = FLIP**distances * (1.0 - FLIP) ** (N_QUBITS - distances)

    probabilities = target.bitstring_probabilities(enumerate_bitstrings(N_QUBITS))
    assert np.allclose(probabilities, expected)


def test_center_is_the_mode_for_a_single_ball():
    target = HammingBallsTarget.nested_family()[0]
    probabilities = target.bitstring_probabilities(enumerate_bitstrings(N_QUBITS))
    center_index = int(_bitstring_indices(target.centers[:1])[0])
    neighbor_indices = np.flatnonzero(
        np.count_nonzero(enumerate_bitstrings(N_QUBITS) != target.centers[0], axis=1)
        == 1,
    )

    assert probabilities[center_index] == probabilities.max()
    assert np.all(probabilities[neighbor_indices] < probabilities[center_index])


def test_sampling_matches_exact_probabilities():
    target = HammingBallsTarget.create(10, 2, FLIP, pattern_seed=3)
    samples = target.sample(np.random.default_rng(0), 1_000_000)
    bitstrings = enumerate_bitstrings(10)
    exact = target.bitstring_probabilities(bitstrings)
    empirical = (
        np.bincount(
            _bitstring_indices(samples),
            minlength=bitstrings.shape[0],
        )
        / samples.shape[0]
    )
    tvd = 0.5 * np.abs(empirical - exact).sum()

    assert tvd < 0.02


def test_sample_is_binary_with_expected_mean_weight():
    target = HammingBallsTarget.create(N_QUBITS, 4, FLIP, pattern_seed=3)
    samples = target.sample(np.random.default_rng(1), 5_000)

    assert samples.dtype == np.int8
    assert np.all((samples == 0) | (samples == 1))
    assert samples.sum(axis=1).mean() == pytest.approx(9.08, abs=0.05)


def test_all_classes_share_identical_shell_profiles():
    family = HammingBallsTarget.nested_family()
    profiles = [target.shell_probabilities for target in family]

    for profile in profiles[1:]:
        assert np.array_equal(profile, profiles[0])
    assert profiles[0].sum() == pytest.approx(1.0)
    assert np.all(profiles[0] > 0)


def test_shell_probabilities_agree_with_bitstring_aggregation():
    target = HammingBallsTarget.create(10, 2, FLIP, pattern_seed=3)
    bitstrings = enumerate_bitstrings(10)
    aggregated = np.bincount(
        bitstrings.sum(axis=1),
        weights=target.bitstring_probabilities(bitstrings),
        minlength=11,
    )

    assert np.allclose(aggregated, target.shell_probabilities)


def test_actual_mean_matches_analytic_value():
    weights = np.arange(N_QUBITS + 1, dtype=np.float64)
    expected = CENTER_WEIGHT + (N_QUBITS - 2 * CENTER_WEIGHT) * FLIP

    for target in HammingBallsTarget.nested_family():
        assert target.actual_mean == pytest.approx(expected, abs=1e-12)
        assert target.actual_mean == pytest.approx(weights @ target.shell_probabilities)
        assert target.actual_std == pytest.approx(
            np.sqrt(((weights - target.actual_mean) ** 2) @ target.shell_probabilities),
        )


def test_validation_rejects_bad_arguments():
    with pytest.raises(ValueError):
        HammingBallsTarget.create(0, 2, FLIP)
    with pytest.raises(ValueError):
        HammingBallsTarget.create(1, 2, FLIP)
    with pytest.raises(ValueError):
        HammingBallsTarget.create(N_QUBITS, 0, FLIP)
    with pytest.raises(ValueError):
        HammingBallsTarget.create(N_QUBITS, 2, 0.0)
    with pytest.raises(ValueError):
        HammingBallsTarget.create(N_QUBITS, 2, 1.0)
    with pytest.raises(ValueError):
        HammingBallsTarget.nested_family(center_counts=(2, 1))
    with pytest.raises(ValueError):
        HammingBallsTarget.nested_family(center_counts=(1, 0))


def test_bitstring_probabilities_rejects_non_binary_input():
    target = HammingBallsTarget.create(N_QUBITS, 2, FLIP)

    with pytest.raises(ValueError):
        target.bitstring_probabilities(np.zeros((3, N_QUBITS), dtype=np.int8) + 2)
    with pytest.raises(ValueError):
        target.bitstring_probabilities(np.zeros((3, N_QUBITS + 1), dtype=np.int8))


def test_sample_rejects_non_positive_count():
    target = HammingBallsTarget.create(N_QUBITS, 2, FLIP)

    with pytest.raises(ValueError):
        target.sample(np.random.default_rng(0), 0)

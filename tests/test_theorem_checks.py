import itertools

import numpy as np
import pytest

from iqp_cond.exact import enumerate_bitstrings, exact_iqp_probabilities
from iqp_cond.response import (
    active_generator_indices,
    binary_span,
    coordinate_bandwidth,
    directional_bandwidth,
    full_frequency_support,
    observable_coordinate_bandwidth,
    observable_frequency_support,
    observable_lightcone,
    surviving_observable_masks,
)


def _expectation(params, generators, angles, observable):
    probabilities = exact_iqp_probabilities(params, generators, angles)
    bitstrings = enumerate_bitstrings(len(observable))
    characters = 1.0 - 2.0 * ((bitstrings @ observable) % 2)
    return float(probabilities @ characters)


def _row_set(values):
    return {tuple(row) for row in np.asarray(values)}


def _expanded_observable_response(params, generators, angles, observable):
    active = active_generator_indices(generators, observable)
    response = 0.0 + 0.0j
    for selection in itertools.product((0, 1), repeat=active.size):
        mask = np.zeros_like(observable)
        coefficient = 1.0 + 0.0j
        for take, generator_index in zip(selection, active, strict=True):
            if take:
                mask ^= generators[generator_index]
                coefficient *= 1j * np.sin(2.0 * params[generator_index])
            else:
                coefficient *= np.cos(2.0 * params[generator_index])
        if np.any(mask & observable):
            continue
        response += (
            coefficient
            * np.prod(np.cos(angles[observable == 1]))
            * np.prod(
                np.sin(angles[mask == 1]),
            )
        )
    return float(np.real_if_close(response))


def test_full_multivariate_support_enumerates_affine_ry_bound():
    weights = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

    support = full_frequency_support(weights)
    support_rows = _row_set(support)

    assert len(support_rows) <= 3 ** weights.shape[0]
    assert (0.0, 0.0) in support_rows
    assert all(tuple(-np.asarray(row)) in support_rows for row in support_rows)
    assert (2.0, 3.0) in support_rows


def test_frequency_collisions_shrink_the_structural_support():
    support = full_frequency_support(np.ones((3, 1)))

    assert support[:, 0] == pytest.approx(np.arange(-3.0, 4.0))
    assert support.shape[0] < 3**3


def test_sampled_output_probability_has_no_frequency_outside_bound():
    weights = np.array([[1.0], [2.0]])
    generators = np.array([[1, 0], [0, 1], [1, 1]])
    params = np.array([0.17, -0.31, 0.27])
    sample_count = 64
    conditions = 2.0 * np.pi * np.arange(sample_count) / sample_count
    values = np.array(
        [
            exact_iqp_probabilities(
                params,
                generators,
                np.array([0.13, -0.21]) + weights[:, 0] * condition,
            )[0]
            for condition in conditions
        ],
    )
    coefficients = np.fft.fft(values) / sample_count
    frequencies = np.rint(np.fft.fftfreq(sample_count) * sample_count).astype(int)
    allowed = {int(row[0]) for row in full_frequency_support(weights)}

    assert np.max(np.abs(coefficients[~np.isin(frequencies, list(allowed))])) < 1e-12


def test_active_generators_and_lightcone_use_anticommutation_parity():
    generators = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
    observable = np.array([1, 0, 0])

    assert active_generator_indices(generators, observable).tolist() == [0, 1]
    assert observable_lightcone(generators, observable).tolist() == [1, 1, 0]
    assert _row_set(binary_span(generators[:2])) == {
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
    }


def test_real_input_y_filter_removes_spurious_neighbor_frequency():
    generators = np.array([[1, 1]])
    observable = np.array([1, 0])
    weights = np.eye(2)

    assert _row_set(surviving_observable_masks(generators, observable)) == {
        (0, 0),
    }
    assert _row_set(observable_frequency_support(weights, generators, observable)) == {
        (-1.0, 0.0),
        (1.0, 0.0),
    }

    theta = 0.37
    alpha_one = 0.41
    values = [
        _expectation(
            np.array([theta]),
            generators,
            np.array([alpha_one, alpha_two]),
            observable,
        )
        for alpha_two in (0.0, 0.29, 1.17)
    ]
    assert values == pytest.approx(
        [np.cos(2.0 * theta) * np.cos(alpha_one)] * 3,
    )


def test_two_active_edges_can_import_neighbors_only_jointly():
    generators = np.array([[1, 1, 0], [1, 0, 1]])
    observable = np.array([1, 0, 0])
    weights = np.eye(3)

    survivors = _row_set(surviving_observable_masks(generators, observable))
    support = _row_set(observable_frequency_support(weights, generators, observable))

    assert survivors == {(0, 0, 0), (0, 1, 1)}
    assert (1.0, 1.0, 1.0) in support
    assert (1.0, 1.0, 0.0) not in support


def test_active_span_expansion_matches_random_exact_small_systems():
    rng = np.random.default_rng(20260810)
    for n_qubits in range(1, 5):
        for _ in range(20):
            n_generators = int(rng.integers(1, 7))
            generators = rng.integers(
                0,
                2,
                size=(n_generators, n_qubits),
                dtype=np.int8,
            )
            generators[np.all(generators == 0, axis=1), 0] = 1
            observable = rng.integers(0, 2, size=n_qubits, dtype=np.int8)
            params = rng.uniform(-1.0, 1.0, size=n_generators)
            angles = rng.uniform(-1.0, 1.0, size=n_qubits)

            expected = _expectation(params, generators, angles, observable)
            actual = _expanded_observable_response(
                params,
                generators,
                angles,
                observable,
            )

            assert actual == pytest.approx(expected, abs=1e-12)


def test_inactive_generator_parameter_cancels_from_exact_expectation():
    generators = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    observable = np.array([1, 0, 0])
    angles = np.array([0.31, 0.47, 0.83])
    baseline = _expectation(
        np.array([0.23, -0.37, 0.0]),
        generators,
        angles,
        observable,
    )
    changed = _expectation(
        np.array([0.23, -0.37, 1.19]),
        generators,
        angles,
        observable,
    )

    assert changed == pytest.approx(baseline, abs=1e-14)


def test_bandwidth_helpers_match_weight_sums_and_lightcone():
    weights = np.array([[1.0, -2.0], [-3.0, 0.5], [7.0, 11.0]])
    generators = np.array([[1, 1, 0], [0, 1, 1]])
    observable = np.array([1, 0, 0])

    assert coordinate_bandwidth(weights) == pytest.approx([11.0, 13.5])
    assert directional_bandwidth(weights, np.array([2.0, -1.0])) == pytest.approx(
        13.5,
    )
    assert observable_coordinate_bandwidth(
        weights,
        generators,
        observable,
    ) == pytest.approx([4.0, 2.5])


@pytest.mark.parametrize(
    "call",
    [
        lambda: full_frequency_support(np.array([1.0, 2.0])),
        lambda: active_generator_indices(np.array([[2, 0]]), np.array([1, 0])),
        lambda: observable_frequency_support(
            np.ones((3, 1)),
            np.array([[1, 1]]),
            np.array([1, 0]),
        ),
        lambda: directional_bandwidth(np.ones((2, 2)), np.ones(3)),
    ],
)
def test_response_checks_reject_invalid_shapes_and_masks(call):
    with pytest.raises(ValueError):
        call()

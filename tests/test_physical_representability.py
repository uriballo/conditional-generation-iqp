from pathlib import Path

import numpy as np

from iqp_cond.physical_representability import (
    curie_weiss_target,
    expectation_errors,
    ferromagnetic_target,
    lattice_gas_target,
    periodic_square_edges,
)


def test_periodic_square_has_two_edges_per_site() -> None:
    edges = periodic_square_edges(4)
    assert len(edges) == 32
    assert len(set(tuple(sorted(edge)) for edge in edges)) == 32


def test_exact_physical_targets_are_normalized_and_symmetric() -> None:
    spin = ferromagnetic_target(0.4406868)
    gas_low = lattice_gas_target(-2.5)
    gas_high = lattice_gas_target(-1.5)
    for target in (spin, gas_low, gas_high):
        assert target.probabilities.shape == (2**16,)
        assert np.isclose(target.probabilities.sum(), 1.0)
        assert np.all(target.probabilities >= 0.0)
    density = gas_low.observables["density"]
    low_mean = np.dot(gas_low.probabilities, density)
    high_mean = np.dot(gas_high.probabilities, density)
    assert low_mean < high_mean


def test_curie_weiss_is_exchange_and_spin_flip_symmetric() -> None:
    target = curie_weiss_target(1.0, n_qubits=8)
    assert target.probabilities.shape == (256,)
    assert np.isclose(target.probabilities.sum(), 1.0)
    assert np.allclose(target.probabilities, target.probabilities[::-1])
    # All configurations in one Hamming-weight shell have equal probability.
    weights = np.fromiter((index.bit_count() for index in range(256)), dtype=np.int8)
    for weight in range(9):
        shell = target.probabilities[weights == weight]
        assert np.allclose(shell, shell[0])


def test_curie_weiss_orders_as_inverse_temperature_increases() -> None:
    warm = curie_weiss_target(0.5, n_qubits=16)
    cold = curie_weiss_target(1.5, n_qubits=16)
    observable = warm.observables["absolute_magnetization"]
    assert np.dot(warm.probabilities, observable) < np.dot(
        cold.probabilities,
        observable,
    )


def test_expectation_errors_vanish_for_exact_target() -> None:
    target = ferromagnetic_target(0.2)
    errors = expectation_errors(target, target.probabilities)
    assert errors
    assert all(abs(value) < 1e-15 for value in errors.values())


def test_pilot_script_is_present() -> None:
    assert Path("scripts/run_physical_representability_pilot.py").is_file()
    assert Path("scripts/run_curie_weiss_representability.py").is_file()
    assert Path("scripts/run_curie_weiss_symmetry_diagnostic.py").is_file()
    assert Path("scripts/run_curie_weiss_capacity_diagnostic.py").is_file()

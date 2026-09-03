import numpy as np
import pytest

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import (
    exact_iqp_probabilities_many,
    exact_iqp_visible_probabilities,
)


def test_fast_two_local_path_matches_generic_fallback():
    rng = np.random.default_rng(4)
    n = 6
    generators = np.zeros((8, n), dtype=np.int8)
    generators[:6, np.arange(6)] = 1
    generators[6, [0, 3]] = 1
    generators[7, [2, 5]] = 1
    params = rng.normal(size=8)
    angles = rng.normal(size=(3, n))
    generic = generators.copy()
    generic[7, 1] = 1  # force the existing arbitrary-support fallback
    expected = exact_iqp_probabilities_many(params, generic, angles)
    # The fast path is tested on the same physical generators after adding a
    # cancelling 3-local pair, which leaves the phase unchanged.
    fast_generators = generators.copy()
    fast_params = params.copy()
    fast_generators = np.vstack([fast_generators, np.zeros((1, n), dtype=np.int8)])
    fast_generators[-1, [1, 2, 5]] = 1
    fast_params = np.r_[fast_params, 0.0]
    actual = exact_iqp_probabilities_many(fast_params, fast_generators, angles)
    assert np.allclose(
        actual,
        exact_iqp_probabilities_many(params, generators, angles),
        atol=1e-12,
    )
    assert expected.shape == actual.shape


@pytest.mark.parametrize("n_qubits", [8, 12, 16])
def test_fast_complete_two_local_evaluator_matches_legacy_path_through_16(n_qubits):
    rng = np.random.default_rng(n_qubits)
    generators = np.asarray(complete_ansatz(n_qubits).generators, dtype=np.int8)
    params = rng.normal(0.0, 0.1, generators.shape[0])
    angles = rng.normal(size=(1, n_qubits))
    fast = exact_iqp_probabilities_many(params, generators, angles)
    generic_generators = np.vstack([generators, np.zeros((1, n_qubits), dtype=np.int8)])
    generic_generators[-1, :3] = 1
    generic = exact_iqp_probabilities_many(
        np.r_[params, 0.0], generic_generators, angles
    )
    assert np.allclose(fast, generic, atol=1e-11)


def test_visible_marginal_matches_full_distribution():
    rng = np.random.default_rng(8)
    n_visible, n_ancillas = 4, 2
    generators = rng.integers(0, 2, size=(12, n_visible + n_ancillas), dtype=np.int8)
    params = rng.normal(size=12)
    angles = rng.normal(size=(2, n_visible + n_ancillas))
    full = exact_iqp_probabilities_many(params, generators, angles)
    expected = full.reshape(2, 2**n_ancillas, 2**n_visible).sum(axis=1)
    actual = exact_iqp_visible_probabilities(
        params, generators, angles, n_visible, n_ancillas
    )
    assert np.allclose(actual, expected, atol=1e-12)
    assert np.allclose(actual.sum(axis=1), 1.0)


def test_visible_marginal_accepts_single_condition_and_validates_width():
    generators = np.eye(3, dtype=np.int8)
    params = np.array([0.1, -0.2, 0.3])
    result = exact_iqp_visible_probabilities(params, generators, np.zeros(3), 2, 1)
    assert result.shape == (1, 4)
    with pytest.raises(ValueError, match="generator width"):
        exact_iqp_visible_probabilities(params, generators, np.zeros(3), 1, 1)
    with pytest.raises(ValueError, match="n_visible"):
        exact_iqp_visible_probabilities(params, generators, np.zeros(3), 0, 3)

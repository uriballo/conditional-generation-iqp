"""Tests for ancilla-assisted conditional IQP generation."""

import numpy as np
import pytest

from iqp_cond.ancilla import ancilla_only_condition_angles, marginalize_ancillas
from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import exact_iqp_probabilities_many
from iqp_cond.mmd import exact_conditional_mmd, exact_target_fouriers


def test_ancilla_only_condition_angles():
    ancilla_angles = np.array([[0.0, np.pi], [np.pi, 0.0]])
    full = ancilla_only_condition_angles(ancilla_angles, n_data_qubits=3)
    assert full.shape == (2, 5)
    np.testing.assert_allclose(full[0], [0.0, 0.0, 0.0, 0.0, np.pi])
    np.testing.assert_allclose(full[1], [0.0, 0.0, 0.0, np.pi, 0.0])


def test_marginalize_ancillas_normalization():
    n_data = 3
    n_ancillas = 2
    ansatz = complete_ansatz(n_data + n_ancillas)
    params = np.zeros(ansatz.n_parameters)
    angles = ancilla_only_condition_angles(np.array([[0.2, 0.2]]), n_data)

    joint = exact_iqp_probabilities_many(params, ansatz.generators, angles)
    marginal_probs = marginalize_ancillas(joint, n_data, n_ancillas)
    assert marginal_probs.shape == (1, 2**n_data)
    np.testing.assert_allclose(marginal_probs.sum(), 1.0, atol=1e-6)


def test_exact_mmd_marginalizes_ancillas():
    """Exact Conditional MMD compares data marginals, not joint tables."""
    n_data = 3
    n_ancillas = 1
    ansatz = complete_ansatz(n_data + n_ancillas)
    params = np.linspace(-0.2, 0.2, ansatz.n_parameters)
    full_angles = ancilla_only_condition_angles(np.array([[0.2]]), n_data)
    joint = exact_iqp_probabilities_many(params, ansatz.generators, full_angles)
    marginal_probs = marginalize_ancillas(joint, n_data, n_ancillas)
    result = exact_conditional_mmd(
        params,
        ansatz.generators,
        full_angles,
        exact_target_fouriers(marginal_probs),
        (1.0, 2.0, 4.0),
    )
    assert float(result.loss) == pytest.approx(0.0, abs=1e-12)

"""Tests for ancilla-assisted conditional IQP generation."""

import numpy as np
import pytest

from iqp_cond.ancilla import (
    ancilla_ansatz,
    ancilla_condition_angles,
    ancilla_only_condition_angles,
    exact_ancilla_marginal_probabilities,
    exact_ancilla_marginal_probabilities_from_full_angles,
    pad_observables_for_ancillas,
)
from iqp_cond.exact import exact_iqp_probabilities_many
from iqp_cond.mmd import exact_conditional_mmd, exact_target_fouriers


def test_ancilla_condition_angles():
    data_angles = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]])
    broadcast = ancilla_condition_angles(
        data_angles,
        n_ancillas=2,
        scheme="broadcast_all",
    )
    assert broadcast.shape == (2, 5)
    np.testing.assert_allclose(broadcast[0], [0.1, 0.1, 0.1, 0.1, 0.1])
    np.testing.assert_allclose(broadcast[1], [0.5, 0.5, 0.5, 0.5, 0.5])

    fixed = ancilla_condition_angles(data_angles, n_ancillas=2, scheme="fixed_zero")
    assert fixed.shape == (2, 5)
    np.testing.assert_allclose(fixed[0], [0.1, 0.1, 0.1, 0.0, 0.0])
    np.testing.assert_allclose(fixed[1], [0.5, 0.5, 0.5, 0.0, 0.0])


def test_ancilla_only_condition_angles():
    ancilla_angles = np.array([[0.0, np.pi], [np.pi, 0.0]])
    full = ancilla_only_condition_angles(ancilla_angles, n_data_qubits=3)
    assert full.shape == (2, 5)
    np.testing.assert_allclose(full[0], [0.0, 0.0, 0.0, 0.0, np.pi])
    np.testing.assert_allclose(full[1], [0.0, 0.0, 0.0, np.pi, 0.0])


def test_pad_observables_for_ancillas():
    data_ops = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)
    padded = pad_observables_for_ancillas(data_ops, n_ancillas=2)
    assert padded.shape == (2, 5)
    np.testing.assert_array_equal(padded[0], [1, 0, 1, 0, 0])
    np.testing.assert_array_equal(padded[1], [0, 1, 1, 0, 0])


def test_marginalize_ancillas_normalization():
    n_data = 3
    n_ancillas = 2
    ansatz = ancilla_ansatz(n_data, n_ancillas)
    params = np.zeros(ansatz.n_parameters)
    angles = np.array([[0.2, 0.2, 0.2]])

    marginal_probs = exact_ancilla_marginal_probabilities(
        params,
        ansatz.generators,
        angles,
        n_data_qubits=n_data,
        n_ancillas=n_ancillas,
    )
    assert marginal_probs.shape == (1, 2**n_data)
    np.testing.assert_allclose(marginal_probs.sum(), 1.0, atol=1e-6)


def test_full_angle_marginal_matches_broadcast_wrapper():
    n_data = 3
    n_ancillas = 2
    ansatz = ancilla_ansatz(n_data, n_ancillas)
    params = np.linspace(-0.1, 0.1, ansatz.n_parameters)
    data_angles = np.array([[0.2, 0.2, 0.2], [-0.1, -0.1, -0.1]])
    full_angles = ancilla_condition_angles(data_angles, n_ancillas, "broadcast_all")
    expected = exact_ancilla_marginal_probabilities(
        params,
        ansatz.generators,
        data_angles,
        n_data_qubits=n_data,
        n_ancillas=n_ancillas,
    )
    actual = exact_ancilla_marginal_probabilities_from_full_angles(
        params,
        ansatz.generators,
        full_angles,
        n_data_qubits=n_data,
        n_ancillas=n_ancillas,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_ancilla_uncoupled_identity():
    """Uncoupled ancillas leave the visible probabilities unchanged."""
    n_data = 3
    n_ancillas = 1

    data_ansatz = ancilla_ansatz(n_data, 0)
    params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 3 single + 3 pair
    data_angles = np.array([[0.3, 0.3, 0.3]])

    data_probs = exact_iqp_probabilities_many(
        params,
        data_ansatz.generators,
        data_angles,
    )

    # Construct N=4 ansatz, but set all generators involving ancilla (qubit 3) to 0
    full_ansatz = ancilla_ansatz(n_data, n_ancillas)
    full_params = np.zeros(full_ansatz.n_parameters)
    # Complete graph generator order: 4 1-body then 6 2-body
    # Copy data 1-body params to first 3
    full_params[:3] = params[:3]
    # 2-body edges for 4 qubits: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # data 2-body edges: (0,1), (0,2), (1,2)
    full_params[4] = params[3]  # (0,1)
    full_params[5] = params[4]  # (0,2)
    full_params[7] = params[5]  # (1,2)

    marginal_probs = exact_ancilla_marginal_probabilities(
        full_params,
        full_ansatz.generators,
        data_angles,
        n_data_qubits=n_data,
        n_ancillas=n_ancillas,
        scheme="fixed_zero",
    )

    np.testing.assert_allclose(marginal_probs[0], data_probs[0], atol=1e-6)


def test_exact_mmd_marginalizes_ancillas():
    """Exact Conditional MMD compares data marginals, not joint tables."""
    n_data = 3
    n_ancillas = 1
    ansatz = ancilla_ansatz(n_data, n_ancillas)
    params = np.linspace(-0.2, 0.2, ansatz.n_parameters)
    data_angles = np.array([[0.2, 0.2, 0.2]])
    full_angles = ancilla_condition_angles(
        data_angles,
        n_ancillas,
        scheme="fixed_zero",
    )
    marginal_probs = exact_ancilla_marginal_probabilities(
        params,
        ansatz.generators,
        data_angles,
        n_data_qubits=n_data,
        n_ancillas=n_ancillas,
        scheme="fixed_zero",
    )
    result = exact_conditional_mmd(
        params,
        ansatz.generators,
        full_angles,
        exact_target_fouriers(marginal_probs),
        (1.0, 2.0, 4.0),
    )
    assert float(result.loss) == pytest.approx(0.0, abs=1e-12)

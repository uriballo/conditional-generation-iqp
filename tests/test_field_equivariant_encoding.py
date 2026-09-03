"""Contracts for the signed-field equivariant input state."""

from __future__ import annotations

from math import comb

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.encoding import (
    FIELD_EQUIVARIANT_ENCODING,
    field_equivariant_shell_amplitudes,
)
from iqp_cond.exact import enumerate_bitstrings, exact_iqp_probabilities_many
from iqp_cond.simulator import estimate_pauli_z

jax.config.update("jax_enable_x64", True)


def _angles(field: float, n_qubits: int) -> np.ndarray:
    angle = -(7.0 * np.pi / 6.0) * field
    return np.full(n_qubits, angle, dtype=np.float64)


def test_shell_amplitudes_are_normalized_and_transform_under_field_flip() -> None:
    n_qubits = 6
    positive = np.asarray(
        field_equivariant_shell_amplitudes(jnp.asarray(_angles(0.2, 1)), n_qubits)
    )
    negative = np.asarray(
        field_equivariant_shell_amplitudes(jnp.asarray(_angles(-0.2, 1)), n_qubits)
    )
    multiplicities = np.asarray(
        [comb(n_qubits, k) for k in range(n_qubits + 1)]
    )
    assert np.isclose(np.sum(multiplicities * np.abs(positive) ** 2), 1.0)
    np.testing.assert_allclose(
        negative,
        ((-1.0) ** np.arange(n_qubits + 1)) * positive,
        atol=1e-12,
    )


def test_exact_output_obeys_conditional_bit_flip_equivariance() -> None:
    n_qubits = 6
    ansatz = complete_ansatz(n_qubits)
    params = np.random.default_rng(4).normal(0.0, 0.2, ansatz.n_parameters)
    probabilities = exact_iqp_probabilities_many(
        params,
        ansatz.generators,
        np.stack([_angles(0.2, n_qubits), _angles(-0.2, n_qubits)]),
        FIELD_EQUIVARIANT_ENCODING,
    )
    np.testing.assert_allclose(probabilities[0], probabilities[1][::-1], atol=1e-12)


def test_classical_estimator_matches_exact_pauli_moments() -> None:
    n_qubits = 5
    ansatz = complete_ansatz(n_qubits)
    rng = np.random.default_rng(8)
    params = rng.normal(0.0, 0.15, ansatz.n_parameters)
    angles = _angles(0.15, n_qubits)[None, :]
    ops = enumerate_bitstrings(n_qubits)[1:]
    probabilities = exact_iqp_probabilities_many(
        params,
        ansatz.generators,
        angles,
        FIELD_EQUIVARIANT_ENCODING,
    )[0]
    bits = enumerate_bitstrings(n_qubits)
    exact = np.asarray(
        [np.sum(probabilities * (1 - 2 * ((bits @ op) % 2))) for op in ops]
    )
    estimate = estimate_pauli_z(
        jnp.asarray(params),
        jnp.asarray(ansatz.generators),
        jnp.asarray(ops),
        jnp.asarray(angles),
        jax.random.PRNGKey(11),
        32_768,
        FIELD_EQUIVARIANT_ENCODING,
    )
    np.testing.assert_allclose(np.asarray(estimate.values[0]), exact, atol=0.035)

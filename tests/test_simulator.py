import functools

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from iqp_cond import (
    estimate_pauli_z,
    exact_iqp_probabilities,
    exact_iqp_probabilities_many,
)


def _observable(mask):
    wires = [wire for wire, bit in enumerate(mask) if bit]
    if not wires:
        return qml.Identity(0)
    return functools.reduce(
        lambda left, right: left @ right,
        [qml.PauliZ(wire) for wire in wires],
    )


def _pennylane_expvals(params, generators, ops, angles):
    n_qubits = generators.shape[1]
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device, interface="jax")
    def circuit(theta):
        for wire, angle in enumerate(angles):
            qml.RY(angle, wires=wire)
        for wire in range(n_qubits):
            qml.Hadamard(wires=wire)
        for value, generator in zip(theta, generators, strict=True):
            wires = np.flatnonzero(generator).tolist()
            if len(wires) == 1:
                qml.RZ(-2.0 * value, wires=wires[0])
            else:
                qml.MultiRZ(-2.0 * value, wires=wires)
        for wire in range(n_qubits):
            qml.Hadamard(wires=wire)
        return [qml.expval(_observable(mask)) for mask in ops]

    return jnp.asarray(circuit(params))


@pytest.mark.parametrize(
    "params,generators,ops,angles",
    [
        ([0.37], [[1]], [[1]], [0.0]),
        ([0.2, -0.31], [[1, 0], [1, 1]], [[1, 0], [1, 1]], [0.2, 0.7]),
        (
            [0.11, -0.23, 0.41, 0.17],
            [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 1, 1]],
            [[1, 0, 0], [0, 1, 1], [1, 1, 0]],
            [0.1, 0.5, 1.1],
        ),
    ],
)
def test_estimates_match_exact_pennylane(params, generators, ops, angles):
    params = jnp.asarray(params, dtype=jnp.float64)
    generators = jnp.asarray(generators, dtype=jnp.float64)
    ops = jnp.asarray(ops, dtype=jnp.float64)
    angles = jnp.asarray(angles, dtype=jnp.float64)

    estimate = estimate_pauli_z(
        params,
        generators,
        ops,
        angles,
        jax.random.PRNGKey(12),
        50_000,
    )
    expected = _pennylane_expvals(
        params,
        np.asarray(generators),
        np.asarray(ops),
        angles,
    )
    tolerance = jnp.maximum(5.0 * estimate.standard_errors[0], 0.01)

    assert jnp.all(jnp.abs(estimate.values[0] - expected) <= tolerance)


def test_exact_fast_walsh_probabilities_match_pennylane():
    params = np.array([0.17, -0.31, 0.27])
    generators = np.array([[1, 0], [0, 1], [1, 1]])
    angles = np.array([0.23, 0.81])
    expected_expvals = _pennylane_expvals(
        jnp.asarray(params),
        generators,
        np.array([[1, 0], [0, 1], [1, 1]]),
        jnp.asarray(angles),
    )
    probabilities = exact_iqp_probabilities(params, generators, angles)
    bitstrings = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    characters = 1 - 2 * ((bitstrings @ np.array([[1, 0], [0, 1], [1, 1]]).T) % 2)

    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities @ characters == pytest.approx(np.asarray(expected_expvals))


def test_batched_exact_probabilities_match_individual_evaluation():
    params = np.array([0.17, -0.31, 0.27])
    generators = np.array([[1, 0], [0, 1], [1, 1]])
    angles = np.array([[0.0, 0.0], [0.23, 0.81], [0.4, 0.4]])

    batched = exact_iqp_probabilities_many(params, generators, angles)
    individual = np.stack(
        [
            exact_iqp_probabilities(params, generators, condition_angles)
            for condition_angles in angles
        ],
    )

    assert batched == pytest.approx(individual)


def test_autodiff_gradient_matches_exact_pennylane():
    params = jnp.array([0.13, -0.24, 0.09])
    generators = jnp.array([[1, 0], [0, 1], [1, 1]], dtype=jnp.float64)
    ops = jnp.array([[1, 0], [1, 1]], dtype=jnp.float64)
    angles = jnp.array([0.2, 0.6], dtype=jnp.float64)
    key = jax.random.PRNGKey(4)

    def estimated_objective(theta):
        return estimate_pauli_z(
            theta,
            generators,
            ops,
            angles,
            key,
            100_000,
        ).values.sum()

    def exact_objective(theta):
        return _pennylane_expvals(
            theta,
            np.asarray(generators),
            np.asarray(ops),
            angles,
        ).sum()

    actual = jax.grad(estimated_objective)(params)
    expected = jax.grad(exact_objective)(params)
    assert actual == pytest.approx(np.asarray(expected), abs=0.025)

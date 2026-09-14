"""Ancilla-assisted conditional IQP Born machine utilities."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def ancilla_only_condition_angles(
    ancilla_angles: jnp.ndarray | np.ndarray,
    n_data_qubits: int,
) -> jnp.ndarray:
    """Place all condition angles on ancillas and fix data angles to zero.

    ``ancilla_angles`` contains one angle per ancilla and has shape
    ``(n_conditions, n_ancillas)`` (or ``(n_ancillas,)`` for one condition).
    The returned vectors follow the repository convention: data qubits occupy
    the first positions and ancillas occupy the final positions.
    """
    if n_data_qubits < 1:
        raise ValueError("n_data_qubits must be positive")
    angles = jnp.asarray(ancilla_angles, dtype=jnp.float64)
    if angles.ndim == 1:
        angles = angles[None, :]
    if angles.ndim != 2 or angles.shape[1] < 1:
        raise ValueError(
            "ancilla_angles must have shape (n_conditions, n_ancillas)",
        )
    data_part = jnp.zeros(
        (angles.shape[0], n_data_qubits),
        dtype=angles.dtype,
    )
    return jnp.concatenate([data_part, angles], axis=1)


def marginalize_ancillas(
    joint_probabilities: np.ndarray,
    n_data_qubits: int,
    n_ancillas: int,
) -> np.ndarray:
    """Sum out ancilla qubits from joint N-qubit bitstring probabilities.

    Assumes little-endian bit ordering where data qubits occupy bits 0..n_data_qubits-1
    and ancillas occupy bits n_data_qubits..n_data_qubits+n_ancillas-1.
    """
    joint_probabilities = np.asarray(joint_probabilities, dtype=np.float64)
    if n_ancillas == 0:
        return joint_probabilities

    if joint_probabilities.ndim == 1:
        reshaped = joint_probabilities.reshape(2**n_ancillas, 2**n_data_qubits)
        return reshaped.sum(axis=0)
    elif joint_probabilities.ndim == 2:
        n_conditions = joint_probabilities.shape[0]
        reshaped = joint_probabilities.reshape(
            n_conditions,
            2**n_ancillas,
            2**n_data_qubits,
        )
        return reshaped.sum(axis=1)
    else:
        raise ValueError("joint_probabilities must be 1D or 2D")

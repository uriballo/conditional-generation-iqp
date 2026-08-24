"""Ancilla-assisted conditional IQP Born machine utilities."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .ansatz import Ansatz, complete_ansatz
from .encoding import ANGLE_ENCODING
from .exact import exact_iqp_probabilities_many


def ancilla_condition_angles(
    data_angles: jnp.ndarray | np.ndarray,
    n_ancillas: int,
    scheme: str = "broadcast_all",
) -> jnp.ndarray:
    """Construct full (N = n_data + n_ancillas) condition angle vectors.

    Parameters
    ----------
    data_angles : jnp.ndarray | np.ndarray
        Broadcast condition angles for data qubits, shape
        ``(n_conditions, n_data)`` or ``(n_data,)``.
    n_ancillas : int
        Number of ancilla qubits.
    scheme : str
        'broadcast_all' applies the same condition angle to ancilla qubits.
        'fixed_zero' fixes ancilla qubit angles at 0.0.

    Returns
    -------
    jnp.ndarray
        Full condition angles of shape (n_conditions, n_data + n_ancillas).
    """
    if n_ancillas < 0:
        raise ValueError("n_ancillas must be non-negative")
    angles = jnp.asarray(data_angles, dtype=jnp.float64)
    if angles.ndim == 1:
        angles = angles[None, :]
    n_conditions, _n_data = angles.shape
    if n_ancillas == 0:
        return angles

    if scheme == "broadcast_all":
        ancilla_part = jnp.tile(angles[:, :1], (1, n_ancillas))
    elif scheme == "fixed_zero":
        ancilla_part = jnp.zeros((n_conditions, n_ancillas), dtype=jnp.float64)
    else:
        raise ValueError(f"unknown ancilla conditioning scheme: {scheme!r}")

    return jnp.concatenate([angles, ancilla_part], axis=1)


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


def pad_observables_for_ancillas(
    data_ops: jnp.ndarray | np.ndarray,
    n_ancillas: int,
) -> jnp.ndarray:
    """Pad n-qubit Pauli-Z mask observables with zeros for ancilla qubits.

    Since zero mask entry corresponds to Identity on that qubit, the padded
    observable acts non-trivially only on data qubits and measures marginal
    expectations over data qubits.
    """
    if n_ancillas < 0:
        raise ValueError("n_ancillas must be non-negative")
    ops = jnp.asarray(data_ops, dtype=jnp.int8)
    if n_ancillas == 0:
        return ops
    zeros = jnp.zeros((ops.shape[0], n_ancillas), dtype=ops.dtype)
    return jnp.concatenate([ops, zeros], axis=1)


def ancilla_ansatz(n_data_qubits: int = 19, n_ancillas: int = 0) -> Ansatz:
    """Return complete two-local ansatz on N = n_data_qubits + n_ancillas qubits."""
    total_qubits = n_data_qubits + n_ancillas
    return complete_ansatz(total_qubits)


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


def exact_ancilla_marginal_probabilities(
    params: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    data_angles: np.ndarray | jnp.ndarray,
    n_data_qubits: int = 19,
    n_ancillas: int = 0,
    scheme: str = "broadcast_all",
    encoding: str = ANGLE_ENCODING,
) -> np.ndarray:
    """Compute exact marginal probabilities for an ancilla-assisted model."""
    full_angles = ancilla_condition_angles(data_angles, n_ancillas, scheme)
    joint_probs = exact_iqp_probabilities_many(
        params,
        generators,
        full_angles,
        encoding,
    )
    return marginalize_ancillas(joint_probs, n_data_qubits, n_ancillas)


def exact_ancilla_marginal_probabilities_from_full_angles(
    params: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    full_angles: np.ndarray | jnp.ndarray,
    n_data_qubits: int = 19,
    n_ancillas: int = 0,
    encoding: str = ANGLE_ENCODING,
) -> np.ndarray:
    """Evaluate a visible marginal from explicit visible and ancilla angles.

    Unlike :func:`exact_ancilla_marginal_probabilities`, this function does not
    construct or broadcast ancilla angles. It is therefore the exact-evaluation
    entry point for role-separated product-state encodings.
    """
    angles = jnp.asarray(full_angles, dtype=jnp.float64)
    if angles.ndim == 1:
        angles = angles[None, :]
    expected = n_data_qubits + n_ancillas
    if angles.ndim != 2 or angles.shape[1] != expected:
        raise ValueError(
            f"full_angles must have shape (n_conditions, {expected})",
        )
    joint_probs = exact_iqp_probabilities_many(
        params,
        generators,
        angles,
        encoding,
    )
    return marginalize_ancillas(joint_probs, n_data_qubits, n_ancillas)

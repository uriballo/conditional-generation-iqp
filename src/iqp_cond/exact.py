"""Exact small-system and 19-qubit evaluation utilities."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def enumerate_bitstrings(n_qubits: int) -> np.ndarray:
    """Enumerate bitstrings in little-endian integer order."""
    if n_qubits < 1 or n_qubits > 24:
        raise ValueError("n_qubits must be between 1 and 24")
    indices = np.arange(2**n_qubits, dtype=np.uint32)[:, None]
    shifts = np.arange(n_qubits, dtype=np.uint32)
    return ((indices >> shifts) & 1).astype(np.int8)


def _fwht(values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.complex128).copy()
    size = transformed.shape[0]
    width = 1
    while width < size:
        blocks = transformed.reshape(-1, 2 * width)
        left = blocks[:, :width].copy()
        right = blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2
    return transformed


def exact_iqp_probabilities(
    params: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    angles: np.ndarray | jnp.ndarray,
) -> np.ndarray:
    """Compute exact probabilities for ``RY -> H -> D_Z -> H``."""
    probabilities = exact_iqp_probabilities_many(
        params,
        generators,
        np.asarray(angles)[None, :],
    )
    return probabilities[0]


def exact_iqp_probabilities_many(
    params: np.ndarray | jnp.ndarray,
    generators: np.ndarray | jnp.ndarray,
    angles: np.ndarray | jnp.ndarray,
) -> np.ndarray:
    """Compute exact probabilities for several input-angle vectors."""
    params = np.asarray(params, dtype=np.float64)
    generators = np.asarray(generators, dtype=np.int8)
    angles = np.asarray(angles, dtype=np.float64)
    if generators.ndim != 2:
        raise ValueError("generators must have shape (n_generators, n_qubits)")
    n_qubits = generators.shape[1]
    if params.shape != (generators.shape[0],):
        raise ValueError("params must have shape (n_generators,)")
    if angles.ndim != 2 or angles.shape[1] != n_qubits:
        raise ValueError("angles must have shape (n_conditions, n_qubits)")

    bitstrings = enumerate_bitstrings(n_qubits)
    phase = np.zeros(bitstrings.shape[0], dtype=np.float64)
    for theta, generator in zip(params, generators, strict=True):
        wires = np.flatnonzero(generator)
        if wires.size == 1:
            character = 1.0 - 2.0 * bitstrings[:, wires[0]]
        elif wires.size == 2:
            character = 1.0 - 2.0 * np.bitwise_xor(
                bitstrings[:, wires[0]],
                bitstrings[:, wires[1]],
            )
        else:
            parity = np.mod(bitstrings[:, wires].sum(axis=1), 2)
            character = 1.0 - 2.0 * parity
        phase += theta * character

    phase_factors = np.exp(1j * phase)
    probabilities = []
    for condition_angles in angles:
        cosine = np.cos(condition_angles / 2.0)
        sine = np.sin(condition_angles / 2.0)
        amp_zero = (cosine + sine) / np.sqrt(2.0)
        amp_one = (cosine - sine) / np.sqrt(2.0)
        reference = np.prod(
            np.where(bitstrings == 0, amp_zero[None, :], amp_one[None, :]),
            axis=1,
        )
        diagonal_state = reference * phase_factors
        final_state = _fwht(diagonal_state) / np.sqrt(2**n_qubits)
        condition_probabilities = np.abs(final_state) ** 2
        probabilities.append(
            condition_probabilities / condition_probabilities.sum(),
        )
    return np.stack(probabilities)


def total_variation_distance(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    """Return half the L1 distance between probability vectors."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("probability vectors must be one-dimensional and aligned")
    return float(0.5 * np.abs(reference - candidate).sum())


def shell_probabilities(
    probabilities: np.ndarray,
    bitstrings: np.ndarray,
) -> np.ndarray:
    """Aggregate bitstring probabilities by Hamming weight."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    bitstrings = np.asarray(bitstrings)
    if probabilities.shape != (bitstrings.shape[0],):
        raise ValueError("probabilities and bitstrings must align")
    return np.bincount(
        bitstrings.sum(axis=1),
        weights=probabilities,
        minlength=bitstrings.shape[1] + 1,
    )

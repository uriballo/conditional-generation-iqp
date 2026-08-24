"""Structural conditional-response checks for affine ``RY`` IQP models.

The routines in this module enumerate finite-resource upper bounds.  They do
not assert that every allowed Fourier coefficient is nonzero: frequency
collisions, circuit parameters, and symmetries can all shrink the realized
support.
"""

from __future__ import annotations

import itertools

import numpy as np


def _as_weights(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("weights must have shape (n_qubits, condition_dimension)")
    if not np.all(np.isfinite(values)):
        raise ValueError("weights must be finite")
    return values


def _as_binary_matrix(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must have shape (n_rows, n_qubits)")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError(f"{name} must contain only zeros and ones")
    return matrix.astype(np.int8, copy=False)


def _as_binary_vector(name: str, values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 1 or vector.shape[0] < 1:
        raise ValueError(f"{name} must have shape (n_qubits,)")
    if not np.all((vector == 0) | (vector == 1)):
        raise ValueError(f"{name} must contain only zeros and ones")
    return vector.astype(np.int8, copy=False)


def _validate_generator_observable(
    generators: np.ndarray,
    observable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    generator_matrix = _as_binary_matrix("generators", generators)
    observable_mask = _as_binary_vector("observable", observable)
    if generator_matrix.shape[1] != observable_mask.shape[0]:
        raise ValueError("generators and observable must use the same qubits")
    return generator_matrix, observable_mask


def _signed_frequency_rows(weights: np.ndarray) -> np.ndarray:
    if weights.shape[0] == 0:
        return np.zeros((1, weights.shape[1]), dtype=np.float64)
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=weights.shape[0])),
        dtype=np.float64,
    )
    return signs @ weights


def full_frequency_support(weights: np.ndarray) -> np.ndarray:
    """Enumerate the affine-``RY`` full-probability support upper bound.

    If row ``i`` of ``weights`` is :math:`w_i`, this returns the distinct
    vectors in ``{sum_i s_i w_i : s_i in {-1, 0, 1}}``.  Biases alter Fourier
    coefficients but not this support.
    """
    weight_matrix = _as_weights(weights)
    coefficients = np.asarray(
        list(itertools.product((-1.0, 0.0, 1.0), repeat=weight_matrix.shape[0])),
        dtype=np.float64,
    )
    return np.unique(coefficients @ weight_matrix, axis=0)


def coordinate_bandwidth(weights: np.ndarray) -> np.ndarray:
    """Return ``sum_i |W[i, k]|`` for every condition coordinate ``k``."""
    return np.sum(np.abs(_as_weights(weights)), axis=0)


def directional_bandwidth(weights: np.ndarray, direction: np.ndarray) -> float:
    """Return the response bandwidth along a condition-space direction."""
    weight_matrix = _as_weights(weights)
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (weight_matrix.shape[1],):
        raise ValueError("direction must have shape (condition_dimension,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError("direction must be finite")
    return float(np.sum(np.abs(weight_matrix @ vector)))


def active_generator_indices(
    generators: np.ndarray,
    observable: np.ndarray,
) -> np.ndarray:
    """Return generators that anticommute with the middle-frame observable."""
    generator_matrix, observable_mask = _validate_generator_observable(
        generators,
        observable,
    )
    parity = (generator_matrix @ observable_mask) % 2
    return np.flatnonzero(parity).astype(np.int64)


def binary_span(masks: np.ndarray) -> np.ndarray:
    """Enumerate the distinct binary masks in an ``F_2`` row span."""
    mask_matrix = _as_binary_matrix("masks", masks)
    span = np.zeros((1, mask_matrix.shape[1]), dtype=np.int8)
    for mask in mask_matrix:
        span = np.unique(np.concatenate((span, span ^ mask)), axis=0)
    return span


def surviving_observable_masks(
    generators: np.ndarray,
    observable: np.ndarray,
) -> np.ndarray:
    """Return active-span masks that survive on real ``RY`` product inputs.

    Conjugating ``X_a`` by active diagonal generators produces Pauli masks
    ``X_a Z_h`` with ``h`` in their binary span.  A mask overlapping ``a``
    produces at least one local ``Y`` and has zero expectation on a real
    ``RY`` product state, so only disjoint ``h`` are returned.
    """
    generator_matrix, observable_mask = _validate_generator_observable(
        generators,
        observable,
    )
    active = active_generator_indices(generator_matrix, observable_mask)
    span = binary_span(generator_matrix[active])
    return span[np.all((span & observable_mask) == 0, axis=1)]


def observable_lightcone(
    generators: np.ndarray,
    observable: np.ndarray,
) -> np.ndarray:
    """Return the qubit lightcone from the observable and active generators."""
    generator_matrix, observable_mask = _validate_generator_observable(
        generators,
        observable,
    )
    active = active_generator_indices(generator_matrix, observable_mask)
    if active.size == 0:
        return observable_mask.copy()
    return np.bitwise_or.reduce(
        np.concatenate((observable_mask[None, :], generator_matrix[active])),
        axis=0,
    ).astype(np.int8)


def observable_frequency_support(
    weights: np.ndarray,
    generators: np.ndarray,
    observable: np.ndarray,
) -> np.ndarray:
    """Enumerate the IQP-specific observable-response support upper bound.

    For each surviving active-span mask ``h``, the corresponding product term
    contains one sine or cosine factor on ``support(observable) union
    support(h)``.  Its Fourier support therefore consists of all signed sums
    of the associated rows of ``weights``.
    """
    weight_matrix = _as_weights(weights)
    generator_matrix, observable_mask = _validate_generator_observable(
        generators,
        observable,
    )
    if generator_matrix.shape[1] != weight_matrix.shape[0]:
        raise ValueError("weights, generators, and observable must use the same qubits")

    frequencies = []
    for mask in surviving_observable_masks(generator_matrix, observable_mask):
        support = np.flatnonzero(observable_mask | mask)
        frequencies.append(_signed_frequency_rows(weight_matrix[support]))
    return np.unique(np.concatenate(frequencies, axis=0), axis=0)


def observable_coordinate_bandwidth(
    weights: np.ndarray,
    generators: np.ndarray,
    observable: np.ndarray,
) -> np.ndarray:
    """Return the coordinate bandwidth bound restricted to the lightcone."""
    weight_matrix = _as_weights(weights)
    lightcone = observable_lightcone(generators, observable)
    if lightcone.shape[0] != weight_matrix.shape[0]:
        raise ValueError("weights, generators, and observable must use the same qubits")
    return np.sum(np.abs(weight_matrix[lightcone.astype(bool)]), axis=0)

"""Encoder maps from conditions to the library's angle encoder.

Every encoder here is a product state, so it is built with a single all-``Y``
layer of :func:`iqp_cond.angle_encoder`.  The maps are frozen designs: the
scalar broadcast map, the anti-phase two-block map that is the current choice
for the one-parameter study, and the two-block and broadcast maps of the
two-parameter study.
"""

from __future__ import annotations

import iqp_cond as ciqp
import numpy as np

from .datasets import as_conditions, condition_angle_matrix

__all__ = [
    "ALPHA_MAX_OVER_PI",
    "BROADCAST_COEFFICIENTS",
    "ENCODER_MAPS",
    "LOCATION_COEFFICIENTS",
    "WIDTH_COEFFICIENTS",
    "anti_block_angles",
    "anti_block_encoder",
    "block_assignment",
    "broadcast_angles",
    "encoder_for",
    "scalar_broadcast_encoder",
    "two_block_angles",
    "two_block_encoder",
]

#: Offsets and slopes of the affine sum map ``Sigma / pi = p0 + p1 u``.
LOCATION_COEFFICIENTS = (0.762466, -1.524932)
#: Offsets and slopes of the affine difference map ``Delta / pi = q0 + q1 v``.
WIDTH_COEFFICIENTS = (-1.105356, 1.786576)
#: Frozen broadcast ablation map ``alpha / pi = c0 + c1 u + c2 v``.
BROADCAST_COEFFICIENTS = (0.411775, -0.823550, 0.0)
#: Box constraint on every block angle over the condition rectangle.
ALPHA_MAX_OVER_PI = 0.38


def _one_y_layer(angles: np.ndarray) -> ciqp.AngleEncoder:
    """Wrap a ``(conditions, qubits)`` angle matrix into the library encoder."""
    values = np.asarray(angles, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("angles must have shape (conditions, qubits)")
    return ciqp.angle_encoder(
        values[:, None, :],
        axes=("Y" * values.shape[1],),
    )


def scalar_broadcast_encoder(
    conditions: np.ndarray,
    n_qubits: int,
) -> ciqp.AngleEncoder:
    """One-parameter study: one broadcast angle per scalar condition."""
    return _one_y_layer(condition_angle_matrix(conditions, n_qubits))


def anti_block_angles(conditions: np.ndarray, total_qubits: int) -> np.ndarray:
    """Return the ``(conditions, total_qubits)`` anti-phase block angles.

    Even-indexed qubits get ``pi c`` and odd-indexed qubits ``pi (1 - c)``, so
    the condition sets the contrast between the two halves while the expected
    total excitation stays at ``total_qubits / 2`` for every condition.  At
    ``c = 0`` the state is the alternating half-filled pattern, at
    ``c = 1/2`` every qubit sits at ``p = 1/2``, and at ``c = 1`` the blocks
    are swapped.
    """
    values = np.asarray(conditions, dtype=np.float64)
    if values.ndim != 1 or not np.all((0.0 <= values) & (values <= 1.0)):
        raise ValueError("conditions must be a one-dimensional array in [0, 1]")
    if total_qubits < 2:
        raise ValueError("the anti-phase encoder needs at least two qubits")
    even = np.pi * values[:, None]
    odd = np.pi * (1.0 - values)[:, None]
    parity = (np.arange(total_qubits) % 2 == 0)[None, :]
    return np.where(parity, even, odd)


def anti_block_encoder(conditions: np.ndarray, n_qubits: int) -> ciqp.AngleEncoder:
    """One-parameter study: anti-phase two-block broadcast map.

    The current choice for the Gaussian studies: no target calibration, and
    the condition enters as a contrast between qubit halves.
    """
    return _one_y_layer(anti_block_angles(conditions, n_qubits))


def block_assignment(total_qubits: int) -> np.ndarray:
    """Return the block index (0 or 1) of every register position.

    Qubit-index parity splits visible and ancillary qubits evenly, so the width
    channel survives ancilla marginalisation.
    """
    if total_qubits < 2:
        raise ValueError("the two-block encoder needs at least two qubits")
    return np.arange(total_qubits) % 2


def _check_angles(angles: np.ndarray) -> None:
    if angles.ndim != 2 or angles.shape[0] < 1 or angles.shape[1] < 2:
        raise ValueError("angle matrix has the wrong shape")
    if not np.all(np.abs(angles) <= np.pi * ALPHA_MAX_OVER_PI + 1e-9):
        raise ValueError("encoder angle leaves the frozen monotone branch")


def two_block_angles(conditions: np.ndarray, total_qubits: int) -> np.ndarray:
    """Return the ``(conditions, total_qubits)`` angles of the two-block map."""
    values = as_conditions(conditions)
    p0, p1 = LOCATION_COEFFICIENTS
    q0, q1 = WIDTH_COEFFICIENTS
    sigma = np.pi * (p0 + p1 * values[:, 0])
    delta = np.pi * (q0 + q1 * values[:, 1])
    alpha_a = (sigma + delta) / 2.0
    alpha_b = (sigma - delta) / 2.0
    blocks = block_assignment(total_qubits)
    angles = np.where(blocks[None, :] == 0, alpha_a[:, None], alpha_b[:, None])
    _check_angles(angles)
    return angles


def broadcast_angles(conditions: np.ndarray, total_qubits: int) -> np.ndarray:
    """Return the single-channel ablation angles on the same grid."""
    values = as_conditions(conditions)
    c0, c1, c2 = BROADCAST_COEFFICIENTS
    angle = np.pi * (c0 + c1 * values[:, 0] + c2 * values[:, 1])
    angles = np.repeat(angle[:, None], total_qubits, axis=1)
    _check_angles(angles)
    return angles


def two_block_encoder(conditions: np.ndarray, n_qubits: int) -> ciqp.AngleEncoder:
    """Two-parameter study: location and width channels in two angle blocks."""
    return _one_y_layer(two_block_angles(conditions, n_qubits))


ENCODER_MAPS = {
    "two_block": two_block_encoder,
    "broadcast": lambda conditions, n_qubits: _one_y_layer(
        broadcast_angles(conditions, n_qubits),
    ),
}


def encoder_for(
    name: str,
    conditions: np.ndarray,
    n_qubits: int,
) -> ciqp.AngleEncoder:
    """Build a named two-parameter encoder over the given conditions."""
    if name not in ENCODER_MAPS:
        raise ValueError(
            f"unknown encoder {name!r}; expected {', '.join(sorted(ENCODER_MAPS))}",
        )
    return ENCODER_MAPS[name](conditions, n_qubits)

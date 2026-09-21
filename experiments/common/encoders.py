"""Encoder maps from conditions to the library's angle encoder.

The maps are frozen designs. Most use a single all-``Y`` product-state layer;
the scalar appendix also contains layered controls that separate the effect of
the paired angle map from depth and noncommuting re-uploading.
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
    "anti_fixed_phase_encoder",
    "anti_perqubit_sine_angles",
    "anti_perqubit_sine_encoder",
    "anti_reupload_encoder",
    "balanced_angles",
    "balanced_encoder",
    "block_assignment",
    "broadcast_angles",
    "broadcast_ramp_angles",
    "broadcast_ramp_encoder",
    "encoder_for",
    "scalar_broadcast_encoder",
    "skew_angles",
    "skew_encoder",
    "target_mean_angles",
    "target_mean_encoder",
    "two_block_angles",
    "two_block_encoder",
    "zero_angles",
    "zero_encoder",
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
    return ciqp.AngleEncoder(
        values[:, None, :],
        axes=("Y" * values.shape[1],),
    )


def _stacked_layers(
    layers: list[np.ndarray],
    axes: tuple[str, ...],
) -> ciqp.AngleEncoder:
    """Wrap equally shaped angle matrices into a multilayer encoder."""
    values = [np.asarray(layer, dtype=np.float64) for layer in layers]
    if not values or any(value.shape != values[0].shape for value in values):
        raise ValueError("encoder layers must have the same non-empty shape")
    if values[0].ndim != 2 or values[0].shape[1] < 1:
        raise ValueError("angles must have shape (conditions, qubits)")
    if len(axes) != len(values):
        raise ValueError("one axis specification is required per layer")
    return ciqp.AngleEncoder(np.stack(values, axis=1), axes=axes)


def scalar_broadcast_encoder(
    conditions: np.ndarray,
    n_qubits: int,
) -> ciqp.AngleEncoder:
    """One-parameter study: one broadcast angle per scalar condition."""
    return _one_y_layer(condition_angle_matrix(conditions, n_qubits))


def anti_block_angles(conditions: np.ndarray, total_qubits: int) -> np.ndarray:
    """Return the ``(conditions, total_qubits)`` anti-phase block angles.

    Consecutive pairs get ``pi c`` and ``pi (1 - c)``. An unpaired final
    qubit, when present, gets ``pi / 2``. The expected total excitation is
    therefore exactly ``total_qubits / 2`` for every condition and the input
    state transforms by global bit complementation under ``c -> 1-c``.
    """
    values = np.asarray(conditions, dtype=np.float64)
    if values.ndim != 1 or not np.all((0.0 <= values) & (values <= 1.0)):
        raise ValueError("conditions must be a one-dimensional array in [0, 1]")
    if total_qubits < 1:
        raise ValueError("total_qubits must be positive")
    angles = np.full((values.size, total_qubits), np.pi / 2, dtype=np.float64)
    paired = total_qubits // 2
    if paired:
        angles[:, : 2 * paired : 2] = np.pi * values[:, None]
        angles[:, 1 : 2 * paired : 2] = np.pi * (1.0 - values[:, None])
    return angles


def anti_block_encoder(conditions: np.ndarray, n_qubits: int) -> ciqp.AngleEncoder:
    """One-parameter study: anti-phase two-block broadcast map.

    The current choice for the Gaussian studies: no target calibration, and
    the condition enters as a contrast between qubit halves.
    """
    return _one_y_layer(anti_block_angles(conditions, n_qubits))


def anti_fixed_phase_encoder(
    conditions: np.ndarray,
    n_qubits: int,
) -> ciqp.AngleEncoder:
    """Depth-matched paired control with a condition-independent Z layer."""
    base = anti_block_angles(conditions, n_qubits)
    phase = np.full_like(base, np.pi / 4.0)
    axes = ("Y" * n_qubits, "Z" * n_qubits, "Y" * n_qubits)
    return _stacked_layers([base / 2.0, phase, base / 2.0], axes)


def anti_reupload_encoder(
    conditions: np.ndarray,
    n_qubits: int,
) -> ciqp.AngleEncoder:
    """Paired map with a noncommuting, condition-dependent Z re-upload."""
    values = _scalar_conditions(conditions, n_qubits)
    base = anti_block_angles(values, n_qubits)
    phase = np.repeat((np.pi * values)[:, None], n_qubits, axis=1)
    axes = ("Y" * n_qubits, "Z" * n_qubits, "Y" * n_qubits)
    return _stacked_layers([base / 2.0, phase, base / 2.0], axes)


def anti_perqubit_sine_angles(
    conditions: np.ndarray,
    n_qubits: int,
    delta: float = 0.1,
) -> np.ndarray:
    """Add a fixed sinusoidal per-wire offset to the paired angle map."""
    values = _scalar_conditions(conditions, n_qubits)
    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    index = np.arange(n_qubits, dtype=np.float64)
    profile = np.sin(2.0 * np.pi * index / n_qubits)
    return anti_block_angles(values, n_qubits) + delta * profile[None, :]


def anti_perqubit_sine_encoder(
    conditions: np.ndarray,
    n_qubits: int,
) -> ciqp.AngleEncoder:
    """Paired map with a fixed 0.1-radian sinusoidal per-wire offset."""
    return _one_y_layer(anti_perqubit_sine_angles(conditions, n_qubits))


def _scalar_conditions(conditions: np.ndarray, n_qubits: int) -> np.ndarray:
    values = np.asarray(conditions, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("conditions must be a non-empty one-dimensional array")
    if not np.all((0.0 <= values) & (values <= 1.0)):
        raise ValueError("conditions must be in [0, 1]")
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    return values


def balanced_angles(conditions: np.ndarray, n_qubits: int) -> np.ndarray:
    """Return condition-free balanced angles for a scalar condition grid."""
    values = _scalar_conditions(conditions, n_qubits)
    return np.full((values.size, n_qubits), np.pi / 2, dtype=np.float64)


def balanced_encoder(conditions: np.ndarray, n_qubits: int) -> ciqp.AngleEncoder:
    """Build the condition-free balanced product-state encoder."""
    return _one_y_layer(balanced_angles(conditions, n_qubits))


def zero_angles(conditions: np.ndarray, n_qubits: int) -> np.ndarray:
    """Return the fixed zero angles of an unencoded IQP input."""
    values = _scalar_conditions(conditions, n_qubits)
    return np.zeros((values.size, n_qubits), dtype=np.float64)


def zero_encoder(conditions: np.ndarray, n_qubits: int) -> ciqp.AngleEncoder:
    """Build the fixed computational-basis input used by IQPopt-style fits."""
    return _one_y_layer(zero_angles(conditions, n_qubits))


def broadcast_ramp_angles(conditions: np.ndarray, n_qubits: int) -> np.ndarray:
    """Return the broadcast ``pi*c`` angles used as an encoder ablation."""
    values = _scalar_conditions(conditions, n_qubits)
    return np.repeat((np.pi * values)[:, None], n_qubits, axis=1)


def broadcast_ramp_encoder(
    conditions: np.ndarray,
    n_qubits: int,
) -> ciqp.AngleEncoder:
    """Build the broadcast ``pi*c`` encoder."""
    return _one_y_layer(broadcast_ramp_angles(conditions, n_qubits))


def target_mean_angles(conditions: np.ndarray, n_qubits: int) -> np.ndarray:
    """Return broadcast angles calibrated to the scalar target mean fraction."""
    values = _scalar_conditions(conditions, n_qubits)
    mean_fraction = 0.15 + 0.70 * values
    angle = 2.0 * np.arcsin(np.sqrt(mean_fraction))
    return np.repeat(angle[:, None], n_qubits, axis=1)


def target_mean_encoder(
    conditions: np.ndarray,
    n_qubits: int,
) -> ciqp.AngleEncoder:
    """Build the target-mean broadcast encoder control."""
    return _one_y_layer(target_mean_angles(conditions, n_qubits))


def skew_angles(
    conditions: np.ndarray,
    total_qubits: int,
    delta: float,
) -> np.ndarray:
    """Return paired constant-excitation angles with a symmetry-breaking skew."""
    values = _scalar_conditions(conditions, total_qubits)
    if not np.isfinite(delta) or abs(delta) >= 1.0 / np.pi:
        raise ValueError("delta must be finite with abs(delta) < 1/pi")
    transformed = values + delta * np.sin(np.pi * values) ** 2
    if np.any((transformed < 0.0) | (transformed > 1.0)):
        raise ValueError("skewed conditions must remain in [0, 1]")
    angles = np.full((values.size, total_qubits), np.pi / 2, dtype=np.float64)
    paired = total_qubits // 2
    if paired:
        angles[:, : 2 * paired : 2] = np.pi * transformed[:, None]
        angles[:, 1 : 2 * paired : 2] = np.pi * (1.0 - transformed[:, None])
    return angles


def skew_encoder(
    conditions: np.ndarray,
    n_qubits: int,
    delta: float,
) -> ciqp.AngleEncoder:
    """Build a skewed paired encoder for a specified symmetry-breaking delta."""
    return _one_y_layer(skew_angles(conditions, n_qubits, delta))


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
    "anti_block": anti_block_encoder,
    "anti_fixed_phase": anti_fixed_phase_encoder,
    "anti_reupload": anti_reupload_encoder,
    "anti_perqubit_sine": anti_perqubit_sine_encoder,
    "balanced": balanced_encoder,
    "broadcast_ramp": broadcast_ramp_encoder,
    "target_mean": target_mean_encoder,
    "skew_minus": lambda conditions, n_qubits: skew_encoder(
        conditions, n_qubits, -0.1,
    ),
    "skew_plus": lambda conditions, n_qubits: skew_encoder(
        conditions, n_qubits, 0.1,
    ),
    "zero": zero_encoder,
}


def encoder_for(
    name: str,
    conditions: np.ndarray,
    n_qubits: int,
) -> ciqp.AngleEncoder:
    """Build a named encoder over the given conditions."""
    if name not in ENCODER_MAPS:
        raise ValueError(
            f"unknown encoder {name!r}; expected {', '.join(sorted(ENCODER_MAPS))}",
        )
    return ENCODER_MAPS[name](conditions, n_qubits)

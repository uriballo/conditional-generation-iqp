"""Frozen scalar Gaussian translation benchmark definitions."""

from __future__ import annotations

from math import pi

import iqp_cond as ciqp
import numpy as np

from experiments.common.datasets import gaussian_target, target_mean
from experiments.common.encoders import anti_block_angles

N_VISIBLE = 8
N_ANCILLAS = 0
WIDTH_FACTOR = 0.5
ANCHOR_COUNTS = (2, 3, 4, 5)
ENCODERS = (
    "broadcast_affine",
    "qubit_affine",
    "anti_block",
    "condition_blind",
    "exponential",
)
INTERVALS = {
    "interpolation": (0.0, 1.0),
    "near": (0.0, 0.8),
    "far": (0.0, 0.5),
    "mirror_near": (0.2, 1.0),
    "mirror_far": (0.5, 1.0),
}
LAYOUT_SEEDS = (1101, 1102, 1103)
EVAL_SEEDS = {
    "dev": (201, 202, 203),
    "validation": (301, 302, 303),
    "audit": (401, 402, 403),
}


def _conditions(conditions: np.ndarray) -> np.ndarray:
    values = np.asarray(conditions, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("conditions must be a non-empty finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("conditions must lie in [0, 1]")
    return values


def protocol_interval(name: str) -> tuple[float, float]:
    """Return a named closed training interval."""
    try:
        return INTERVALS[name]
    except KeyError as error:
        raise ValueError(f"unknown interval {name!r}") from error


def training_set_label(
    name: str,
    count: int,
    placement: str,
    layout_seed: int | None,
) -> str:
    """Return the stable run-identity label for one anchor configuration."""
    protocol_interval(name)
    if count not in ANCHOR_COUNTS:
        raise ValueError(f"count must be one of {ANCHOR_COUNTS}")
    if placement == "even":
        if layout_seed is not None:
            raise ValueError("even placement must not carry a layout seed")
        suffix = "even"
    elif placement == "random":
        if layout_seed not in LAYOUT_SEEDS:
            raise ValueError(f"random layout_seed must be one of {LAYOUT_SEEDS}")
        suffix = f"random-{layout_seed}"
    else:
        raise ValueError("placement must be 'even' or 'random'")
    return f"{name}-k{count}-{suffix}"


def anchor_conditions(
    protocol: str,
    count: int,
    placement: str = "even",
    layout_seed: int | None = None,
) -> np.ndarray:
    """Generate endpoint-inclusive evenly spaced or reproducible random anchors."""
    low, high = protocol_interval(protocol)
    if count not in ANCHOR_COUNTS:
        raise ValueError(f"count must be one of {ANCHOR_COUNTS}")
    if placement == "even":
        result = np.linspace(low, high, count, dtype=np.float64)
    elif placement == "random":
        if layout_seed is None:
            raise ValueError("random placement requires a separate layout_seed")
        interior = np.random.default_rng(layout_seed).uniform(low, high, count - 2)
        result = np.sort(np.concatenate(([low], interior, [high])))
    else:
        raise ValueError("placement must be 'even' or 'random'")
    result.setflags(write=False)
    return result


def evaluation_grid(split: str) -> np.ndarray:
    """Return the immutable, disjoint 20-point split grid."""
    offsets = {"dev": 1, "validation": 2, "audit": 3}
    try:
        offset = offsets[split]
    except KeyError as error:
        raise ValueError(f"unknown evaluation split {split!r}") from error
    base = np.asarray([(6 * k + offset) / 128 for k in range(10)], dtype=np.float64)
    values = np.sort(np.concatenate((base, 1.0 - base)))
    values.setflags(write=False)
    return values


def classify_condition(
    condition: float,
    protocol: str,
    anchors: np.ndarray,
) -> str:
    """Classify a point against exact anchors and its protocol interval."""
    value = float(_conditions(np.asarray([condition]))[0])
    low, high = protocol_interval(protocol)
    anchor_values = _conditions(anchors)
    if np.any(np.isclose(value, anchor_values, atol=1e-12, rtol=0.0)):
        return "training"
    if low <= value <= high:
        return "held_out_interpolation"
    if protocol.startswith("mirror_"):
        return "mirrored_diagnostic"
    return "near_extrapolation" if protocol == "near" else "far_extrapolation"


def encoder_angles(
    conditions: np.ndarray,
    name: str,
    *,
    n_qubits: int = N_VISIBLE,
    fallback_k: int = 2,
    fallback_s: float = pi / 54,
) -> np.ndarray:
    """Return angle matrices for the frozen encoder families."""
    values = _conditions(conditions)
    if n_qubits != N_VISIBLE:
        raise ValueError("the scalar benchmark fixes n_visible=8 and n_ancillas=0")
    if name == "broadcast_affine":
        theta = pi * (0.32 - 0.68 * values)
        return np.repeat(theta[:, None], n_qubits, axis=1)
    if name == "qubit_affine":
        t = np.arange(n_qubits, dtype=np.float64) / (n_qubits - 1)
        theta = pi * (
            (0.28 + 0.06 * t[None, :]) - (0.62 + 0.08 * t[None, :]) * values[:, None]
        )
        return theta
    if name == "anti_block":
        return anti_block_angles(values, n_qubits)
    if name == "condition_blind":
        return np.zeros((values.size, n_qubits), dtype=np.float64)
    if name == "exponential":
        if fallback_k not in (2, 3, 4) or fallback_s not in (pi / 54, pi / 27):
            raise ValueError("fallback choices are k in {2,3,4} and s in {pi/54,pi/27}")
        phase = np.zeros((values.size, n_qubits), dtype=np.float64)
        phase[:, :fallback_k] = (
            fallback_s * (3.0 ** np.arange(fallback_k))[None, :] * values[:, None]
        )
        return phase
    raise ValueError(f"unknown encoder {name!r}")


def make_encoder(
    conditions: np.ndarray,
    name: str,
    *,
    fallback_k: int = 2,
    fallback_s: float = pi / 54,
) -> ciqp.AngleEncoder:
    """Build a public-library product encoder; exponential starts in |+>^8."""
    values = _conditions(conditions)
    if name == "exponential":
        z_angles = encoder_angles(
            values, name, fallback_k=fallback_k, fallback_s=fallback_s
        )
        y_angles = np.full_like(z_angles, pi / 2)
        return ciqp.AngleEncoder(
            np.stack((y_angles, z_angles), axis=1),
            axes=("Y" * N_VISIBLE, "Z" * N_VISIBLE),
        )
    angles = encoder_angles(values, name)
    return ciqp.AngleEncoder(angles[:, None, :], axes=("Y" * N_VISIBLE,))


def target_probabilities(
    conditions: np.ndarray, n_visible: int = N_VISIBLE
) -> np.ndarray:
    """Return exact bitstring probabilities from the shared truncated target."""
    if n_visible != N_VISIBLE:
        raise ValueError("the scalar benchmark fixes n_visible=8")
    values = _conditions(conditions)
    indices = np.arange(2**n_visible, dtype=np.int64)
    states = ((indices[:, None] >> np.arange(n_visible)) & 1).astype(np.float64)
    return np.stack(
        [
            gaussian_target(n_visible, WIDTH_FACTOR, float(c)).bitstring_probabilities(
                states
            )
            for c in values
        ]
    )


def target_moments(condition: float, n_visible: int = N_VISIBLE) -> tuple[float, float]:
    """Return exact truncated target mean and standard deviation."""
    if n_visible != N_VISIBLE:
        raise ValueError("the scalar benchmark fixes n_visible=8")
    target = gaussian_target(
        n_visible, WIDTH_FACTOR, float(_conditions(np.asarray([condition]))[0])
    )
    return target.actual_mean, target.actual_std


def target_diagnostics(
    condition: float, n_visible: int = N_VISIBLE
) -> dict[str, float]:
    """Return nominal and calibrated moments, width, and boundary shell mass."""
    if n_visible != N_VISIBLE:
        raise ValueError("the scalar benchmark fixes n_visible=8")
    value = float(_conditions(np.asarray([condition]))[0])
    target = gaussian_target(n_visible, WIDTH_FACTOR, value)
    return {
        "nominal_mean": target_mean(n_visible, value),
        "exponent_width": target.width,
        "actual_mean": target.actual_mean,
        "actual_std": target.actual_std,
        "boundary_shell_mass": float(
            target.shell_probabilities[0] + target.shell_probabilities[-1]
        ),
    }


def matched_exposure_steps(fixed_updates: int, anchor_count: int) -> int:
    """Scale updates by anchor count; the quotient must be integral."""
    if anchor_count not in ANCHOR_COUNTS or fixed_updates < 1:
        raise ValueError("invalid matched-exposure budget")
    numerator = fixed_updates * anchor_count
    if numerator % 3:
        raise ValueError("fixed_updates * anchor_count must be divisible by three")
    return numerator // 3

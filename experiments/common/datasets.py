"""Discrete-Gaussian target families over Hamming weight.

The one-parameter family varies the prescribed mean weight with a scalar
condition; the two-parameter family varies the prescribed mean and standard
deviation with a pair ``(u, v) = (mean / n, sd / sqrt(n))``.  Both are exact
truncated discrete Gaussians on the weight shells, uniform within a shell.

The shell location and width are calibrated to the prescribed moments with a
bisection root finder, so the module needs only NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, pi, sqrt

import numpy as np

__all__ = [
    "GRID_MEAN_FRACTIONS",
    "GRID_WIDTH_FRACTIONS",
    "GaussianHammingTarget",
    "as_conditions",
    "calibrate_location",
    "condition_angle",
    "condition_angle_matrix",
    "gaussian_target",
    "grid_conditions",
    "held_out_conditions",
    "is_perimeter",
    "prescribed_target",
    "target_location_scale",
    "target_mean",
    "target_width",
    "training_conditions",
]

#: Frozen mean-coordinate grid of the two-parameter study.
GRID_MEAN_FRACTIONS = (0.375, 0.4375, 0.500, 0.5625, 0.625)
#: Frozen width-coordinate grid of the two-parameter study.
GRID_WIDTH_FRACTIONS = (0.300, 0.3375, 0.375, 0.4125, 0.450)


def _bisect(
    objective,
    low: float,
    high: float,
    *,
    iterations: int = 200,
    xtol: float = 1e-14,
) -> float:
    """Return a root of a sign-changing scalar function by bisection."""
    f_low = objective(low)
    f_high = objective(high)
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if f_low * f_high > 0.0:
        raise ValueError("objective does not change sign on the bracket")
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        f_middle = objective(middle)
        if f_middle == 0.0 or (high - low) < xtol * max(1.0, abs(middle)):
            return middle
        if f_low * f_middle < 0.0:
            high = middle
        else:
            low, f_low = middle, f_middle
    return 0.5 * (low + high)


def _shell_probabilities(
    n_qubits: int,
    location: float,
    width: float,
) -> np.ndarray:
    """Return the truncated discrete-Gaussian law over weights ``0..n``."""
    weights = np.arange(n_qubits + 1, dtype=np.float64)
    log_mass = -0.5 * ((weights - location) / width) ** 2
    log_mass -= np.max(log_mass)
    mass = np.exp(log_mass)
    return mass / mass.sum()


def _expected_weight(n_qubits: int, location: float, width: float) -> float:
    weights = np.arange(n_qubits + 1, dtype=np.float64)
    return float(weights @ _shell_probabilities(n_qubits, location, width))


def calibrate_location(n_qubits: int, mean: float, width: float) -> float:
    """Find the shell-law location with the requested truncated mean.

    The bracket expands until the monotone residual changes sign, so extreme
    means at large widths are supported.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    if not 0.0 < mean < n_qubits:
        raise ValueError("mean must lie strictly between zero and n_qubits")
    if width <= 0.0:
        raise ValueError("width must be positive")

    def objective(location: float) -> float:
        return _expected_weight(n_qubits, location, width) - mean

    low, high = -1.0, 1.0
    for _ in range(60):
        if objective(low) * objective(high) < 0.0:
            return _bisect(objective, low, high)
        low *= 2.0
        high *= 2.0
    raise RuntimeError("no location bracket reproduces the requested mean")


@dataclass(frozen=True)
class GaussianHammingTarget:
    """A discrete Gaussian over weight shells, uniform within each shell."""

    n_qubits: int
    mean: float
    width: float
    location: float
    shell_probabilities: np.ndarray

    @classmethod
    def create(
        cls,
        n_qubits: int,
        mean: float,
        width: float,
    ) -> GaussianHammingTarget:
        """Construct a target and calibrate its location to the exact mean."""
        location = calibrate_location(n_qubits, mean, width)
        return cls(
            n_qubits,
            mean,
            width,
            location,
            _shell_probabilities(n_qubits, location, width),
        )

    @property
    def actual_mean(self) -> float:
        """Expected Hamming weight."""
        weights = np.arange(self.n_qubits + 1, dtype=np.float64)
        return float(weights @ self.shell_probabilities)

    @property
    def actual_std(self) -> float:
        """Hamming-weight standard deviation."""
        weights = np.arange(self.n_qubits + 1, dtype=np.float64)
        variance = ((weights - self.actual_mean) ** 2) @ self.shell_probabilities
        return float(np.sqrt(variance))

    def bitstring_probabilities(self, bitstrings: np.ndarray) -> np.ndarray:
        """Return the exact probability of each supplied binary string."""
        bits = np.asarray(bitstrings)
        if bits.ndim != 2 or bits.shape[1] != self.n_qubits:
            raise ValueError("bitstrings must have shape (n_strings, n_qubits)")
        if not np.all((bits == 0) | (bits == 1)):
            raise ValueError("bitstrings must be binary")
        weights = bits.sum(axis=1).astype(int)
        shell_sizes = np.asarray(
            [comb(self.n_qubits, int(weight)) for weight in weights],
            dtype=np.float64,
        )
        return self.shell_probabilities[weights] / shell_sizes

    def sample(self, rng: np.random.Generator, n_samples: int) -> np.ndarray:
        """Draw bitstrings from the target."""
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        weights = rng.choice(
            self.n_qubits + 1,
            size=n_samples,
            p=self.shell_probabilities,
        )
        samples = np.zeros((n_samples, self.n_qubits), dtype=np.int8)
        for row, weight in enumerate(weights):
            if weight:
                ones = rng.choice(self.n_qubits, size=int(weight), replace=False)
                samples[row, ones] = 1
        return samples


def target_mean(n_qubits: int, condition: float) -> float:
    """Prescribed mean weight of the one-parameter family."""
    if n_qubits < 2 or not 0.0 <= condition <= 1.0:
        raise ValueError("target coordinates are out of range")
    return n_qubits * (0.15 + 0.70 * condition)


def target_width(n_qubits: int, width_factor: float) -> float:
    """Prescribed shell width of the one-parameter family."""
    if n_qubits < 2 or width_factor <= 0.0:
        raise ValueError("target width coordinates are out of range")
    return width_factor * sqrt(n_qubits)


def gaussian_target(
    n_qubits: int,
    width_factor: float,
    condition: float,
) -> GaussianHammingTarget:
    """Return the one-parameter target at a scalar condition."""
    return GaussianHammingTarget.create(
        n_qubits,
        target_mean(n_qubits, condition),
        target_width(n_qubits, width_factor),
    )


def condition_angle(condition: float) -> float:
    """Broadcast encoder angle ``pi (0.35 - 0.70 c)`` of the scalar family."""
    if not 0.0 <= condition <= 1.0:
        raise ValueError("condition must lie in [0, 1]")
    return pi * (0.35 - 0.70 * condition)


def condition_angle_matrix(conditions: np.ndarray, n_qubits: int) -> np.ndarray:
    """Return the ``(conditions, n_qubits)`` broadcast angle matrix."""
    values = np.asarray(conditions, dtype=np.float64)
    if values.ndim != 1 or not np.all((0.0 <= values) & (values <= 1.0)):
        raise ValueError("conditions must be a one-dimensional array in [0, 1]")
    return np.repeat(
        np.asarray([condition_angle(float(value)) for value in values])[:, None],
        n_qubits,
        axis=1,
    )


def as_conditions(conditions: np.ndarray) -> np.ndarray:
    """Return conditions as a ``(n_conditions, 2)`` float array."""
    values = np.asarray(conditions, dtype=np.float64)
    if values.ndim == 1 and values.shape == (2,):
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("conditions must have shape (n_conditions, 2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("conditions must be finite")
    return values


def grid_conditions() -> np.ndarray:
    """Return the frozen 25-point condition grid in row-major order."""
    return np.asarray(
        [(u, v) for u in GRID_MEAN_FRACTIONS for v in GRID_WIDTH_FRACTIONS],
        dtype=np.float64,
    )


def is_perimeter(condition: np.ndarray) -> bool:
    """Return True if a condition lies on the boundary of the frozen grid."""
    u, v = (float(value) for value in np.asarray(condition).reshape(2))
    return (
        np.isclose(u, GRID_MEAN_FRACTIONS[0])
        or np.isclose(u, GRID_MEAN_FRACTIONS[-1])
        or np.isclose(v, GRID_WIDTH_FRACTIONS[0])
        or np.isclose(v, GRID_WIDTH_FRACTIONS[-1])
    )


def training_conditions() -> np.ndarray:
    """Return the 16 perimeter conditions of the two-parameter study."""
    return np.asarray(
        [condition for condition in grid_conditions() if is_perimeter(condition)],
        dtype=np.float64,
    )


def held_out_conditions() -> np.ndarray:
    """Return the nine interior conditions of the two-parameter study."""
    return np.asarray(
        [condition for condition in grid_conditions() if not is_perimeter(condition)],
        dtype=np.float64,
    )


def target_location_scale(
    n_qubits: int,
    mean: float,
    std: float,
) -> tuple[float, float]:
    """Return the shell ``(location, width)`` reproducing mean and deviation."""
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least two")
    if not 0.0 < mean < n_qubits:
        raise ValueError("mean must lie strictly inside the support")
    if std <= 0.0:
        raise ValueError("std must be positive")

    def residual(width: float) -> float:
        return GaussianHammingTarget.create(n_qubits, mean, width).actual_std - std

    low, high = 1e-9, 1.0
    for _ in range(80):
        if residual(high) >= 0.0:
            break
        high *= 2.0
    else:  # pragma: no cover - the family is unbounded in width
        raise RuntimeError("no width bracket covers the requested deviation")
    if residual(low) > 0.0:
        raise ValueError("requested standard deviation is unreachable")
    width = _bisect(residual, low, high)
    return calibrate_location(n_qubits, mean, width), width


def prescribed_target(
    n_qubits: int,
    condition: np.ndarray,
) -> GaussianHammingTarget:
    """Return the exact two-parameter target at one ``(u, v)`` condition."""
    u, v = (float(value) for value in as_conditions(condition)[0])
    mean = u * n_qubits
    std = v * sqrt(n_qubits)
    _, width = target_location_scale(n_qubits, mean, std)
    return GaussianHammingTarget.create(n_qubits, mean, width)

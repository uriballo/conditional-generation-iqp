"""Discrete Gaussian target distributions over Hamming-weight shells."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from scipy.optimize import brentq


def _shell_probabilities(
    n_qubits: int,
    location: float,
    width: float,
) -> np.ndarray:
    weights = np.arange(n_qubits + 1, dtype=np.float64)
    log_mass = -0.5 * ((weights - location) / width) ** 2
    log_mass -= np.max(log_mass)
    mass = np.exp(log_mass)
    return mass / mass.sum()


def _expected_weight(n_qubits: int, location: float, width: float) -> float:
    weights = np.arange(n_qubits + 1, dtype=np.float64)
    return float(weights @ _shell_probabilities(n_qubits, location, width))


def calibrate_location(n_qubits: int, mean: float, width: float) -> float:
    """Find the discrete-Gaussian location with the requested truncated mean."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    if not 0.0 < mean < n_qubits:
        raise ValueError("mean must lie strictly between zero and n_qubits")
    if width <= 0.0:
        raise ValueError("width must be positive")

    objective = lambda location: _expected_weight(  # noqa: E731
        n_qubits,
        location,
        width,
    ) - mean
    return float(brentq(objective, -20.0 * n_qubits, 20.0 * n_qubits))


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
        shell_probs = _shell_probabilities(n_qubits, location, width)
        return cls(n_qubits, mean, width, location, shell_probs)

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

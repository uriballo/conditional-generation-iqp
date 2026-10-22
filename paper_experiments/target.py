"""Zero-field Curie--Weiss laws represented by Hamming-weight shells.

All calculations use the ``n + 1`` shell weights, so exact target observables
remain inexpensive even when the underlying bitstring space is enormous.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

import numpy as np

__all__ = ["CurieWeissTarget", "curie_weiss_target"]


def _validate(n_qubits: int, beta: float) -> None:
    if isinstance(n_qubits, bool) or n_qubits < 1 or int(n_qubits) != n_qubits:
        raise ValueError("n_qubits must be a positive integer")
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")


@dataclass(frozen=True)
class CurieWeissTarget:
    """Exact zero-field Curie--Weiss distribution for ``n_qubits`` spins."""

    n_qubits: int
    beta: float

    def __post_init__(self) -> None:
        _validate(self.n_qubits, self.beta)
        object.__setattr__(self, "n_qubits", int(self.n_qubits))
        weights = np.arange(self.n_qubits + 1, dtype=np.float64)
        magnetization = (2.0 * weights - self.n_qubits) / self.n_qubits
        log_mass = np.array(
            [
                lgamma(self.n_qubits + 1)
                - lgamma(k + 1)
                - lgamma(self.n_qubits - k + 1)
                + self.beta * (2 * k - self.n_qubits) ** 2 / (2.0 * self.n_qubits)
                for k in range(self.n_qubits + 1)
            ],
            dtype=np.float64,
        )
        log_mass -= log_mass.max()
        shell = np.exp(log_mass)
        shell /= shell.sum()
        object.__setattr__(self, "shell_probabilities", shell)
        object.__setattr__(self, "magnetizations", magnetization)

    shell_probabilities: np.ndarray = None  # type: ignore[assignment]
    magnetizations: np.ndarray = None  # type: ignore[assignment]

    @property
    def probabilities(self) -> np.ndarray:
        """Probability of each shell, indexed by Hamming weight."""
        return self.shell_probabilities.copy()

    def bitstring_probabilities(self, states: np.ndarray) -> np.ndarray:
        """Return exact probabilities for rows of zero/one spin states."""
        values = np.asarray(states)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.n_qubits:
            raise ValueError("states must have shape (samples, n_qubits)")
        if np.any((values != 0) & (values != 1)):
            raise ValueError("states must contain only zeroes and ones")
        weights = values.sum(axis=1).astype(int)
        counts = np.array(
            [
                np.exp(
                    lgamma(self.n_qubits + 1)
                    - lgamma(k + 1)
                    - lgamma(self.n_qubits - k + 1)
                )
                for k in weights
            ]
        )
        return self.shell_probabilities[weights] / counts

    def sample(self, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample bitstrings uniformly within the selected shell."""
        if isinstance(size, bool) or size < 0 or int(size) != size:
            raise ValueError("size must be a non-negative integer")
        generator = np.random.default_rng() if rng is None else rng
        weights = generator.choice(
            self.n_qubits + 1, size=int(size), p=self.shell_probabilities
        )
        samples = np.zeros((int(size), self.n_qubits), dtype=np.int8)
        for row, weight in enumerate(weights):
            if weight:
                samples[
                    row,
                    generator.choice(self.n_qubits, size=int(weight), replace=False),
                ] = 1
        return samples

    @property
    def m2(self) -> float:
        return float(self.shell_probabilities @ (self.magnetizations**2))

    @property
    def m4(self) -> float:
        return float(self.shell_probabilities @ (self.magnetizations**4))

    @property
    def susceptibility(self) -> float:
        return self.n_qubits * self.m2

    @property
    def binder_cumulant(self) -> float:
        return 1.0 - self.m4 / (3.0 * self.m2**2)


def curie_weiss_target(n_qubits: int, beta: float) -> CurieWeissTarget:
    """Construct an exact zero-field Curie--Weiss target."""
    return CurieWeissTarget(n_qubits, beta)

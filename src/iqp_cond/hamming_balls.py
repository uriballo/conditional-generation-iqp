"""Nested fixed-weight Hamming-ball targets for conditional IQP generation."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

DEFAULT_FLIP_PROBABILITY = 0.08
NESTED_CENTER_COUNTS = (1, 2, 4, 8)


def _draw_weighted_centers(
    n_qubits: int,
    n_centers: int,
    pattern_seed: int,
) -> np.ndarray:
    """Draw distinct random centers, each with exactly ``n_qubits // 2`` ones."""
    n_qubits = int(n_qubits)
    n_centers = int(n_centers)
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least two")
    if n_centers < 1:
        raise ValueError("n_centers must be positive")
    weight = n_qubits // 2
    distinct = comb(n_qubits, weight)
    if n_centers > distinct:
        raise ValueError(
            f"n_centers exceeds the {distinct} distinct weight-{weight} strings",
        )
    rng = np.random.default_rng(int(pattern_seed))
    centers = []
    seen = set()
    while len(centers) < n_centers:
        ones = rng.choice(n_qubits, size=weight, replace=False)
        center = np.zeros(n_qubits, dtype=np.int8)
        center[ones] = 1
        key = center.tobytes()
        if key in seen:
            continue
        seen.add(key)
        centers.append(center)
    return np.stack(centers)


@dataclass(frozen=True)
class HammingBallsTarget:
    """A mixture of Hamming balls: ``K`` planted centers with bit-flip noise.

    Every center has exactly ``n_qubits // 2`` ones, so the Hamming-weight
    marginal of a ball depends only on the shared center weight.  As a result
    the shell profile is identical for every number of centers ``K``, leaving
    ``K`` visible only in bitstring-level multi-modality.
    """

    n_qubits: int
    centers: np.ndarray
    flip_probability: float = DEFAULT_FLIP_PROBABILITY
    pattern_seed: int = 0

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=np.int8)
        object.__setattr__(self, "centers", centers)
        if self.n_qubits < 2:
            raise ValueError("n_qubits must be at least two")
        if not 0.0 < self.flip_probability < 1.0:
            raise ValueError("flip_probability must lie in (0, 1)")
        if centers.ndim != 2 or centers.shape[1] != self.n_qubits:
            raise ValueError("centers must have shape (n_centers, n_qubits)")
        if not np.all((centers == 0) | (centers == 1)):
            raise ValueError("centers must be binary")
        if not np.all(centers.sum(axis=1) == self.n_qubits // 2):
            raise ValueError("every center must have weight n_qubits // 2")

    @classmethod
    def create(
        cls,
        n_qubits: int,
        n_centers: int,
        flip_probability: float = DEFAULT_FLIP_PROBABILITY,
        pattern_seed: int = 0,
    ) -> HammingBallsTarget:
        """Construct a target with seeded random weight-``n_qubits // 2`` centers."""
        centers = _draw_weighted_centers(n_qubits, n_centers, pattern_seed)
        return cls(n_qubits, centers, flip_probability, pattern_seed)

    @classmethod
    def nested_family(
        cls,
        n_qubits: int = 19,
        center_counts: tuple[int, ...] = NESTED_CENTER_COUNTS,
        flip_probability: float = DEFAULT_FLIP_PROBABILITY,
        pattern_seed: int = 0,
    ) -> tuple[HammingBallsTarget, ...]:
        """Return one target per center count, sharing nested center prefixes.

        All targets draw from a single seeded center stream and use the first
        ``count`` centers, so the ``K=8`` centers contain the ``K=4`` centers,
        which contain the ``K=2`` centers, which contain the ``K=1`` center.
        """
        counts = tuple(int(count) for count in center_counts)
        if any(count < 1 for count in counts):
            raise ValueError("center_counts must be positive")
        if counts != tuple(sorted(counts)):
            raise ValueError("center_counts must be in ascending order")
        centers = _draw_weighted_centers(n_qubits, counts[-1], pattern_seed)
        return tuple(
            cls(n_qubits, centers[:count], flip_probability, pattern_seed)
            for count in counts
        )

    @property
    def n_centers(self) -> int:
        return self.centers.shape[0]

    @property
    def center_weight(self) -> int:
        return int(self.centers[0].sum())

    @property
    def shell_probabilities(self) -> np.ndarray:
        """Analytic Hamming-weight shell profile, identical for any center count."""
        n = self.n_qubits
        p = self.flip_probability
        weight = self.center_weight
        n_zeros = n - weight
        shell = np.zeros(n + 1, dtype=np.float64)
        for m in range(n + 1):
            total = 0.0
            for j in range(max(0, m - n_zeros), min(weight, m) + 1):
                flips = m - j
                total += (
                    comb(weight, j)
                    * comb(n_zeros, flips)
                    * (1.0 - p) ** j
                    * p ** (weight - j)
                    * p**flips
                    * (1.0 - p) ** (n_zeros - flips)
                )
            shell[m] = total
        return shell

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

    def _distances_to_centers(self, bitstrings: np.ndarray) -> np.ndarray:
        bits = np.asarray(bitstrings, dtype=np.int8)
        if bits.ndim != 2 or bits.shape[1] != self.n_qubits:
            raise ValueError("bitstrings must have shape (n_strings, n_qubits)")
        if not np.all((bits == 0) | (bits == 1)):
            raise ValueError("bitstrings must be binary")
        return np.count_nonzero(bits[:, None, :] != self.centers[None, :, :], axis=2)

    def bitstring_probabilities(self, bitstrings: np.ndarray) -> np.ndarray:
        """Return the exact probability of each supplied binary string."""
        distances = self._distances_to_centers(bitstrings)
        per_center = self.flip_probability**distances * (
            1.0 - self.flip_probability
        ) ** (self.n_qubits - distances)
        return per_center.mean(axis=1)

    def sample(self, rng: np.random.Generator, n_samples: int) -> np.ndarray:
        """Draw bitstrings from a random center with bit-flip noise."""
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        choices = rng.integers(0, self.n_centers, size=n_samples)
        samples = self.centers[choices].copy()
        flips = rng.random((n_samples, self.n_qubits)) < self.flip_probability
        samples ^= flips.astype(np.int8)
        return samples

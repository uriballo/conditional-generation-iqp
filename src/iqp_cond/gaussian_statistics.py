"""Paired seed-level statistics for retained campaign summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def median_iqr(values: Sequence[float]) -> dict[str, float | int]:
    """Return the declared descriptive statistics without changing units."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be a non-empty finite vector")
    return {
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "n_replication_units": int(array.size),
        "replication_unit": "seed",
    }


@dataclass(frozen=True, slots=True)
class PairedInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    bootstrap_replicates: int
    bootstrap_seed: int
    paired_seed_values: tuple[tuple[int, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "replication_unit": "optimizer_seed",
            "paired_seed_values": {
                str(seed): value for seed, value in self.paired_seed_values
            },
        }


def paired_bootstrap_interval(
    paired_seed_values: dict[int, float],
    *,
    replicates: int = 10_000,
    seed: int = 20_260_811,
    confidence_level: float = 0.95,
) -> PairedInterval:
    if len(paired_seed_values) < 2 or len(set(paired_seed_values)) != len(
        paired_seed_values
    ):
        raise ValueError("paired bootstrap requires unique seed values")
    if replicates < 1 or not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap settings are invalid")
    ordered = tuple(sorted((int(k), float(v)) for k, v in paired_seed_values.items()))
    values = np.asarray([value for _, value in ordered], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("paired values must be finite")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(replicates, values.size))].mean(
        axis=1
    )
    tail = (1.0 - confidence_level) / 2.0
    return PairedInterval(
        estimate=float(values.mean()),
        lower=float(np.quantile(draws, tail)),
        upper=float(np.quantile(draws, 1.0 - tail)),
        confidence_level=confidence_level,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
        paired_seed_values=ordered,
    )


def paired_interval(
    values: Mapping[int, float],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, object]:
    """Bootstrap a paired seed-level contrast with frozen randomness."""
    return paired_bootstrap_interval(
        dict(values),
        replicates=replicates,
        seed=seed,
        confidence_level=confidence_level,
    ).to_dict()

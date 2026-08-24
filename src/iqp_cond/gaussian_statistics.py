"""Paired statistics and bank baselines for the Gaussian campaign."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .exact import shell_probabilities, total_variation_distance


@dataclass(frozen=True, slots=True)
class GaussianMetricRecord:
    """Typed exact-evaluation metrics for one method and Condition."""

    profile_tvd: float
    bitstring_tvd: float
    mean_absolute_error: float
    width_absolute_error: float
    conditional_mmd: float

    def __post_init__(self) -> None:
        values = (
            self.profile_tvd,
            self.bitstring_tvd,
            self.mean_absolute_error,
            self.width_absolute_error,
            self.conditional_mmd,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(
                "Gaussian evaluation metrics must be finite and non-negative"
            )
        if self.profile_tvd > 1.0 + 1e-12 or self.bitstring_tvd > 1.0 + 1e-12:
            raise ValueError("Gaussian TVD metrics cannot exceed one")

    def to_dict(self) -> dict[str, float]:
        return {
            "profile_tvd": self.profile_tvd,
            "bitstring_tvd": self.bitstring_tvd,
            "mean_absolute_error": self.mean_absolute_error,
            "width_absolute_error": self.width_absolute_error,
            "conditional_mmd": self.conditional_mmd,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GaussianMetricRecord:
        expected = {
            "profile_tvd",
            "bitstring_tvd",
            "mean_absolute_error",
            "width_absolute_error",
            "conditional_mmd",
        }
        if set(value) != expected:
            raise ValueError("Gaussian metric record schema differs")
        return cls(**{name: float(value[name]) for name in expected})


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


def adjacent_indices(
    training_conditions: np.ndarray, condition: float
) -> tuple[int, int]:
    training = np.asarray(training_conditions, dtype=np.float64)
    if training.ndim != 1 or training.size < 2 or not np.all(np.diff(training) > 0):
        raise ValueError("training conditions must be strictly increasing")
    if not training[0] < condition < training[-1] or np.any(
        np.isclose(training, condition, atol=1e-14, rtol=0.0)
    ):
        raise ValueError("held-out condition must be strictly interior and unseen")
    right = int(np.searchsorted(training, condition, side="right"))
    return right - 1, right


def barycentric_mixture(
    training_conditions: np.ndarray,
    stored_probabilities: np.ndarray,
    condition: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    training = np.asarray(training_conditions, dtype=np.float64)
    stored = np.asarray(stored_probabilities, dtype=np.float64)
    if stored.ndim != 2 or stored.shape[0] != training.size:
        raise ValueError("stored probabilities must align with training conditions")
    left, right = adjacent_indices(training, condition)
    weight_right = float(
        (condition - training[left]) / (training[right] - training[left])
    )
    mixture = (1.0 - weight_right) * stored[left] + weight_right * stored[right]
    return mixture, {
        "left_index": left,
        "right_index": right,
        "left_weight": 1.0 - weight_right,
        "right_weight": weight_right,
    }


def oracle_adjacent_lookup(
    training_conditions: np.ndarray,
    stored_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    condition: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    training = np.asarray(training_conditions, dtype=np.float64)
    stored = np.asarray(stored_probabilities, dtype=np.float64)
    target = np.asarray(target_probabilities, dtype=np.float64)
    left, right = adjacent_indices(training, condition)
    tvds = (
        total_variation_distance(target, stored[left]),
        total_variation_distance(target, stored[right]),
    )
    selected = left if tvds[0] <= tvds[1] else right
    return stored[selected], {
        "left_index": left,
        "right_index": right,
        "selected_index": selected,
        "left_bitstring_tvd": tvds[0],
        "right_bitstring_tvd": tvds[1],
    }


def oracle_adjacent_profile_lookup(
    training_conditions: np.ndarray,
    stored_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    bitstrings: np.ndarray,
    condition: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Select the adjacent output with lower task-primary profile TVD."""
    training = np.asarray(training_conditions, dtype=np.float64)
    stored = np.asarray(stored_probabilities, dtype=np.float64)
    target = np.asarray(target_probabilities, dtype=np.float64)
    bitstrings = np.asarray(bitstrings, dtype=np.int8)
    left, right = adjacent_indices(training, condition)
    target_profile = shell_probabilities(target, bitstrings)
    tvds = (
        total_variation_distance(
            target_profile, shell_probabilities(stored[left], bitstrings)
        ),
        total_variation_distance(
            target_profile, shell_probabilities(stored[right], bitstrings)
        ),
    )
    selected = left if tvds[0] <= tvds[1] else right
    return stored[selected], {
        "left_index": left,
        "right_index": right,
        "selected_index": selected,
        "selection_metric": "profile_tvd",
        "left_profile_tvd": tvds[0],
        "right_profile_tvd": tvds[1],
    }

"""Exact metrics, reference baselines, and paired Curie--Weiss statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curie_weiss_data import CurieWeissTarget
from .exact import shell_probabilities, total_variation_distance
from .gaussian_statistics import PairedInterval, paired_bootstrap_interval


@dataclass(frozen=True, slots=True)
class ThermodynamicMetrics:
    profile_tvd: float
    bitstring_tvd: float
    absolute_magnetization: float
    energy_per_spin: float
    susceptibility: float
    binder_cumulant: float
    heat_capacity_per_spin: float

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def thermodynamic_metrics(
    target: CurieWeissTarget,
    model_probabilities: np.ndarray,
) -> ThermodynamicMetrics:
    model = np.asarray(model_probabilities, dtype=np.float64)
    if model.shape != target.probabilities.shape or not np.isclose(model.sum(), 1.0):
        raise ValueError("model probabilities are not a normalized target-shaped law")
    bits = np.arange(model.size, dtype=np.uint32)[:, None]
    bit_array = ((bits >> np.arange(target.n_qubits)) & 1).astype(np.int8)
    profile = shell_probabilities(model, bit_array)
    m = target.magnetization
    energy = target.energy
    mean_m = float(model @ m)
    mean_m2 = float(model @ (m**2))
    mean_m4 = float(model @ (m**4))
    mean_energy = float(model @ energy)
    energy_variance = float(model @ (energy**2) - mean_energy**2)
    binder = 0.0 if mean_m2 == 0.0 else 1.0 - mean_m4 / (3.0 * mean_m2**2)
    return ThermodynamicMetrics(
        total_variation_distance(target.profile, profile),
        total_variation_distance(target.probabilities, model),
        float(model @ np.abs(m)),
        mean_energy / target.n_qubits,
        target.beta * target.n_qubits * (mean_m2 - mean_m**2),
        binder,
        target.beta**2 * energy_variance / target.n_qubits,
    )


def adjacent_reference_indices(
    training_betas: np.ndarray, beta: float
) -> tuple[int, int]:
    training = np.asarray(training_betas, dtype=np.float64)
    if training.ndim != 1 or not np.all(np.diff(training) > 0):
        raise ValueError("training betas must be strictly increasing")
    if np.any(np.isclose(training, beta)) or not training[0] < beta < training[-1]:
        raise ValueError("beta must be an unseen interior midpoint")
    right = int(np.searchsorted(training, beta))
    if not np.isclose(beta, (training[right - 1] + training[right]) / 2):
        raise ValueError("held-out beta must be the adjacent midpoint")
    return right - 1, right


def midpoint_mixture(
    training_betas: np.ndarray,
    reference_probabilities: np.ndarray,
    beta: float,
) -> np.ndarray:
    left, right = adjacent_reference_indices(training_betas, beta)
    return 0.5 * (reference_probabilities[left] + reference_probabilities[right])


def adjacent_profile_oracle(
    training_betas: np.ndarray,
    reference_probabilities: np.ndarray,
    target: CurieWeissTarget,
) -> tuple[np.ndarray, int]:
    left, right = adjacent_reference_indices(training_betas, target.beta)
    bitstrings = np.arange(target.probabilities.size, dtype=np.uint32)[:, None]
    bits = ((bitstrings >> np.arange(target.n_qubits)) & 1).astype(np.int8)
    scores = [
        total_variation_distance(
            target.profile, shell_probabilities(reference_probabilities[index], bits)
        )
        for index in (left, right)
    ]
    selected = left if scores[0] <= scores[1] else right
    return reference_probabilities[selected], selected


def paired_interval(values: dict[int, float]) -> PairedInterval:
    return paired_bootstrap_interval(values, replicates=10_000, seed=20_260_812)


def evidence_status(interval: PairedInterval) -> str:
    if interval.lower > 0:
        return "supported"
    if interval.upper < 0:
        return "unsupported"
    return "mixed"

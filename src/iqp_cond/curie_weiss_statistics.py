"""Exact metrics and thermodynamic observables for Curie--Weiss."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curie_weiss_data import CurieWeissTarget
from .exact import shell_probabilities, total_variation_distance


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

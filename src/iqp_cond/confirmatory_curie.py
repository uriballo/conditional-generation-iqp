"""Frozen Curie--Weiss confirmatory design and exact target utilities."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import networkx as nx
import numpy as np

from .ansatz import Ansatz, complete_ansatz, generators_from_graph
from .exact import enumerate_bitstrings
from .protocol import ProtocolValidationError, content_digest

CAMPAIGN_VERSION = "curie_weiss_confirmatory_v1"
SYSTEM_SIZES = (8, 12, 16)
SEEDS = tuple(range(100, 110))
ANCILLA_RUNGS = (1, 3, 5)
TRAINING_BETAS = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
HELD_OUT_BETAS = (0.6, 0.8, 1.0, 1.2, 1.4)
CRITICAL_BETAS = (0.8, 1.0, 1.2)
FIELD_TRAINING = (0.0, 0.15, 0.30)
FIELD_REFLECTED_TRAINING = (-0.30, -0.15, 0.0)
FIELD_HELD_OUT = (-0.225, -0.075, 0.075, 0.225)


class CurieConfirmatoryArm(StrEnum):
    """Model arms in the temperature and ancilla study."""

    M0_PRODUCT = "m0_product"
    M0_SPIN_FLIP = "m0_spin_flip"
    FULL_JOINT_R1 = "full_joint_r1"
    FULL_JOINT_R3 = "full_joint_r3"
    FULL_JOINT_R5 = "full_joint_r5"
    DISCONNECTED_R1 = "disconnected_r1"
    DISCONNECTED_R3 = "disconnected_r3"
    DISCONNECTED_R5 = "disconnected_r5"
    VISIBLE_MATCHED_R1 = "visible_matched_r1"
    VISIBLE_MATCHED_R3 = "visible_matched_r3"
    VISIBLE_MATCHED_R5 = "visible_matched_r5"
    NO_CONDITION_M0 = "no_condition_m0"
    NO_CONDITION_M5 = "no_condition_m5"

    @property
    def ancillas(self) -> int:
        suffix = self.value.rsplit("r", maxsplit=1)
        if len(suffix) == 2 and suffix[1] in {"1", "3", "5"}:
            return int(suffix[1])
        return 5 if self is CurieConfirmatoryArm.NO_CONDITION_M5 else 0


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ProtocolValidationError(
            f"{context} keys differ; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}",
        )


@dataclass(frozen=True, slots=True)
class CurieConfirmatoryManifest:
    """Immutable, content-addressed confirmatory campaign manifest."""

    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        _exact_keys(
            self.raw,
            {
                "schema_version",
                "campaign_version",
                "system_sizes",
                "seeds",
                "ancilla_rungs",
                "models",
                "temperatures",
                "temperature_encoder",
                "condition_map",
                "field_study",
                "training",
                "statistics",
            },
            "confirmatory Curie--Weiss manifest",
        )
        if self.raw["schema_version"] != 1:
            raise ProtocolValidationError("manifest schema_version must be 1")
        if self.campaign_version != CAMPAIGN_VERSION:
            raise ProtocolValidationError("campaign version differs")
        if self.system_sizes != SYSTEM_SIZES or self.seeds != SEEDS:
            raise ProtocolValidationError("system sizes or fresh seeds differ")
        if self.ancilla_rungs != ANCILLA_RUNGS:
            raise ProtocolValidationError("ancilla rungs differ")
        declared_models = tuple(self.raw["models"])
        expected_models = tuple(arm.value for arm in CurieConfirmatoryArm)
        if declared_models != expected_models:
            raise ProtocolValidationError("model arms differ")
        temperatures = self.raw["temperatures"]
        _exact_keys(temperatures, {"training", "held_out", "critical"}, "temperatures")
        if (
            tuple(float(v) for v in temperatures["training"]) != TRAINING_BETAS
            or tuple(float(v) for v in temperatures["held_out"]) != HELD_OUT_BETAS
            or tuple(float(v) for v in temperatures["critical"]) != CRITICAL_BETAS
        ):
            raise ProtocolValidationError("temperature grid differs")
        if self.raw["temperature_encoder"] != "spin_flip_branch":
            raise ProtocolValidationError("temperature encoder differs")
        self._validate_condition_map()
        self._validate_field_study()
        self._validate_training_and_statistics()
        object.__setattr__(self, "raw", _freeze(self.raw))

    @property
    def campaign_version(self) -> str:
        return str(self.raw["campaign_version"])

    @property
    def system_sizes(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["system_sizes"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["seeds"])

    @property
    def ancilla_rungs(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["ancilla_rungs"])

    @property
    def training_betas(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.raw["temperatures"]["training"])

    @property
    def held_out_betas(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.raw["temperatures"]["held_out"])

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.raw)

    @classmethod
    def from_json(cls, path: Path) -> CurieConfirmatoryManifest:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(f"invalid manifest: {path}") from error
        if not isinstance(value, dict):
            raise ProtocolValidationError("manifest must be a JSON object")
        return cls(value)

    def _validate_condition_map(self) -> None:
        expected = {
            "normalized_coordinate": "beta",
            "map_kind": "broadcast_piecewise_linear",
            "anchor_units": "pi",
            "pieces": [
                {"start": 0.5, "end": 0.9, "slope": 0.175, "anchor": -0.14},
                {"start": 0.9, "end": 1.0, "slope": 0.70, "anchor": -0.07},
                {"start": 1.0, "end": 1.2, "slope": 0.40, "anchor": 0.0},
                {"start": 1.2, "end": 1.5, "slope": 0.20, "anchor": 0.08},
            ],
            "ancilla_scheme": "broadcast_all",
            "outcome_tuned": False,
        }
        if _thaw(self.raw["condition_map"]) != expected:
            raise ProtocolValidationError("frozen temperature condition map differs")

    def _validate_field_study(self) -> None:
        expected = {
            "beta": 1.0,
            "training_fields": [0.0, 0.15, 0.30],
            "reflected_training_fields": [-0.30, -0.15, 0.0],
            "held_out_fields": [-0.225, -0.075, 0.075, 0.225],
            "angle_slope_pi": -7.0 / 6.0,
            "equivariance_tolerance": 1e-12,
            "two_sided_noninferiority_tvd": 0.02,
        }
        actual = _thaw(self.raw["field_study"])
        if set(actual) != set(expected):
            raise ProtocolValidationError("field study schema differs")
        for key in expected:
            if isinstance(expected[key], float):
                if not math.isclose(float(actual[key]), expected[key], abs_tol=1e-15):
                    raise ProtocolValidationError("field study differs")
            elif actual[key] != expected[key]:
                raise ProtocolValidationError("field study differs")

    def _validate_training_and_statistics(self) -> None:
        if _thaw(self.raw["training"]) != {
            "optimizer": "adam_cosine_decay",
            "updates": 2000,
            "initial_learning_rate": 0.01,
            "final_learning_rate": 0.001,
            "total_target_minibatch": 450,
            "reference_minibatch": 450,
            "validation_interval": 25,
            "checkpoint_rule": "lowest_validation_conditional_mmd",
            "bandwidths": [1.0, 2.0, 4.0],
            "visible_character_masks": 256,
            "training_model_samples": 8192,
            "validation_model_samples": 16384,
        }:
            raise ProtocolValidationError("training protocol differs")
        if _thaw(self.raw["statistics"]) != {
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260825,
            "confidence_level": 0.95,
            "replication_unit": "seed",
            "summary": "median_and_interquartile_range",
            "interval": "paired_percentile_bootstrap",
        }:
            raise ProtocolValidationError("statistical protocol differs")


@dataclass(frozen=True, slots=True)
class CurieConfirmatoryRunIdentity:
    """One seed-matched temperature-study fit."""

    campaign_version: str
    n_qubits: int
    arm: CurieConfirmatoryArm
    ancillas: int
    seed: int
    configuration_digest: str

    def __post_init__(self) -> None:
        if self.campaign_version != CAMPAIGN_VERSION:
            raise ProtocolValidationError("run campaign version differs")
        if self.n_qubits not in SYSTEM_SIZES or self.seed not in SEEDS:
            raise ProtocolValidationError("run coordinates are outside the manifest")
        if self.ancillas != self.arm.ancillas:
            raise ProtocolValidationError(
                "run ancilla count differs from its model arm"
            )
        if len(self.configuration_digest) != 64:
            raise ProtocolValidationError("configuration_digest must be SHA-256")

    @property
    def run_id(self) -> str:
        return content_digest(self.to_dict())

    @property
    def parameter_count(self) -> int:
        if self.arm.value.startswith("disconnected_r"):
            return disconnected_ansatz(self.n_qubits, self.ancillas).n_parameters
        total = self.n_qubits + self.ancillas
        return total * (total + 1) // 2

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_version": self.campaign_version,
            "n_qubits": self.n_qubits,
            "arm": self.arm.value,
            "ancillas": self.ancillas,
            "seed": self.seed,
            "configuration_digest": self.configuration_digest,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> CurieConfirmatoryRunIdentity:
        _exact_keys(
            value,
            {
                "campaign_version",
                "n_qubits",
                "arm",
                "ancillas",
                "seed",
                "configuration_digest",
            },
            "confirmatory Curie--Weiss run identity",
        )
        return cls(
            campaign_version=str(value["campaign_version"]),
            n_qubits=int(value["n_qubits"]),
            arm=CurieConfirmatoryArm(str(value["arm"])),
            ancillas=int(value["ancillas"]),
            seed=int(value["seed"]),
            configuration_digest=str(value["configuration_digest"]),
        )


def plan_curie_confirmatory_runs(
    manifest: CurieConfirmatoryManifest,
) -> tuple[CurieConfirmatoryRunIdentity, ...]:
    """Enumerate the frozen 3 x 10 x 13 temperature-study fits."""
    runs = tuple(
        CurieConfirmatoryRunIdentity(
            manifest.campaign_version,
            n_qubits,
            arm,
            arm.ancillas,
            seed,
            manifest.digest,
        )
        for n_qubits in manifest.system_sizes
        for seed in manifest.seeds
        for arm in CurieConfirmatoryArm
    )
    if len(runs) != 390 or len({run.run_id for run in runs}) != len(runs):
        raise RuntimeError("confirmatory planner must produce 390 unique fits")
    return runs


def curie_weiss_probabilities(n_qubits: int, beta: float) -> np.ndarray:
    """Return ``p_beta(x) proportional to exp(beta M(x)^2 / (2n))``."""
    if n_qubits not in SYSTEM_SIZES or not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("invalid Curie--Weiss target coordinates")
    bits = enumerate_bitstrings(n_qubits)
    magnetization = (2.0 * bits - 1.0).sum(axis=1)
    return _normalized_exponential(beta * magnetization**2 / (2.0 * n_qubits))


def signed_field_probabilities(n_qubits: int, field: float) -> np.ndarray:
    """Return the beta=1 target ``p_h(x) proportional to exp(M^2/(2n)+hM)``."""
    if n_qubits not in SYSTEM_SIZES or not math.isfinite(field):
        raise ValueError("invalid signed-field target coordinates")
    bits = enumerate_bitstrings(n_qubits)
    magnetization = (2.0 * bits - 1.0).sum(axis=1)
    return _normalized_exponential(
        magnetization**2 / (2.0 * n_qubits) + field * magnetization,
    )


def _normalized_exponential(log_weights: np.ndarray) -> np.ndarray:
    weights = np.exp(np.asarray(log_weights, dtype=np.float64) - np.max(log_weights))
    probabilities = weights / weights.sum()
    if not np.all(np.isfinite(probabilities)) or not np.isclose(
        probabilities.sum(), 1.0, atol=1e-14
    ):
        raise FloatingPointError("target normalization failed")
    return probabilities


def equivariance_defect(
    probabilities_at_h: np.ndarray,
    probabilities_at_minus_h: np.ndarray,
) -> float:
    """Return one half of the L1 defect under ``(x,h) -> (complement(x),-h)``."""
    at_h = _validated_probabilities(probabilities_at_h)
    at_minus_h = _validated_probabilities(probabilities_at_minus_h)
    if at_h.shape != at_minus_h.shape:
        raise ValueError("probability vectors must have the same shape")
    return float(0.5 * np.abs(at_h - at_minus_h[::-1]).sum())


def _validated_probabilities(value: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(value, dtype=np.float64)
    if probabilities.ndim != 1 or len(probabilities) < 2:
        raise ValueError("probabilities must be a one-dimensional vector")
    if len(probabilities) & (len(probabilities) - 1):
        raise ValueError("probability vector length must be a power of two")
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-10)
    ):
        raise ValueError("probabilities must be finite, nonnegative, and normalized")
    return probabilities


def disconnected_ansatz(n_visible_qubits: int, n_ancillas: int) -> Ansatz:
    """Return complete within-register banks with no visible--ancilla terms."""
    if n_visible_qubits not in SYSTEM_SIZES or n_ancillas not in ANCILLA_RUNGS:
        raise ValueError("unsupported disconnected-control coordinates")
    graph = nx.Graph()
    graph.add_nodes_from(range(n_visible_qubits + n_ancillas))
    graph.add_edges_from(combinations(range(n_visible_qubits), 2))
    graph.add_edges_from(
        combinations(range(n_visible_qubits, n_visible_qubits + n_ancillas), 2)
    )
    return Ansatz(
        name=f"disconnected_n{n_visible_qubits}_r{n_ancillas}",
        graph=graph,
        generators=generators_from_graph(graph),
    )


def visible_matched_ansatz(n_visible_qubits: int, n_ancillas: int) -> Ansatz:
    """Match the full joint parameter count with visible three-local terms."""
    if n_visible_qubits not in SYSTEM_SIZES or n_ancillas not in ANCILLA_RUNGS:
        raise ValueError("unsupported matched-control coordinates")
    base = complete_ansatz(n_visible_qubits)
    total_qubits = n_visible_qubits + n_ancillas
    target_count = total_qubits * (total_qubits + 1) // 2
    extra_count = target_count - base.n_parameters
    triples = tuple(combinations(range(n_visible_qubits), 3))
    if extra_count > len(triples):
        raise ValueError("not enough distinct visible three-local generators")
    extra = np.zeros((extra_count, n_visible_qubits), dtype=np.float64)
    for row, support in zip(extra, triples[:extra_count], strict=True):
        row[list(support)] = 1.0
    generators = np.concatenate((np.asarray(base.generators), extra), axis=0)
    return Ansatz(
        name=f"visible_matched_n{n_visible_qubits}_r{n_ancillas}",
        graph=base.graph.copy(),
        generators=jnp.asarray(generators, dtype=jnp.float64),
    )

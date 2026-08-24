"""Typed protocol and content-addressed identities for the Quantum campaign."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any


class ProtocolValidationError(ValueError):
    """Raised when a campaign manifest or run identity violates the protocol."""


class Stage(StrEnum):
    """Training-run stages represented in the immutable run ledger."""

    RUNTIME_FEASIBILITY = "runtime_feasibility"
    REPRESENTABILITY_SCREEN = "representability_screen"
    TUNING = "tuning"
    PRIMARY = "primary"
    ANCILLA = "ancilla"
    REPRESENTABILITY_DIAGNOSTIC = "representability_diagnostic"


class BudgetPool(StrEnum):
    """Non-transferable full-training-run allocation pools."""

    FEASIBILITY_REPRESENTABILITY = "feasibility_representability"
    TUNING = "tuning"
    PRIMARY = "primary"
    ANCILLA = "ancilla"


class ModelFamily(StrEnum):
    """Complete trained-model vocabulary for the campaign."""

    FIXED_INPUT = "fixed_input"
    CONDITION_MODULATED_PHASE = "condition_modulated_phase"
    NO_CONDITION = "no_condition"
    SEPARATE = "separate"
    MATCHED_CAPACITY_CONTROL = "matched_capacity_control"


class PhaseFeature(StrEnum):
    """Preregistered parameter-matched Condition-to-phase assignments."""

    COMMON_AFFINE_MULTIPLIER = "common_affine_multiplier"
    SIGNED_AFFINE_MULTIPLIERS = "signed_affine_multipliers"
    BALANCED_CONSTANT_SLOPE_BANK = "balanced_constant_slope_bank"


MODEL_FAMILY_VOCABULARY = tuple(item.value for item in ModelFamily)
PHASE_FEATURE_VOCABULARY = tuple(item.value for item in PhaseFeature)
SHARED_PRIMARY_FAMILIES = (
    ModelFamily.FIXED_INPUT,
    ModelFamily.CONDITION_MODULATED_PHASE,
    ModelFamily.NO_CONDITION,
)
EXPECTED_RUN_ALLOCATION = {
    BudgetPool.FEASIBILITY_REPRESENTABILITY.value: 20,
    BudgetPool.TUNING.value: 36,
    BudgetPool.PRIMARY.value: 320,
    BudgetPool.ANCILLA.value: 24,
}
EXPECTED_BANDWIDTHS = (1.0, 2.0, 4.0)
EXPECTED_SAMPLER_PROTOCOL = {
    "n_chains": 8,
    "burn_in_sweeps": 4_000,
    "thinning_sweeps": 32,
    "training_samples_per_condition": 5_000,
    "validation_samples_per_condition": 2_000,
    "min_ess_per_chain": 100.0,
    "max_chain_energy_z": 0.30,
    "max_energy_mean_z": 0.10,
    "max_magnetization_error": 0.08,
    "max_correlation_rmse": 0.08,
    "max_character_rmse": 0.08,
}
OPTIMIZER_NAME = "adam_cosine_decay"
CHECKPOINT_RULE = "lowest_validation_conditional_mmd"
TUNING_SELECTION_RULE = "minimum_median_validation_conditional_mmd_then_candidate_id"
EXPECTED_TUNING_CANDIDATES = (
    (
        "fixed_input_lr_low",
        ModelFamily.FIXED_INPUT,
        3e-3,
        3e-4,
        None,
    ),
    (
        "fixed_input_lr_reference",
        ModelFamily.FIXED_INPUT,
        1e-2,
        1e-3,
        None,
    ),
    (
        "fixed_input_lr_high",
        ModelFamily.FIXED_INPUT,
        3e-2,
        3e-3,
        None,
    ),
    (
        "phase_common_lr_low",
        ModelFamily.CONDITION_MODULATED_PHASE,
        3e-3,
        3e-4,
        PhaseFeature.COMMON_AFFINE_MULTIPLIER.value,
    ),
    (
        "phase_signed_lr_reference",
        ModelFamily.CONDITION_MODULATED_PHASE,
        1e-2,
        1e-3,
        PhaseFeature.SIGNED_AFFINE_MULTIPLIERS.value,
    ),
    (
        "phase_balanced_lr_high",
        ModelFamily.CONDITION_MODULATED_PHASE,
        3e-2,
        3e-3,
        PhaseFeature.BALANCED_CONSTANT_SLOPE_BANK.value,
    ),
    (
        "no_condition_lr_low",
        ModelFamily.NO_CONDITION,
        3e-3,
        3e-4,
        None,
    ),
    (
        "no_condition_lr_reference",
        ModelFamily.NO_CONDITION,
        1e-2,
        1e-3,
        None,
    ),
    (
        "no_condition_lr_high",
        ModelFamily.NO_CONDITION,
        3e-2,
        3e-3,
        None,
    ),
)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical mappings require string keys")
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON does not permit non-finite floats")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the unique UTF-8 JSON representation used for campaign hashes."""
    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    """Return a SHA-256 digest of a canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ProtocolValidationError(
            f"{context} keys do not match schema; missing={missing}, unknown={unknown}",
        )


def _require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolValidationError(f"{context} must be an integer")
    return value


def _require_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolValidationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolValidationError(f"{context} must be finite")
    return result


def _require_digest(value: str, context: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ProtocolValidationError(f"{context} must be a lowercase SHA-256 digest")


def _require_unique_ints(
    values: tuple[int, ...],
    expected_length: int,
    context: str,
) -> None:
    if len(values) != expected_length:
        raise ProtocolValidationError(
            f"{context} must contain exactly {expected_length} values",
        )
    if len(set(values)) != len(values):
        raise ProtocolValidationError(f"{context} must not contain duplicates")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ProtocolValidationError(
            f"{context} must contain non-negative integers",
        )


@dataclass(frozen=True)
class CalibrationProvenance:
    """Provenance for the Target-only beta-grid calibration."""

    method: str
    artifact_path: str
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.method != "target_only_calibration_v1":
            raise ProtocolValidationError(
                "calibration method must be 'target_only_calibration_v1'",
            )
        path = Path(self.artifact_path)
        if not self.artifact_path or path.is_absolute() or ".." in path.parts:
            raise ProtocolValidationError(
                "calibration artifact_path must be a non-empty "
                "repository-relative path",
            )
        if self.artifact_sha256 is not None:
            _require_digest(self.artifact_sha256, "calibration artifact_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalibrationProvenance:
        _require_exact_keys(
            value,
            {"method", "artifact_path", "artifact_sha256"},
            "calibration",
        )
        digest = value["artifact_sha256"]
        if digest is not None and not isinstance(digest, str):
            raise ProtocolValidationError(
                "calibration artifact_sha256 must be a string or null",
            )
        return cls(
            method=str(value["method"]),
            artifact_path=str(value["artifact_path"]),
            artifact_sha256=digest,
        )


@dataclass(frozen=True)
class BetaGrid:
    """Five Training betas and their four adjacent Held-out midpoints."""

    training: tuple[float, ...]
    held_out: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.training) != 5 or len(self.held_out) != 4:
            raise ProtocolValidationError(
                "beta grid must contain five Training and four Held-out values",
            )
        if any(beta <= 0.0 or not math.isfinite(beta) for beta in self.training):
            raise ProtocolValidationError("Training betas must be finite and positive")
        if any(beta <= 0.0 or not math.isfinite(beta) for beta in self.held_out):
            raise ProtocolValidationError("Held-out betas must be finite and positive")
        if any(
            right <= left
            for left, right in zip(self.training[:-1], self.training[1:], strict=True)
        ):
            raise ProtocolValidationError("Training betas must be strictly increasing")
        if not any(
            math.isclose(beta, 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-14)
            for beta in self.training
        ):
            raise ProtocolValidationError(
                "Training betas must include the beta=1/3 anchor",
            )
        for index, held_out in enumerate(self.held_out):
            expected = (self.training[index] + self.training[index + 1]) / 2.0
            if not math.isclose(held_out, expected, rel_tol=0.0, abs_tol=1e-14):
                raise ProtocolValidationError(
                    "each Held-out beta must be the adjacent Training-beta midpoint",
                )

    def to_dict(self) -> dict[str, object]:
        return {"training": list(self.training), "held_out": list(self.held_out)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BetaGrid:
        _require_exact_keys(value, {"training", "held_out"}, "beta_grid")
        training = tuple(
            _require_float(item, "Training beta") for item in value["training"]
        )
        held_out = tuple(
            _require_float(item, "Held-out beta") for item in value["held_out"]
        )
        return cls(training=training, held_out=held_out)


@dataclass(frozen=True)
class ModelProtocol:
    """Frozen topology, parameter count, and model-family vocabulary."""

    n_qubits: int
    n_parameters: int
    families: tuple[str, ...]
    phase_features: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.n_qubits != 16 or self.n_parameters != 136:
            raise ProtocolValidationError(
                "the primary protocol requires 16 qubits and 136 parameters",
            )
        expected_parameters = self.n_qubits + self.n_qubits * (self.n_qubits - 1) // 2
        if self.n_parameters != expected_parameters:
            raise ProtocolValidationError(
                "parameter count does not match the Complete 2-local topology",
            )
        if len(self.families) != len(set(self.families)) or set(
            self.families,
        ) != set(MODEL_FAMILY_VOCABULARY):
            raise ProtocolValidationError("model-family vocabulary is not exact")
        if len(self.phase_features) != len(set(self.phase_features)) or set(
            self.phase_features,
        ) != set(PHASE_FEATURE_VOCABULARY):
            raise ProtocolValidationError("phase-feature vocabulary is not exact")

    def to_dict(self) -> dict[str, object]:
        return {
            "n_qubits": self.n_qubits,
            "n_parameters": self.n_parameters,
            "families": list(self.families),
            "phase_features": list(self.phase_features),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelProtocol:
        _require_exact_keys(
            value,
            {
                "n_qubits",
                "n_parameters",
                "families",
                "phase_features",
            },
            "model_protocol",
        )
        return cls(
            n_qubits=_require_int(value["n_qubits"], "n_qubits"),
            n_parameters=_require_int(value["n_parameters"], "n_parameters"),
            families=tuple(str(item) for item in value["families"]),
            phase_features=tuple(str(item) for item in value["phase_features"]),
        )


@dataclass(frozen=True)
class EstimatorBudget:
    """Frozen sampled Conditional MMD estimator budget."""

    bandwidths: tuple[float, ...]
    operators_per_bandwidth: int
    reference_samples_per_operator_condition: int

    def __post_init__(self) -> None:
        if self.bandwidths != EXPECTED_BANDWIDTHS:
            raise ProtocolValidationError(
                "production estimator bandwidths must be exactly (1, 2, 4)",
            )
        if (
            self.operators_per_bandwidth != 2_000
            or self.reference_samples_per_operator_condition != 2_000
        ):
            raise ProtocolValidationError(
                "production estimator budget must be exactly 2,000 by 2,000",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "bandwidths": list(self.bandwidths),
            "operators_per_bandwidth": self.operators_per_bandwidth,
            "reference_samples_per_operator_condition": (
                self.reference_samples_per_operator_condition
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EstimatorBudget:
        _require_exact_keys(
            value,
            {
                "bandwidths",
                "operators_per_bandwidth",
                "reference_samples_per_operator_condition",
            },
            "estimator_budget",
        )
        return cls(
            bandwidths=tuple(
                _require_float(item, "estimator bandwidth")
                for item in value["bandwidths"]
            ),
            operators_per_bandwidth=_require_int(
                value["operators_per_bandwidth"],
                "operators_per_bandwidth",
            ),
            reference_samples_per_operator_condition=_require_int(
                value["reference_samples_per_operator_condition"],
                "reference_samples_per_operator_condition",
            ),
        )


@dataclass(frozen=True)
class SamplerProtocol:
    """Frozen independent-chain schedule and exact-reference audit thresholds."""

    n_chains: int
    burn_in_sweeps: int
    thinning_sweeps: int
    training_samples_per_condition: int
    validation_samples_per_condition: int
    min_ess_per_chain: float
    max_chain_energy_z: float
    max_energy_mean_z: float
    max_magnetization_error: float
    max_correlation_rmse: float
    max_character_rmse: float

    def __post_init__(self) -> None:
        if self.to_dict() != EXPECTED_SAMPLER_PROTOCOL:
            raise ProtocolValidationError(
                "sampler schedule and quality thresholds are not frozen values",
            )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "n_chains": self.n_chains,
            "burn_in_sweeps": self.burn_in_sweeps,
            "thinning_sweeps": self.thinning_sweeps,
            "training_samples_per_condition": self.training_samples_per_condition,
            "validation_samples_per_condition": (self.validation_samples_per_condition),
            "min_ess_per_chain": self.min_ess_per_chain,
            "max_chain_energy_z": self.max_chain_energy_z,
            "max_energy_mean_z": self.max_energy_mean_z,
            "max_magnetization_error": self.max_magnetization_error,
            "max_correlation_rmse": self.max_correlation_rmse,
            "max_character_rmse": self.max_character_rmse,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SamplerProtocol:
        _require_exact_keys(value, set(EXPECTED_SAMPLER_PROTOCOL), "sampler_protocol")
        return cls(
            n_chains=_require_int(value["n_chains"], "n_chains"),
            burn_in_sweeps=_require_int(
                value["burn_in_sweeps"],
                "burn_in_sweeps",
            ),
            thinning_sweeps=_require_int(
                value["thinning_sweeps"],
                "thinning_sweeps",
            ),
            training_samples_per_condition=_require_int(
                value["training_samples_per_condition"],
                "training_samples_per_condition",
            ),
            validation_samples_per_condition=_require_int(
                value["validation_samples_per_condition"],
                "validation_samples_per_condition",
            ),
            min_ess_per_chain=_require_float(
                value["min_ess_per_chain"],
                "min_ess_per_chain",
            ),
            max_chain_energy_z=_require_float(
                value["max_chain_energy_z"],
                "max_chain_energy_z",
            ),
            max_energy_mean_z=_require_float(
                value["max_energy_mean_z"],
                "max_energy_mean_z",
            ),
            max_magnetization_error=_require_float(
                value["max_magnetization_error"],
                "max_magnetization_error",
            ),
            max_correlation_rmse=_require_float(
                value["max_correlation_rmse"],
                "max_correlation_rmse",
            ),
            max_character_rmse=_require_float(
                value["max_character_rmse"],
                "max_character_rmse",
            ),
        )


@dataclass(frozen=True)
class TuningCandidate:
    """One preregistered optimizer/phase-feature candidate."""

    candidate_id: str
    model_family: ModelFamily
    initial_learning_rate: float
    final_learning_rate: float
    phase_feature_assignment: str | None

    def __post_init__(self) -> None:
        if (
            not self.candidate_id
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in self.candidate_id
            )
            or self.model_family not in SHARED_PRIMARY_FAMILIES
        ):
            raise ProtocolValidationError("invalid tuning candidate identity")
        initial = _require_float(
            self.initial_learning_rate,
            "initial_learning_rate",
        )
        final = _require_float(self.final_learning_rate, "final_learning_rate")
        if not 0.0 < final <= initial:
            raise ProtocolValidationError(
                "tuning learning rates must satisfy 0 < final <= initial",
            )
        is_phase = self.model_family is ModelFamily.CONDITION_MODULATED_PHASE
        if is_phase != (self.phase_feature_assignment is not None):
            raise ProtocolValidationError(
                "only phase-family candidates carry a phase feature",
            )
        if (
            self.phase_feature_assignment is not None
            and self.phase_feature_assignment not in PHASE_FEATURE_VOCABULARY
        ):
            raise ProtocolValidationError("unknown tuning phase feature")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "model_family": self.model_family.value,
            "initial_learning_rate": self.initial_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "phase_feature_assignment": self.phase_feature_assignment,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TuningCandidate:
        _require_exact_keys(
            value,
            {
                "candidate_id",
                "model_family",
                "initial_learning_rate",
                "final_learning_rate",
                "phase_feature_assignment",
            },
            "tuning_candidate",
        )
        try:
            family = ModelFamily(str(value["model_family"]))
        except ValueError as error:
            raise ProtocolValidationError("unknown tuning model family") from error
        feature = value["phase_feature_assignment"]
        if feature is not None and not isinstance(feature, str):
            raise ProtocolValidationError(
                "tuning phase_feature_assignment must be a string or null",
            )
        return cls(
            candidate_id=str(value["candidate_id"]),
            model_family=family,
            initial_learning_rate=_require_float(
                value["initial_learning_rate"],
                "initial_learning_rate",
            ),
            final_learning_rate=_require_float(
                value["final_learning_rate"],
                "final_learning_rate",
            ),
            phase_feature_assignment=feature,
        )


@dataclass(frozen=True)
class TrainingProtocol:
    """Frozen optimization, checkpoint, and equal-opportunity tuning rules."""

    batch_size: int
    max_updates: int
    optimizer: str
    reference_initial_learning_rate: float
    reference_final_learning_rate: float
    checkpoint_rule: str
    tuning_selection_rule: str
    tuning_coupling_seeds: tuple[int, ...]
    tuning_optimizer_seeds: tuple[int, ...]
    tuning_candidates: tuple[TuningCandidate, ...]

    def __post_init__(self) -> None:
        if (
            self.batch_size != 256
            or self.max_updates != 2_000
            or self.optimizer != OPTIMIZER_NAME
            or self.reference_initial_learning_rate != 1e-2
            or self.reference_final_learning_rate != 1e-3
            or self.checkpoint_rule != CHECKPOINT_RULE
            or self.tuning_selection_rule != TUNING_SELECTION_RULE
        ):
            raise ProtocolValidationError(
                "training budget, optimizer, and selection rules are not frozen",
            )
        if self.tuning_coupling_seeds != (666, 101):
            raise ProtocolValidationError(
                "tuning coupling seeds must be exactly (666, 101)",
            )
        if self.tuning_optimizer_seeds != (0, 1):
            raise ProtocolValidationError(
                "tuning optimizer seeds must be exactly (0, 1)",
            )
        signatures = tuple(
            (
                candidate.candidate_id,
                candidate.model_family,
                candidate.initial_learning_rate,
                candidate.final_learning_rate,
                candidate.phase_feature_assignment,
            )
            for candidate in self.tuning_candidates
        )
        if signatures != EXPECTED_TUNING_CANDIDATES:
            raise ProtocolValidationError(
                "tuning candidates must be the frozen equal 3/3/3 slices",
            )

    def validate_against(self, manifest: CampaignManifest) -> None:
        if not set(self.tuning_coupling_seeds).issubset(manifest.coupling_seeds):
            raise ProtocolValidationError(
                "tuning coupling seeds must be campaign coupling seeds",
            )
        if not set(self.tuning_optimizer_seeds).issubset(
            manifest.optimizer_seeds,
        ):
            raise ProtocolValidationError(
                "tuning optimizer seeds must be campaign optimizer seeds",
            )

    def candidate(self, candidate_id: str) -> TuningCandidate:
        """Resolve one immutable candidate by its globally unique identifier."""
        for candidate in self.tuning_candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise ProtocolValidationError(f"unknown tuning candidate: {candidate_id}")

    def candidates_for(self, family: ModelFamily) -> tuple[TuningCandidate, ...]:
        """Return the three candidates allocated to one shared family."""
        return tuple(
            candidate
            for candidate in self.tuning_candidates
            if candidate.model_family is family
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "max_updates": self.max_updates,
            "optimizer": self.optimizer,
            "reference_initial_learning_rate": self.reference_initial_learning_rate,
            "reference_final_learning_rate": self.reference_final_learning_rate,
            "checkpoint_rule": self.checkpoint_rule,
            "tuning_selection_rule": self.tuning_selection_rule,
            "tuning_coupling_seeds": list(self.tuning_coupling_seeds),
            "tuning_optimizer_seeds": list(self.tuning_optimizer_seeds),
            "tuning_candidates": [
                candidate.to_dict() for candidate in self.tuning_candidates
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingProtocol:
        _require_exact_keys(
            value,
            {
                "batch_size",
                "max_updates",
                "optimizer",
                "reference_initial_learning_rate",
                "reference_final_learning_rate",
                "checkpoint_rule",
                "tuning_selection_rule",
                "tuning_coupling_seeds",
                "tuning_optimizer_seeds",
                "tuning_candidates",
            },
            "training_protocol",
        )
        return cls(
            batch_size=_require_int(value["batch_size"], "batch_size"),
            max_updates=_require_int(value["max_updates"], "max_updates"),
            optimizer=str(value["optimizer"]),
            reference_initial_learning_rate=_require_float(
                value["reference_initial_learning_rate"],
                "reference_initial_learning_rate",
            ),
            reference_final_learning_rate=_require_float(
                value["reference_final_learning_rate"],
                "reference_final_learning_rate",
            ),
            checkpoint_rule=str(value["checkpoint_rule"]),
            tuning_selection_rule=str(value["tuning_selection_rule"]),
            tuning_coupling_seeds=tuple(
                _require_int(seed, "tuning coupling seed")
                for seed in value["tuning_coupling_seeds"]
            ),
            tuning_optimizer_seeds=tuple(
                _require_int(seed, "tuning optimizer seed")
                for seed in value["tuning_optimizer_seeds"]
            ),
            tuning_candidates=tuple(
                TuningCandidate.from_dict(candidate)
                for candidate in value["tuning_candidates"]
            ),
        )


@dataclass(frozen=True)
class RunAllocation:
    """The four non-transferable full-training-run budgets."""

    feasibility_representability: int
    tuning: int
    primary: int
    ancilla: int

    def __post_init__(self) -> None:
        if self.to_dict() != EXPECTED_RUN_ALLOCATION:
            raise ProtocolValidationError(
                "run allocation must be exactly 20/36/320/24",
            )
        if sum(self.to_dict().values()) != 400:
            raise ProtocolValidationError("run allocation must sum to 400")

    def to_dict(self) -> dict[str, int]:
        return {
            "feasibility_representability": self.feasibility_representability,
            "tuning": self.tuning,
            "primary": self.primary,
            "ancilla": self.ancilla,
        }

    def for_pool(self, pool: BudgetPool) -> int:
        return self.to_dict()[pool.value]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunAllocation:
        _require_exact_keys(value, set(EXPECTED_RUN_ALLOCATION), "run_allocation")
        return cls(
            feasibility_representability=_require_int(
                value["feasibility_representability"],
                "feasibility_representability allocation",
            ),
            tuning=_require_int(value["tuning"], "tuning allocation"),
            primary=_require_int(value["primary"], "primary allocation"),
            ancilla=_require_int(value["ancilla"], "ancilla allocation"),
        )


@dataclass(frozen=True)
class StatisticalProtocol:
    """Frozen primary statistical and empirical-gate constants."""

    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    representability_median_tvd_max: float
    representability_cell_tvd_max: float
    benefit_retention_median_min: float
    benefit_retention_lower_bound_min: float
    cell_win_fraction_min: float

    def __post_init__(self) -> None:
        expected = (10_000, 0.95, 0.25, 0.35, 0.5, 0.0, 0.75)
        actual = (
            self.bootstrap_replicates,
            self.confidence_level,
            self.representability_median_tvd_max,
            self.representability_cell_tvd_max,
            self.benefit_retention_median_min,
            self.benefit_retention_lower_bound_min,
            self.cell_win_fraction_min,
        )
        if actual != expected:
            raise ProtocolValidationError(
                "statistical thresholds are not frozen values",
            )
        _require_int(self.bootstrap_seed, "bootstrap_seed")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "representability_median_tvd_max": (self.representability_median_tvd_max),
            "representability_cell_tvd_max": self.representability_cell_tvd_max,
            "benefit_retention_median_min": self.benefit_retention_median_min,
            "benefit_retention_lower_bound_min": (
                self.benefit_retention_lower_bound_min
            ),
            "cell_win_fraction_min": self.cell_win_fraction_min,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StatisticalProtocol:
        expected = {
            "bootstrap_replicates",
            "bootstrap_seed",
            "confidence_level",
            "representability_median_tvd_max",
            "representability_cell_tvd_max",
            "benefit_retention_median_min",
            "benefit_retention_lower_bound_min",
            "cell_win_fraction_min",
        }
        _require_exact_keys(value, expected, "statistics")
        return cls(
            bootstrap_replicates=_require_int(
                value["bootstrap_replicates"],
                "bootstrap_replicates",
            ),
            bootstrap_seed=_require_int(value["bootstrap_seed"], "bootstrap_seed"),
            confidence_level=_require_float(
                value["confidence_level"],
                "confidence_level",
            ),
            representability_median_tvd_max=_require_float(
                value["representability_median_tvd_max"],
                "representability_median_tvd_max",
            ),
            representability_cell_tvd_max=_require_float(
                value["representability_cell_tvd_max"],
                "representability_cell_tvd_max",
            ),
            benefit_retention_median_min=_require_float(
                value["benefit_retention_median_min"],
                "benefit_retention_median_min",
            ),
            benefit_retention_lower_bound_min=_require_float(
                value["benefit_retention_lower_bound_min"],
                "benefit_retention_lower_bound_min",
            ),
            cell_win_fraction_min=_require_float(
                value["cell_win_fraction_min"],
                "cell_win_fraction_min",
            ),
        )


@dataclass(frozen=True)
class AncillaProtocol:
    """Frozen two-instance, three-seed, two-rung ancilla extension."""

    coupling_seeds: tuple[int, ...]
    optimizer_seeds: tuple[int, ...]
    rungs: tuple[int, ...]

    def validate_against(self, manifest: CampaignManifest) -> None:
        _require_unique_ints(self.coupling_seeds, 2, "ancilla coupling seeds")
        _require_unique_ints(self.optimizer_seeds, 3, "ancilla optimizer seeds")
        if 666 not in self.coupling_seeds:
            raise ProtocolValidationError("ancilla instances must include seed 666")
        if not set(self.coupling_seeds).issubset(manifest.coupling_seeds):
            raise ProtocolValidationError(
                "ancilla coupling seeds must be primary coupling seeds",
            )
        if not set(self.optimizer_seeds).issubset(manifest.optimizer_seeds):
            raise ProtocolValidationError(
                "ancilla optimizer seeds must be primary optimizer seeds",
            )
        if self.rungs != (3, 5):
            raise ProtocolValidationError("ancilla rungs must be exactly m=3 and m=5")

    def to_dict(self) -> dict[str, object]:
        return {
            "coupling_seeds": list(self.coupling_seeds),
            "optimizer_seeds": list(self.optimizer_seeds),
            "rungs": list(self.rungs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AncillaProtocol:
        _require_exact_keys(
            value,
            {"coupling_seeds", "optimizer_seeds", "rungs"},
            "ancilla",
        )
        return cls(
            coupling_seeds=tuple(
                _require_int(item, "ancilla coupling seed")
                for item in value["coupling_seeds"]
            ),
            optimizer_seeds=tuple(
                _require_int(item, "ancilla optimizer seed")
                for item in value["optimizer_seeds"]
            ),
            rungs=tuple(_require_int(item, "ancilla rung") for item in value["rungs"]),
        )


@dataclass(frozen=True)
class CampaignManifest:
    """Complete immutable production protocol for one campaign version."""

    schema_version: int
    campaign_version: str
    calibration: CalibrationProvenance
    coupling_seeds: tuple[int, ...]
    optimizer_seeds: tuple[int, ...]
    data_seeds: tuple[int, ...]
    beta_grid: BetaGrid
    model_protocol: ModelProtocol
    estimator_budget: EstimatorBudget
    sampler_protocol: SamplerProtocol
    training_protocol: TrainingProtocol
    run_allocation: RunAllocation
    statistics: StatisticalProtocol
    ancilla: AncillaProtocol

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ProtocolValidationError("manifest schema_version must be 1")
        if not self.campaign_version or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in self.campaign_version
        ):
            raise ProtocolValidationError(
                "campaign_version must use lowercase letters, digits, '_' or '-'",
            )
        _require_unique_ints(self.coupling_seeds, 8, "coupling seeds")
        _require_unique_ints(self.optimizer_seeds, 5, "optimizer seeds")
        _require_unique_ints(self.data_seeds, 8, "data seeds")
        if 666 not in self.coupling_seeds:
            raise ProtocolValidationError(
                "coupling seeds must include canonical seed 666",
            )
        self.ancilla.validate_against(self)
        self.training_protocol.validate_against(self)

    @property
    def digest(self) -> str:
        """Content digest locking every production choice in this manifest."""
        return content_digest(self.to_dict())

    def data_seed_for(self, coupling_seed: int) -> int:
        """Return the frozen data seed paired with one coupling instance."""
        try:
            index = self.coupling_seeds.index(coupling_seed)
        except ValueError as error:
            raise ProtocolValidationError(
                f"unknown coupling seed: {coupling_seed}",
            ) from error
        return self.data_seeds[index]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "campaign_version": self.campaign_version,
            "calibration": self.calibration.to_dict(),
            "coupling_seeds": list(self.coupling_seeds),
            "optimizer_seeds": list(self.optimizer_seeds),
            "data_seeds": list(self.data_seeds),
            "beta_grid": self.beta_grid.to_dict(),
            "model_protocol": self.model_protocol.to_dict(),
            "estimator_budget": self.estimator_budget.to_dict(),
            "sampler_protocol": self.sampler_protocol.to_dict(),
            "training_protocol": self.training_protocol.to_dict(),
            "run_allocation": self.run_allocation.to_dict(),
            "statistics": self.statistics.to_dict(),
            "ancilla": self.ancilla.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CampaignManifest:
        expected = {
            "schema_version",
            "campaign_version",
            "calibration",
            "coupling_seeds",
            "optimizer_seeds",
            "data_seeds",
            "beta_grid",
            "model_protocol",
            "estimator_budget",
            "sampler_protocol",
            "training_protocol",
            "run_allocation",
            "statistics",
            "ancilla",
        }
        _require_exact_keys(value, expected, "manifest")
        return cls(
            schema_version=_require_int(value["schema_version"], "schema_version"),
            campaign_version=str(value["campaign_version"]),
            calibration=CalibrationProvenance.from_dict(value["calibration"]),
            coupling_seeds=tuple(
                _require_int(item, "coupling seed") for item in value["coupling_seeds"]
            ),
            optimizer_seeds=tuple(
                _require_int(item, "optimizer seed")
                for item in value["optimizer_seeds"]
            ),
            data_seeds=tuple(
                _require_int(item, "data seed") for item in value["data_seeds"]
            ),
            beta_grid=BetaGrid.from_dict(value["beta_grid"]),
            model_protocol=ModelProtocol.from_dict(value["model_protocol"]),
            estimator_budget=EstimatorBudget.from_dict(value["estimator_budget"]),
            sampler_protocol=SamplerProtocol.from_dict(value["sampler_protocol"]),
            training_protocol=TrainingProtocol.from_dict(value["training_protocol"]),
            run_allocation=RunAllocation.from_dict(value["run_allocation"]),
            statistics=StatisticalProtocol.from_dict(value["statistics"]),
            ancilla=AncillaProtocol.from_dict(value["ancilla"]),
        )

    @classmethod
    def from_json(cls, path: Path) -> CampaignManifest:
        """Load and validate a manifest from JSON."""
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(
                f"could not load manifest {path}: {error}",
            ) from error
        if not isinstance(payload, dict):
            raise ProtocolValidationError("manifest JSON root must be an object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class RunIdentity:
    """Every scientific coordinate that determines one full training run."""

    campaign_version: str
    stage: Stage
    coupling_seed: int
    condition_scope: str
    model_family: ModelFamily
    phase_feature_assignment: str | None
    tuning_candidate_id: str | None
    ancilla_rung: int | None
    optimizer_seed: int
    data_seed: int
    configuration_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, Stage) or not isinstance(
            self.model_family,
            ModelFamily,
        ):
            raise ProtocolValidationError(
                "run stage and model_family must use typed protocol vocabulary",
            )
        if (
            not isinstance(self.campaign_version, str)
            or not self.campaign_version
            or not isinstance(self.condition_scope, str)
            or not self.condition_scope
        ):
            raise ProtocolValidationError(
                "run campaign_version and condition_scope must be non-empty",
            )
        _require_int(self.coupling_seed, "run coupling_seed")
        _require_int(self.optimizer_seed, "run optimizer_seed")
        _require_int(self.data_seed, "run data_seed")
        _require_digest(self.configuration_digest, "run configuration_digest")
        is_phase = self.model_family is ModelFamily.CONDITION_MODULATED_PHASE
        if is_phase != (self.phase_feature_assignment is not None):
            raise ProtocolValidationError(
                "phase feature is required only for Condition-modulated-phase runs",
            )
        if (
            self.phase_feature_assignment is not None
            and self.phase_feature_assignment not in PHASE_FEATURE_VOCABULARY
        ):
            raise ProtocolValidationError("run uses an unknown phase feature")
        if self.tuning_candidate_id is not None and (
            not isinstance(self.tuning_candidate_id, str)
            or not self.tuning_candidate_id
        ):
            raise ProtocolValidationError(
                "tuning_candidate_id must be a non-empty string or null",
            )
        if self.stage is Stage.TUNING:
            if (
                self.tuning_candidate_id is None
                or self.model_family not in SHARED_PRIMARY_FAMILIES
            ):
                raise ProtocolValidationError(
                    "tuning runs require a shared-family tuning candidate",
                )
        elif self.stage is Stage.PRIMARY:
            is_shared = self.model_family in SHARED_PRIMARY_FAMILIES
            if is_shared != (self.tuning_candidate_id is not None):
                raise ProtocolValidationError(
                    "primary shared runs, and only those runs, require a "
                    "tuning candidate",
                )
        elif self.stage is Stage.ANCILLA:
            if self.tuning_candidate_id is None:
                raise ProtocolValidationError(
                    "ancilla runs require the selected fixed-input candidate",
                )
        elif self.tuning_candidate_id is not None:
            raise ProtocolValidationError(
                "pre-tuning runs cannot carry a tuning candidate",
            )
        if self.stage is Stage.ANCILLA:
            if self.ancilla_rung not in (3, 5):
                raise ProtocolValidationError(
                    "ancilla run must identify rung m=3 or m=5",
                )
            if self.model_family not in {
                ModelFamily.FIXED_INPUT,
                ModelFamily.MATCHED_CAPACITY_CONTROL,
            }:
                raise ProtocolValidationError(
                    "ancilla run uses an invalid model family",
                )
        elif self.ancilla_rung is not None:
            raise ProtocolValidationError("only ancilla runs may carry an ancilla rung")

    @property
    def run_id(self) -> str:
        """Stable SHA-256 identity for filenames, ledgers, and provenance."""
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_version": self.campaign_version,
            "stage": self.stage.value,
            "coupling_seed": self.coupling_seed,
            "condition_scope": self.condition_scope,
            "model_family": self.model_family.value,
            "phase_feature_assignment": self.phase_feature_assignment,
            "tuning_candidate_id": self.tuning_candidate_id,
            "ancilla_rung": self.ancilla_rung,
            "optimizer_seed": self.optimizer_seed,
            "data_seed": self.data_seed,
            "configuration_digest": self.configuration_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunIdentity:
        expected = {
            "campaign_version",
            "stage",
            "coupling_seed",
            "condition_scope",
            "model_family",
            "phase_feature_assignment",
            "tuning_candidate_id",
            "ancilla_rung",
            "optimizer_seed",
            "data_seed",
            "configuration_digest",
        }
        _require_exact_keys(value, expected, "run identity")
        phase_feature = value["phase_feature_assignment"]
        if phase_feature is not None and not isinstance(phase_feature, str):
            raise ProtocolValidationError(
                "phase_feature_assignment must be a string or null",
            )
        tuning_candidate = value["tuning_candidate_id"]
        if tuning_candidate is not None and not isinstance(tuning_candidate, str):
            raise ProtocolValidationError(
                "tuning_candidate_id must be a string or null",
            )
        ancilla_rung = value["ancilla_rung"]
        if ancilla_rung is not None:
            ancilla_rung = _require_int(ancilla_rung, "ancilla_rung")
        try:
            stage = Stage(str(value["stage"]))
            model_family = ModelFamily(str(value["model_family"]))
        except ValueError as error:
            raise ProtocolValidationError(f"unknown run vocabulary: {error}") from error
        return cls(
            campaign_version=str(value["campaign_version"]),
            stage=stage,
            coupling_seed=_require_int(value["coupling_seed"], "run coupling_seed"),
            condition_scope=str(value["condition_scope"]),
            model_family=model_family,
            phase_feature_assignment=phase_feature,
            tuning_candidate_id=tuning_candidate,
            ancilla_rung=ancilla_rung,
            optimizer_seed=_require_int(value["optimizer_seed"], "run optimizer_seed"),
            data_seed=_require_int(value["data_seed"], "run data_seed"),
            configuration_digest=str(value["configuration_digest"]),
        )

"""Leakage-safe training and exact evaluation for the frozen Ising campaign.

This module is the production seam between the immutable campaign protocol and
the lower-level IQP simulator.  Training and checkpoint selection can only see
role-tagged Training and Validation samples.  Exact Held-out Targets enter
through a separate post-checkpoint evaluation interface.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ancilla import marginalize_ancillas
from .ansatz import Ansatz, complete_ansatz
from .artifacts import (
    CheckpointProvenance,
    CompletionStatus,
    ParameterAccounting,
    ResultBundle,
)
from .exact import (
    exact_iqp_probabilities,
    exact_iqp_probabilities_many,
    generator_characters,
)
from .ising import BetaGrid as IsingBetaGrid
from .ising import beta_to_broadcast_angles, exact_ising_target
from .mmd import conditional_mmd, exact_conditional_mmd, exact_target_fouriers
from .phase_conditioning import (
    N_PHASE_COEFFICIENTS,
    PHASE_FEATURE_MAPS,
    assert_fixed_input_contract,
    assert_no_condition_contract,
    assert_phase_conditioned_contract,
    exact_effective_phases,
    normalize_conditions,
    parameter_signature,
    sampled_effective_phases,
)
from .protocol import (
    CampaignManifest,
    ModelFamily,
    RunIdentity,
    Stage,
    canonical_json_bytes,
)
from .training import initialize_parameters

PRODUCTION_N_QUBITS = 16
PRODUCTION_N_STATES = 2**PRODUCTION_N_QUBITS
PRODUCTION_N_PARAMETERS = 136
PRODUCTION_BATCH_SIZE_PER_CONDITION = 256
PRODUCTION_OPERATORS_PER_BANDWIDTH = 2_000
PRODUCTION_REFERENCE_SAMPLES_PER_OPERATOR_CONDITION = 2_000
PRODUCTION_BANDWIDTHS = (1.0, 2.0, 4.0)
PRODUCTION_MAX_UPDATES = 2_000
PRODUCTION_VALIDATION_INTERVAL = 25
PRODUCTION_LEARNING_RATE_PAIRS = (
    (3e-3, 3e-4),
    (1e-2, 1e-3),
    (3e-2, 3e-3),
)
PRODUCTION_TRAINING_SAMPLES_PER_CONDITION = 5_000
PRODUCTION_VALIDATION_SAMPLES_PER_CONDITION = 2_000
REPRESENTABILITY_COUPLING_SEEDS = (666, 101)
REPRESENTABILITY_MEDIAN_TVD_MAX = 0.25
REPRESENTABILITY_CELL_TVD_MAX = 0.35

SUPPORTED_MODEL_FAMILIES = (
    ModelFamily.FIXED_INPUT,
    ModelFamily.CONDITION_MODULATED_PHASE,
    ModelFamily.NO_CONDITION,
    ModelFamily.SEPARATE,
    ModelFamily.MATCHED_CAPACITY_CONTROL,
)

MATCHED_CAPACITY_THREE_LOCAL_SEED = 12_345


class DatasetRole(StrEnum):
    """Data-access roles enforced at the training/evaluation boundary."""

    TRAINING = "training"
    VALIDATION = "validation"
    HELD_OUT = "held_out"


@dataclass(frozen=True, slots=True)
class RoleTaggedDataset:
    """One immutable finite-data Condition with an explicit access role."""

    role: DatasetRole
    coupling_seed: int
    beta: float
    data_seed: int
    samples: np.ndarray
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, DatasetRole):
            raise ValueError("dataset role must use the typed DatasetRole vocabulary")
        samples = np.asarray(self.samples, dtype=np.int8).copy()
        if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 1:
            raise ValueError("samples must have shape (at least 2, n_qubits)")
        if not np.all((samples == 0) | (samples == 1)):
            raise ValueError("samples must be binary")
        if not np.isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("beta must be finite and positive")
        if self.coupling_seed < 0 or self.data_seed < 0:
            raise ValueError("coupling_seed and data_seed must be non-negative")
        if self.source_sha256 is not None and not _is_sha256(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)

    @property
    def digest(self) -> str:
        """Return a role-sensitive content digest of this split and Condition."""
        digest = hashlib.sha256()
        digest.update(self.role.value.encode("ascii"))
        digest.update(np.asarray(self.coupling_seed, dtype=np.int64).tobytes())
        digest.update(np.asarray(self.beta, dtype=np.float64).tobytes())
        digest.update(np.asarray(self.data_seed, dtype=np.int64).tobytes())
        digest.update(repr(self.samples.shape).encode("ascii"))
        digest.update(self.samples.tobytes(order="C"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingValidationData:
    """Matched Training/Validation Conditions with no Held-out access path."""

    training: tuple[RoleTaggedDataset, ...]
    validation: tuple[RoleTaggedDataset, ...]

    def __post_init__(self) -> None:
        if not self.training or not self.validation:
            raise ValueError("Training and Validation datasets must not be empty")
        if any(item.role is not DatasetRole.TRAINING for item in self.training):
            raise ValueError("Training inputs must be tagged role='training'")
        if any(item.role is not DatasetRole.VALIDATION for item in self.validation):
            raise ValueError(
                "checkpoint selection accepts role='validation' data only; "
                "Held-out access is forbidden",
            )
        if len(self.training) != len(self.validation):
            raise ValueError("Training and Validation Conditions must align")
        training_betas = tuple(item.beta for item in self.training)
        validation_betas = tuple(item.beta for item in self.validation)
        if training_betas != validation_betas or len(set(training_betas)) != len(
            training_betas,
        ):
            raise ValueError("Training and Validation betas must align uniquely")
        coupling_seeds = {
            item.coupling_seed for item in (*self.training, *self.validation)
        }
        qubit_counts = {
            item.samples.shape[1] for item in (*self.training, *self.validation)
        }
        if len(coupling_seeds) != 1 or len(qubit_counts) != 1:
            raise ValueError(
                "all dataset Conditions must share instance and qubit count",
            )
        for training, validation in zip(
            self.training,
            self.validation,
            strict=True,
        ):
            if training.data_seed == validation.data_seed:
                raise ValueError("Training and Validation streams must be independent")

    @property
    def betas(self) -> tuple[float, ...]:
        return tuple(item.beta for item in self.training)

    @property
    def coupling_seed(self) -> int:
        return self.training[0].coupling_seed

    @property
    def n_qubits(self) -> int:
        return self.training[0].samples.shape[1]

    def stacked_training(self) -> np.ndarray:
        counts = {item.samples.shape[0] for item in self.training}
        if len(counts) != 1:
            raise ValueError("Training Conditions must have equal sample counts")
        return np.stack([item.samples for item in self.training])

    def stacked_validation(self) -> np.ndarray:
        counts = {item.samples.shape[0] for item in self.validation}
        if len(counts) != 1:
            raise ValueError("Validation Conditions must have equal sample counts")
        return np.stack([item.samples for item in self.validation])

    def dataset_hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for split in (self.training, self.validation):
            for index, item in enumerate(split):
                hashes[f"{item.role.value}_beta_{index}"] = item.digest
                if item.source_sha256 is not None:
                    hashes[f"{item.role.value}_source_beta_{index}"] = (
                        item.source_sha256
                    )
        return hashes


@dataclass(frozen=True, slots=True)
class EstimatorProtocol:
    """Sampled Conditional-MMD budget and declared bandwidths."""

    bandwidths: tuple[float, ...] = PRODUCTION_BANDWIDTHS
    operators_per_bandwidth: int = PRODUCTION_OPERATORS_PER_BANDWIDTH
    reference_samples_per_operator_condition: int = (
        PRODUCTION_REFERENCE_SAMPLES_PER_OPERATOR_CONDITION
    )
    production: bool = True

    def __post_init__(self) -> None:
        if not self.bandwidths or any(
            not np.isfinite(value) or value <= 0.0 for value in self.bandwidths
        ):
            raise ValueError("bandwidths must be finite, positive, and non-empty")
        if len(set(self.bandwidths)) != len(self.bandwidths):
            raise ValueError("bandwidths must not contain duplicates")
        if (
            min(
                self.operators_per_bandwidth,
                self.reference_samples_per_operator_condition,
            )
            < 2
        ):
            raise ValueError("unbiased sampled MMD needs at least two samples")
        if self.production and (
            self.bandwidths != PRODUCTION_BANDWIDTHS
            or self.operators_per_bandwidth != PRODUCTION_OPERATORS_PER_BANDWIDTH
            or self.reference_samples_per_operator_condition
            != PRODUCTION_REFERENCE_SAMPLES_PER_OPERATOR_CONDITION
        ):
            raise ValueError(
                "production objectives require all frozen bandwidths and the "
                "2,000-by-2,000 estimator budget",
            )

    @classmethod
    def from_manifest(cls, manifest: CampaignManifest) -> EstimatorProtocol:
        """Bind the numerical estimator budget to the immutable manifest."""
        budget = manifest.estimator_budget
        manifest_bandwidths = tuple(budget.bandwidths)
        return cls(
            bandwidths=manifest_bandwidths,
            operators_per_bandwidth=budget.operators_per_bandwidth,
            reference_samples_per_operator_condition=(
                budget.reference_samples_per_operator_condition
            ),
            production=True,
        )

    @classmethod
    def testing(
        cls,
        *,
        bandwidths: tuple[float, ...] = (1.0,),
        operators_per_bandwidth: int = 8,
        reference_samples_per_operator_condition: int = 8,
    ) -> EstimatorProtocol:
        """Construct an explicit non-production budget for bounded tests."""
        return cls(
            bandwidths=bandwidths,
            operators_per_bandwidth=operators_per_bandwidth,
            reference_samples_per_operator_condition=(
                reference_samples_per_operator_condition
            ),
            production=False,
        )


@dataclass(frozen=True, slots=True)
class OptimizationProtocol:
    """Frozen optimizer schedule; the loop never performs extra updates."""

    max_updates: int = PRODUCTION_MAX_UPDATES
    batch_size_per_condition: int = PRODUCTION_BATCH_SIZE_PER_CONDITION
    learning_rate: float = 1e-2
    final_learning_rate: float = 1e-3
    validation_interval: int = PRODUCTION_VALIDATION_INTERVAL
    production: bool = True

    def __post_init__(self) -> None:
        if (
            min(
                self.max_updates,
                self.batch_size_per_condition,
                self.validation_interval,
            )
            < 1
        ):
            raise ValueError(
                "updates, batch size, and validation interval must be positive",
            )
        if (
            not np.isfinite(self.learning_rate)
            or not np.isfinite(self.final_learning_rate)
            or self.learning_rate <= 0.0
            or self.final_learning_rate <= 0.0
            or self.final_learning_rate > self.learning_rate
        ):
            raise ValueError("learning rates must be positive with final <= initial")
        if self.production and self.batch_size_per_condition != (
            PRODUCTION_BATCH_SIZE_PER_CONDITION
        ):
            raise ValueError(
                "production requires 256 samples from every Condition per minibatch",
            )
        if self.production and (
            self.max_updates != PRODUCTION_MAX_UPDATES
            or self.validation_interval != PRODUCTION_VALIDATION_INTERVAL
            or (self.learning_rate, self.final_learning_rate)
            not in PRODUCTION_LEARNING_RATE_PAIRS
        ):
            raise ValueError(
                "production requires the frozen update, Validation, and "
                "learning-rate schedule",
            )

    @classmethod
    def for_run(
        cls,
        manifest: CampaignManifest,
        run: RunIdentity,
    ) -> OptimizationProtocol:
        """Resolve frozen learning rates and update limits for a declared run."""
        training = manifest.training_protocol
        if run.configuration_digest != manifest.digest:
            raise ValueError("run identity belongs to a different manifest")
        if run.tuning_candidate_id is None:
            initial = training.reference_initial_learning_rate
            final = training.reference_final_learning_rate
        else:
            candidate = training.candidate(run.tuning_candidate_id)
            if candidate.model_family is not run.model_family:
                # Ancilla Matched-capacity controls reuse the fixed-input winner;
                # this visible-only runner intentionally does not execute ancillas.
                if run.model_family is not ModelFamily.MATCHED_CAPACITY_CONTROL:
                    raise ValueError("run and tuning candidate families differ")
            initial = candidate.initial_learning_rate
            final = candidate.final_learning_rate
        return cls(
            max_updates=training.max_updates,
            batch_size_per_condition=training.batch_size,
            learning_rate=initial,
            final_learning_rate=final,
            validation_interval=PRODUCTION_VALIDATION_INTERVAL,
            production=True,
        )

    @classmethod
    def testing(
        cls,
        *,
        max_updates: int = 2,
        batch_size_per_condition: int = 4,
        validation_interval: int = 1,
    ) -> OptimizationProtocol:
        """Construct a bounded non-production optimizer schedule."""
        return cls(
            max_updates=max_updates,
            batch_size_per_condition=batch_size_per_condition,
            validation_interval=validation_interval,
            production=False,
        )


@dataclass(frozen=True, slots=True)
class EstimatorStreams:
    """Independent deterministic streams for batches, Training, and Validation."""

    minibatches: int
    training_objective: int
    validation_objective: int

    def __post_init__(self) -> None:
        values = (self.minibatches, self.training_objective, self.validation_objective)
        if any(value < 0 or value >= 2**32 for value in values):
            raise ValueError("estimator seeds must be unsigned 32-bit integers")
        if len(set(values)) != len(values):
            raise ValueError("estimator streams must use distinct seeds")

    @classmethod
    def for_run(cls, run: RunIdentity) -> EstimatorStreams:
        """Derive stable streams from the complete declared run identity."""

        def derive(label: str) -> int:
            payload = f"{run.run_id}|{label}".encode("ascii")
            return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

        values = []
        for label in ("minibatches", "training", "validation"):
            nonce = 0
            value = derive(label)
            while value in values:
                nonce += 1
                value = derive(f"{label}-{nonce}")
            values.append(value)
        return cls(*values)

    def to_dict(self) -> dict[str, int]:
        return {
            "minibatches": self.minibatches,
            "training_objective": self.training_objective,
            "validation_objective": self.validation_objective,
        }


@dataclass(frozen=True, slots=True)
class ModelDesign:
    """Frozen model-family semantics at a declared set of Training betas."""

    family: ModelFamily
    betas: tuple[float, ...]
    condition_angles: np.ndarray
    normalized_conditions: np.ndarray
    phase_feature: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.family, ModelFamily):
            raise ValueError("family must use the typed ModelFamily vocabulary")
        if self.family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(f"unsupported Ising model family: {self.family.value}")
        if not self.betas or len(set(self.betas)) != len(self.betas):
            raise ValueError("model betas must be non-empty and unique")
        angles = np.asarray(self.condition_angles, dtype=np.float64).copy()
        normalized = np.asarray(self.normalized_conditions, dtype=np.float64).copy()
        if angles.ndim != 2 or angles.shape[0] != len(self.betas):
            raise ValueError("condition_angles must align with model betas")
        if normalized.shape != (len(self.betas),):
            raise ValueError("normalized_conditions must align with model betas")
        if not np.all(np.isfinite(angles)) or not np.all(
            (normalized >= 0.0) & (normalized <= 1.0),
        ):
            raise ValueError("model inputs must be finite and normalized")
        is_phase = self.family is ModelFamily.CONDITION_MODULATED_PHASE
        if is_phase != (self.phase_feature is not None):
            raise ValueError("phase_feature is required only for the phase family")
        if (
            self.phase_feature is not None
            and self.phase_feature not in PHASE_FEATURE_MAPS
        ):
            raise ValueError("unknown preregistered phase feature")
        if self.family is ModelFamily.SEPARATE and len(self.betas) != 1:
            raise ValueError("a Separate-model run contains exactly one Training beta")
        angles.setflags(write=False)
        normalized.setflags(write=False)
        object.__setattr__(self, "condition_angles", angles)
        object.__setattr__(self, "normalized_conditions", normalized)

    @property
    def n_qubits(self) -> int:
        return self.condition_angles.shape[1]


@dataclass(frozen=True, slots=True)
class FeasibilityTelemetry:
    """Wall-clock and process peak-RSS telemetry for one full run."""

    runtime_seconds: float
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.runtime_seconds) or self.runtime_seconds < 0.0:
            raise ValueError("runtime_seconds must be finite and non-negative")
        if self.peak_rss_bytes < 0:
            raise ValueError("peak_rss_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Complete bounded update and Validation-evaluation history."""

    train_steps: np.ndarray
    train_losses: np.ndarray
    validation_steps: np.ndarray
    validation_losses: np.ndarray

    def __post_init__(self) -> None:
        arrays = []
        for values in (
            self.train_steps,
            self.train_losses,
            self.validation_steps,
            self.validation_losses,
        ):
            array = np.asarray(values).copy()
            if array.ndim != 1:
                raise ValueError("history arrays must be one-dimensional")
            array.setflags(write=False)
            arrays.append(array)
        if arrays[0].size != arrays[1].size or arrays[2].size != arrays[3].size:
            raise ValueError("history step and loss arrays must align")
        object.__setattr__(self, "train_steps", arrays[0])
        object.__setattr__(self, "train_losses", arrays[1])
        object.__setattr__(self, "validation_steps", arrays[2])
        object.__setattr__(self, "validation_losses", arrays[3])


@dataclass(frozen=True, slots=True)
class IsingTrainingResult:
    """Validation-selected checkpoint and auditable production history."""

    design: ModelDesign
    coupling_seed: int
    optimizer_seed: int
    generators: np.ndarray
    best_parameters: np.ndarray
    final_parameters: np.ndarray
    best_step: int
    best_validation_loss: float
    history: TrainingHistory
    estimator: EstimatorProtocol
    optimization: OptimizationProtocol
    streams: EstimatorStreams
    telemetry: FeasibilityTelemetry

    def __post_init__(self) -> None:
        generators = np.asarray(self.generators, dtype=np.int8).copy()
        best = np.asarray(self.best_parameters, dtype=np.float64).copy()
        final = np.asarray(self.final_parameters, dtype=np.float64).copy()
        if (
            generators.ndim != 2
            or best.shape != final.shape
            or best.shape != (generators.shape[0],)
        ):
            raise ValueError("result generators and parameter vectors must align")
        if self.coupling_seed < 0 or self.optimizer_seed < 0:
            raise ValueError("result seeds must be non-negative")
        if self.best_step not in set(self.history.validation_steps.tolist()):
            raise ValueError("best checkpoint must be a recorded Validation step")
        if len(self.history.train_steps) != self.optimization.max_updates:
            raise ValueError("training history must use exactly the frozen max updates")
        for array in (generators, best, final):
            array.setflags(write=False)
        object.__setattr__(self, "generators", generators)
        object.__setattr__(self, "best_parameters", best)
        object.__setattr__(self, "final_parameters", final)


@dataclass(frozen=True, slots=True)
class ExactTargetSet:
    """Post-checkpoint exact target laws for one role and coupling instance."""

    role: DatasetRole
    coupling_seed: int
    betas: tuple[float, ...]
    probabilities: np.ndarray
    source_sha256: str

    def __post_init__(self) -> None:
        if self.role not in {DatasetRole.TRAINING, DatasetRole.HELD_OUT}:
            raise ValueError("exact targets must be tagged Training or Held-out")
        if self.coupling_seed < 0 or any(
            not np.isfinite(beta) or beta <= 0.0 for beta in self.betas
        ):
            raise ValueError("exact Target instance and betas must be valid")
        probabilities = np.asarray(self.probabilities, dtype=np.float64).copy()
        if probabilities.ndim != 2 or probabilities.shape[0] != len(self.betas):
            raise ValueError("exact probabilities must align with betas")
        if not self.betas or len(set(self.betas)) != len(self.betas):
            raise ValueError("exact target betas must be non-empty and unique")
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
        ):
            raise ValueError(
                "each exact target row must be a normalized probability law",
            )
        if not _is_sha256(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        probabilities.setflags(write=False)
        object.__setattr__(self, "probabilities", probabilities)


@dataclass(frozen=True, slots=True)
class ExactTVDRecord:
    """Exact full-distribution TVD for one post-checkpoint evaluation cell."""

    role: DatasetRole
    beta: float
    tvd: float
    n_states: int

    def __post_init__(self) -> None:
        if self.role not in {DatasetRole.TRAINING, DatasetRole.HELD_OUT}:
            raise ValueError("exact TVD role must be Training or Held-out")
        if not np.isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("exact TVD beta must be finite and positive")
        if not np.isfinite(self.tvd) or not 0.0 <= self.tvd <= 1.0:
            raise ValueError("exact TVD must lie in [0, 1]")
        if self.n_states < 2:
            raise ValueError("exact TVD must compare at least two states")


@dataclass(frozen=True, slots=True)
class RepresentabilityCell:
    """One preregistered Separate-model representability result."""

    coupling_seed: int
    beta: float
    tvd: float

    def __post_init__(self) -> None:
        if self.coupling_seed not in REPRESENTABILITY_COUPLING_SEEDS:
            raise ValueError("representability uses only coupling seeds 666 and 101")
        if not np.isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("representability beta must be finite and positive")
        if not np.isfinite(self.tvd) or not 0.0 <= self.tvd <= 1.0:
            raise ValueError("representability TVD must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class RepresentabilityResult:
    """Exact six-cell representability decision at frozen boundaries."""

    cells: tuple[RepresentabilityCell, ...]
    median_tvd: float
    maximum_tvd: float
    median_rule_passed: bool
    every_cell_rule_passed: bool
    passed: bool


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _campaign_grid(manifest: CampaignManifest) -> IsingBetaGrid:
    return IsingBetaGrid(
        training=manifest.beta_grid.training,
        heldout=manifest.beta_grid.held_out,
        provenance=manifest.calibration.method,
    )


def build_model_design(
    family: ModelFamily,
    betas: Sequence[float],
    grid: IsingBetaGrid,
    *,
    phase_feature: str | None = None,
    n_qubits: int = PRODUCTION_N_QUBITS,
) -> ModelDesign:
    """Construct the frozen deployment semantics for one model family."""
    beta_values = tuple(float(beta) for beta in betas)
    normalized = normalize_conditions(
        np.asarray(beta_values),
        grid.training[0],
        grid.training[-1],
    )
    varying_angles = beta_to_broadcast_angles(
        np.asarray(beta_values),
        grid,
        n_qubits,
    )
    anchor_angle = beta_to_broadcast_angles(grid.anchor, grid, n_qubits)
    if family is ModelFamily.FIXED_INPUT:
        angles = varying_angles
    elif family in {ModelFamily.CONDITION_MODULATED_PHASE, ModelFamily.NO_CONDITION}:
        angles = np.broadcast_to(anchor_angle, (len(beta_values), n_qubits)).copy()
    elif family in {ModelFamily.SEPARATE, ModelFamily.MATCHED_CAPACITY_CONTROL}:
        angles = varying_angles
    else:
        raise ValueError(f"unsupported Ising model family: {family.value}")
    return ModelDesign(
        family=family,
        betas=beta_values,
        condition_angles=angles,
        normalized_conditions=normalized,
        phase_feature=phase_feature,
    )


def build_model_design_for_run(
    manifest: CampaignManifest,
    run: RunIdentity,
) -> ModelDesign:
    """Resolve Conditions, inputs, and phase assignment from a declared run."""
    if (
        run.configuration_digest != manifest.digest
        or run.campaign_version != manifest.campaign_version
    ):
        raise ValueError("run identity belongs to a different campaign manifest")
    if run.model_family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError("the Ising runner cannot execute this family")
    if run.condition_scope == "all_training_betas":
        betas = manifest.beta_grid.training
    elif run.condition_scope.startswith("train_beta_"):
        try:
            index = int(run.condition_scope.removeprefix("train_beta_"))
            betas = (manifest.beta_grid.training[index],)
        except (ValueError, IndexError) as error:
            raise ValueError("run uses an invalid Training-beta scope") from error
    else:
        raise ValueError("run uses an unsupported Condition scope")
    n_qubits = PRODUCTION_N_QUBITS
    if run.stage is Stage.ANCILLA and run.model_family is ModelFamily.FIXED_INPUT:
        if run.ancilla_rung is None:
            raise ValueError("ancilla run has no declared rung")
        n_qubits += run.ancilla_rung
    return build_model_design(
        run.model_family,
        betas,
        _campaign_grid(manifest),
        phase_feature=run.phase_feature_assignment,
        n_qubits=n_qubits,
    )


def matched_capacity_ansatz(
    n_visible_qubits: int,
    ancilla_rung: int,
) -> Ansatz:
    """Match an ancilla rung with visible two- plus deterministic three-local terms.

    Duplicate diagonal generators would collapse into a single effective phase,
    so the visible control extends the Complete two-local bank with distinct
    three-local masks.  Its trainable coefficient count exactly equals that of
    a Complete two-local circuit on ``n_visible_qubits + ancilla_rung`` qubits.
    """
    if n_visible_qubits < 3:
        raise ValueError("matched-capacity controls need at least three qubits")
    if ancilla_rung not in (3, 5):
        raise ValueError("campaign matched-capacity rungs are exactly m=3 and m=5")
    base = complete_ansatz(n_visible_qubits)
    target_qubits = n_visible_qubits + ancilla_rung
    target_parameters = target_qubits * (target_qubits + 1) // 2
    extra_count = target_parameters - base.n_parameters
    triples = list(combinations(range(n_visible_qubits), 3))
    if extra_count > len(triples):
        raise ValueError("not enough distinct three-local visible generators")
    rng = np.random.default_rng(MATCHED_CAPACITY_THREE_LOCAL_SEED)
    selected = [triples[index] for index in rng.permutation(len(triples))[:extra_count]]
    extra = np.zeros((extra_count, n_visible_qubits), dtype=np.float64)
    for row, triple in zip(extra, selected, strict=True):
        row[list(triple)] = 1.0
    return Ansatz(
        name=f"matched_capacity_m{ancilla_rung}",
        graph=base.graph.copy(),
        generators=jnp.asarray(
            np.concatenate((np.asarray(base.generators), extra), axis=0),
            dtype=jnp.float64,
        ),
    )


def ansatz_for_run(run: RunIdentity) -> Ansatz:
    """Return the unique frozen topology declared by a campaign run identity."""
    if run.stage is not Stage.ANCILLA:
        return complete_ansatz(PRODUCTION_N_QUBITS)
    if run.ancilla_rung is None:
        raise ValueError("ancilla run has no declared rung")
    if run.model_family is ModelFamily.FIXED_INPUT:
        return complete_ansatz(PRODUCTION_N_QUBITS + run.ancilla_rung)
    if run.model_family is ModelFamily.MATCHED_CAPACITY_CONTROL:
        return matched_capacity_ansatz(PRODUCTION_N_QUBITS, run.ancilla_rung)
    raise ValueError("ancilla run uses an unsupported family")


def balanced_minibatch_indices(
    data: TrainingValidationData,
    optimization: OptimizationProtocol,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Precompute exactly ``batch_size`` indices for every Condition and step."""
    rng = np.random.default_rng(seed)
    schedules = []
    for dataset in data.training:
        schedules.append(
            rng.integers(
                0,
                dataset.samples.shape[0],
                size=(
                    optimization.max_updates,
                    optimization.batch_size_per_condition,
                ),
            ),
        )
    return tuple(schedules)


def condition_balanced_minibatch(
    data: TrainingValidationData,
    schedules: tuple[np.ndarray, ...],
    step: int,
) -> np.ndarray:
    """Return one equally sized sample block from every Training Condition."""
    if len(schedules) != len(data.training):
        raise ValueError("minibatch schedules must align with Training Conditions")
    return np.stack(
        [
            dataset.samples[schedule[step]]
            for dataset, schedule in zip(data.training, schedules, strict=True)
        ],
    )


def sampled_training_objective(
    parameters: jnp.ndarray,
    generators: jnp.ndarray,
    design: ModelDesign,
    targets: jnp.ndarray,
    estimator: EstimatorProtocol,
    key: jax.Array,
) -> jnp.ndarray:
    """Evaluate a differentiable, deterministic-stream sampled objective."""
    if targets.ndim != 3 or targets.shape[0] != len(design.betas):
        raise ValueError("targets must align with the model's Training Conditions")
    if design.family is ModelFamily.CONDITION_MODULATED_PHASE:
        effective = sampled_effective_phases(
            parameters,
            design.normalized_conditions,
            design.phase_feature,
        )
        losses = []
        # Reusing the key intentionally pairs operator masks across Conditions.
        for condition in range(len(design.betas)):
            losses.append(
                conditional_mmd(
                    effective[condition],
                    generators,
                    jnp.asarray(design.condition_angles[[condition]]),
                    targets[condition : condition + 1],
                    estimator.bandwidths,
                    estimator.operators_per_bandwidth,
                    estimator.reference_samples_per_operator_condition,
                    key,
                ).loss,
            )
        return jnp.mean(jnp.stack(losses))
    return conditional_mmd(
        parameters,
        generators,
        jnp.asarray(design.condition_angles),
        targets,
        estimator.bandwidths,
        estimator.operators_per_bandwidth,
        estimator.reference_samples_per_operator_condition,
        key,
    ).loss


def _validate_training_contract(
    ansatz: Ansatz,
    design: ModelDesign,
    data: TrainingValidationData,
    estimator: EstimatorProtocol,
    optimization: OptimizationProtocol,
    run: RunIdentity | None,
) -> None:
    if design.betas != data.betas:
        raise ValueError("model design and dataset Conditions must align exactly")
    if ansatz.n_qubits != design.n_qubits or data.n_qubits > design.n_qubits:
        raise ValueError(
            "ansatz and model inputs must align and include every visible data qubit",
        )
    if estimator.production != optimization.production:
        raise ValueError("estimator and optimizer must agree on production mode")
    if optimization.production:
        if data.n_qubits != PRODUCTION_N_QUBITS:
            raise ValueError(
                "production requires the 16-qubit, 136-generator visible baseline",
            )
        if run is None:
            expected_ansatz = complete_ansatz(PRODUCTION_N_QUBITS)
        else:
            if run.configuration_digest == "":
                raise ValueError("production run identity has no manifest digest")
            if run.model_family is not design.family:
                raise ValueError("run identity and model design families differ")
            expected_ansatz = ansatz_for_run(run)
        if (
            ansatz.n_qubits != expected_ansatz.n_qubits
            or ansatz.n_parameters != expected_ansatz.n_parameters
            or design.n_qubits != expected_ansatz.n_qubits
        ):
            raise ValueError(
                "production model dimensions do not match the declared topology",
            )
        expected_generators = np.asarray(expected_ansatz.generators)
        if not np.array_equal(np.asarray(ansatz.generators), expected_generators):
            raise ValueError("production requires the frozen declared generator bank")
        expected_conditions = 1 if design.family is ModelFamily.SEPARATE else 5
        if len(design.betas) != expected_conditions:
            raise ValueError("production shared/Separate condition counts are frozen")
        if any(
            item.samples.shape
            != (
                PRODUCTION_TRAINING_SAMPLES_PER_CONDITION,
                PRODUCTION_N_QUBITS,
            )
            for item in data.training
        ):
            raise ValueError(
                "production Training cells must contain exactly 5,000 strings",
            )
        if any(
            item.samples.shape
            != (
                PRODUCTION_VALIDATION_SAMPLES_PER_CONDITION,
                PRODUCTION_N_QUBITS,
            )
            for item in data.validation
        ):
            raise ValueError(
                "production Validation cells must contain exactly 2,000 strings",
            )


def process_peak_rss_bytes() -> int:
    """Return the process peak resident-set size using platform-correct units."""
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _assert_result_deployment(result: IsingTrainingResult) -> None:
    design = result.design
    parameters = result.best_parameters
    generators = result.generators
    if parameters.size == N_PHASE_COEFFICIENTS:
        parameter_signature(parameters, generators)
    elif parameters.shape != (generators.shape[0],):
        raise ValueError("deployment parameters and generators must align")
    if (
        design.family is ModelFamily.FIXED_INPUT
        and parameters.size == N_PHASE_COEFFICIENTS
    ):
        assert_fixed_input_contract(parameters, generators, design.condition_angles)
    elif design.family is ModelFamily.NO_CONDITION:
        assert_no_condition_contract(parameters, generators, design.condition_angles)
    elif design.family is ModelFamily.CONDITION_MODULATED_PHASE:
        assert_phase_conditioned_contract(
            parameters,
            generators,
            design.normalized_conditions,
            design.condition_angles[0],
            design.phase_feature,
        )
    elif design.family not in {
        ModelFamily.FIXED_INPUT,
        ModelFamily.MATCHED_CAPACITY_CONTROL,
        ModelFamily.SEPARATE,
    }:
        raise ValueError("unknown production deployment family")


def train_ising_model(
    design: ModelDesign,
    data: TrainingValidationData,
    optimizer_seed: int,
    *,
    estimator: EstimatorProtocol,
    optimization: OptimizationProtocol,
    streams: EstimatorStreams,
    ansatz: Ansatz | None = None,
    run: RunIdentity | None = None,
) -> IsingTrainingResult:
    """Train for exactly the frozen updates and select by Validation only."""
    if optimizer_seed < 0:
        raise ValueError("optimizer_seed must be non-negative")
    if ansatz is None:
        ansatz = complete_ansatz(PRODUCTION_N_QUBITS)
    _validate_training_contract(
        ansatz,
        design,
        data,
        estimator,
        optimization,
        run,
    )

    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    parameters = initialize_parameters(ansatz.n_parameters, optimizer_seed)
    learning_rate = optax.cosine_decay_schedule(
        init_value=optimization.learning_rate,
        decay_steps=optimization.max_updates,
        alpha=optimization.final_learning_rate / optimization.learning_rate,
    )
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(parameters)
    schedules = balanced_minibatch_indices(data, optimization, streams.minibatches)
    validation_targets = jnp.asarray(data.stacked_validation(), dtype=jnp.float64)
    training_key = jax.random.PRNGKey(streams.training_objective)
    validation_key = jax.random.PRNGKey(streams.validation_objective)

    def objective(
        current: jnp.ndarray,
        targets: jnp.ndarray,
        key: jax.Array,
    ) -> jnp.ndarray:
        return sampled_training_objective(
            current,
            generators,
            design,
            targets,
            estimator,
            key,
        )

    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    validation_objective = jax.jit(
        lambda current: objective(current, validation_targets, validation_key),
    )

    started = time.perf_counter()
    best_parameters = parameters
    best_step = 0
    best_validation = float(validation_objective(parameters))
    train_steps: list[int] = []
    train_losses: list[float] = []
    validation_steps = [0]
    validation_losses = [best_validation]

    for step in range(optimization.max_updates):
        batch = condition_balanced_minibatch(data, schedules, step)
        step_key = jax.random.fold_in(training_key, step)
        loss, gradient = value_and_gradient(
            parameters,
            jnp.asarray(batch, dtype=jnp.float64),
            step_key,
        )
        updates, optimizer_state = optimizer.update(
            gradient,
            optimizer_state,
            parameters,
        )
        parameters = optax.apply_updates(parameters, updates)
        completed = step + 1
        train_steps.append(completed)
        train_losses.append(float(loss))
        if (
            completed % optimization.validation_interval == 0
            or completed == optimization.max_updates
        ):
            validation = float(validation_objective(parameters))
            validation_steps.append(completed)
            validation_losses.append(validation)
            if validation < best_validation:
                best_validation = validation
                best_parameters = parameters
                best_step = completed

    telemetry = FeasibilityTelemetry(
        runtime_seconds=time.perf_counter() - started,
        peak_rss_bytes=process_peak_rss_bytes(),
    )
    result = IsingTrainingResult(
        design=design,
        coupling_seed=data.coupling_seed,
        optimizer_seed=optimizer_seed,
        generators=np.asarray(ansatz.generators),
        best_parameters=np.asarray(best_parameters),
        final_parameters=np.asarray(parameters),
        best_step=best_step,
        best_validation_loss=best_validation,
        history=TrainingHistory(
            train_steps=np.asarray(train_steps, dtype=np.int64),
            train_losses=np.asarray(train_losses, dtype=np.float64),
            validation_steps=np.asarray(validation_steps, dtype=np.int64),
            validation_losses=np.asarray(validation_losses, dtype=np.float64),
        ),
        estimator=estimator,
        optimization=optimization,
        streams=streams,
        telemetry=telemetry,
    )
    if optimization.production:
        _assert_result_deployment(result)
    return result


def train_exact_objective_diagnostic(
    design: ModelDesign,
    data: TrainingValidationData,
    target: ExactTargetSet,
    optimizer_seed: int,
    *,
    estimator: EstimatorProtocol,
    optimization: OptimizationProtocol,
    streams: EstimatorStreams,
    ansatz: Ansatz | None = None,
    run: RunIdentity | None = None,
) -> IsingTrainingResult:
    """Run the sole exact-objective representability diagnostic.

    This path deliberately uses the complete exact Target law in the objective
    and never accesses Held-out Conditions.  It is a diagnostic for estimator
    versus model-family limitation, not an alternative production objective.
    """
    if design.family is not ModelFamily.SEPARATE or len(design.betas) != 1:
        raise ValueError("the exact diagnostic requires one Separate-model cell")
    if target.role is not DatasetRole.TRAINING:
        raise ValueError("the exact diagnostic cannot access Held-out Targets")
    if (
        target.coupling_seed != data.coupling_seed
        or target.betas != design.betas
        or data.betas != design.betas
    ):
        raise ValueError("diagnostic design, data, and exact Target must align")
    if optimizer_seed < 0:
        raise ValueError("optimizer_seed must be non-negative")
    if optimization.production and target.probabilities.shape != (
        1,
        PRODUCTION_N_STATES,
    ):
        raise ValueError("production exact diagnostic requires all 65,536 states")
    if ansatz is None:
        ansatz = complete_ansatz(PRODUCTION_N_QUBITS)
    _validate_training_contract(
        ansatz,
        design,
        data,
        estimator,
        optimization,
        run,
    )

    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    parameters = initialize_parameters(ansatz.n_parameters, optimizer_seed)
    target_fouriers = exact_target_fouriers(target.probabilities)
    characters = jnp.asarray(generator_characters(np.asarray(ansatz.generators)))
    learning_rate = optax.cosine_decay_schedule(
        init_value=optimization.learning_rate,
        decay_steps=optimization.max_updates,
        alpha=optimization.final_learning_rate / optimization.learning_rate,
    )
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(parameters)

    def objective(current: jnp.ndarray) -> jnp.ndarray:
        return exact_conditional_mmd(
            current,
            generators,
            jnp.asarray(design.condition_angles),
            target_fouriers,
            estimator.bandwidths,
            characters=characters,
        ).loss

    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    exact_value = jax.jit(objective)
    started = time.perf_counter()
    best_parameters = parameters
    best_step = 0
    best_validation = float(exact_value(parameters))
    train_steps: list[int] = []
    train_losses: list[float] = []
    validation_steps = [0]
    validation_losses = [best_validation]
    for step in range(optimization.max_updates):
        loss, gradient = value_and_gradient(parameters)
        updates, optimizer_state = optimizer.update(
            gradient,
            optimizer_state,
            parameters,
        )
        parameters = optax.apply_updates(parameters, updates)
        completed = step + 1
        train_steps.append(completed)
        train_losses.append(float(loss))
        if (
            completed % optimization.validation_interval == 0
            or completed == optimization.max_updates
        ):
            validation = float(exact_value(parameters))
            validation_steps.append(completed)
            validation_losses.append(validation)
            if validation < best_validation:
                best_validation = validation
                best_parameters = parameters
                best_step = completed

    result = IsingTrainingResult(
        design=design,
        coupling_seed=data.coupling_seed,
        optimizer_seed=optimizer_seed,
        generators=np.asarray(ansatz.generators),
        best_parameters=np.asarray(best_parameters),
        final_parameters=np.asarray(parameters),
        best_step=best_step,
        best_validation_loss=best_validation,
        history=TrainingHistory(
            train_steps=np.asarray(train_steps, dtype=np.int64),
            train_losses=np.asarray(train_losses, dtype=np.float64),
            validation_steps=np.asarray(validation_steps, dtype=np.int64),
            validation_losses=np.asarray(validation_losses, dtype=np.float64),
        ),
        estimator=estimator,
        optimization=optimization,
        streams=streams,
        telemetry=FeasibilityTelemetry(
            runtime_seconds=time.perf_counter() - started,
            peak_rss_bytes=process_peak_rss_bytes(),
        ),
    )
    if optimization.production:
        _assert_result_deployment(result)
    return result


def _model_probabilities(
    result: IsingTrainingResult,
    betas: tuple[float, ...],
    grid: IsingBetaGrid,
    n_visible_qubits: int,
) -> np.ndarray:
    parameters = result.best_parameters
    generators = result.generators
    n_qubits = result.design.n_qubits
    normalized = normalize_conditions(betas, grid.training[0], grid.training[-1])
    if result.design.family in {
        ModelFamily.FIXED_INPUT,
        ModelFamily.MATCHED_CAPACITY_CONTROL,
    }:
        angles = beta_to_broadcast_angles(np.asarray(betas), grid, n_qubits)
        probabilities = exact_iqp_probabilities_many(parameters, generators, angles)
        return marginalize_ancillas(
            probabilities,
            n_visible_qubits,
            n_qubits - n_visible_qubits,
        )
    if result.design.family is ModelFamily.CONDITION_MODULATED_PHASE:
        phases = exact_effective_phases(
            parameters,
            normalized,
            result.design.phase_feature,
        )
        fixed_input = result.design.condition_angles[0]
        return np.stack(
            [
                exact_iqp_probabilities(phase, generators, fixed_input)
                for phase in phases
            ],
        )
    if result.design.family is ModelFamily.NO_CONDITION:
        fixed = result.design.condition_angles[0]
        angles = np.broadcast_to(fixed, (len(betas), n_qubits))
        return exact_iqp_probabilities_many(parameters, generators, angles)
    separate_input = result.design.condition_angles[0]
    probability = exact_iqp_probabilities(parameters, generators, separate_input)
    return np.broadcast_to(probability, (len(betas), probability.size)).copy()


def evaluate_exact_tvd(
    result: IsingTrainingResult,
    training_targets: ExactTargetSet,
    heldout_targets: ExactTargetSet,
    grid: IsingBetaGrid,
    *,
    production: bool = True,
) -> tuple[ExactTVDRecord, ...]:
    """Evaluate a frozen checkpoint on exact Training and untouched Held-out laws."""
    if training_targets.role is not DatasetRole.TRAINING:
        raise ValueError("training_targets must be tagged role='training'")
    if heldout_targets.role is not DatasetRole.HELD_OUT:
        raise ValueError("Held-out evaluation requires role='held_out'")
    if (
        training_targets.coupling_seed != heldout_targets.coupling_seed
        or training_targets.coupling_seed != result.coupling_seed
    ):
        raise ValueError("exact target coupling instances must align")
    if training_targets.betas != grid.training or heldout_targets.betas != grid.heldout:
        raise ValueError("exact target betas must equal the frozen campaign grid")
    if production and (
        training_targets.probabilities.shape[1] != PRODUCTION_N_STATES
        or heldout_targets.probabilities.shape[1] != PRODUCTION_N_STATES
    ):
        raise ValueError("production fidelity must enumerate all 65,536 states")

    return evaluate_exact_target_set(
        result,
        training_targets,
        grid,
        production=production,
    ) + evaluate_exact_target_set(
        result,
        heldout_targets,
        grid,
        production=production,
    )


def evaluate_exact_target_set(
    result: IsingTrainingResult,
    targets: ExactTargetSet,
    grid: IsingBetaGrid,
    *,
    production: bool = True,
) -> tuple[ExactTVDRecord, ...]:
    """Evaluate only the explicitly supplied post-checkpoint exact Target role."""
    if targets.coupling_seed != result.coupling_seed:
        raise ValueError("exact Target and trained result instances must align")
    if production and targets.probabilities.shape[1] != PRODUCTION_N_STATES:
        raise ValueError("production fidelity must enumerate all 65,536 states")
    n_states = targets.probabilities.shape[1]
    n_visible_qubits = round(np.log2(n_states))
    if 2**n_visible_qubits != n_states:
        raise ValueError("exact Target width must be a power of two")
    model = _model_probabilities(
        result,
        targets.betas,
        grid,
        n_visible_qubits,
    )
    if model.shape != targets.probabilities.shape:
        raise ValueError("model and exact target probability tables must align")
    return tuple(
        ExactTVDRecord(
            role=targets.role,
            beta=beta,
            tvd=float(0.5 * np.abs(reference - candidate).sum()),
            n_states=reference.size,
        )
        for beta, reference, candidate in zip(
            targets.betas,
            targets.probabilities,
            model,
            strict=True,
        )
    )


def exact_training_target_set(coupling_seed: int, beta: float) -> ExactTargetSet:
    """Enumerate one declared Training Target without exposing Held-out laws."""
    target = exact_ising_target(coupling_seed, beta)
    return ExactTargetSet(
        DatasetRole.TRAINING,
        coupling_seed,
        (float(beta),),
        target.probabilities[None, :],
        target.digest,
    )


def representability_gate(
    cells: Sequence[RepresentabilityCell],
    grid: IsingBetaGrid,
) -> RepresentabilityResult:
    """Apply median <= 0.25 and every-cell <= 0.35 to the exact six cells."""
    values = tuple(cells)
    screened_betas = (grid.training[0], grid.anchor, grid.training[-1])
    expected = {
        (coupling_seed, beta)
        for coupling_seed in REPRESENTABILITY_COUPLING_SEEDS
        for beta in screened_betas
    }
    actual = {(cell.coupling_seed, cell.beta) for cell in values}
    if len(values) != 6 or actual != expected:
        raise ValueError(
            "representability requires seeds 666/101 at low/anchor/high exactly once",
        )
    tvds = np.asarray([cell.tvd for cell in values], dtype=np.float64)
    median = float(np.median(tvds))
    maximum = float(np.max(tvds))
    median_passed = median <= REPRESENTABILITY_MEDIAN_TVD_MAX
    cells_passed = maximum <= REPRESENTABILITY_CELL_TVD_MAX
    return RepresentabilityResult(
        cells=tuple(sorted(values, key=lambda cell: (cell.coupling_seed, cell.beta))),
        median_tvd=median,
        maximum_tvd=maximum,
        median_rule_passed=median_passed,
        every_cell_rule_passed=cells_passed,
        passed=median_passed and cells_passed,
    )


def run_representability_screen(
    manifest: CampaignManifest,
    evaluate_cell: Callable[[int, float], float],
) -> RepresentabilityResult:
    """Execute the exact preregistered six-cell representability screen."""
    grid = _campaign_grid(manifest)
    cells = tuple(
        RepresentabilityCell(coupling_seed, beta, evaluate_cell(coupling_seed, beta))
        for coupling_seed in REPRESENTABILITY_COUPLING_SEEDS
        for beta in (grid.training[0], grid.anchor, grid.training[-1])
    )
    return representability_gate(cells, grid)


def load_training_validation_data(
    directory: Path,
    coupling_seed: int,
    betas: Sequence[float],
    *,
    expected_base_data_seed: int | None = None,
    training_grid: Sequence[float] | None = None,
) -> TrainingValidationData:
    """Load verified role-tagged finite datasets from prepared NPZ archives."""
    if (expected_base_data_seed is None) != (training_grid is None):
        raise ValueError(
            "expected_base_data_seed and training_grid must be supplied together",
        )
    training = []
    validation = []
    for beta in betas:
        path = directory / f"coupling_{coupling_seed}_beta_{float(beta):.12g}.npz"
        source_digest = _sha256_file(path)
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "training",
                "validation",
                "training_seed",
                "validation_seed",
                "beta",
            }
            if set(archive.files) != required:
                raise ValueError(f"dataset archive has unexpected keys: {path}")
            stored_beta = float(archive["beta"])
            if not np.isclose(stored_beta, beta, atol=1e-14, rtol=0.0):
                raise ValueError(
                    f"dataset beta does not match filename/request: {path}",
                )
            training_seed = int(archive["training_seed"])
            validation_seed = int(archive["validation_seed"])
            if training_grid is not None and expected_base_data_seed is not None:
                matches = [
                    index
                    for index, grid_beta in enumerate(training_grid)
                    if np.isclose(grid_beta, beta, atol=1e-14, rtol=0.0)
                ]
                if len(matches) != 1:
                    raise ValueError("dataset beta is not a unique manifest Condition")
                expected_training_seed = expected_base_data_seed * 100 + 2 * matches[0]
                if (
                    training_seed != expected_training_seed
                    or validation_seed != expected_training_seed + 1
                ):
                    raise ValueError(
                        "dataset streams do not match the manifest data-seed schedule",
                    )
            training.append(
                RoleTaggedDataset(
                    DatasetRole.TRAINING,
                    coupling_seed,
                    stored_beta,
                    training_seed,
                    archive["training"],
                    source_digest,
                ),
            )
            validation.append(
                RoleTaggedDataset(
                    DatasetRole.VALIDATION,
                    coupling_seed,
                    stored_beta,
                    validation_seed,
                    archive["validation"],
                    source_digest,
                ),
            )
    return TrainingValidationData(tuple(training), tuple(validation))


def load_exact_target_sets(
    path: Path,
    manifest: CampaignManifest,
    coupling_seed: int,
) -> tuple[ExactTargetSet, ExactTargetSet]:
    """Load exact 65,536-state Training and Held-out target tables."""
    source_digest = _sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "coupling_seed",
            "betas",
            "couplings",
            "bitstrings",
            "energies",
            "probabilities",
        }
        if set(archive.files) != required:
            raise ValueError("exact target archive has unexpected keys")
        if int(archive["coupling_seed"]) != coupling_seed:
            raise ValueError("exact target archive uses the wrong coupling seed")
        expected_betas = manifest.beta_grid.training + manifest.beta_grid.held_out
        betas = tuple(float(value) for value in archive["betas"])
        if len(betas) != len(expected_betas) or not np.allclose(
            betas,
            expected_betas,
            atol=1e-14,
            rtol=0.0,
        ):
            raise ValueError("exact target archive does not match the manifest grid")
        probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
    split = len(manifest.beta_grid.training)
    return (
        ExactTargetSet(
            DatasetRole.TRAINING,
            coupling_seed,
            manifest.beta_grid.training,
            probabilities[:split],
            source_digest,
        ),
        ExactTargetSet(
            DatasetRole.HELD_OUT,
            coupling_seed,
            manifest.beta_grid.held_out,
            probabilities[split:],
            source_digest,
        ),
    )


def write_checkpoint(path: Path, result: IsingTrainingResult) -> str:
    """Create an immutable NPZ checkpoint and return its SHA-256 digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_parameters": result.best_parameters,
        "final_parameters": result.final_parameters,
        "generators": result.generators,
        "best_step": np.asarray(result.best_step, dtype=np.int64),
        "best_validation_loss": np.asarray(
            result.best_validation_loss,
            dtype=np.float64,
        ),
        "train_steps": result.history.train_steps,
        "train_losses": result.history.train_losses,
        "validation_steps": result.history.validation_steps,
        "validation_losses": result.history.validation_losses,
    }
    try:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        # A resumed reservation may already have completed checkpoint I/O before
        # interruption.  Reuse only if the exact bytes remain immutable.
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(payload) or any(
                not np.array_equal(existing[name], value)
                for name, value in payload.items()
            ):
                raise FileExistsError(
                    f"checkpoint path already contains different content: {path}",
                ) from None
    return _sha256_file(path)


def result_bundle_from_training(
    run: RunIdentity,
    result: IsingTrainingResult,
    data: TrainingValidationData,
    exact_targets_sha256: str | None,
    evaluation: Sequence[ExactTVDRecord],
    checkpoint_path: Path,
    *,
    checkpoint_artifact_path: str,
    code_revision: str,
    environment: Mapping[str, str],
    invocation: Sequence[str],
    started_at: str,
    completed_at: str,
    manifest: CampaignManifest | None = None,
    objective_name: str = "sampled_conditional_mmd",
    reused_m0_checkpoint: CheckpointProvenance | None = None,
) -> ResultBundle:
    """Build a complete immutable ResultBundle for a declared campaign run."""
    evaluation = tuple(evaluation)
    if run.model_family is not result.design.family:
        raise ValueError("run identity and trained model family differ")
    if run.optimizer_seed != result.optimizer_seed:
        raise ValueError("run identity and optimizer seed differ")
    if run.coupling_seed != data.coupling_seed:
        raise ValueError("run identity and finite-data instance differ")
    if run.coupling_seed != result.coupling_seed:
        raise ValueError("run identity and trained result instance differ")
    if run.phase_feature_assignment != result.design.phase_feature:
        raise ValueError("run identity and phase-feature assignment differ")
    if result.optimization.production:
        if manifest is None:
            raise ValueError("production bundle construction requires the manifest")
        if result.estimator != EstimatorProtocol.from_manifest(manifest):
            raise ValueError("result estimator does not match the manifest")
        if result.optimization != OptimizationProtocol.for_run(manifest, run):
            raise ValueError("result optimizer does not match its declared run")
    if exact_targets_sha256 is not None and not _is_sha256(exact_targets_sha256):
        raise ValueError("exact_targets_sha256 must be a lowercase digest")
    if objective_name not in {
        "sampled_conditional_mmd",
        "exact_conditional_mmd_diagnostic",
    }:
        raise ValueError("unknown training objective provenance")
    if run.stage is Stage.PRIMARY:
        if exact_targets_sha256 is None:
            raise ValueError("primary bundles require exact Target provenance")
        role_counts = {
            role: sum(record.role is role for record in evaluation)
            for role in (DatasetRole.TRAINING, DatasetRole.HELD_OUT)
        }
        if role_counts != {DatasetRole.TRAINING: 5, DatasetRole.HELD_OUT: 4}:
            raise ValueError("production bundles require all five/four exact TVD cells")
        if result.optimization.production and any(
            record.n_states != PRODUCTION_N_STATES for record in evaluation
        ):
            raise ValueError("production bundles require 65,536-state exact TVDs")
    elif run.stage is Stage.REPRESENTABILITY_SCREEN:
        if exact_targets_sha256 is None:
            raise ValueError("representability bundles require exact Target provenance")
        if (
            len(evaluation) != 1
            or evaluation[0].role is not DatasetRole.TRAINING
            or (
                result.optimization.production
                and evaluation[0].n_states != PRODUCTION_N_STATES
            )
        ):
            raise ValueError(
                "representability requires one 65,536-state Training TVD",
            )
    elif run.stage is Stage.REPRESENTABILITY_DIAGNOSTIC:
        if objective_name != "exact_conditional_mmd_diagnostic":
            raise ValueError("representability diagnostic must use the exact objective")
        if (
            exact_targets_sha256 is None
            or len(evaluation) != 1
            or evaluation[0].role is not DatasetRole.TRAINING
            or (
                result.optimization.production
                and evaluation[0].n_states != PRODUCTION_N_STATES
            )
        ):
            raise ValueError(
                "exact diagnostic requires one 65,536-state Training TVD",
            )
    elif run.stage is Stage.ANCILLA:
        if exact_targets_sha256 is None or reused_m0_checkpoint is None:
            raise ValueError(
                "ancilla bundles require exact Targets and a reused m=0 checkpoint",
            )
        if reused_m0_checkpoint.manifest_digest != run.configuration_digest:
            raise ValueError("reused m=0 checkpoint belongs to another manifest")
        role_counts = {
            role: sum(record.role is role for record in evaluation)
            for role in (DatasetRole.TRAINING, DatasetRole.HELD_OUT)
        }
        if role_counts != {DatasetRole.TRAINING: 5, DatasetRole.HELD_OUT: 4}:
            raise ValueError("ancilla bundles require all five/four exact TVD cells")
        if result.optimization.production and any(
            record.n_states != PRODUCTION_N_STATES for record in evaluation
        ):
            raise ValueError("production ancilla bundles require exact visible TVDs")
    elif evaluation or exact_targets_sha256 is not None:
        raise ValueError(
            "pre-primary tuning/feasibility bundles cannot access exact Targets",
        )
    checkpoint_sha256 = write_checkpoint(checkpoint_path, result)
    checkpoint = CheckpointProvenance(
        source_run_id=run.run_id,
        manifest_digest=run.configuration_digest,
        selection_rule=(
            "lowest exact Training Conditional MMD; ties keep earliest step"
            if objective_name == "exact_conditional_mmd_diagnostic"
            else "lowest frozen Validation Conditional MMD; ties keep earliest step"
        ),
        selected_step=result.best_step,
        validation_metric=(
            "exact_conditional_mmd"
            if objective_name == "exact_conditional_mmd_diagnostic"
            else "conditional_mmd"
        ),
        validation_value=result.best_validation_loss,
        artifact_path=checkpoint_artifact_path,
        artifact_sha256=checkpoint_sha256,
    )
    dataset_hashes = data.dataset_hashes()
    if exact_targets_sha256 is not None:
        dataset_hashes["exact_targets"] = exact_targets_sha256
    training_metrics = [
        {"role": record.role.value, "beta": record.beta, "tvd": record.tvd}
        for record in evaluation
    ]
    raw_metrics = {
        "model_family": result.design.family.value,
        "ancilla_rung": run.ancilla_rung,
        "ancilla_conditioning_scheme": (
            "broadcast_all"
            if run.stage is Stage.ANCILLA
            and run.model_family is ModelFamily.FIXED_INPUT
            else None
        ),
        "matched_capacity_generator_protocol": (
            f"complete_2local_plus_seed_{MATCHED_CAPACITY_THREE_LOCAL_SEED}_3local"
            if run.stage is Stage.ANCILLA
            and run.model_family is ModelFamily.MATCHED_CAPACITY_CONTROL
            else None
        ),
        "objective": objective_name,
        "phase_feature": result.design.phase_feature,
        "tuning_candidate_id": run.tuning_candidate_id,
        "best_step": result.best_step,
        "best_validation_conditional_mmd": result.best_validation_loss,
        "max_updates": result.optimization.max_updates,
        "batch_size_per_condition": result.optimization.batch_size_per_condition,
        "initial_learning_rate": result.optimization.learning_rate,
        "final_learning_rate": result.optimization.final_learning_rate,
        "validation_interval": result.optimization.validation_interval,
        "bandwidths": list(result.estimator.bandwidths),
        "operators_per_bandwidth": result.estimator.operators_per_bandwidth,
        "reference_samples_per_operator_condition": (
            result.estimator.reference_samples_per_operator_condition
        ),
        "exact_evaluation_states": sorted({record.n_states for record in evaluation}),
        "exact_tvd": training_metrics,
        "reused_m0_checkpoint": (
            None if reused_m0_checkpoint is None else reused_m0_checkpoint.to_dict()
        ),
    }
    if run.stage is Stage.ANCILLA:
        if run.ancilla_rung is None:
            raise ValueError("ancilla result has no rung")
        if run.model_family is ModelFamily.FIXED_INPUT:
            accounting = ParameterAccounting(
                n_qubits=result.design.n_qubits,
                n_generators=result.generators.shape[0],
                n_trainable_parameters=result.best_parameters.size,
                n_ancillas=run.ancilla_rung,
            )
        else:
            accounting = ParameterAccounting(
                n_qubits=PRODUCTION_N_QUBITS,
                n_generators=result.generators.shape[0],
                n_trainable_parameters=result.best_parameters.size,
                n_ancillas=0,
            )
    else:
        accounting = ParameterAccounting(
            n_qubits=result.design.n_qubits,
            n_generators=result.generators.shape[0],
            n_trainable_parameters=result.best_parameters.size,
            n_ancillas=0,
        )
    return ResultBundle.create(
        run=run,
        manifest_digest=run.configuration_digest,
        code_revision=code_revision,
        environment=environment,
        invocation=tuple(invocation),
        started_at=started_at,
        completed_at=completed_at,
        runtime_seconds=result.telemetry.runtime_seconds,
        peak_memory_bytes=result.telemetry.peak_rss_bytes,
        parameter_accounting=accounting,
        dataset_hashes=dataset_hashes,
        estimator_randomness=result.streams.to_dict(),
        checkpoint=checkpoint,
        raw_metrics=raw_metrics,
        completion_status=CompletionStatus.COMPLETED,
    )


def runtime_environment(lockfile: Path | None = None) -> dict[str, str]:
    """Return deterministic environment provenance for ResultBundles."""
    values = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "numpy": np.__version__,
    }
    if lockfile is not None:
        values["lock_sha256"] = _sha256_file(lockfile)
    return values


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for artifact provenance."""
    return datetime.now(UTC).isoformat()


def canonical_metrics_digest(metrics: Mapping[str, object]) -> str:
    """Return a stable evidence digest for a gate decision."""
    return hashlib.sha256(canonical_json_bytes(dict(metrics))).hexdigest()


def representability_result_json(result: RepresentabilityResult) -> str:
    """Serialize the complete representability decision for CLI reporting."""
    payload = {
        "cells": [
            {
                "coupling_seed": cell.coupling_seed,
                "beta": cell.beta,
                "tvd": cell.tvd,
            }
            for cell in result.cells
        ],
        "median_tvd": result.median_tvd,
        "maximum_tvd": result.maximum_tvd,
        "median_rule_passed": result.median_rule_passed,
        "every_cell_rule_passed": result.every_cell_rule_passed,
        "passed": result.passed,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

"""Frozen protocol and deterministic planning for the arXiv foundations study.

This module is deliberately separate from :mod:`iqp_cond.protocol`.  The
``quantum_campaign_v1`` schema and digest remain immutable repository history;
the compact foundations study has its own manifest and run matrix while
reusing the content-addressed :class:`~iqp_cond.protocol.RunIdentity` seam.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import (
    BetaGrid,
    ModelFamily,
    ProtocolValidationError,
    RunIdentity,
    Stage,
    TuningCandidate,
    content_digest,
)

FOUNDATIONS_SHARED_FAMILIES = (
    ModelFamily.FIXED_INPUT,
    ModelFamily.CONDITION_MODULATED_PHASE,
    ModelFamily.NO_CONDITION,
)
FOUNDATIONS_PHASE_FEATURE = "balanced_constant_slope_bank"


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ProtocolValidationError(
            f"{context} keys do not match schema; "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}",
        )


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolValidationError(f"{context} must be an integer")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolValidationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolValidationError(f"{context} must be finite")
    return result


def _digest(value: str, context: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ProtocolValidationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _unique_nonnegative(values: tuple[int, ...], context: str) -> None:
    if (
        not values
        or len(values) != len(set(values))
        or any(value < 0 for value in values)
    ):
        raise ProtocolValidationError(
            f"{context} must contain unique non-negative integers",
        )


@dataclass(frozen=True, slots=True)
class SourceCampaign:
    version: str
    manifest_digest: str

    def __post_init__(self) -> None:
        if self.version != "quantum_campaign_v1":
            raise ProtocolValidationError(
                "source campaign must remain quantum_campaign_v1"
            )
        _digest(self.manifest_digest, "source manifest digest")

    def to_dict(self) -> dict[str, object]:
        return {"version": self.version, "manifest_digest": self.manifest_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceCampaign:
        _exact_keys(value, {"version", "manifest_digest"}, "source_campaign")
        return cls(str(value["version"]), str(value["manifest_digest"]))


@dataclass(frozen=True, slots=True)
class InstanceProtocol:
    development: tuple[int, ...]
    primary: tuple[int, ...]
    reserved: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, values, expected in (
            ("development", self.development, 2),
            ("primary", self.primary, 4),
            ("reserved", self.reserved, 2),
        ):
            _unique_nonnegative(values, f"{name} instances")
            if len(values) != expected:
                raise ProtocolValidationError(
                    f"{name} instances must contain exactly {expected} seeds",
                )
        groups = (set(self.development), set(self.primary), set(self.reserved))
        if any(
            groups[left] & groups[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ProtocolValidationError("instance seed groups must be disjoint")
        if (
            self.development != (606, 707)
            or self.primary != (666, 101, 202, 303)
            or self.reserved != (404, 505)
        ):
            raise ProtocolValidationError(
                "instance seed groups differ from the frozen compact study"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "development": list(self.development),
            "primary": list(self.primary),
            "reserved": list(self.reserved),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InstanceProtocol:
        _exact_keys(value, {"development", "primary", "reserved"}, "instances")
        return cls(
            development=tuple(
                _integer(item, "development seed") for item in value["development"]
            ),
            primary=tuple(_integer(item, "primary seed") for item in value["primary"]),
            reserved=tuple(
                _integer(item, "reserved seed") for item in value["reserved"]
            ),
        )


@dataclass(frozen=True, slots=True)
class OptimizerSeedProtocol:
    representability: tuple[int, ...]
    tuning: tuple[int, ...]
    primary: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.representability != (0, 1)
            or self.tuning != (0,)
            or self.primary != (0, 1, 2)
        ):
            raise ProtocolValidationError(
                "optimizer seed groups differ from the frozen compact study"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "representability": list(self.representability),
            "tuning": list(self.tuning),
            "primary": list(self.primary),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OptimizerSeedProtocol:
        _exact_keys(value, {"representability", "tuning", "primary"}, "optimizer_seeds")
        return cls(
            representability=tuple(
                _integer(item, "representability optimizer seed")
                for item in value["representability"]
            ),
            tuning=tuple(
                _integer(item, "tuning optimizer seed") for item in value["tuning"]
            ),
            primary=tuple(
                _integer(item, "primary optimizer seed") for item in value["primary"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ExactDataProtocol:
    source: str
    training_samples: int
    validation_samples: int
    root_seed: int

    def __post_init__(self) -> None:
        if self.source != "exact_gibbs_draws":
            raise ProtocolValidationError(
                "foundations data source must be exact_gibbs_draws"
            )
        if self.training_samples != 5_000 or self.validation_samples != 2_000:
            raise ProtocolValidationError(
                "exact data split must be 5,000 Training / 2,000 Validation"
            )
        if self.root_seed != 20_260_811:
            raise ProtocolValidationError("exact data root seed is not frozen")

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "training_samples": self.training_samples,
            "validation_samples": self.validation_samples,
            "root_seed": self.root_seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExactDataProtocol:
        _exact_keys(
            value,
            {"source", "training_samples", "validation_samples", "root_seed"},
            "data",
        )
        return cls(
            source=str(value["source"]),
            training_samples=_integer(value["training_samples"], "training_samples"),
            validation_samples=_integer(
                value["validation_samples"], "validation_samples"
            ),
            root_seed=_integer(value["root_seed"], "data root_seed"),
        )


@dataclass(frozen=True, slots=True)
class EstimatorSelectionProtocol:
    bandwidths: tuple[float, ...]
    candidate_budgets: tuple[int, ...]
    exact_loss_bias_max: float
    exact_gradient_cosine_min: float
    exact_gradient_nrmse_max: float
    large_loss_cv_max: float
    large_gradient_cosine_min: float

    def __post_init__(self) -> None:
        if self.bandwidths != (1.0, 2.0, 4.0):
            raise ProtocolValidationError(
                "foundations bandwidths must be exactly (1, 2, 4)"
            )
        if self.candidate_budgets != (128, 256, 512, 1024):
            raise ProtocolValidationError(
                "estimator candidates must be 128/256/512/1024"
            )
        expected = (0.01, 0.95, 0.20, 0.10, 0.90)
        actual = (
            self.exact_loss_bias_max,
            self.exact_gradient_cosine_min,
            self.exact_gradient_nrmse_max,
            self.large_loss_cv_max,
            self.large_gradient_cosine_min,
        )
        if actual != expected:
            raise ProtocolValidationError(
                "estimator-selection thresholds are not frozen"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "bandwidths": list(self.bandwidths),
            "candidate_budgets": list(self.candidate_budgets),
            "exact_loss_bias_max": self.exact_loss_bias_max,
            "exact_gradient_cosine_min": self.exact_gradient_cosine_min,
            "exact_gradient_nrmse_max": self.exact_gradient_nrmse_max,
            "large_loss_cv_max": self.large_loss_cv_max,
            "large_gradient_cosine_min": self.large_gradient_cosine_min,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EstimatorSelectionProtocol:
        expected = {
            "bandwidths",
            "candidate_budgets",
            "exact_loss_bias_max",
            "exact_gradient_cosine_min",
            "exact_gradient_nrmse_max",
            "large_loss_cv_max",
            "large_gradient_cosine_min",
        }
        _exact_keys(value, expected, "estimator")
        return cls(
            bandwidths=tuple(
                _number(item, "bandwidth") for item in value["bandwidths"]
            ),
            candidate_budgets=tuple(
                _integer(item, "estimator candidate")
                for item in value["candidate_budgets"]
            ),
            exact_loss_bias_max=_number(
                value["exact_loss_bias_max"], "exact_loss_bias_max"
            ),
            exact_gradient_cosine_min=_number(
                value["exact_gradient_cosine_min"], "exact_gradient_cosine_min"
            ),
            exact_gradient_nrmse_max=_number(
                value["exact_gradient_nrmse_max"], "exact_gradient_nrmse_max"
            ),
            large_loss_cv_max=_number(value["large_loss_cv_max"], "large_loss_cv_max"),
            large_gradient_cosine_min=_number(
                value["large_gradient_cosine_min"], "large_gradient_cosine_min"
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeProtocol:
    smoke_updates: int
    max_projected_seconds_per_run: float
    max_peak_rss_bytes: int

    def __post_init__(self) -> None:
        if (
            self.smoke_updates != 100
            or self.max_projected_seconds_per_run != 3_600.0
            or self.max_peak_rss_bytes != 28 * 1024**3
        ):
            raise ProtocolValidationError("runtime preflight thresholds are not frozen")

    def to_dict(self) -> dict[str, object]:
        return {
            "smoke_updates": self.smoke_updates,
            "max_projected_seconds_per_run": self.max_projected_seconds_per_run,
            "max_peak_rss_bytes": self.max_peak_rss_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeProtocol:
        _exact_keys(
            value,
            {"smoke_updates", "max_projected_seconds_per_run", "max_peak_rss_bytes"},
            "runtime",
        )
        return cls(
            smoke_updates=_integer(value["smoke_updates"], "smoke_updates"),
            max_projected_seconds_per_run=_number(
                value["max_projected_seconds_per_run"], "max_projected_seconds_per_run"
            ),
            max_peak_rss_bytes=_integer(
                value["max_peak_rss_bytes"], "max_peak_rss_bytes"
            ),
        )


@dataclass(frozen=True, slots=True)
class FoundationsTrainingProtocol:
    batch_size: int
    max_updates: int
    validation_interval: int
    optimizer: str
    checkpoint_rule: str
    tuning_selection_rule: str
    candidates: tuple[TuningCandidate, ...]

    def __post_init__(self) -> None:
        if (self.batch_size, self.max_updates, self.validation_interval) != (
            256,
            1_000,
            50,
        ):
            raise ProtocolValidationError(
                "training batch/update/Validation schedule is not frozen"
            )
        if (
            self.optimizer != "adam_cosine_decay"
            or self.checkpoint_rule != "lowest_validation_conditional_mmd"
        ):
            raise ProtocolValidationError(
                "training optimizer or checkpoint rule is not frozen"
            )
        if (
            self.tuning_selection_rule
            != "minimum_median_validation_conditional_mmd_then_candidate_id"
        ):
            raise ProtocolValidationError("tuning selection rule is not frozen")
        if (
            len(self.candidates) != 6
            or len({candidate.candidate_id for candidate in self.candidates}) != 6
        ):
            raise ProtocolValidationError(
                "tuning requires exactly two candidates per shared family"
            )
        for family in FOUNDATIONS_SHARED_FAMILIES:
            candidates = self.candidates_for(family)
            if len(candidates) != 2:
                raise ProtocolValidationError(
                    "tuning requires exactly two candidates per shared family"
                )
            if tuple(
                (item.initial_learning_rate, item.final_learning_rate)
                for item in candidates
            ) != ((0.003, 0.0003), (0.01, 0.001)):
                raise ProtocolValidationError(
                    "tuning learning-rate schedules are not frozen"
                )
            if family is ModelFamily.CONDITION_MODULATED_PHASE and any(
                item.phase_feature_assignment != FOUNDATIONS_PHASE_FEATURE
                for item in candidates
            ):
                raise ProtocolValidationError(
                    "phase baseline must use the balanced constant/slope feature bank"
                )

    def candidates_for(self, family: ModelFamily) -> tuple[TuningCandidate, ...]:
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.model_family is family
        )

    def candidate(self, candidate_id: str) -> TuningCandidate:
        matches = tuple(
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == candidate_id
        )
        if len(matches) != 1:
            raise ProtocolValidationError(
                f"unknown foundations tuning candidate: {candidate_id}"
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "max_updates": self.max_updates,
            "validation_interval": self.validation_interval,
            "optimizer": self.optimizer,
            "checkpoint_rule": self.checkpoint_rule,
            "tuning_selection_rule": self.tuning_selection_rule,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FoundationsTrainingProtocol:
        expected = {
            "batch_size",
            "max_updates",
            "validation_interval",
            "optimizer",
            "checkpoint_rule",
            "tuning_selection_rule",
            "candidates",
        }
        _exact_keys(value, expected, "training")
        return cls(
            batch_size=_integer(value["batch_size"], "batch_size"),
            max_updates=_integer(value["max_updates"], "max_updates"),
            validation_interval=_integer(
                value["validation_interval"], "validation_interval"
            ),
            optimizer=str(value["optimizer"]),
            checkpoint_rule=str(value["checkpoint_rule"]),
            tuning_selection_rule=str(value["tuning_selection_rule"]),
            candidates=tuple(
                TuningCandidate.from_dict(item) for item in value["candidates"]
            ),
        )


@dataclass(frozen=True, slots=True)
class RepresentabilityProtocol:
    beta_indices: tuple[int, ...]
    median_tvd_max: float
    cell_tvd_max: float

    def __post_init__(self) -> None:
        if (
            self.beta_indices != (0, 2, 4)
            or self.median_tvd_max != 0.25
            or self.cell_tvd_max != 0.35
        ):
            raise ProtocolValidationError("representability screen is not frozen")

    def to_dict(self) -> dict[str, object]:
        return {
            "beta_indices": list(self.beta_indices),
            "median_tvd_max": self.median_tvd_max,
            "cell_tvd_max": self.cell_tvd_max,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RepresentabilityProtocol:
        _exact_keys(
            value,
            {"beta_indices", "median_tvd_max", "cell_tvd_max"},
            "representability",
        )
        return cls(
            beta_indices=tuple(
                _integer(item, "representability beta index")
                for item in value["beta_indices"]
            ),
            median_tvd_max=_number(value["median_tvd_max"], "median_tvd_max"),
            cell_tvd_max=_number(value["cell_tvd_max"], "cell_tvd_max"),
        )


@dataclass(frozen=True, slots=True)
class PublicationProtocol:
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    training_median_tvd_max: float
    cell_win_fraction_min: float
    min_instances_improved: int

    def __post_init__(self) -> None:
        if (
            self.bootstrap_replicates != 10_000
            or self.bootstrap_seed != 20_260_811
            or self.confidence_level != 0.95
        ):
            raise ProtocolValidationError("bootstrap protocol is not frozen")
        if (
            self.training_median_tvd_max != 0.35
            or self.cell_win_fraction_min != 0.75
            or self.min_instances_improved != 3
        ):
            raise ProtocolValidationError("publication thresholds are not frozen")

    def to_dict(self) -> dict[str, object]:
        return {
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "training_median_tvd_max": self.training_median_tvd_max,
            "cell_win_fraction_min": self.cell_win_fraction_min,
            "min_instances_improved": self.min_instances_improved,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PublicationProtocol:
        expected = {
            "bootstrap_replicates",
            "bootstrap_seed",
            "confidence_level",
            "training_median_tvd_max",
            "cell_win_fraction_min",
            "min_instances_improved",
        }
        _exact_keys(value, expected, "publication")
        return cls(
            bootstrap_replicates=_integer(
                value["bootstrap_replicates"], "bootstrap_replicates"
            ),
            bootstrap_seed=_integer(value["bootstrap_seed"], "bootstrap_seed"),
            confidence_level=_number(value["confidence_level"], "confidence_level"),
            training_median_tvd_max=_number(
                value["training_median_tvd_max"], "training_median_tvd_max"
            ),
            cell_win_fraction_min=_number(
                value["cell_win_fraction_min"], "cell_win_fraction_min"
            ),
            min_instances_improved=_integer(
                value["min_instances_improved"], "min_instances_improved"
            ),
        )


@dataclass(frozen=True, slots=True)
class FoundationsRunAllocation:
    representability: int
    tuning: int
    primary: int

    def __post_init__(self) -> None:
        if (self.representability, self.tuning, self.primary) != (12, 12, 36):
            raise ProtocolValidationError(
                "foundations run allocation must be exactly 12/12/36"
            )

    @property
    def total(self) -> int:
        return self.representability + self.tuning + self.primary

    def to_dict(self) -> dict[str, int]:
        return {
            "representability": self.representability,
            "tuning": self.tuning,
            "primary": self.primary,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FoundationsRunAllocation:
        _exact_keys(value, {"representability", "tuning", "primary"}, "run_allocation")
        return cls(
            *(
                _integer(value[name], name)
                for name in ("representability", "tuning", "primary")
            )
        )


@dataclass(frozen=True, slots=True)
class FoundationsManifest:
    schema_version: int
    campaign_version: str
    source_campaign: SourceCampaign
    instances: InstanceProtocol
    optimizer_seeds: OptimizerSeedProtocol
    beta_grid: BetaGrid
    data: ExactDataProtocol
    estimator: EstimatorSelectionProtocol
    runtime: RuntimeProtocol
    training: FoundationsTrainingProtocol
    representability: RepresentabilityProtocol
    publication: PublicationProtocol
    run_allocation: FoundationsRunAllocation

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.campaign_version != "arxiv_foundations_v1":
            raise ProtocolValidationError("foundations manifest identity is not frozen")

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "campaign_version": self.campaign_version,
            "source_campaign": self.source_campaign.to_dict(),
            "instances": self.instances.to_dict(),
            "optimizer_seeds": self.optimizer_seeds.to_dict(),
            "beta_grid": self.beta_grid.to_dict(),
            "data": self.data.to_dict(),
            "estimator": self.estimator.to_dict(),
            "runtime": self.runtime.to_dict(),
            "training": self.training.to_dict(),
            "representability": self.representability.to_dict(),
            "publication": self.publication.to_dict(),
            "run_allocation": self.run_allocation.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FoundationsManifest:
        expected = {
            "schema_version",
            "campaign_version",
            "source_campaign",
            "instances",
            "optimizer_seeds",
            "beta_grid",
            "data",
            "estimator",
            "runtime",
            "training",
            "representability",
            "publication",
            "run_allocation",
        }
        _exact_keys(value, expected, "foundations manifest")
        return cls(
            schema_version=_integer(value["schema_version"], "schema_version"),
            campaign_version=str(value["campaign_version"]),
            source_campaign=SourceCampaign.from_dict(value["source_campaign"]),
            instances=InstanceProtocol.from_dict(value["instances"]),
            optimizer_seeds=OptimizerSeedProtocol.from_dict(value["optimizer_seeds"]),
            beta_grid=BetaGrid.from_dict(value["beta_grid"]),
            data=ExactDataProtocol.from_dict(value["data"]),
            estimator=EstimatorSelectionProtocol.from_dict(value["estimator"]),
            runtime=RuntimeProtocol.from_dict(value["runtime"]),
            training=FoundationsTrainingProtocol.from_dict(value["training"]),
            representability=RepresentabilityProtocol.from_dict(
                value["representability"]
            ),
            publication=PublicationProtocol.from_dict(value["publication"]),
            run_allocation=FoundationsRunAllocation.from_dict(value["run_allocation"]),
        )

    @classmethod
    def from_json(cls, path: Path) -> FoundationsManifest:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(
                f"could not load foundations manifest {path}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ProtocolValidationError("foundations manifest root must be an object")
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class FoundationsTuningSelection:
    model_family: ModelFamily
    candidate_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "model_family": self.model_family.value,
            "candidate_id": self.candidate_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FoundationsTuningSelection:
        _exact_keys(value, {"model_family", "candidate_id"}, "tuning selection")
        try:
            family = ModelFamily(str(value["model_family"]))
        except ValueError as error:
            raise ProtocolValidationError("unknown tuning selection family") from error
        return cls(family, str(value["candidate_id"]))


@dataclass(frozen=True, slots=True)
class FoundationsTuningDecision:
    manifest_digest: str
    evidence_digest: str
    selections: tuple[FoundationsTuningSelection, ...]

    def __post_init__(self) -> None:
        _digest(self.manifest_digest, "tuning manifest_digest")
        _digest(self.evidence_digest, "tuning evidence_digest")
        families = tuple(selection.model_family for selection in self.selections)
        if (
            len(self.selections) != 3
            or set(families) != set(FOUNDATIONS_SHARED_FAMILIES)
            or len(set(families)) != 3
        ):
            raise ProtocolValidationError(
                "tuning decision requires exactly one selection per shared family"
            )

    @classmethod
    def reference(cls, manifest: FoundationsManifest) -> FoundationsTuningDecision:
        selections = tuple(
            FoundationsTuningSelection(
                family,
                manifest.training.candidates_for(family)[1].candidate_id,
            )
            for family in FOUNDATIONS_SHARED_FAMILIES
        )
        return cls(manifest.digest, "0" * 64, selections)

    def validate_against(self, manifest: FoundationsManifest) -> None:
        if self.manifest_digest != manifest.digest:
            raise ProtocolValidationError(
                "tuning decision belongs to a different foundations manifest"
            )
        for selection in self.selections:
            candidate = manifest.training.candidate(selection.candidate_id)
            if candidate.model_family is not selection.model_family:
                raise ProtocolValidationError(
                    "tuning selection family and candidate differ"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_digest": self.manifest_digest,
            "evidence_digest": self.evidence_digest,
            "selections": [selection.to_dict() for selection in self.selections],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FoundationsTuningDecision:
        _exact_keys(
            value,
            {"manifest_digest", "evidence_digest", "selections"},
            "tuning decision",
        )
        return cls(
            manifest_digest=str(value["manifest_digest"]),
            evidence_digest=str(value["evidence_digest"]),
            selections=tuple(
                FoundationsTuningSelection.from_dict(item)
                for item in value["selections"]
            ),
        )

    def candidate_for(
        self, manifest: FoundationsManifest, family: ModelFamily
    ) -> TuningCandidate:
        self.validate_against(manifest)
        selection = next(
            item for item in self.selections if item.model_family is family
        )
        return manifest.training.candidate(selection.candidate_id)


def _run(
    manifest: FoundationsManifest,
    *,
    stage: Stage,
    coupling_seed: int,
    condition_scope: str,
    family: ModelFamily,
    optimizer_seed: int,
    candidate: TuningCandidate | None = None,
) -> RunIdentity:
    return RunIdentity(
        campaign_version=manifest.campaign_version,
        stage=stage,
        coupling_seed=coupling_seed,
        condition_scope=condition_scope,
        model_family=family,
        phase_feature_assignment=(
            candidate.phase_feature_assignment if candidate else None
        ),
        tuning_candidate_id=(candidate.candidate_id if candidate else None),
        ancilla_rung=None,
        optimizer_seed=optimizer_seed,
        data_seed=manifest.data.root_seed,
        configuration_digest=manifest.digest,
    )


def plan_foundations_runtime(manifest: FoundationsManifest) -> tuple[RunIdentity, ...]:
    return (
        _run(
            manifest,
            stage=Stage.RUNTIME_FEASIBILITY,
            coupling_seed=manifest.instances.development[0],
            condition_scope="all_training_smoke",
            family=ModelFamily.FIXED_INPUT,
            optimizer_seed=manifest.optimizer_seeds.tuning[0],
        ),
    )


def plan_foundations_representability(
    manifest: FoundationsManifest,
) -> tuple[RunIdentity, ...]:
    result = tuple(
        _run(
            manifest,
            stage=Stage.REPRESENTABILITY_SCREEN,
            coupling_seed=coupling_seed,
            condition_scope=f"train_beta_{beta_index}",
            family=ModelFamily.SEPARATE,
            optimizer_seed=optimizer_seed,
        )
        for coupling_seed in manifest.instances.development
        for beta_index in manifest.representability.beta_indices
        for optimizer_seed in manifest.optimizer_seeds.representability
    )
    if len(result) != manifest.run_allocation.representability:
        raise ProtocolValidationError(
            "representability planner count differs from manifest"
        )
    return result


def plan_foundations_tuning(manifest: FoundationsManifest) -> tuple[RunIdentity, ...]:
    result = tuple(
        _run(
            manifest,
            stage=Stage.TUNING,
            coupling_seed=coupling_seed,
            condition_scope="all_training",
            family=candidate.model_family,
            optimizer_seed=manifest.optimizer_seeds.tuning[0],
            candidate=candidate,
        )
        for coupling_seed in manifest.instances.development
        for candidate in manifest.training.candidates
    )
    if len(result) != manifest.run_allocation.tuning:
        raise ProtocolValidationError("tuning planner count differs from manifest")
    return result


def plan_foundations_primary(
    manifest: FoundationsManifest,
    tuning: FoundationsTuningDecision,
) -> tuple[RunIdentity, ...]:
    tuning.validate_against(manifest)
    result = tuple(
        _run(
            manifest,
            stage=Stage.PRIMARY,
            coupling_seed=coupling_seed,
            condition_scope="all_training",
            family=family,
            optimizer_seed=optimizer_seed,
            candidate=tuning.candidate_for(manifest, family),
        )
        for coupling_seed in manifest.instances.primary
        for optimizer_seed in manifest.optimizer_seeds.primary
        for family in FOUNDATIONS_SHARED_FAMILIES
    )
    if len(result) != manifest.run_allocation.primary:
        raise ProtocolValidationError("primary planner count differs from manifest")
    return result

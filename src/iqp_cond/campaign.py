"""Deterministic planning, gate release, and resumable campaign orchestration."""

from __future__ import annotations

import fcntl
import hashlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactStore,
    CompletionStatus,
    CorruptArtifactError,
    DuplicateArtifactError,
    IncompatibleArtifactError,
    ResultBundle,
    read_immutable_json,
    sha256_file,
    write_immutable_bytes,
    write_immutable_json,
)
from .protocol import (
    SHARED_PRIMARY_FAMILIES,
    BudgetPool,
    CampaignManifest,
    ModelFamily,
    ProtocolValidationError,
    RunIdentity,
    Stage,
    TuningCandidate,
    content_digest,
)


class CampaignError(RuntimeError):
    """Base error for campaign planning and state transitions."""


class UndeclaredRunError(CampaignError):
    """Raised when a run is not in the frozen stage plan."""


class GateBlockedError(CampaignError):
    """Raised when a predecessor gate has not passed."""


class DuplicateGateError(CampaignError):
    """Raised when code attempts to rewrite a terminal gate decision."""


class BudgetExceededError(CampaignError):
    """Raised when a run would exceed its non-transferable allocation pool."""


class RunNotReservedError(CampaignError):
    """Raised when a result has no matching immutable ledger reservation."""


class Gate(StrEnum):
    """Monotonic gates in the binding Ising campaign kill tree."""

    THEOREM_NOVELTY = "theorem_novelty"
    TARGET_CALIBRATION = "target_calibration"
    SAMPLING_QUALITY = "sampling_quality"
    RUNTIME_FEASIBILITY = "runtime_feasibility"
    REPRESENTABILITY = "representability"
    TUNING_PROTOCOL = "tuning_protocol"
    BENEFIT_RETENTION = "benefit_retention"
    INTERPOLATION = "interpolation"
    ANCILLA_EXTENSION = "ancilla_extension"


class GateDecision(StrEnum):
    """Terminal decision for one gate in one campaign version."""

    PASSED = "passed"
    FAILED = "failed"


GATE_PREDECESSORS: dict[Gate, tuple[Gate, ...]] = {
    Gate.THEOREM_NOVELTY: (),
    Gate.TARGET_CALIBRATION: (Gate.THEOREM_NOVELTY,),
    Gate.SAMPLING_QUALITY: (Gate.TARGET_CALIBRATION,),
    Gate.RUNTIME_FEASIBILITY: (Gate.SAMPLING_QUALITY,),
    Gate.REPRESENTABILITY: (Gate.RUNTIME_FEASIBILITY,),
    Gate.TUNING_PROTOCOL: (Gate.REPRESENTABILITY,),
    Gate.BENEFIT_RETENTION: (Gate.TUNING_PROTOCOL,),
    Gate.INTERPOLATION: (Gate.TUNING_PROTOCOL,),
    Gate.ANCILLA_EXTENSION: (Gate.BENEFIT_RETENTION, Gate.INTERPOLATION),
}


def gate_ancestors(gate: Gate) -> tuple[Gate, ...]:
    """Return every transitive predecessor in deterministic gate order."""
    result: list[Gate] = []

    def visit(current: Gate) -> None:
        for predecessor in GATE_PREDECESSORS[current]:
            visit(predecessor)
            if predecessor not in result:
                result.append(predecessor)

    visit(gate)
    return tuple(result)


STAGE_RELEASE_GATES: dict[Stage, tuple[Gate, ...]] = {
    Stage.RUNTIME_FEASIBILITY: (Gate.SAMPLING_QUALITY,),
    Stage.REPRESENTABILITY_SCREEN: (Gate.RUNTIME_FEASIBILITY,),
    Stage.TUNING: (Gate.REPRESENTABILITY,),
    Stage.PRIMARY: (Gate.TUNING_PROTOCOL,),
    Stage.ANCILLA: (Gate.ANCILLA_EXTENSION,),
}

STAGE_BUDGET_POOLS: dict[Stage, BudgetPool] = {
    Stage.RUNTIME_FEASIBILITY: BudgetPool.FEASIBILITY_REPRESENTABILITY,
    Stage.REPRESENTABILITY_SCREEN: BudgetPool.FEASIBILITY_REPRESENTABILITY,
    Stage.TUNING: BudgetPool.TUNING,
    Stage.PRIMARY: BudgetPool.PRIMARY,
    Stage.ANCILLA: BudgetPool.ANCILLA,
    Stage.REPRESENTABILITY_DIAGNOSTIC: BudgetPool.FEASIBILITY_REPRESENTABILITY,
}


class PlanStatus(StrEnum):
    """Dry-run state of a declared immutable run identity."""

    ELIGIBLE = "eligible"
    BLOCKED = "blocked"
    RESERVED = "reserved"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class PlanEntry:
    """One dry-run planning row with no state mutation."""

    run: RunIdentity
    status: PlanStatus
    reason: str | None = None


@dataclass(frozen=True)
class TuningSelection:
    """One selected manifest candidate, repeated in the immutable gate record."""

    model_family: ModelFamily
    candidate_id: str
    initial_learning_rate: float
    final_learning_rate: float
    phase_feature_assignment: str | None

    def __post_init__(self) -> None:
        TuningCandidate(
            candidate_id=self.candidate_id,
            model_family=self.model_family,
            initial_learning_rate=self.initial_learning_rate,
            final_learning_rate=self.final_learning_rate,
            phase_feature_assignment=self.phase_feature_assignment,
        )

    @classmethod
    def from_candidate(cls, candidate: TuningCandidate) -> TuningSelection:
        """Copy a preregistered candidate into a terminal selection record."""
        return cls(
            model_family=candidate.model_family,
            candidate_id=candidate.candidate_id,
            initial_learning_rate=candidate.initial_learning_rate,
            final_learning_rate=candidate.final_learning_rate,
            phase_feature_assignment=candidate.phase_feature_assignment,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_family": self.model_family.value,
            "candidate_id": self.candidate_id,
            "initial_learning_rate": self.initial_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "phase_feature_assignment": self.phase_feature_assignment,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TuningSelection:
        expected = {
            "model_family",
            "candidate_id",
            "initial_learning_rate",
            "final_learning_rate",
            "phase_feature_assignment",
        }
        if set(value) != expected:
            raise ProtocolValidationError("tuning selection keys do not match schema")
        try:
            candidate = TuningCandidate.from_dict(
                {
                    "candidate_id": value["candidate_id"],
                    "model_family": value["model_family"],
                    "initial_learning_rate": value["initial_learning_rate"],
                    "final_learning_rate": value["final_learning_rate"],
                    "phase_feature_assignment": value["phase_feature_assignment"],
                },
            )
        except ProtocolValidationError as error:
            raise ProtocolValidationError("invalid tuning selection") from error
        return cls.from_candidate(candidate)


@dataclass(frozen=True)
class TuningDecision:
    """Evidence-bound winners selected only after the 36-run tuning screen."""

    manifest_digest: str
    evidence_digest: str
    selection_rule: str
    selections: tuple[TuningSelection, ...]

    def __post_init__(self) -> None:
        for name, digest in (
            ("manifest_digest", self.manifest_digest),
            ("evidence_digest", self.evidence_digest),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ProtocolValidationError(
                    f"tuning decision {name} must be a lowercase SHA-256 digest",
                )
        families = tuple(selection.model_family for selection in self.selections)
        if families != SHARED_PRIMARY_FAMILIES:
            raise ProtocolValidationError(
                "tuning decision requires one ordered selection per shared family",
            )
        if not self.selection_rule:
            raise ProtocolValidationError(
                "tuning decision selection_rule must be non-empty",
            )

    @classmethod
    def create(
        cls,
        manifest: CampaignManifest,
        candidate_ids: Mapping[ModelFamily, str],
        *,
        evidence_digest: str,
    ) -> TuningDecision:
        """Create a terminal decision from three preregistered candidate IDs."""
        if set(candidate_ids) != set(SHARED_PRIMARY_FAMILIES):
            raise ProtocolValidationError(
                "tuning decision must select every shared model family",
            )
        selections = []
        for family in SHARED_PRIMARY_FAMILIES:
            candidate = manifest.training_protocol.candidate(candidate_ids[family])
            if candidate.model_family is not family:
                raise ProtocolValidationError(
                    f"candidate {candidate.candidate_id} does not belong to {family}",
                )
            selections.append(TuningSelection.from_candidate(candidate))
        return cls(
            manifest_digest=manifest.digest,
            evidence_digest=evidence_digest,
            selection_rule=manifest.training_protocol.tuning_selection_rule,
            selections=tuple(selections),
        )

    def validate_against(self, manifest: CampaignManifest) -> None:
        """Reject a decision that is not an exact selection from this manifest."""
        if self.manifest_digest != manifest.digest:
            raise IncompatibleArtifactError(
                "tuning decision belongs to a different campaign manifest",
            )
        if self.selection_rule != manifest.training_protocol.tuning_selection_rule:
            raise IncompatibleArtifactError(
                "tuning decision uses a different selection rule",
            )
        for selection in self.selections:
            candidate = manifest.training_protocol.candidate(selection.candidate_id)
            if selection != TuningSelection.from_candidate(candidate):
                raise IncompatibleArtifactError(
                    "tuning decision does not match its manifest candidate",
                )

    def selection_for(self, family: ModelFamily) -> TuningSelection:
        """Return the frozen winner for one shared family."""
        for selection in self.selections:
            if selection.model_family is family:
                return selection
        raise ProtocolValidationError(f"no tuning selection for {family.value}")

    @property
    def digest(self) -> str:
        """Content digest used to bind the terminal selection record."""
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_digest": self.manifest_digest,
            "evidence_digest": self.evidence_digest,
            "selection_rule": self.selection_rule,
            "selections": [selection.to_dict() for selection in self.selections],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TuningDecision:
        expected = {
            "manifest_digest",
            "evidence_digest",
            "selection_rule",
            "selections",
        }
        if set(value) != expected:
            raise ProtocolValidationError("tuning decision keys do not match schema")
        return cls(
            manifest_digest=str(value["manifest_digest"]),
            evidence_digest=str(value["evidence_digest"]),
            selection_rule=str(value["selection_rule"]),
            selections=tuple(
                TuningSelection.from_dict(selection)
                for selection in value["selections"]
            ),
        )


@dataclass(frozen=True)
class GateRecord:
    """Immutable evidence-backed gate decision."""

    gate: Gate
    decision: GateDecision
    manifest_digest: str
    evidence_digest: str
    decided_at: str
    reason: str
    tuning_decision: TuningDecision | None = None

    def __post_init__(self) -> None:
        for name, digest in (
            ("manifest_digest", self.manifest_digest),
            ("evidence_digest", self.evidence_digest),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ProtocolValidationError(
                    f"gate {name} must be a lowercase SHA-256 digest",
                )
        _parse_timestamp(self.decided_at, "gate decided_at")
        if not self.reason:
            raise ProtocolValidationError("gate reason must be non-empty")
        requires_decision = (
            self.gate is Gate.TUNING_PROTOCOL and self.decision is GateDecision.PASSED
        )
        if requires_decision != (self.tuning_decision is not None):
            raise ProtocolValidationError(
                "only a passed tuning-protocol gate carries a tuning decision",
            )
        if self.tuning_decision is not None:
            if self.tuning_decision.manifest_digest != self.manifest_digest:
                raise IncompatibleArtifactError(
                    "gate and tuning decision manifest digests differ",
                )
            if self.tuning_decision.evidence_digest != self.evidence_digest:
                raise IncompatibleArtifactError(
                    "gate and tuning decision evidence digests differ",
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate.value,
            "decision": self.decision.value,
            "manifest_digest": self.manifest_digest,
            "evidence_digest": self.evidence_digest,
            "decided_at": self.decided_at,
            "reason": self.reason,
            "tuning_decision": (
                self.tuning_decision.to_dict() if self.tuning_decision else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GateRecord:
        expected = {
            "gate",
            "decision",
            "manifest_digest",
            "evidence_digest",
            "decided_at",
            "reason",
            "tuning_decision",
        }
        if set(value) != expected:
            raise ProtocolValidationError("gate record keys do not match schema")
        try:
            gate = Gate(str(value["gate"]))
            decision = GateDecision(str(value["decision"]))
        except ValueError as error:
            raise ProtocolValidationError("unknown gate record vocabulary") from error
        tuning_value = value["tuning_decision"]
        tuning_decision = (
            None if tuning_value is None else TuningDecision.from_dict(tuning_value)
        )
        return cls(
            gate=gate,
            decision=decision,
            manifest_digest=str(value["manifest_digest"]),
            evidence_digest=str(value["evidence_digest"]),
            decided_at=str(value["decided_at"]),
            reason=str(value["reason"]),
            tuning_decision=tuning_decision,
        )

    @property
    def digest(self) -> str:
        """Return the digest bound by a separately archived evidence record."""
        return content_digest(self.to_dict())


@dataclass(frozen=True)
class GateEvidenceBinding:
    """Immutable path-and-SHA binding for one unchanged terminal gate record."""

    schema_version: int
    gate: Gate
    manifest_digest: str
    gate_record_digest: str
    evidence_path: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ProtocolValidationError(
                "gate evidence binding schema_version must be 1",
            )
        for name, digest in (
            ("manifest_digest", self.manifest_digest),
            ("gate_record_digest", self.gate_record_digest),
            ("evidence_sha256", self.evidence_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ProtocolValidationError(
                    f"gate evidence {name} must be a lowercase SHA-256 digest",
                )
        path = Path(self.evidence_path)
        if not self.evidence_path or path.is_absolute() or ".." in path.parts:
            raise ProtocolValidationError(
                "gate evidence_path must be a campaign-relative path",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "gate": self.gate.value,
            "manifest_digest": self.manifest_digest,
            "gate_record_digest": self.gate_record_digest,
            "evidence_path": self.evidence_path,
            "evidence_sha256": self.evidence_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GateEvidenceBinding:
        expected = {
            "schema_version",
            "gate",
            "manifest_digest",
            "gate_record_digest",
            "evidence_path",
            "evidence_sha256",
        }
        if set(value) != expected:
            raise ProtocolValidationError(
                "gate evidence binding keys do not match schema",
            )
        try:
            gate = Gate(str(value["gate"]))
        except ValueError as error:
            raise ProtocolValidationError(
                "unknown gate evidence binding vocabulary",
            ) from error
        return cls(
            schema_version=value["schema_version"],
            gate=gate,
            manifest_digest=str(value["manifest_digest"]),
            gate_record_digest=str(value["gate_record_digest"]),
            evidence_path=str(value["evidence_path"]),
            evidence_sha256=str(value["evidence_sha256"]),
        )


@dataclass(frozen=True)
class RunReservation:
    """Immutable ledger event reserving one full run from one budget pool."""

    run: RunIdentity
    manifest_digest: str
    budget_pool: BudgetPool
    reserved_at: str

    def __post_init__(self) -> None:
        if self.manifest_digest != self.run.configuration_digest:
            raise IncompatibleArtifactError(
                "reservation manifest and run configuration digests differ",
            )
        _parse_timestamp(self.reserved_at, "reservation reserved_at")
        if STAGE_BUDGET_POOLS[self.run.stage] is not self.budget_pool:
            raise ProtocolValidationError("reservation uses the wrong budget pool")

    def to_dict(self) -> dict[str, object]:
        return {
            "run": self.run.to_dict(),
            "manifest_digest": self.manifest_digest,
            "budget_pool": self.budget_pool.value,
            "reserved_at": self.reserved_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunReservation:
        expected = {"run", "manifest_digest", "budget_pool", "reserved_at"}
        if set(value) != expected:
            raise ProtocolValidationError("reservation keys do not match schema")
        try:
            budget_pool = BudgetPool(str(value["budget_pool"]))
        except ValueError as error:
            raise ProtocolValidationError("unknown reservation budget pool") from error
        return cls(
            run=RunIdentity.from_dict(value["run"]),
            manifest_digest=str(value["manifest_digest"]),
            budget_pool=budget_pool,
            reserved_at=str(value["reserved_at"]),
        )


def _parse_timestamp(value: str, context: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProtocolValidationError(f"{context} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProtocolValidationError(f"{context} must include a UTC offset")
    return parsed


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _run_identity(
    manifest: CampaignManifest,
    *,
    stage: Stage,
    coupling_seed: int,
    condition_scope: str,
    model_family: ModelFamily,
    optimizer_seed: int,
    phase_feature_assignment: str | None = None,
    tuning_candidate_id: str | None = None,
    ancilla_rung: int | None = None,
) -> RunIdentity:
    return RunIdentity(
        campaign_version=manifest.campaign_version,
        stage=stage,
        coupling_seed=coupling_seed,
        condition_scope=condition_scope,
        model_family=model_family,
        phase_feature_assignment=phase_feature_assignment,
        tuning_candidate_id=tuning_candidate_id,
        ancilla_rung=ancilla_rung,
        optimizer_seed=optimizer_seed,
        data_seed=manifest.data_seed_for(coupling_seed),
        configuration_digest=manifest.digest,
    )


def plan_runtime_feasibility(
    manifest: CampaignManifest,
) -> tuple[RunIdentity, ...]:
    """Return the sole production-budget feasibility benchmark."""
    return (
        _run_identity(
            manifest,
            stage=Stage.RUNTIME_FEASIBILITY,
            coupling_seed=666,
            condition_scope="all_training_betas",
            model_family=ModelFamily.FIXED_INPUT,
            optimizer_seed=manifest.optimizer_seeds[0],
        ),
    )


def plan_representability_screen(
    manifest: CampaignManifest,
) -> tuple[RunIdentity, ...]:
    """Return the exact 2-instance x low/anchor/high Separate-model screen."""
    anchor_index = next(
        index
        for index, beta in enumerate(manifest.beta_grid.training)
        if abs(beta - 1.0 / 3.0) <= 1e-14
    )
    beta_indices = (0, anchor_index, len(manifest.beta_grid.training) - 1)
    result = tuple(
        _run_identity(
            manifest,
            stage=Stage.REPRESENTABILITY_SCREEN,
            coupling_seed=coupling_seed,
            condition_scope=f"train_beta_{beta_index}",
            model_family=ModelFamily.SEPARATE,
            optimizer_seed=manifest.optimizer_seeds[0],
        )
        for coupling_seed in manifest.training_protocol.tuning_coupling_seeds
        for beta_index in beta_indices
    )
    if len(result) != 6 or len({run.run_id for run in result}) != 6:
        raise ProtocolValidationError(
            "representability plan is not six unique run identities",
        )
    return result


def plan_tuning(manifest: CampaignManifest) -> tuple[RunIdentity, ...]:
    """Return the frozen 9 candidates x 2 instances x 2 seeds tuning matrix."""
    result = tuple(
        _run_identity(
            manifest,
            stage=Stage.TUNING,
            coupling_seed=coupling_seed,
            condition_scope="all_training_betas",
            model_family=candidate.model_family,
            optimizer_seed=optimizer_seed,
            phase_feature_assignment=candidate.phase_feature_assignment,
            tuning_candidate_id=candidate.candidate_id,
        )
        for candidate in manifest.training_protocol.tuning_candidates
        for coupling_seed in manifest.training_protocol.tuning_coupling_seeds
        for optimizer_seed in manifest.training_protocol.tuning_optimizer_seeds
    )
    if len(result) != manifest.run_allocation.tuning or len(
        {run.run_id for run in result},
    ) != len(result):
        raise ProtocolValidationError("tuning plan is not 36 unique run identities")
    return result


def plan_primary(
    manifest: CampaignManifest,
    tuning_decision: TuningDecision,
) -> tuple[RunIdentity, ...]:
    """Return the exact deterministic 8 x 5 x (3 + 5) primary matrix."""
    tuning_decision.validate_against(manifest)
    runs = []
    for coupling_seed in manifest.coupling_seeds:
        for optimizer_seed in manifest.optimizer_seeds:
            for family in SHARED_PRIMARY_FAMILIES:
                selection = tuning_decision.selection_for(family)
                runs.append(
                    _run_identity(
                        manifest,
                        stage=Stage.PRIMARY,
                        coupling_seed=coupling_seed,
                        condition_scope="all_training_betas",
                        model_family=family,
                        optimizer_seed=optimizer_seed,
                        phase_feature_assignment=selection.phase_feature_assignment,
                        tuning_candidate_id=selection.candidate_id,
                    ),
                )
            for beta_index in range(len(manifest.beta_grid.training)):
                runs.append(
                    _run_identity(
                        manifest,
                        stage=Stage.PRIMARY,
                        coupling_seed=coupling_seed,
                        condition_scope=f"train_beta_{beta_index}",
                        model_family=ModelFamily.SEPARATE,
                        optimizer_seed=optimizer_seed,
                    ),
                )
    result = tuple(runs)
    if len(result) != manifest.run_allocation.primary or len(
        {run.run_id for run in result},
    ) != len(result):
        raise ProtocolValidationError("primary plan is not 320 unique run identities")
    return result


def plan_ancilla(
    manifest: CampaignManifest,
    tuning_decision: TuningDecision,
) -> tuple[RunIdentity, ...]:
    """Return the exact deterministic 2 x 3 x 2 x 2 ancilla matrix."""
    tuning_decision.validate_against(manifest)
    fixed_selection = tuning_decision.selection_for(ModelFamily.FIXED_INPUT)
    runs = []
    for coupling_seed in manifest.ancilla.coupling_seeds:
        for optimizer_seed in manifest.ancilla.optimizer_seeds:
            for rung in manifest.ancilla.rungs:
                for family in (
                    ModelFamily.FIXED_INPUT,
                    ModelFamily.MATCHED_CAPACITY_CONTROL,
                ):
                    runs.append(
                        _run_identity(
                            manifest,
                            stage=Stage.ANCILLA,
                            coupling_seed=coupling_seed,
                            condition_scope="all_training_betas",
                            model_family=family,
                            optimizer_seed=optimizer_seed,
                            tuning_candidate_id=fixed_selection.candidate_id,
                            ancilla_rung=rung,
                        ),
                    )
    result = tuple(runs)
    if len(result) != manifest.run_allocation.ancilla or len(
        {run.run_id for run in result},
    ) != len(result):
        raise ProtocolValidationError("ancilla plan is not 24 unique run identities")
    return result


def plan_representability_diagnostic(
    manifest: CampaignManifest,
) -> tuple[RunIdentity, ...]:
    """Return the sole exact-objective diagnostic allowed after gate failure."""
    return (
        _run_identity(
            manifest,
            stage=Stage.REPRESENTABILITY_DIAGNOSTIC,
            coupling_seed=666,
            condition_scope="screened_low_anchor_high_betas",
            model_family=ModelFamily.SEPARATE,
            optimizer_seed=manifest.optimizer_seeds[0],
        ),
    )


def planned_runs(
    manifest: CampaignManifest,
    stage: Stage,
    tuning_decision: TuningDecision | None = None,
) -> tuple[RunIdentity, ...]:
    """Return every declared run identity for a supported campaign stage."""
    if stage is Stage.RUNTIME_FEASIBILITY:
        return plan_runtime_feasibility(manifest)
    if stage is Stage.REPRESENTABILITY_SCREEN:
        return plan_representability_screen(manifest)
    if stage is Stage.TUNING:
        return plan_tuning(manifest)
    if stage is Stage.PRIMARY:
        if tuning_decision is None:
            raise ProtocolValidationError(
                "primary planning requires a passed tuning decision",
            )
        return plan_primary(manifest, tuning_decision)
    if stage is Stage.ANCILLA:
        if tuning_decision is None:
            raise ProtocolValidationError(
                "ancilla planning requires a passed tuning decision",
            )
        return plan_ancilla(manifest, tuning_decision)
    if stage is Stage.REPRESENTABILITY_DIAGNOSTIC:
        return plan_representability_diagnostic(manifest)
    raise ProtocolValidationError(f"unsupported campaign stage: {stage}")


class CampaignStore:
    """Filesystem-backed acceptance seam for one frozen campaign manifest."""

    def __init__(self, root: Path, manifest: CampaignManifest) -> None:
        self.root = root
        self.manifest = manifest
        self.manifest_path = root / "manifest.json"
        self.gates_directory = root / "gates"
        self.gate_evidence_bindings_directory = root / "gate-evidence-bindings"
        self.gate_evidence_archive_directory = root / "evidence" / "gates"
        self.reservations_directory = root / "ledger" / "reservations"
        self.lock_path = root / "ledger" / ".lock"
        self.gates_directory.mkdir(parents=True, exist_ok=True)
        self.reservations_directory.mkdir(parents=True, exist_ok=True)
        self._ensure_manifest_snapshot()
        self.artifacts = ArtifactStore(root / "artifacts", manifest.digest)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _ensure_manifest_snapshot(self) -> None:
        payload = {
            "manifest": self.manifest.to_dict(),
            "manifest_digest": self.manifest.digest,
        }
        if not self.manifest_path.exists():
            try:
                write_immutable_json(self.manifest_path, payload)
            except DuplicateArtifactError:
                pass
        stored = read_immutable_json(self.manifest_path)
        if set(stored) != {"manifest", "manifest_digest"}:
            raise CorruptArtifactError("campaign manifest snapshot has invalid keys")
        try:
            stored_manifest = CampaignManifest.from_dict(stored["manifest"])
        except ProtocolValidationError as error:
            raise CorruptArtifactError(
                "campaign manifest snapshot is invalid",
            ) from error
        if (
            stored["manifest_digest"] != stored_manifest.digest
            or stored_manifest.digest != self.manifest.digest
        ):
            raise IncompatibleArtifactError(
                "campaign store is already bound to a different manifest",
            )

    def _gate_path(self, gate: Gate) -> Path:
        return self.gates_directory / f"{gate.value}.json"

    def gate_record(self, gate: Gate) -> GateRecord | None:
        """Return one verified terminal gate record, if it exists."""
        path = self._gate_path(gate)
        if not path.exists():
            return None
        try:
            record = GateRecord.from_dict(read_immutable_json(path))
        except ProtocolValidationError as error:
            raise CorruptArtifactError(f"invalid gate record: {gate.value}") from error
        if record.gate is not gate:
            raise CorruptArtifactError("gate filename and embedded gate differ")
        if record.manifest_digest != self.manifest.digest:
            raise IncompatibleArtifactError("gate belongs to a different manifest")
        if record.tuning_decision is not None:
            record.tuning_decision.validate_against(self.manifest)
        return record

    def _gate_evidence_binding_path(self, gate: Gate) -> Path:
        return self.gate_evidence_bindings_directory / f"{gate.value}.json"

    def gate_evidence_binding(self, gate: Gate) -> GateEvidenceBinding | None:
        """Return and fully verify one archived gate-evidence binding."""
        binding_path = self._gate_evidence_binding_path(gate)
        if not binding_path.exists():
            return None
        record = self.gate_record(gate)
        if record is None:
            raise CorruptArtifactError(
                f"gate evidence exists without terminal decision: {gate.value}",
            )
        try:
            binding = GateEvidenceBinding.from_dict(
                read_immutable_json(binding_path),
            )
        except ProtocolValidationError as error:
            raise CorruptArtifactError(
                f"invalid gate evidence binding: {gate.value}",
            ) from error
        if binding.gate is not gate:
            raise CorruptArtifactError(
                "gate evidence filename and embedded gate differ",
            )
        if binding.manifest_digest != self.manifest.digest:
            raise IncompatibleArtifactError(
                "gate evidence belongs to a different manifest",
            )
        if binding.gate_record_digest != record.digest:
            raise CorruptArtifactError(
                "gate evidence is not bound to the terminal gate record",
            )
        if binding.evidence_sha256 != record.evidence_digest:
            raise CorruptArtifactError(
                "gate evidence SHA differs from the terminal decision",
            )
        evidence_path = (self.root / binding.evidence_path).resolve()
        try:
            evidence_path.relative_to(self.root.resolve())
        except ValueError as error:
            raise CorruptArtifactError(
                "gate evidence resolves outside the campaign root",
            ) from error
        if not evidence_path.is_file():
            raise CorruptArtifactError(
                f"archived gate evidence is missing: {binding.evidence_path}",
            )
        if sha256_file(evidence_path) != binding.evidence_sha256:
            raise CorruptArtifactError(
                f"archived gate evidence digest mismatch: {gate.value}",
            )
        return binding

    def archive_gate_evidence(
        self,
        gate: Gate,
        source: Path,
    ) -> GateEvidenceBinding:
        """Archive evidence for a terminal decision without rewriting that decision."""
        source_bytes = source.read_bytes()
        source_digest = hashlib.sha256(source_bytes).hexdigest()
        suffix = source.suffix.lower() if source.suffix else ".bin"
        if not suffix[1:].isalnum() or len(suffix) > 12:
            suffix = ".bin"
        with self._locked():
            record = self.gate_record(gate)
            if record is None:
                raise GateBlockedError(
                    f"gate {gate.value} has no terminal decision to bind",
                )
            if source_digest != record.evidence_digest:
                raise IncompatibleArtifactError(
                    "evidence bytes do not match the terminal gate digest",
                )
            existing = self.gate_evidence_binding(gate)
            if existing is not None:
                if existing.evidence_sha256 != source_digest:
                    raise DuplicateArtifactError(
                        f"gate {gate.value} already binds different evidence",
                    )
                return existing

            relative_archive = (
                Path("evidence") / "gates" / gate.value / f"{source_digest}{suffix}"
            )
            archive_path = self.root / relative_archive
            if archive_path.exists():
                if sha256_file(archive_path) != source_digest:
                    raise CorruptArtifactError(
                        f"conflicting archived gate evidence: {archive_path}",
                    )
            else:
                write_immutable_bytes(archive_path, source_bytes)
            binding = GateEvidenceBinding(
                schema_version=1,
                gate=gate,
                manifest_digest=self.manifest.digest,
                gate_record_digest=record.digest,
                evidence_path=str(relative_archive),
                evidence_sha256=source_digest,
            )
            write_immutable_json(
                self._gate_evidence_binding_path(gate),
                binding.to_dict(),
            )
            return self.gate_evidence_binding(gate) or binding

    def restore_gate_evidence(self, gate: Gate, destination: Path) -> Path:
        """Restore verified archived evidence without modifying the gate decision."""
        binding = self.gate_evidence_binding(gate)
        if binding is None:
            raise CorruptArtifactError(
                f"gate {gate.value} has no archived evidence binding",
            )
        archive_path = self.root / binding.evidence_path
        if destination.exists():
            if not destination.is_file() or sha256_file(destination) != (
                binding.evidence_sha256
            ):
                raise DuplicateArtifactError(
                    f"restore destination already contains different content: "
                    f"{destination}",
                )
            return destination
        write_immutable_bytes(destination, archive_path.read_bytes())
        return destination

    def release_gate(
        self,
        gate: Gate,
        decision: GateDecision,
        *,
        evidence_digest: str,
        reason: str,
        decided_at: str | None = None,
        tuning_decision: TuningDecision | None = None,
    ) -> GateRecord:
        """Record an external theorem decision or an evidence-backed failure.

        Successful downstream decisions are deliberately unavailable through
        this raw primitive.  They must be derived from archived scientific
        outputs by :mod:`iqp_cond.campaign_finalizers`, which calls the private
        finalized-decision seam below only after constructing canonical gate
        evidence.  Keeping failures available here ensures that an operational
        or scientific stop can always be recorded without fabricating a pass.
        """
        if gate is not Gate.THEOREM_NOVELTY and decision is GateDecision.PASSED:
            raise ProtocolValidationError(
                f"passed {gate.value} decisions must be derived by its "
                "campaign finalizer",
            )
        return self._record_gate(
            gate,
            decision,
            evidence_digest=evidence_digest,
            reason=reason,
            decided_at=decided_at,
            tuning_decision=tuning_decision,
        )

    def _release_finalized_gate(
        self,
        gate: Gate,
        decision: GateDecision,
        *,
        evidence_digest: str,
        reason: str,
        decided_at: str | None = None,
        tuning_decision: TuningDecision | None = None,
    ) -> GateRecord:
        """Persist one decision already derived by a gate-specific finalizer."""
        if gate is Gate.THEOREM_NOVELTY:
            raise ProtocolValidationError(
                "the externally adjudicated theorem gate uses release_gate",
            )
        return self._record_gate(
            gate,
            decision,
            evidence_digest=evidence_digest,
            reason=reason,
            decided_at=decided_at,
            tuning_decision=tuning_decision,
        )

    def _record_gate(
        self,
        gate: Gate,
        decision: GateDecision,
        *,
        evidence_digest: str,
        reason: str,
        decided_at: str | None,
        tuning_decision: TuningDecision | None,
    ) -> GateRecord:
        """Implement the shared monotonic gate-record transition."""
        with self._locked():
            if self._gate_path(gate).exists():
                raise DuplicateGateError(
                    f"gate {gate.value} already has a terminal decision",
                )
            for predecessor in gate_ancestors(gate):
                record = self.gate_record(predecessor)
                if record is not None and record.decision is GateDecision.FAILED:
                    raise GateBlockedError(
                        f"gate {gate.value} is terminally blocked by "
                        f"failed {predecessor.value}",
                    )
            for predecessor in GATE_PREDECESSORS[gate]:
                record = self.gate_record(predecessor)
                if record is None:
                    raise GateBlockedError(
                        f"gate {gate.value} awaits {predecessor.value}",
                    )
            if tuning_decision is not None:
                tuning_decision.validate_against(self.manifest)
            record = GateRecord(
                gate=gate,
                decision=decision,
                manifest_digest=self.manifest.digest,
                evidence_digest=evidence_digest,
                decided_at=decided_at or _now(),
                reason=reason,
                tuning_decision=tuning_decision,
            )
            try:
                write_immutable_json(self._gate_path(gate), record.to_dict())
            except DuplicateArtifactError as error:
                raise DuplicateGateError(
                    f"gate {gate.value} already has a terminal decision",
                ) from error
            return record

    def _stage_block_reason(self, stage: Stage) -> str | None:
        if stage is Stage.REPRESENTABILITY_DIAGNOSTIC:
            for ancestor in gate_ancestors(Gate.REPRESENTABILITY):
                ancestor_record = self.gate_record(ancestor)
                if (
                    ancestor_record is not None
                    and ancestor_record.decision is GateDecision.FAILED
                ):
                    return (
                        f"diagnostic is terminally blocked by failed {ancestor.value}"
                    )
            representability = self.gate_record(Gate.REPRESENTABILITY)
            if representability is None:
                return "representability gate has no decision"
            if representability.decision is not GateDecision.FAILED:
                return "diagnostic is released only after representability failure"
            return None
        for gate in STAGE_RELEASE_GATES[stage]:
            for ancestor in gate_ancestors(gate):
                ancestor_record = self.gate_record(ancestor)
                if (
                    ancestor_record is not None
                    and ancestor_record.decision is GateDecision.FAILED
                ):
                    return (
                        f"stage {stage.value} is terminally blocked by failed "
                        f"{ancestor.value}"
                    )
            record = self.gate_record(gate)
            if record is None:
                return f"required gate {gate.value} has no decision"
            if record.decision is GateDecision.FAILED:
                return f"required gate {gate.value} failed terminally"
        return None

    def _assert_stage_released(self, stage: Stage) -> None:
        reason = self._stage_block_reason(stage)
        if reason is not None:
            raise GateBlockedError(reason)

    def tuning_decision(self) -> TuningDecision | None:
        """Return the immutable passed tuning decision, if it exists."""
        record = self.gate_record(Gate.TUNING_PROTOCOL)
        if record is None or record.decision is GateDecision.FAILED:
            return None
        if record.tuning_decision is None:
            raise CorruptArtifactError(
                "passed tuning gate is missing its tuning decision",
            )
        return record.tuning_decision

    def _planned_runs(self, stage: Stage) -> tuple[RunIdentity, ...]:
        return planned_runs(
            self.manifest,
            stage,
            tuning_decision=self.tuning_decision(),
        )

    def _declared_by_id(self, stage: Stage) -> dict[str, RunIdentity]:
        return {run.run_id: run for run in self._planned_runs(stage)}

    def _assert_declared(self, run: RunIdentity) -> None:
        declared = self._declared_by_id(run.stage).get(run.run_id)
        if declared is None or declared != run:
            raise UndeclaredRunError(
                f"run {run.run_id} is not declared by the frozen manifest",
            )

    def _reservation_path(self, run_id: str) -> Path:
        return self.reservations_directory / f"{run_id}.json"

    def reservation(self, run_id: str) -> RunReservation | None:
        """Return one verified immutable ledger reservation, if present."""
        path = self._reservation_path(run_id)
        if not path.exists():
            return None
        try:
            reservation = RunReservation.from_dict(read_immutable_json(path))
        except ProtocolValidationError as error:
            raise CorruptArtifactError(f"invalid reservation: {run_id}") from error
        if reservation.run.run_id != run_id:
            raise CorruptArtifactError("reservation filename and run identity differ")
        if reservation.manifest_digest != self.manifest.digest:
            raise IncompatibleArtifactError(
                "reservation belongs to a different campaign manifest",
            )
        self._assert_declared(reservation.run)
        return reservation

    def _reservations(self) -> tuple[RunReservation, ...]:
        return tuple(
            reservation
            for path in sorted(self.reservations_directory.glob("*.json"))
            if (reservation := self.reservation(path.stem)) is not None
        )

    def used_budget(self, pool: BudgetPool) -> int:
        """Count reservations in one pool without borrowing from another pool."""
        return sum(
            reservation.budget_pool is pool for reservation in self._reservations()
        )

    def remaining_budget(self, pool: BudgetPool) -> int:
        """Return the independently bounded remaining run count for one pool."""
        return self.manifest.run_allocation.for_pool(pool) - self.used_budget(pool)

    def reserve_run(
        self,
        run: RunIdentity,
        *,
        reserved_at: str | None = None,
    ) -> RunReservation:
        """Atomically reserve one declared run after its predecessor gate passes."""
        self._assert_stage_released(run.stage)
        self._assert_declared(run)
        with self._locked():
            if self._reservation_path(run.run_id).exists() or self.artifacts.has_bundle(
                run.run_id,
            ):
                raise DuplicateArtifactError(
                    f"run identity has already been reserved: {run.run_id}",
                )
            pool = STAGE_BUDGET_POOLS[run.stage]
            if self.remaining_budget(pool) <= 0:
                raise BudgetExceededError(
                    f"non-transferable {pool.value} allocation is exhausted",
                )
            reservation = RunReservation(
                run=run,
                manifest_digest=self.manifest.digest,
                budget_pool=pool,
                reserved_at=reserved_at or _now(),
            )
            write_immutable_json(
                self._reservation_path(run.run_id),
                reservation.to_dict(),
            )
            return reservation

    def _validate_parameter_accounting(self, bundle: ResultBundle) -> None:
        accounting = bundle.parameter_accounting
        run = bundle.run
        if run.stage is not Stage.ANCILLA:
            expected = (16, 136, 136, 0)
        else:
            rung = run.ancilla_rung
            if rung is None:
                raise ProtocolValidationError("ancilla result has no rung")
            total_qubits = 16 + rung
            parameters = total_qubits * (total_qubits + 1) // 2
            if run.model_family is ModelFamily.FIXED_INPUT:
                expected = (total_qubits, parameters, parameters, rung)
            else:
                expected = (16, parameters, parameters, 0)
        actual = (
            accounting.n_qubits,
            accounting.n_generators,
            accounting.n_trainable_parameters,
            accounting.n_ancillas,
        )
        if actual != expected:
            raise IncompatibleArtifactError(
                f"parameter accounting {actual} does not match declared run {expected}",
            )

    def record_result(self, bundle: ResultBundle) -> Path:
        """Persist a terminal bundle only for its exact prior reservation."""
        self._assert_declared(bundle.run)
        if bundle.manifest_digest != self.manifest.digest:
            raise IncompatibleArtifactError(
                "result bundle belongs to a different campaign manifest",
            )
        self._validate_parameter_accounting(bundle)
        with self._locked():
            reservation = self.reservation(bundle.run.run_id)
            if reservation is None:
                raise RunNotReservedError(
                    f"run {bundle.run.run_id} has no ledger reservation",
                )
            if reservation.run != bundle.run:
                raise IncompatibleArtifactError(
                    "result run identity differs from its reservation",
                )
            return self.artifacts.write_bundle(bundle)

    def result(self, run_id: str) -> ResultBundle | None:
        """Return a verified result, or ``None`` when the run is not terminal."""
        if not self.artifacts.has_bundle(run_id):
            return None
        return self.artifacts.load_bundle(run_id)

    def dry_run(self, stage: Stage) -> tuple[PlanEntry, ...]:
        """Describe the deterministic stage plan without changing the ledger."""
        block_reason = self._stage_block_reason(stage)
        if stage in {Stage.PRIMARY, Stage.ANCILLA} and self.tuning_decision() is None:
            raise GateBlockedError(
                block_reason or "planning requires a passed tuning decision",
            )
        entries = []
        for run in self._planned_runs(stage):
            result = self.result(run.run_id)
            reservation = self.reservation(run.run_id)
            if result is not None:
                status = (
                    PlanStatus.COMPLETED
                    if result.completion_status is CompletionStatus.COMPLETED
                    else PlanStatus.FAILED
                )
                reason = None
            elif reservation is not None:
                status = PlanStatus.RESERVED
                reason = "reserved without a terminal result"
            elif block_reason is not None:
                status = PlanStatus.BLOCKED
                reason = block_reason
            else:
                status = PlanStatus.ELIGIBLE
                reason = None
            entries.append(PlanEntry(run=run, status=status, reason=reason))
        return tuple(entries)

    def execute_one(
        self,
        run: RunIdentity,
        executor: Callable[[RunIdentity], ResultBundle],
        *,
        resume: bool = False,
        reserved_at: str | None = None,
    ) -> ResultBundle:
        """Reserve, execute, and immutably record one run through a caller adapter."""
        existing_reservation = self.reservation(run.run_id)
        if existing_reservation is None:
            self.reserve_run(run, reserved_at=reserved_at)
        elif not resume:
            raise DuplicateArtifactError(
                f"run {run.run_id} is already reserved; use resume=True",
            )
        elif self.result(run.run_id) is not None:
            raise DuplicateArtifactError(f"run {run.run_id} already has a result")
        bundle = executor(run)
        if bundle.run != run:
            raise IncompatibleArtifactError(
                "executor returned a bundle for a different run identity",
            )
        self.record_result(bundle)
        return bundle

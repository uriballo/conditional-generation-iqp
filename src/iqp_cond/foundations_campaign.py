"""Immutable kill tree and run ledger for ``arxiv_foundations_v1``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from .artifacts import (
    ArtifactStore,
    DuplicateArtifactError,
    ResultBundle,
    read_immutable_json,
    write_immutable_json,
)
from .foundations_protocol import (
    FoundationsManifest,
    FoundationsTuningDecision,
    plan_foundations_primary,
    plan_foundations_representability,
    plan_foundations_runtime,
    plan_foundations_tuning,
)
from .protocol import RunIdentity, Stage


class FoundationsCampaignError(RuntimeError):
    """Base error for the compact foundations campaign."""


class GateBlockedError(FoundationsCampaignError):
    """A frozen predecessor has not passed."""


class UndeclaredRunError(FoundationsCampaignError):
    """A run identity is outside the immutable compact plan."""


class FoundationsGate(StrEnum):
    DATA_READY = "data_ready"
    ESTIMATOR_CALIBRATION = "estimator_calibration"
    RUNTIME_FEASIBILITY = "runtime_feasibility"
    REPRESENTABILITY = "representability"
    TUNING = "tuning"
    CONDITIONAL_CONTROL = "conditional_control"
    INTERPOLATION = "interpolation"
    RELEASE_METADATA = "release_metadata"


class GateDecision(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


_PREDECESSORS: dict[FoundationsGate, tuple[FoundationsGate, ...]] = {
    FoundationsGate.DATA_READY: (),
    FoundationsGate.ESTIMATOR_CALIBRATION: (FoundationsGate.DATA_READY,),
    FoundationsGate.RUNTIME_FEASIBILITY: (
        FoundationsGate.DATA_READY,
        FoundationsGate.ESTIMATOR_CALIBRATION,
    ),
    FoundationsGate.REPRESENTABILITY: (FoundationsGate.RUNTIME_FEASIBILITY,),
    FoundationsGate.TUNING: (FoundationsGate.REPRESENTABILITY,),
    FoundationsGate.CONDITIONAL_CONTROL: (FoundationsGate.TUNING,),
    FoundationsGate.INTERPOLATION: (FoundationsGate.TUNING,),
    FoundationsGate.RELEASE_METADATA: (
        FoundationsGate.CONDITIONAL_CONTROL,
        FoundationsGate.INTERPOLATION,
    ),
}

_STAGE_GATES: dict[Stage, tuple[FoundationsGate, ...]] = {
    Stage.RUNTIME_FEASIBILITY: (
        FoundationsGate.DATA_READY,
        FoundationsGate.ESTIMATOR_CALIBRATION,
    ),
    Stage.REPRESENTABILITY_SCREEN: (FoundationsGate.RUNTIME_FEASIBILITY,),
    Stage.TUNING: (FoundationsGate.REPRESENTABILITY,),
    Stage.PRIMARY: (FoundationsGate.TUNING,),
}


@dataclass(frozen=True, slots=True)
class FoundationsGateRecord:
    gate: FoundationsGate
    decision: GateDecision
    manifest_digest: str
    evidence_digest: str
    finalized_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "gate": self.gate.value,
            "decision": self.decision.value,
            "manifest_digest": self.manifest_digest,
            "evidence_digest": self.evidence_digest,
            "finalized_at": self.finalized_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> FoundationsGateRecord:
        expected = {
            "gate",
            "decision",
            "manifest_digest",
            "evidence_digest",
            "finalized_at",
        }
        if set(value) != expected:
            raise ValueError("foundations gate record schema differs")
        return cls(
            gate=FoundationsGate(str(value["gate"])),
            decision=GateDecision(str(value["decision"])),
            manifest_digest=str(value["manifest_digest"]),
            evidence_digest=str(value["evidence_digest"]),
            finalized_at=str(value["finalized_at"]),
        )


class FoundationsCampaignStore:
    """Separate immutable store; it never reads or writes the failed v1 ledger."""

    def __init__(self, root: Path, manifest: FoundationsManifest) -> None:
        self.root = root
        self.manifest = manifest
        self.manifest_snapshot = root / "manifest.json"
        self.gates_directory = root / "gates"
        self.ledger_directory = root / "ledger"
        self.tuning_path = root / "decisions" / "tuning.json"
        self.artifacts = ArtifactStore(root / "artifacts", manifest.digest)
        root.mkdir(parents=True, exist_ok=True)
        if self.manifest_snapshot.exists():
            if read_immutable_json(self.manifest_snapshot) != manifest.to_dict():
                raise ValueError("campaign store manifest snapshot differs")
        else:
            write_immutable_json(self.manifest_snapshot, manifest.to_dict())

    def gate_path(self, gate: FoundationsGate) -> Path:
        return self.gates_directory / f"{gate.value}.json"

    def gate_record(self, gate: FoundationsGate) -> FoundationsGateRecord | None:
        path = self.gate_path(gate)
        if not path.exists():
            return None
        record = FoundationsGateRecord.from_dict(read_immutable_json(path))
        if record.manifest_digest != self.manifest.digest or record.gate is not gate:
            raise ValueError("foundations gate record is incompatible")
        return record

    def _require_passed(self, gates: tuple[FoundationsGate, ...]) -> None:
        for gate in gates:
            record = self.gate_record(gate)
            if record is None:
                raise GateBlockedError(f"gate {gate.value} is not finalized")
            if record.decision is GateDecision.FAILED:
                raise GateBlockedError(f"gate {gate.value} failed")

    def finalize_gate(
        self,
        gate: FoundationsGate,
        decision: GateDecision,
        *,
        evidence_digest: str,
    ) -> FoundationsGateRecord:
        if self.gate_path(gate).exists():
            raise ValueError(f"gate {gate.value} is already terminal")
        if len(evidence_digest) != 64 or any(
            character not in "0123456789abcdef" for character in evidence_digest
        ):
            raise ValueError("gate evidence_digest must be a lowercase SHA-256")
        self._require_passed(_PREDECESSORS[gate])
        record = FoundationsGateRecord(
            gate=gate,
            decision=decision,
            manifest_digest=self.manifest.digest,
            evidence_digest=evidence_digest,
            finalized_at=datetime.now(UTC).isoformat(),
        )
        write_immutable_json(self.gate_path(gate), record.to_dict())
        return record

    def finalize_tuning(
        self, decision: FoundationsTuningDecision
    ) -> FoundationsGateRecord:
        decision.validate_against(self.manifest)
        if self.tuning_path.exists():
            raise ValueError("tuning decision is already immutable")
        self._require_passed(_PREDECESSORS[FoundationsGate.TUNING])
        write_immutable_json(self.tuning_path, decision.to_dict())
        return self.finalize_gate(
            FoundationsGate.TUNING,
            GateDecision.PASSED,
            evidence_digest=decision.evidence_digest,
        )

    def tuning_decision(self) -> FoundationsTuningDecision:
        self._require_passed((FoundationsGate.TUNING,))
        decision = FoundationsTuningDecision.from_dict(
            read_immutable_json(self.tuning_path)
        )
        decision.validate_against(self.manifest)
        return decision

    def planned_runs(self, stage: Stage) -> tuple[RunIdentity, ...]:
        if stage is Stage.RUNTIME_FEASIBILITY:
            return plan_foundations_runtime(self.manifest)
        if stage is Stage.REPRESENTABILITY_SCREEN:
            return plan_foundations_representability(self.manifest)
        if stage is Stage.TUNING:
            return plan_foundations_tuning(self.manifest)
        if stage is Stage.PRIMARY:
            return plan_foundations_primary(self.manifest, self.tuning_decision())
        raise ValueError(f"unsupported foundations stage: {stage.value}")

    def reservation_path(self, run: RunIdentity) -> Path:
        return self.ledger_directory / f"{run.run_id}.json"

    def _declared(self, run: RunIdentity) -> bool:
        return any(
            candidate.run_id == run.run_id for candidate in self.planned_runs(run.stage)
        )

    def reserve(self, run: RunIdentity) -> Path:
        if run.configuration_digest != self.manifest.digest:
            raise UndeclaredRunError("run belongs to a different manifest")
        if run.stage not in _STAGE_GATES or not self._declared(run):
            raise UndeclaredRunError("run is outside the immutable foundations plan")
        self._require_passed(_STAGE_GATES[run.stage])
        path = self.reservation_path(run)
        if path.exists():
            raise ValueError("run is already reserved")
        write_immutable_json(
            path,
            {
                "manifest_digest": self.manifest.digest,
                "run": run.to_dict(),
                "reserved_at": datetime.now(UTC).isoformat(),
            },
        )
        return path

    def record_result(self, bundle: ResultBundle) -> Path:
        run = bundle.run
        if not self._declared(run):
            raise UndeclaredRunError("result run is outside the immutable plan")
        if not self.reservation_path(run).exists():
            raise ValueError("result has no immutable reservation")
        try:
            return self.artifacts.write_bundle(bundle)
        except DuplicateArtifactError as error:
            raise ValueError("duplicate run identity result") from error

    def require_complete(self, run: RunIdentity) -> ResultBundle:
        if not self.reservation_path(run).exists():
            raise ValueError("run was never reserved")
        if not self.artifacts.has_bundle(run.run_id):
            raise ValueError("partial result reuse is forbidden")
        return self.artifacts.load_bundle(run.run_id)

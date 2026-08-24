"""Immutable store for the eight-visible Curie--Weiss ancilla campaign."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import read_immutable_json, sha256_file, write_immutable_json
from .curie_weiss_ancilla_protocol import (
    CurieWeissAncillaManifest,
    CurieWeissAncillaRunIdentity,
    plan_curie_weiss_ancilla_runs,
)


class CurieWeissAncillaCampaignError(RuntimeError):
    pass


class UndeclaredAncillaRunError(CurieWeissAncillaCampaignError):
    pass


class PartialAncillaResultError(CurieWeissAncillaCampaignError):
    pass


@dataclass(frozen=True, slots=True)
class CurieWeissAncillaResultBundle:
    run: CurieWeissAncillaRunIdentity
    manifest_digest: str
    code_revision: str
    environment: dict[str, str]
    invocation: tuple[str, ...]
    started_at: str
    completed_at: str
    runtime_seconds: float
    peak_memory_bytes: int
    dataset_path: str
    dataset_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    probabilities_path: str
    probabilities_sha256: str
    estimator_randomness: dict[str, int]
    metrics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.manifest_digest != self.run.configuration_digest:
            raise ValueError("bundle manifest differs from run")
        for value in (
            self.manifest_digest,
            self.dataset_sha256,
            self.checkpoint_sha256,
            self.probabilities_sha256,
        ):
            if len(value) != 64:
                raise ValueError("artifact digests must be SHA-256")
        if not self.code_revision or not self.environment or not self.invocation:
            raise ValueError("bundle provenance is incomplete")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "completion_status": "completed",
            "run": self.run.to_dict(),
            "manifest_digest": self.manifest_digest,
            "code_revision": self.code_revision,
            "environment": dict(sorted(self.environment.items())),
            "invocation": list(self.invocation),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "runtime_seconds": self.runtime_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "probabilities_path": self.probabilities_path,
            "probabilities_sha256": self.probabilities_sha256,
            "estimator_randomness": dict(sorted(self.estimator_randomness.items())),
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CurieWeissAncillaResultBundle:
        if (
            value.get("schema_version") != 1
            or value.get("completion_status") != "completed"
        ):
            raise ValueError("result is not completed schema v1")
        return cls(
            run=CurieWeissAncillaRunIdentity.from_dict(value["run"]),
            manifest_digest=str(value["manifest_digest"]),
            code_revision=str(value["code_revision"]),
            environment={str(k): str(v) for k, v in value["environment"].items()},
            invocation=tuple(str(v) for v in value["invocation"]),
            started_at=str(value["started_at"]),
            completed_at=str(value["completed_at"]),
            runtime_seconds=float(value["runtime_seconds"]),
            peak_memory_bytes=int(value["peak_memory_bytes"]),
            dataset_path=str(value["dataset_path"]),
            dataset_sha256=str(value["dataset_sha256"]),
            checkpoint_path=str(value["checkpoint_path"]),
            checkpoint_sha256=str(value["checkpoint_sha256"]),
            probabilities_path=str(value["probabilities_path"]),
            probabilities_sha256=str(value["probabilities_sha256"]),
            estimator_randomness={
                str(k): int(v) for k, v in value["estimator_randomness"].items()
            },
            metrics=dict(value["metrics"]),
        )


class CurieWeissAncillaCampaignStore:
    def __init__(self, root: Path, manifest: CurieWeissAncillaManifest) -> None:
        self.root, self.manifest = root, manifest
        self.manifest_path = root / "manifest.json"
        self.preflight_path = root / "preflight.json"
        self.ledger_directory = root / "ledger"
        self.results_directory = root / "results"
        self.checkpoints_directory = root / "checkpoints"
        self.probabilities_directory = root / "probabilities"
        self._planned = {
            run.run_id: run for run in plan_curie_weiss_ancilla_runs(manifest)
        }
        root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            if read_immutable_json(self.manifest_path) != manifest.to_dict():
                raise CurieWeissAncillaCampaignError("manifest snapshot differs")
        else:
            write_immutable_json(self.manifest_path, manifest.to_dict())

    def validate_declared(self, run: CurieWeissAncillaRunIdentity) -> None:
        if self._planned.get(run.run_id) != run:
            raise UndeclaredAncillaRunError("run is outside the frozen fit plan")

    def reservation_path(self, run: CurieWeissAncillaRunIdentity) -> Path:
        return self.ledger_directory / f"{run.run_id}.json"

    def result_path(self, run: CurieWeissAncillaRunIdentity) -> Path:
        return self.results_directory / f"{run.run_id}.json"

    def checkpoint_path(self, run: CurieWeissAncillaRunIdentity) -> Path:
        return self.checkpoints_directory / f"{run.run_id}.npz"

    def probabilities_path(self, run: CurieWeissAncillaRunIdentity) -> Path:
        return self.probabilities_directory / f"{run.run_id}.npz"

    def record_preflight(self, evidence: dict[str, Any]) -> Path:
        if evidence.get("manifest_digest") != self.manifest.digest or not isinstance(
            evidence.get("passed"), bool
        ):
            raise CurieWeissAncillaCampaignError("preflight evidence is incompatible")
        if self.preflight_path.exists():
            if read_immutable_json(self.preflight_path) != evidence:
                raise CurieWeissAncillaCampaignError("preflight evidence differs")
        else:
            write_immutable_json(self.preflight_path, evidence)
        return self.preflight_path

    def require_preflight(self) -> dict[str, Any]:
        if not self.preflight_path.exists():
            raise CurieWeissAncillaCampaignError("runtime preflight has not passed")
        evidence = read_immutable_json(self.preflight_path)
        if (
            evidence.get("manifest_digest") != self.manifest.digest
            or evidence.get("passed") is not True
        ):
            raise CurieWeissAncillaCampaignError("runtime preflight is incompatible")
        return evidence

    def reserve(
        self, run: CurieWeissAncillaRunIdentity, *, require_preflight: bool = True
    ) -> Path:
        self.validate_declared(run)
        if require_preflight:
            self.require_preflight()
        path = self.reservation_path(run)
        if path.exists():
            if not self.result_path(run).exists():
                raise PartialAncillaResultError("reserved run has no complete result")
            return path
        write_immutable_json(
            path,
            {
                "manifest_digest": self.manifest.digest,
                "run": run.to_dict(),
                "reserved_at": datetime.now(UTC).isoformat(),
            },
        )
        return path

    def record_result(self, bundle: CurieWeissAncillaResultBundle) -> Path:
        self.validate_declared(bundle.run)
        if (
            bundle.manifest_digest != self.manifest.digest
            or not self.reservation_path(bundle.run).exists()
        ):
            raise CurieWeissAncillaCampaignError("result is incompatible or unreserved")
        repository = self.root.parents[2]
        for raw, digest in (
            (bundle.checkpoint_path, bundle.checkpoint_sha256),
            (bundle.probabilities_path, bundle.probabilities_sha256),
        ):
            path = Path(raw)
            path = path if path.is_absolute() else repository / path
            if not path.exists() or sha256_file(path) != digest:
                raise CurieWeissAncillaCampaignError("result artifact digest differs")
        destination = self.result_path(bundle.run)
        if destination.exists():
            raise CurieWeissAncillaCampaignError("duplicate run result")
        write_immutable_json(destination, bundle.to_dict())
        return destination

    def has_complete(self, run: CurieWeissAncillaRunIdentity) -> bool:
        self.validate_declared(run)
        return self.result_path(run).exists()

    def load_result(
        self, run: CurieWeissAncillaRunIdentity
    ) -> CurieWeissAncillaResultBundle:
        self.validate_declared(run)
        if not self.result_path(run).exists():
            if (
                self.reservation_path(run).exists()
                or self.checkpoint_path(run).exists()
                or self.probabilities_path(run).exists()
            ):
                raise PartialAncillaResultError("partial result reuse is forbidden")
            raise CurieWeissAncillaCampaignError("run is incomplete")
        result = CurieWeissAncillaResultBundle.from_dict(
            read_immutable_json(self.result_path(run))
        )
        if result.run != run or result.manifest_digest != self.manifest.digest:
            raise CurieWeissAncillaCampaignError("stored result is incompatible")
        return result

    def completed_runs(self) -> tuple[CurieWeissAncillaRunIdentity, ...]:
        return tuple(
            run for run in self._planned.values() if self.result_path(run).exists()
        )

"""Immutable store and result envelope for the Curie--Weiss campaign."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import read_immutable_json, sha256_file, write_immutable_json
from .curie_weiss_protocol import (
    CurieWeissCampaignManifest,
    CurieWeissRunIdentity,
    plan_curie_weiss_runs,
)


class CurieWeissCampaignError(RuntimeError):
    pass


class UndeclaredCurieWeissRunError(CurieWeissCampaignError):
    pass


class PartialCurieWeissResultError(CurieWeissCampaignError):
    pass


@dataclass(frozen=True, slots=True)
class CurieWeissResultBundle:
    run: CurieWeissRunIdentity
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
            raise ValueError("result and run manifest digests differ")
        digests = (
            self.manifest_digest,
            self.dataset_sha256,
            self.checkpoint_sha256,
            self.probabilities_sha256,
        )
        if any(len(value) != 64 for value in digests):
            raise ValueError("result digests must be SHA-256")
        if not self.code_revision or not self.environment or not self.invocation:
            raise ValueError("result provenance is incomplete")
        if self.runtime_seconds < 0 or self.peak_memory_bytes < 0:
            raise ValueError("resource values are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
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
            "completion_status": "completed",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CurieWeissResultBundle:
        if (
            value.get("schema_version") != 1
            or value.get("completion_status") != "completed"
        ):
            raise ValueError("Curie--Weiss result bundle is not complete schema v1")
        expected = set(cls.__dataclass_fields__) | {
            "schema_version",
            "completion_status",
        }
        if set(value) != expected:
            raise ValueError("Curie--Weiss result bundle schema differs")
        return cls(
            run=CurieWeissRunIdentity.from_dict(value["run"]),
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


class CurieWeissCampaignStore:
    def __init__(self, root: Path, manifest: CurieWeissCampaignManifest) -> None:
        self.root, self.manifest = root, manifest
        self.manifest_path = root / "manifest.json"
        self.preflight_path = root / "preflight.json"
        self.ledger_directory = root / "ledger"
        self.results_directory = root / "results"
        self.checkpoints_directory = root / "checkpoints"
        self.probabilities_directory = root / "probabilities"
        self._planned = {r.run_id: r for r in plan_curie_weiss_runs(manifest)}
        root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            if read_immutable_json(self.manifest_path) != manifest.to_dict():
                raise CurieWeissCampaignError("manifest snapshot differs")
        else:
            write_immutable_json(self.manifest_path, manifest.to_dict())

    def validate_declared(self, run: CurieWeissRunIdentity) -> None:
        if self._planned.get(run.run_id) != run:
            raise UndeclaredCurieWeissRunError("run is outside the 270-fit plan")

    def reservation_path(self, run: CurieWeissRunIdentity) -> Path:
        return self.ledger_directory / f"{run.run_id}.json"

    def result_path(self, run: CurieWeissRunIdentity) -> Path:
        return self.results_directory / f"{run.run_id}.json"

    def checkpoint_path(self, run: CurieWeissRunIdentity) -> Path:
        return self.checkpoints_directory / f"{run.run_id}.npz"

    def probabilities_path(self, run: CurieWeissRunIdentity) -> Path:
        return self.probabilities_directory / f"{run.run_id}.npz"

    def record_preflight(self, evidence: dict[str, Any]) -> Path:
        if (
            evidence.get("manifest_digest") != self.manifest.digest
            or evidence.get("passed") is not True
        ):
            raise CurieWeissCampaignError("preflight did not release training")
        if self.preflight_path.exists():
            if read_immutable_json(self.preflight_path) != evidence:
                raise CurieWeissCampaignError("preflight evidence differs")
        else:
            write_immutable_json(self.preflight_path, evidence)
        return self.preflight_path

    def require_preflight(self) -> dict[str, Any]:
        if not self.preflight_path.exists():
            raise CurieWeissCampaignError("runtime preflight has not passed")
        evidence = read_immutable_json(self.preflight_path)
        if (
            evidence.get("manifest_digest") != self.manifest.digest
            or evidence.get("passed") is not True
        ):
            raise CurieWeissCampaignError("runtime preflight is incompatible")
        return evidence

    def reserve(
        self, run: CurieWeissRunIdentity, *, require_preflight: bool = True
    ) -> Path:
        self.validate_declared(run)
        if require_preflight:
            self.require_preflight()
        path = self.reservation_path(run)
        payload = {
            "manifest_digest": self.manifest.digest,
            "run": run.to_dict(),
            "reserved_at": datetime.now(UTC).isoformat(),
        }
        if path.exists():
            existing = read_immutable_json(path)
            if (
                existing["manifest_digest"] != self.manifest.digest
                or existing["run"] != run.to_dict()
            ):
                raise CurieWeissCampaignError("reservation identity differs")
            if not self.result_path(run).exists():
                raise PartialCurieWeissResultError(
                    "reserved run has no complete result; partial restart is forbidden"
                )
        else:
            write_immutable_json(path, payload)
        return path

    def record_result(self, bundle: CurieWeissResultBundle) -> Path:
        self.validate_declared(bundle.run)
        if (
            bundle.manifest_digest != self.manifest.digest
            or not self.reservation_path(bundle.run).exists()
        ):
            raise CurieWeissCampaignError("result is incompatible or unreserved")
        repository = self.root.parents[2]
        for raw, digest in (
            (bundle.checkpoint_path, bundle.checkpoint_sha256),
            (bundle.probabilities_path, bundle.probabilities_sha256),
        ):
            path = Path(raw)
            if not path.is_absolute():
                path = repository / path
            if not path.exists() or sha256_file(path) != digest:
                raise CurieWeissCampaignError("result artifact digest differs")
        destination = self.result_path(bundle.run)
        if destination.exists():
            raise CurieWeissCampaignError("duplicate run result")
        write_immutable_json(destination, bundle.to_dict())
        return destination

    def has_complete(self, run: CurieWeissRunIdentity) -> bool:
        self.validate_declared(run)
        return self.result_path(run).exists()

    def load_result(self, run: CurieWeissRunIdentity) -> CurieWeissResultBundle:
        self.validate_declared(run)
        if not self.result_path(run).exists():
            if (
                self.reservation_path(run).exists()
                or self.checkpoint_path(run).exists()
                or self.probabilities_path(run).exists()
            ):
                raise PartialCurieWeissResultError("partial result reuse is forbidden")
            raise CurieWeissCampaignError("run is incomplete")
        result = CurieWeissResultBundle.from_dict(
            read_immutable_json(self.result_path(run))
        )
        if result.run != run or result.manifest_digest != self.manifest.digest:
            raise CurieWeissCampaignError("stored result is incompatible")
        return result

    def completed_runs(self) -> tuple[CurieWeissRunIdentity, ...]:
        return tuple(r for r in self._planned.values() if self.result_path(r).exists())

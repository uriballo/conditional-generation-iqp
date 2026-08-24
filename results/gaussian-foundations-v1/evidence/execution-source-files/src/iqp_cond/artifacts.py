"""Immutable, content-verified result bundles for a frozen campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .protocol import (
    ProtocolValidationError,
    RunIdentity,
    canonical_json_bytes,
    content_digest,
)


class ArtifactError(RuntimeError):
    """Base error for immutable campaign artifacts."""


class DuplicateArtifactError(ArtifactError):
    """Raised when an immutable artifact already exists."""


class CorruptArtifactError(ArtifactError):
    """Raised when stored content does not match its recorded digest."""


class IncompatibleArtifactError(ArtifactError):
    """Raised when provenance belongs to a different campaign or run."""


class CompletionStatus(StrEnum):
    """Terminal status of one reserved full training run."""

    COMPLETED = "completed"
    FAILED = "failed"


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of an on-disk artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _require_sha256(value: str, context: str) -> None:
    if not _is_sha256(value):
        raise ProtocolValidationError(f"{context} must be a lowercase SHA-256 digest")


def _parse_timestamp(value: str, context: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProtocolValidationError(f"{context} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProtocolValidationError(f"{context} must include a UTC offset")
    return parsed


def _pairs(
    values: Mapping[str, Any],
    *,
    integer_values: bool,
    context: str,
) -> tuple[tuple[str, str | int], ...]:
    pairs: list[tuple[str, str | int]] = []
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ProtocolValidationError(f"{context} names must be non-empty strings")
        if integer_values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ProtocolValidationError(f"{context} values must be integers")
            pairs.append((name, value))
        else:
            if not isinstance(value, str) or not value:
                raise ProtocolValidationError(
                    f"{context} values must be non-empty strings",
                )
            pairs.append((name, value))
    return tuple(sorted(pairs))


def write_immutable_json(destination: Path, payload: Mapping[str, Any]) -> str:
    """Atomically create one content-verified JSON record without overwriting."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload_digest = content_digest(payload)
    envelope = {"payload": dict(payload), "payload_sha256": payload_digest}
    encoded = canonical_json_bytes(envelope) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise DuplicateArtifactError(
                f"immutable artifact already exists: {destination}",
            ) from error
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return payload_digest


def write_immutable_bytes(destination: Path, payload: bytes) -> str:
    """Atomically create one arbitrary immutable artifact without overwriting."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload_digest = hashlib.sha256(payload).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise DuplicateArtifactError(
                f"immutable artifact already exists: {destination}",
            ) from error
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return payload_digest


def read_immutable_json(path: Path) -> dict[str, Any]:
    """Load an immutable JSON record and verify its canonical content digest."""
    try:
        envelope = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CorruptArtifactError(
            f"could not read immutable artifact {path}",
        ) from error
    if not isinstance(envelope, dict) or set(envelope) != {
        "payload",
        "payload_sha256",
    }:
        raise CorruptArtifactError(f"invalid immutable artifact envelope: {path}")
    payload = envelope["payload"]
    recorded_digest = envelope["payload_sha256"]
    if not isinstance(payload, dict) or not isinstance(recorded_digest, str):
        raise CorruptArtifactError(f"invalid immutable artifact values: {path}")
    if not _is_sha256(recorded_digest) or content_digest(payload) != recorded_digest:
        raise CorruptArtifactError(f"content digest mismatch: {path}")
    return payload


@dataclass(frozen=True)
class ParameterAccounting:
    """Explicit model-size accounting captured with every result."""

    n_qubits: int
    n_generators: int
    n_trainable_parameters: int
    n_ancillas: int

    def __post_init__(self) -> None:
        values = (
            self.n_qubits,
            self.n_generators,
            self.n_trainable_parameters,
            self.n_ancillas,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ProtocolValidationError(
                "parameter accounting values must be integers",
            )
        if (
            self.n_qubits < 1
            or self.n_generators < 1
            or self.n_trainable_parameters < 1
            or self.n_ancillas < 0
        ):
            raise ProtocolValidationError(
                "parameter accounting values are out of range",
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "n_qubits": self.n_qubits,
            "n_generators": self.n_generators,
            "n_trainable_parameters": self.n_trainable_parameters,
            "n_ancillas": self.n_ancillas,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ParameterAccounting:
        expected = {
            "n_qubits",
            "n_generators",
            "n_trainable_parameters",
            "n_ancillas",
        }
        if set(value) != expected:
            raise ProtocolValidationError(
                "parameter_accounting keys do not match schema",
            )
        return cls(**{name: value[name] for name in expected})


@dataclass(frozen=True)
class CheckpointProvenance:
    """Identity and selection provenance of the persisted best checkpoint."""

    source_run_id: str
    manifest_digest: str
    selection_rule: str
    selected_step: int
    validation_metric: str
    validation_value: float
    artifact_path: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.source_run_id, "checkpoint source_run_id")
        _require_sha256(self.manifest_digest, "checkpoint manifest_digest")
        _require_sha256(self.artifact_sha256, "checkpoint artifact_sha256")
        if (
            not self.selection_rule
            or not self.validation_metric
            or not self.artifact_path
        ):
            raise ProtocolValidationError(
                "checkpoint rule, metric, and artifact path must be non-empty",
            )
        if (
            isinstance(self.selected_step, bool)
            or not isinstance(self.selected_step, int)
            or self.selected_step < 0
        ):
            raise ProtocolValidationError(
                "checkpoint selected_step must be non-negative",
            )
        if isinstance(self.validation_value, bool) or not isinstance(
            self.validation_value, (int, float)
        ):
            raise ProtocolValidationError("checkpoint validation_value must be numeric")
        if not math.isfinite(float(self.validation_value)):
            raise ProtocolValidationError("checkpoint validation_value must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_run_id": self.source_run_id,
            "manifest_digest": self.manifest_digest,
            "selection_rule": self.selection_rule,
            "selected_step": self.selected_step,
            "validation_metric": self.validation_metric,
            "validation_value": float(self.validation_value),
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointProvenance:
        expected = {
            "source_run_id",
            "manifest_digest",
            "selection_rule",
            "selected_step",
            "validation_metric",
            "validation_value",
            "artifact_path",
            "artifact_sha256",
        }
        if set(value) != expected:
            raise ProtocolValidationError("checkpoint keys do not match schema")
        return cls(
            source_run_id=str(value["source_run_id"]),
            manifest_digest=str(value["manifest_digest"]),
            selection_rule=str(value["selection_rule"]),
            selected_step=value["selected_step"],
            validation_metric=str(value["validation_metric"]),
            validation_value=value["validation_value"],
            artifact_path=str(value["artifact_path"]),
            artifact_sha256=str(value["artifact_sha256"]),
        )


@dataclass(frozen=True)
class ResultBundle:
    """Complete immutable provenance and raw metrics for one reserved run."""

    schema_version: int
    run: RunIdentity
    manifest_digest: str
    code_revision: str
    environment: tuple[tuple[str, str], ...]
    invocation: tuple[str, ...]
    started_at: str
    completed_at: str
    runtime_seconds: float
    peak_memory_bytes: int
    parameter_accounting: ParameterAccounting
    dataset_hashes: tuple[tuple[str, str], ...]
    estimator_randomness: tuple[tuple[str, int], ...]
    checkpoint: CheckpointProvenance | None
    raw_metrics_json: str
    completion_status: CompletionStatus
    error: str | None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ProtocolValidationError("result bundle schema_version must be 1")
        _require_sha256(self.manifest_digest, "result manifest_digest")
        if self.manifest_digest != self.run.configuration_digest:
            raise IncompatibleArtifactError(
                "result manifest digest differs from its run configuration digest",
            )
        if not self.code_revision or not self.invocation:
            raise ProtocolValidationError(
                "code revision and invocation must be non-empty",
            )
        if any(
            not isinstance(argument, str) or not argument
            for argument in self.invocation
        ):
            raise ProtocolValidationError(
                "invocation arguments must be non-empty strings",
            )
        if not self.environment or len(dict(self.environment)) != len(self.environment):
            raise ProtocolValidationError(
                "environment must be non-empty with unique keys",
            )
        if not self.dataset_hashes or len(dict(self.dataset_hashes)) != len(
            self.dataset_hashes,
        ):
            raise ProtocolValidationError("dataset hashes must be non-empty and unique")
        for name, digest in self.dataset_hashes:
            if not name:
                raise ProtocolValidationError("dataset hash names must be non-empty")
            _require_sha256(digest, f"dataset hash {name}")
        if not self.estimator_randomness or len(dict(self.estimator_randomness)) != len(
            self.estimator_randomness,
        ):
            raise ProtocolValidationError(
                "estimator randomness must be non-empty with unique keys",
            )
        if any(
            not name or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for name, seed in self.estimator_randomness
        ):
            raise ProtocolValidationError(
                "estimator randomness requires non-negative integer seeds",
            )
        started = _parse_timestamp(self.started_at, "started_at")
        completed = _parse_timestamp(self.completed_at, "completed_at")
        if completed < started:
            raise ProtocolValidationError("completed_at cannot precede started_at")
        if (
            isinstance(self.runtime_seconds, bool)
            or not isinstance(self.runtime_seconds, (int, float))
            or not math.isfinite(float(self.runtime_seconds))
            or self.runtime_seconds < 0
        ):
            raise ProtocolValidationError(
                "runtime_seconds must be finite and non-negative",
            )
        if (
            isinstance(self.peak_memory_bytes, bool)
            or not isinstance(self.peak_memory_bytes, int)
            or self.peak_memory_bytes < 0
        ):
            raise ProtocolValidationError("peak_memory_bytes must be non-negative")
        try:
            metrics = json.loads(self.raw_metrics_json)
        except json.JSONDecodeError as error:
            raise ProtocolValidationError("raw_metrics_json is invalid") from error
        if not isinstance(metrics, dict):
            raise ProtocolValidationError("raw metrics must be a JSON object")
        canonical_metrics = canonical_json_bytes(metrics).decode("utf-8")
        if self.raw_metrics_json != canonical_metrics:
            raise ProtocolValidationError("raw_metrics_json must use canonical JSON")
        if self.checkpoint is not None:
            if self.checkpoint.manifest_digest != self.manifest_digest:
                raise IncompatibleArtifactError(
                    "checkpoint belongs to a different campaign manifest",
                )
            if self.checkpoint.source_run_id != self.run.run_id:
                raise IncompatibleArtifactError(
                    "checkpoint source run differs from result run",
                )
        if self.completion_status is CompletionStatus.COMPLETED:
            if self.checkpoint is None or self.error is not None:
                raise ProtocolValidationError(
                    "completed result requires a checkpoint and no error",
                )
        elif not self.error:
            raise ProtocolValidationError("failed result requires a non-empty error")

    @property
    def raw_metrics(self) -> dict[str, Any]:
        """Return a fresh mutable view of the canonical raw metrics."""
        return json.loads(self.raw_metrics_json)

    @property
    def digest(self) -> str:
        """Content digest of the complete result bundle."""
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run": self.run.to_dict(),
            "manifest_digest": self.manifest_digest,
            "code_revision": self.code_revision,
            "environment": dict(self.environment),
            "invocation": list(self.invocation),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "runtime_seconds": float(self.runtime_seconds),
            "peak_memory_bytes": self.peak_memory_bytes,
            "parameter_accounting": self.parameter_accounting.to_dict(),
            "dataset_hashes": dict(self.dataset_hashes),
            "estimator_randomness": dict(self.estimator_randomness),
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "raw_metrics": self.raw_metrics,
            "completion_status": self.completion_status.value,
            "error": self.error,
        }

    @classmethod
    def create(
        cls,
        *,
        run: RunIdentity,
        manifest_digest: str,
        code_revision: str,
        environment: Mapping[str, str],
        invocation: tuple[str, ...] | list[str],
        started_at: str,
        completed_at: str,
        runtime_seconds: float,
        peak_memory_bytes: int,
        parameter_accounting: ParameterAccounting,
        dataset_hashes: Mapping[str, str],
        estimator_randomness: Mapping[str, int],
        checkpoint: CheckpointProvenance | None,
        raw_metrics: Mapping[str, Any],
        completion_status: CompletionStatus,
        error: str | None = None,
    ) -> ResultBundle:
        """Construct a bundle while canonicalizing map-valued provenance."""
        return cls(
            schema_version=1,
            run=run,
            manifest_digest=manifest_digest,
            code_revision=code_revision,
            environment=tuple(
                (str(name), str(value))
                for name, value in _pairs(
                    environment,
                    integer_values=False,
                    context="environment",
                )
            ),
            invocation=tuple(invocation),
            started_at=started_at,
            completed_at=completed_at,
            runtime_seconds=runtime_seconds,
            peak_memory_bytes=peak_memory_bytes,
            parameter_accounting=parameter_accounting,
            dataset_hashes=tuple(
                (str(name), str(value))
                for name, value in _pairs(
                    dataset_hashes,
                    integer_values=False,
                    context="dataset hashes",
                )
            ),
            estimator_randomness=tuple(
                (str(name), int(value))
                for name, value in _pairs(
                    estimator_randomness,
                    integer_values=True,
                    context="estimator randomness",
                )
            ),
            checkpoint=checkpoint,
            raw_metrics_json=canonical_json_bytes(raw_metrics).decode("utf-8"),
            completion_status=completion_status,
            error=error,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResultBundle:
        expected = {
            "schema_version",
            "run",
            "manifest_digest",
            "code_revision",
            "environment",
            "invocation",
            "started_at",
            "completed_at",
            "runtime_seconds",
            "peak_memory_bytes",
            "parameter_accounting",
            "dataset_hashes",
            "estimator_randomness",
            "checkpoint",
            "raw_metrics",
            "completion_status",
            "error",
        }
        if set(value) != expected:
            raise ProtocolValidationError("result bundle keys do not match schema")
        if value["schema_version"] != 1:
            raise ProtocolValidationError("result bundle schema_version must be 1")
        checkpoint_value = value["checkpoint"]
        checkpoint = (
            None
            if checkpoint_value is None
            else CheckpointProvenance.from_dict(checkpoint_value)
        )
        try:
            status = CompletionStatus(str(value["completion_status"]))
        except ValueError as error:
            raise ProtocolValidationError("unknown result completion status") from error
        error_value = value["error"]
        if error_value is not None and not isinstance(error_value, str):
            raise ProtocolValidationError("result error must be a string or null")
        return cls.create(
            run=RunIdentity.from_dict(value["run"]),
            manifest_digest=str(value["manifest_digest"]),
            code_revision=str(value["code_revision"]),
            environment=value["environment"],
            invocation=value["invocation"],
            started_at=str(value["started_at"]),
            completed_at=str(value["completed_at"]),
            runtime_seconds=value["runtime_seconds"],
            peak_memory_bytes=value["peak_memory_bytes"],
            parameter_accounting=ParameterAccounting.from_dict(
                value["parameter_accounting"],
            ),
            dataset_hashes=value["dataset_hashes"],
            estimator_randomness=value["estimator_randomness"],
            checkpoint=checkpoint,
            raw_metrics=value["raw_metrics"],
            completion_status=status,
            error=error_value,
        )


class ArtifactStore:
    """Content-verified immutable storage for campaign result bundles."""

    def __init__(self, root: Path, manifest_digest: str) -> None:
        _require_sha256(manifest_digest, "artifact store manifest_digest")
        self.root = root
        self.manifest_digest = manifest_digest
        self.runs_directory = root / "runs"
        self.runs_directory.mkdir(parents=True, exist_ok=True)

    def bundle_path(self, run_id: str) -> Path:
        """Return the immutable JSON path for one run identity."""
        _require_sha256(run_id, "run_id")
        return self.runs_directory / f"{run_id}.json"

    def has_bundle(self, run_id: str) -> bool:
        """Return whether a result bundle exists for a run identity."""
        return self.bundle_path(run_id).exists()

    def write_bundle(self, bundle: ResultBundle) -> Path:
        """Atomically persist a new bundle, rejecting every duplicate identity."""
        if bundle.manifest_digest != self.manifest_digest:
            raise IncompatibleArtifactError(
                "result bundle belongs to a different campaign manifest",
            )
        destination = self.bundle_path(bundle.run.run_id)
        write_immutable_json(destination, bundle.to_dict())
        return destination

    def load_bundle(self, run_id: str) -> ResultBundle:
        """Load and validate one bundle against this campaign and run identity."""
        payload = read_immutable_json(self.bundle_path(run_id))
        try:
            bundle = ResultBundle.from_dict(payload)
        except (ProtocolValidationError, IncompatibleArtifactError) as error:
            raise CorruptArtifactError(f"invalid result bundle for {run_id}") from error
        if bundle.run.run_id != run_id:
            raise CorruptArtifactError(
                "result filename and embedded run identity differ",
            )
        if bundle.manifest_digest != self.manifest_digest:
            raise IncompatibleArtifactError(
                "stored result belongs to a different campaign manifest",
            )
        return bundle

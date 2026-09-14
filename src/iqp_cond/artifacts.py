"""Immutable, content-verified artifact files for a frozen campaign."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .protocol import canonical_json_bytes, content_digest


class ArtifactError(RuntimeError):
    """Base error for immutable campaign artifacts."""


class DuplicateArtifactError(ArtifactError):
    """Raised when an immutable artifact already exists."""


class CorruptArtifactError(ArtifactError):
    """Raised when stored content does not match its recorded digest."""


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

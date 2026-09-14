"""Canonical JSON encoding and content-addressed digests."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any


class ProtocolValidationError(ValueError):
    """Raised when a typed protocol record violates its schema."""


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

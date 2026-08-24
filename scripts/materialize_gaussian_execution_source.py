#!/usr/bin/env python3
"""Materialize the content-addressed source used by the Gaussian campaign."""

from __future__ import annotations

import shutil
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json, sha256_file

CAMPAIGN = Path("results/gaussian-foundations-v1")
SOURCE_RECORD = CAMPAIGN / "evidence" / "execution-source.json"
SNAPSHOT = CAMPAIGN / "evidence" / "execution-source-files"
SCALING_RECORD = Path("results/quantum-campaign-v1/evidence/estimator-scaling.json")
SCALING_SNAPSHOT = Path("results/quantum-campaign-v1/evidence/execution-source-files")


def materialize(repository: Path) -> tuple[int, Path]:
    repository = repository.resolve()
    record = read_immutable_json(repository / SOURCE_RECORD)
    files = record.get("files", {})
    if not files:
        raise RuntimeError("execution-source record contains no files")

    copied = 0
    for relative, expected in sorted(files.items()):
        destination = repository / SNAPSHOT / relative
        if destination.is_file() and sha256_file(destination) == expected:
            continue
        source = repository / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(
                f"historical bytes unavailable for {relative}: expected {expected}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != expected:
            raise RuntimeError(f"copy verification failed for {relative}")
        copied += 1

    mismatches = [
        relative
        for relative, expected in sorted(files.items())
        if sha256_file(repository / SNAPSHOT / relative) != expected
    ]
    if mismatches:
        raise RuntimeError(f"snapshot digest mismatches: {mismatches}")
    return copied, repository / SNAPSHOT


def materialize_scaling(repository: Path) -> tuple[int, Path]:
    repository = repository.resolve()
    record = read_immutable_json(repository / SCALING_RECORD)
    files = record["provenance"]["source_digests"]
    copied = 0
    for relative, expected in sorted(files.items()):
        destination = repository / SCALING_SNAPSHOT / relative
        if destination.is_file() and sha256_file(destination) == expected:
            continue
        candidates = (repository / relative, repository / SNAPSHOT / relative)
        source = next(
            (
                candidate
                for candidate in candidates
                if candidate.is_file() and sha256_file(candidate) == expected
            ),
            None,
        )
        if source is None:
            raise RuntimeError(
                f"historical estimator-audit bytes unavailable for {relative}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied += 1
    return copied, repository / SCALING_SNAPSHOT


def main() -> int:
    copied, snapshot = materialize(Path.cwd())
    print(f"verified historical Gaussian source snapshot: {snapshot} ({copied} copied)")
    scaling_copied, scaling_snapshot = materialize_scaling(Path.cwd())
    print(
        "verified historical estimator-audit source snapshot: "
        f"{scaling_snapshot} ({scaling_copied} copied)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

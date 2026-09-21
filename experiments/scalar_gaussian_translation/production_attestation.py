"""Post-run verification and attestation for the scalar production campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import analysis, campaign, production, protocol, review_summaries

RUN_ARTIFACTS = (
    "checkpoint.npz",
    "restarts.npz",
    "config.json",
    "metrics.json",
    "run.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _normalize(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _artifact(path: Path) -> dict[str, Any]:
    path = Path(path)
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(campaign.ROOT.resolve()))
    except ValueError:
        display = str(path)
    return {"path": display, "sha256": _sha256(path), "bytes": path.stat().st_size}


def _load_authorized_evidence(
    manifest_path: Path, authorization_path: Path
) -> tuple[dict[str, Any], list[campaign.CampaignJob], dict[str, Any]]:
    """Validate authorization evidence without requiring the old checkout to be HEAD."""
    authorization, _ = production._content_addressed(
        authorization_path, "production authorization"
    )
    if (
        authorization.get("schema_version") != 1
        or authorization.get("decision") != "authorized"
    ):
        raise ValueError("production authorization decision is invalid")
    state = authorization.get("authorized_repository_state", {})
    if state.get("paper_dirty") is not False:
        raise ValueError("production was not authorized from a clean repository")
    artifacts = authorization.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("production authorization artifacts are malformed")
    resolved = {
        name: production._resolve_artifact(record) for name, record in artifacts.items()
    }
    if resolved.get("proposal") != Path(manifest_path).resolve():
        raise ValueError("authorization references a different production proposal")
    # The post-run verifier is new analysis code, so it can make the working
    # tree dirty or live at a later paper commit. The authorization already
    # records the clean execution state. Reuse that state only for the bridge's
    # cleanliness/revision precondition; _validate_evidence still checks every
    # authorized file hash and the current runtime-source snapshot bytewise.
    original_repo_state = analysis._repo_state
    analysis._repo_state = lambda: dict(state)
    try:
        manifest, jobs, audit_attestation = production._validate_evidence(
            resolved["proposal"],
            resolved["audit_manifest"],
            resolved["audit_attestation"],
            resolved["audit_rows"],
            resolved["audit_gate_result"],
            resolved["source_equivalence"],
            resolved["provenance_root"],
            resolved["frozen_gates"],
        )
    finally:
        analysis._repo_state = original_repo_state
    if authorization.get("matrix") != production._manifest_summary(jobs):
        raise ValueError("authorized production matrix changed")
    expected_canary = [job.job_id for job in production.select_canary_jobs(jobs)]
    observed_canary = [item.get("job_id") for item in authorization["canary_jobs"]]
    if observed_canary != expected_canary:
        raise ValueError("authorized canary identities changed")
    if authorization.get("runtime_source_snapshot_sha256") != manifest.get(
        "source_snapshot_sha256"
    ):
        raise ValueError("authorization source snapshot differs from the manifest")
    audit_attestation.pop("_verified_content_sha256", None)
    return manifest, jobs, authorization


def _validate_npz(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if not archive.files:
            raise ValueError(f"empty NumPy archive: {path}")
        for name in archive.files:
            value = archive[name]
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError(f"non-finite values in {path}:{name}")


def _artifact_inventory(
    entries: list[dict[str, Any]],
    jobs: list[campaign.CampaignJob],
    records_dir: Path,
    authorization: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_names = {f"{job.job_id}.json" for job in jobs}
    observed_names = {path.name for path in Path(records_dir).glob("*.json")}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise ValueError(
            f"production record set mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    entry_by_id = {entry["job"].job_id: entry for entry in entries}
    records: dict[str, Any] = {}
    shard_rows: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    run_ids: set[str] = set()
    expected_state = authorization["authorized_repository_state"]
    for index, job in enumerate(jobs):
        entry = entry_by_id[job.job_id]
        record_path = Path(records_dir) / f"{job.job_id}.json"
        payload = json.loads(record_path.read_text())
        if payload.get("job") != _normalize(asdict(job)):
            raise ValueError(f"job payload mismatch for {job.job_id}")
        if payload.get("audit_gates_sha256") != manifest.get("frozen_gates_sha256"):
            raise ValueError(f"frozen-gate hash mismatch for {job.job_id}")
        run_record = payload.get("run_record", {})
        production._assert_finite(run_record)
        if run_record.get("resources", {}).get("optimization_restarts") != 2:
            raise ValueError(f"job {job.job_id} did not complete two restarts")
        run_id = run_record.get("run_id")
        if not isinstance(run_id, str) or run_id in run_ids:
            raise ValueError(f"duplicate or malformed run id for {job.job_id}")
        run_ids.add(run_id)
        run_dir = Path(entry["checkpoint"]).parent
        artifact_paths = [run_dir / name for name in RUN_ARTIFACTS]
        if not all(path.is_file() for path in artifact_paths):
            raise FileNotFoundError(f"job {job.job_id} is missing a run artifact")
        config = json.loads((run_dir / "config.json").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        run = json.loads((run_dir / "run.json").read_text())
        production._assert_finite(config)
        production._assert_finite(metrics)
        production._assert_finite(run)
        if metrics.get("run_id") != run_id or run.get("run_id") != run_id:
            raise ValueError(f"run identity mismatch for {job.job_id}")
        if run.get("state") != expected_state:
            raise ValueError(f"repository revision mismatch for {job.job_id}")
        expected_config = campaign._run_config(job)
        if config != _normalize(expected_config.as_dict()):
            raise ValueError(f"run configuration mismatch for {job.job_id}")
        if metrics.get("config_digest") != expected_config.digest():
            raise ValueError(f"configuration digest mismatch for {job.job_id}")
        if (
            run.get("seed") != job.optimizer_seed
            or run.get("training_conditions") != list(job.anchors)
            or run.get("evaluation_conditions")
            != protocol.evaluation_grid(job.split).tolist()
        ):
            raise ValueError(f"condition or seed mismatch for {job.job_id}")
        _validate_npz(run_dir / "checkpoint.npz")
        _validate_npz(run_dir / "restarts.npz")
        artifacts = {
            "job_record_sha256": _sha256(record_path),
            **{
                f"{name.removesuffix('.json').removesuffix('.npz')}_sha256": _sha256(
                    run_dir / name
                )
                for name in RUN_ARTIFACTS
            },
        }
        records[job.job_id] = {"run_id": run_id, "shard": index % 2, **artifacts}
        shard_rows[index % 2].append(
            {"job_id": job.job_id, "run_id": run_id, **artifacts}
        )
    shards = {
        str(shard): {
            "expected_jobs": len(jobs[shard::2]),
            "observed_jobs": len(rows),
            "artifact_inventory_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
        }
        for shard, rows in shard_rows.items()
    }
    return records, shards


def _gate_status(gate_result: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for encoder, expected_count in (
        ("broadcast_affine", 28),
        ("qubit_affine", 28),
        ("anti_block", 12),
    ):
        encoder_result = gate_result.get("encoders", {}).get(encoder, {})
        checks = encoder_result.get("checks", {})
        all_true = len(checks) == expected_count and all(
            value is True for value in checks.values()
        )
        if not all_true:
            raise ValueError(f"production gate checks failed for {encoder}")
        if encoder != "anti_block" and encoder_result.get("pass") is not True:
            raise ValueError(f"production qualification failed for {encoder}")
        result[encoder] = {
            "check_count": len(checks),
            "all_checks_native_true": True,
            "reported_pass": encoder_result.get("pass"),
        }
    return result


def attest_production(
    manifest_path: Path,
    authorization_path: Path,
    records_dir: Path,
    gates_path: Path,
    rows_csv: Path,
    aggregates_path: Path,
    gate_result_path: Path,
    summaries_path: Path,
    direct_rows_path: Path,
    figure_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Recompute production evidence and write a content-addressed attestation."""
    manifest_path = Path(manifest_path)
    records_dir = Path(records_dir)
    manifest, jobs, authorization = _load_authorized_evidence(
        manifest_path, Path(authorization_path)
    )
    if manifest.get("phase") != "production" or manifest.get("split") != "audit":
        raise ValueError("production attestation requires the authorized audit grid")
    loaded_manifest, entries = analysis.load_records(manifest_path, records_dir)
    if loaded_manifest != manifest or len(entries) != len(jobs):
        raise ValueError("production record loading changed the authorized matrix")
    records, shards = _artifact_inventory(
        entries, jobs, records_dir, authorization, manifest
    )
    regenerated_rows = analysis.extract_rows(
        entries, manifest["phase"], manifest["split"]
    )
    regenerated_csv = analysis._csv_payload(regenerated_rows).encode("utf-8")
    supplied_csv = Path(rows_csv).read_bytes()
    if regenerated_csv != supplied_csv:
        raise ValueError("production rows differ from checkpoint regeneration")
    aggregates = analysis.aggregate_rows(regenerated_rows)
    supplied_aggregates = json.loads(Path(aggregates_path).read_text())
    if _normalize(aggregates) != _normalize(supplied_aggregates):
        raise ValueError("production aggregates differ from recomputation")
    gates_hash = _sha256(Path(gates_path))
    if manifest.get("frozen_gates_sha256") != gates_hash:
        raise ValueError("production manifest does not bind the supplied gates")
    gate_result = analysis.evaluate_gates(regenerated_rows, Path(gates_path))
    supplied_gate_result = json.loads(Path(gate_result_path).read_text())
    if _normalize(gate_result) != _normalize(supplied_gate_result):
        raise ValueError("production gate result differs from recomputation")
    status = _gate_status(gate_result)
    summaries = review_summaries.summarize(
        review_summaries._read_csv(Path(rows_csv)),
        review_summaries._read_csv(Path(direct_rows_path)),
    )
    supplied_summaries = json.loads(Path(summaries_path).read_text())
    if _normalize(summaries) != _normalize(supplied_summaries):
        raise ValueError("production review summaries differ from recomputation")
    fixed_point = [
        row["bitstring_tvd"]
        for row in regenerated_rows
        if row["encoder"] == "anti_block"
        and row["category"] == "training"
        and np.isclose(row["condition"], 0.5, rtol=0, atol=1e-12)
    ]
    if not fixed_point:
        raise ValueError("anti-block fixed-point rows are missing")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "production_valid": True,
        "created_utc": datetime.now(UTC).isoformat(),
        "authorized_repository_state": authorization["authorized_repository_state"],
        "authorization": {
            **_artifact(Path(authorization_path)),
            "content_sha256": authorization["content_sha256"],
        },
        "production_manifest": _artifact(manifest_path),
        "frozen_gates": _artifact(Path(gates_path)),
        "matrix": production._manifest_summary(jobs),
        "job_counts": {"expected": len(jobs), "observed": len(entries)},
        "shards": shards,
        "gate_verification": {
            "recomputed_result_matches_supplied": True,
            "encoders": status,
        },
        "row_verification": {
            "matches_regenerated_rows": True,
            "row_count": len(regenerated_rows),
            "canonical_csv_sha256": hashlib.sha256(regenerated_csv).hexdigest(),
        },
        "derived_artifacts": {
            "rows": _artifact(Path(rows_csv)),
            "aggregates": _artifact(Path(aggregates_path)),
            "gate_result": _artifact(Path(gate_result_path)),
            "review_summaries": _artifact(Path(summaries_path)),
            "direct_reference_rows": _artifact(Path(direct_rows_path)),
            "figures": [_artifact(path) for path in figure_paths],
        },
        "anti_block_fixed_point": {
            "condition": 0.5,
            "input_state": "|+>^8",
            "row_count": len(fixed_point),
            "tvd_min": float(np.min(fixed_point)),
            "tvd_median": float(np.median(fixed_point)),
            "tvd_max": float(np.max(fixed_point)),
        },
        "attestation_software": _artifact(Path(__file__).resolve()),
        "job_records": records,
    }
    payload["content_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    analysis._atomic_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--aggregates", type=Path, required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--direct-rows", type=Path, required=True)
    parser.add_argument("--figure", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = attest_production(
        args.manifest,
        args.authorization,
        args.records_dir,
        args.gates,
        args.rows,
        args.aggregates,
        args.gate_result,
        args.summaries,
        args.direct_rows,
        args.figure,
        args.output,
    )
    print(payload["content_sha256"])


if __name__ == "__main__":
    main()

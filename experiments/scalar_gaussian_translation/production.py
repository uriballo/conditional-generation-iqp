"""Fail-closed authorization and execution for the scalar production campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.common import runner

from . import analysis, campaign

ROOT = campaign.ROOT
DEFAULT_MANIFEST = Path(__file__).with_name("manifests") / "production_proposal.json"
DEFAULT_AUDIT_MANIFEST = Path(__file__).with_name("manifests") / "audit.json"
DEFAULT_REVIEW = Path(__file__).with_name("review")
LAUNCHER_PATH = Path(__file__).with_name("launch_production_strix.sh")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError as error:
        raise ValueError(f"authorization input must be inside {ROOT}") from error


def _artifact(path: Path, *, content_digest: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"path": _relative(path), "sha256": _sha256(path)}
    if content_digest is not None:
        record["content_sha256"] = content_digest
    return record


def _resolve_artifact(record: dict[str, Any]) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("authorization artifact record is malformed")
    path = (ROOT / record["path"]).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("authorization artifact escapes the repository") from error
    if not path.is_file() or record.get("sha256") != _sha256(path):
        raise ValueError(f"authorization artifact changed: {record['path']}")
    return path


def _content_addressed(path: Path, label: str) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is malformed")
    digest = payload.get("content_sha256")
    if not isinstance(digest, str) or digest != analysis._content_digest(payload):
        raise ValueError(f"{label} content digest is invalid")
    return payload, digest


def _manifest_summary(jobs: list[campaign.CampaignJob]) -> dict[str, Any]:
    return {
        "job_count": len(jobs),
        "unique_job_ids": len({job.job_id for job in jobs}),
        "encoders": dict(sorted(Counter(job.encoder for job in jobs).items())),
        "budget_modes": dict(sorted(Counter(job.budget_mode for job in jobs).items())),
        "protocols": dict(sorted(Counter(job.protocol for job in jobs).items())),
        "optimizer_seeds": sorted({job.optimizer_seed for job in jobs}),
    }


def _assert_production_manifest(
    payload: dict[str, Any], jobs: list[campaign.CampaignJob]
) -> None:
    expected = {
        "study": "scalar_gaussian_translation",
        "phase": "production",
        "split": "audit",
        "enabled": False,
        "production": True,
        "production_enabled": False,
        "job_count": 3120,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"production manifest {key} must equal {value!r}")
    summary = _manifest_summary(jobs)
    if summary != {
        "job_count": 3120,
        "unique_job_ids": 3120,
        "encoders": {
            "anti_block": 1000,
            "broadcast_affine": 1000,
            "condition_blind": 120,
            "qubit_affine": 1000,
        },
        "budget_modes": {"fixed_total": 1680, "matched_exposure": 1440},
        "protocols": {
            "far": 1000,
            "interpolation": 1000,
            "mirror_far": 60,
            "mirror_near": 60,
            "near": 1000,
        },
        "optimizer_seeds": list(range(501, 511)),
    }:
        raise ValueError("production manifest matrix differs from the reviewed plan")


def _one_job(
    jobs: list[campaign.CampaignJob],
    *,
    encoder: str,
    protocol_name: str,
    anchor_count: int,
    budget_mode: str,
) -> campaign.CampaignJob:
    selected = [
        job
        for job in jobs
        if job.encoder == encoder
        and job.protocol == protocol_name
        and job.anchor_count == anchor_count
        and job.placement == "even"
        and job.layout_seed is None
        and job.optimizer_seed == 501
        and job.budget_mode == budget_mode
    ]
    if len(selected) != 1:
        raise ValueError(f"canary identity did not select one {encoder} job")
    return selected[0]


def select_canary_jobs(
    jobs: list[campaign.CampaignJob],
) -> list[campaign.CampaignJob]:
    """Return the predeclared operational canary matrix."""
    return [
        _one_job(
            jobs,
            encoder="broadcast_affine",
            protocol_name="interpolation",
            anchor_count=2,
            budget_mode="fixed_total",
        ),
        _one_job(
            jobs,
            encoder="qubit_affine",
            protocol_name="near",
            anchor_count=3,
            budget_mode="fixed_total",
        ),
        _one_job(
            jobs,
            encoder="anti_block",
            protocol_name="far",
            anchor_count=5,
            budget_mode="matched_exposure",
        ),
        _one_job(
            jobs,
            encoder="condition_blind",
            protocol_name="interpolation",
            anchor_count=3,
            budget_mode="fixed_total",
        ),
    ]


def _validate_evidence(
    manifest_path: Path,
    audit_manifest_path: Path,
    attestation_path: Path,
    rows_path: Path,
    gate_result_path: Path,
    bridge_path: Path,
    provenance_root_path: Path,
    gates_path: Path,
) -> tuple[dict[str, Any], list[campaign.CampaignJob], dict[str, Any]]:
    manifest, jobs = campaign._load_manifest(manifest_path)
    _assert_production_manifest(manifest, jobs)
    root = analysis.validate_provenance_root(
        audit_manifest_path, provenance_root_path, gates_path
    )
    bridge = analysis.validate_source_equivalence(
        audit_manifest_path, bridge_path, provenance_root_path
    )
    attestation, attestation_digest = _content_addressed(
        attestation_path, "audit attestation"
    )
    if attestation.get("launch_valid") is not True:
        raise ValueError("audit attestation is not launch-valid")
    if attestation.get("row_verification") != {
        "matches_regenerated_rows": True,
        "row_count": 22014,
        "canonical_csv_sha256": _sha256(rows_path),
    }:
        raise ValueError("audit row verification is invalid")
    gate_verification = attestation.get("gate_verification", {})
    if (
        gate_verification.get("valid") is not True
        or gate_verification.get("recomputed_result_matches_supplied") is not True
    ):
        raise ValueError("audit gate verification is invalid")
    for encoder in ("broadcast_affine", "qubit_affine"):
        status = gate_verification.get("general_encoders", {}).get(encoder, {})
        if status != {
            "check_count": 28,
            "all_28_checks_native_true": True,
            "reported_pass": True,
        }:
            raise ValueError(f"audit gate status is invalid for {encoder}")
    if attestation.get("job_counts") != {"expected": 936, "observed": 936}:
        raise ValueError("audit attestation job count is invalid")
    if attestation.get("audit_manifest", {}).get("sha256") != _sha256(
        audit_manifest_path
    ):
        raise ValueError("audit attestation manifest hash mismatch")
    if attestation.get("frozen_gates", {}).get("sha256") != _sha256(gates_path):
        raise ValueError("audit attestation frozen-gate hash mismatch")
    if attestation.get("long_form_rows_csv", {}).get("sha256") != _sha256(rows_path):
        raise ValueError("audit attestation rows hash mismatch")
    if attestation.get("gate_result_json", {}).get("sha256") != _sha256(
        gate_result_path
    ):
        raise ValueError("audit attestation gate-result hash mismatch")
    source = attestation.get("source_provenance", {})
    if source.get("bridge_file_sha256") != _sha256(bridge_path) or source.get(
        "bridge_content_sha256"
    ) != bridge.get("content_sha256"):
        raise ValueError("audit attestation bridge binding mismatch")
    attested_root = attestation.get("provenance_root", {})
    if (
        attested_root.get("file_sha256") != root["file_sha256"]
        or attested_root.get("content_sha256") != root["content_sha256"]
    ):
        raise ValueError("audit attestation provenance-root binding mismatch")
    evaluator = attestation.get("evaluator", {})
    if evaluator.get("sha256") != _sha256(Path(analysis.__file__).resolve()):
        raise ValueError("attested evaluator differs from the current evaluator")
    if manifest.get("frozen_gates_sha256") != _sha256(gates_path):
        raise ValueError("production manifest frozen-gate binding mismatch")
    current_files, current_snapshot = campaign._source_snapshot()
    if (
        manifest.get("source_file_sha256") != current_files
        or manifest.get("source_snapshot_sha256") != current_snapshot
    ):
        raise ValueError("production source snapshot mismatch")
    attestation["_verified_content_sha256"] = attestation_digest
    return manifest, jobs, attestation


def create_authorization(
    manifest_path: Path,
    audit_manifest_path: Path,
    attestation_path: Path,
    rows_path: Path,
    gate_result_path: Path,
    bridge_path: Path,
    provenance_root_path: Path,
    gates_path: Path,
    output_path: Path,
    authorization_basis: str,
) -> dict[str, Any]:
    """Create an external authorization bound to one clean checkout."""
    state = runner.repo_state()
    if state.get("paper_dirty") is not False:
        raise RuntimeError("production authorization requires a clean repository")
    _, jobs, attestation = _validate_evidence(
        manifest_path,
        audit_manifest_path,
        attestation_path,
        rows_path,
        gate_result_path,
        bridge_path,
        provenance_root_path,
        gates_path,
    )
    if not authorization_basis.strip():
        raise ValueError("authorization basis cannot be empty")
    _, source_snapshot = campaign._source_snapshot()
    canary = select_canary_jobs(jobs)
    root_payload, root_digest = _content_addressed(
        provenance_root_path, "audit provenance root"
    )
    bridge_payload, bridge_digest = _content_addressed(
        bridge_path, "source-equivalence bridge"
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "decision": "authorized",
        "authorization_basis": authorization_basis.strip(),
        "created_utc": datetime.now(UTC).isoformat(),
        "authorized_repository_state": state,
        "runtime_source_snapshot_sha256": source_snapshot,
        "matrix": _manifest_summary(jobs),
        "canary_jobs": [
            {
                "job_id": job.job_id,
                "encoder": job.encoder,
                "protocol": job.protocol,
                "anchor_count": job.anchor_count,
                "budget_mode": job.budget_mode,
                "optimizer_seed": job.optimizer_seed,
            }
            for job in canary
        ],
        "artifacts": {
            "proposal": _artifact(manifest_path),
            "audit_manifest": _artifact(audit_manifest_path),
            "audit_attestation": _artifact(
                attestation_path,
                content_digest=attestation["_verified_content_sha256"],
            ),
            "audit_rows": _artifact(rows_path),
            "audit_gate_result": _artifact(gate_result_path),
            "source_equivalence": _artifact(bridge_path, content_digest=bridge_digest),
            "provenance_root": _artifact(
                provenance_root_path, content_digest=root_digest
            ),
            "frozen_gates": _artifact(gates_path),
            "production_runner": _artifact(Path(__file__).resolve()),
            "launcher": _artifact(LAUNCHER_PATH),
        },
        "reviewed_root_identity": {
            "audit_manifest_sha256": root_payload["audit_manifest_sha256"],
            "frozen_gates_sha256": root_payload["frozen_gates_sha256"],
            "protocol_sha256": root_payload["protocol_sha256"],
            "source_snapshot_sha256": root_payload["source_snapshot_sha256"],
        },
        "bridge_identity": {
            "reference_manifest_sha256": bridge_payload["reference_manifest"]["sha256"],
            "original_snapshot_sha256": bridge_payload["original"][
                "source_snapshot_sha256"
            ],
            "current_snapshot_sha256": bridge_payload["current"][
                "source_snapshot_sha256"
            ],
        },
    }
    payload["content_sha256"] = analysis._content_digest(payload)
    _atomic_json(output_path, payload)
    return payload


def validate_authorization(
    manifest_path: Path, authorization_path: Path
) -> tuple[dict[str, Any], list[campaign.CampaignJob], dict[str, Any]]:
    """Validate an authorization and every artifact it binds."""
    authorization, _ = _content_addressed(
        authorization_path, "production authorization"
    )
    if (
        authorization.get("schema_version") != 1
        or authorization.get("decision") != "authorized"
    ):
        raise ValueError("production authorization decision is invalid")
    state = runner.repo_state()
    if state.get("paper_dirty") is not False:
        raise RuntimeError("production execution requires a clean repository")
    if authorization.get("authorized_repository_state") != state:
        raise RuntimeError("current repository state differs from authorization")
    artifacts = authorization.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("production authorization artifacts are malformed")
    resolved = {name: _resolve_artifact(record) for name, record in artifacts.items()}
    if resolved.get("proposal") != Path(manifest_path).resolve():
        raise ValueError("authorization references a different proposal")
    if resolved.get("production_runner") != Path(__file__).resolve():
        raise ValueError("authorization references a different production runner")
    if resolved.get("launcher") != LAUNCHER_PATH.resolve():
        raise ValueError("authorization references a different launcher")
    manifest, jobs, attestation = _validate_evidence(
        resolved["proposal"],
        resolved["audit_manifest"],
        resolved["audit_attestation"],
        resolved["audit_rows"],
        resolved["audit_gate_result"],
        resolved["source_equivalence"],
        resolved["provenance_root"],
        resolved["frozen_gates"],
    )
    content_records = {
        "audit_attestation": attestation["_verified_content_sha256"],
        "source_equivalence": json.loads(resolved["source_equivalence"].read_text())[
            "content_sha256"
        ],
        "provenance_root": json.loads(resolved["provenance_root"].read_text())[
            "content_sha256"
        ],
    }
    for name, digest in content_records.items():
        if artifacts[name].get("content_sha256") != digest:
            raise ValueError(f"authorization {name} content digest mismatch")
    _, current_snapshot = campaign._source_snapshot()
    if authorization.get("runtime_source_snapshot_sha256") != current_snapshot:
        raise RuntimeError("authorized runtime source snapshot changed")
    if authorization.get("matrix") != _manifest_summary(jobs):
        raise ValueError("authorized production matrix changed")
    expected_canary = [job.job_id for job in select_canary_jobs(jobs)]
    observed_canary = [item.get("job_id") for item in authorization["canary_jobs"]]
    if observed_canary != expected_canary:
        raise ValueError("authorized canary identities changed")
    return manifest, jobs, authorization


def _assert_finite(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("run record contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _assert_finite(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)


def _verify_run_record(record: dict[str, Any]) -> None:
    _assert_finite(record)
    if record.get("resources", {}).get("optimization_restarts") != 2:
        raise ValueError("production job did not complete both restarts")
    run_dir = Path(record.get("directory", ""))
    if not run_dir.is_dir() or not (run_dir / "checkpoint.npz").is_file():
        raise FileNotFoundError("production job did not write a checkpoint")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--authorization", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    authorize = sub.add_parser("authorize")
    authorize.add_argument(
        "--audit-manifest", type=Path, default=DEFAULT_AUDIT_MANIFEST
    )
    authorize.add_argument(
        "--attestation", type=Path, default=DEFAULT_REVIEW / "audit_attestation.json"
    )
    authorize.add_argument(
        "--rows", type=Path, default=DEFAULT_REVIEW / "audit_rows.csv"
    )
    authorize.add_argument(
        "--gate-result", type=Path, default=DEFAULT_REVIEW / "audit_gate_check.json"
    )
    authorize.add_argument(
        "--source-equivalence",
        type=Path,
        default=DEFAULT_REVIEW / "source_equivalence.json",
    )
    authorize.add_argument(
        "--provenance-root",
        type=Path,
        default=DEFAULT_REVIEW / "audit_provenance_root.json",
    )
    authorize.add_argument(
        "--gates", type=Path, default=DEFAULT_REVIEW / "frozen_gates.json"
    )
    authorize.add_argument("--basis", required=True)
    authorize.add_argument("--output", type=Path, required=True)
    sub.add_parser("verify")
    sub.add_parser("list-canary")
    listing = sub.add_parser("list-shard")
    listing.add_argument("shard_index", type=int)
    listing.add_argument("shard_count", type=int)
    one = sub.add_parser("run-job")
    one.add_argument("job_id")
    one.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)

    if args.command == "authorize":
        payload = create_authorization(
            args.manifest,
            args.audit_manifest,
            args.attestation,
            args.rows,
            args.gate_result,
            args.source_equivalence,
            args.provenance_root,
            args.gates,
            args.output,
            args.basis,
        )
        print(f"authorized {payload['matrix']['job_count']} jobs: {args.output}")
        return
    if args.authorization is None:
        raise ValueError("production commands require --authorization")
    _, jobs, authorization = validate_authorization(args.manifest, args.authorization)
    if args.command == "verify":
        print(authorization["content_sha256"])
        return
    if args.command == "list-canary":
        for item in authorization["canary_jobs"]:
            print(item["job_id"])
        return
    if args.command == "list-shard":
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("invalid shard index/count")
        for index, job in enumerate(jobs):
            if index % args.shard_count == args.shard_index:
                print(job.job_id)
        return
    selected = next((job for job in jobs if job.job_id == args.job_id), None)
    if selected is None:
        raise ValueError(f"unknown production job id {args.job_id!r}")
    record = campaign._execute(selected, args.output_root, args.manifest)
    _verify_run_record(record)


if __name__ == "__main__":
    main()

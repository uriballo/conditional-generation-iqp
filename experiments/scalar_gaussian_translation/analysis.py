"""Deterministic post-processing for scalar Gaussian campaign artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import iqp_cond as ciqp
import numpy as np

from experiments.common import metrics
from experiments.common.config import RunConfig

from . import campaign, protocol

N = protocol.N_VISIBLE
AUDIT_PROVENANCE_ROOT_CONTENT_SHA256 = (
    "61e01f81018599943f2e735d388dcb275c761373da50703ca8e2b86a6907a905"
)
METRIC_KEYS = (
    "bitstring_tvd",
    "hamming_profile_tvd",
    "generated_mean",
    "target_mean",
    "absolute_mean_error",
    "generated_std",
    "target_std",
    "absolute_std_error",
    "within_shell_residual",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _job_dict(job: dict[str, Any]) -> campaign.CampaignJob:
    data = dict(job)
    data["anchors"] = tuple(data["anchors"])
    if data.get("fit_condition") is not None:
        data["fit_condition"] = tuple(data["fit_condition"])
    return campaign.CampaignJob(**data)


def _snapshot_digest(file_hashes: dict[str, str], library_revision: str) -> str:
    return hashlib.sha256(
        campaign._canonical(
            {"files": file_hashes, "library_revision": library_revision}
        )
    ).hexdigest()


def _repo_state() -> dict[str, Any]:
    return campaign.runner.repo_state()


def _validate_source_manifest(manifest: dict[str, Any]) -> None:
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("jobs"), list)
        or manifest.get("job_count") != len(manifest["jobs"])
        or any(not isinstance(job, dict) for job in manifest["jobs"])
    ):
        raise ValueError("manifest campaign payload is malformed")
    files = manifest.get("source_file_sha256")
    revision = manifest.get("library_revision")
    state = manifest.get("state")
    if (
        not isinstance(files, dict)
        or not files
        or not all(
            isinstance(name, str)
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for name, digest in files.items()
        )
        or not isinstance(revision, str)
        or not revision
        or not isinstance(state, dict)
        or state.get("library_revision") != revision
    ):
        raise ValueError("manifest source provenance is malformed")
    if manifest.get("source_snapshot_sha256") != _snapshot_digest(files, revision):
        raise ValueError("manifest source snapshot digest is invalid")
    protocol_hash = campaign._sha256(campaign.PROTOCOL_PATH)
    if (
        manifest.get("protocol_sha256") != protocol_hash
        or files.get(str(campaign.PROTOCOL_PATH.relative_to(campaign.ROOT)))
        != protocol_hash
    ):
        raise ValueError("manifest protocol hash is invalid")


def _content_digest(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_provenance_root(
    manifest_path: Path,
    provenance_root_path: Path,
    gates_path: Path | None = None,
) -> dict[str, Any]:
    """Validate the reviewed root against the frozen audit inputs."""
    manifest_path = Path(manifest_path).resolve()
    root_path = Path(provenance_root_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = json.loads(root_path.read_text())
    _validate_source_manifest(manifest)
    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise ValueError("audit provenance root has an unsupported schema")
    if root.get("content_sha256") != _content_digest(root):
        raise ValueError("audit provenance root content digest is invalid")
    if root.get("content_sha256") != AUDIT_PROVENANCE_ROOT_CONTENT_SHA256:
        raise ValueError("audit provenance root is not the reviewed trust root")
    if manifest.get("phase") != "audit" or manifest.get("split") != "audit":
        raise ValueError("provenance root requires an audit manifest")
    expected_identity = {
        "study": manifest.get("study"),
        "phase": manifest.get("phase"),
        "split": manifest.get("split"),
    }
    if root.get("identity") != expected_identity:
        raise ValueError("audit provenance root identity mismatch")
    paths = root.get("paths")
    if not isinstance(paths, dict) or not all(
        isinstance(paths.get(key), str)
        for key in ("audit_manifest", "frozen_gates", "protocol")
    ):
        raise ValueError("audit provenance root paths are malformed")
    rooted_manifest = (root_path.parent / paths["audit_manifest"]).resolve()
    rooted_gates = (root_path.parent / paths["frozen_gates"]).resolve()
    rooted_protocol = (root_path.parent / paths["protocol"]).resolve()
    if rooted_manifest != manifest_path:
        raise ValueError("audit provenance root manifest path mismatch")
    if rooted_protocol != campaign.PROTOCOL_PATH.resolve():
        raise ValueError("audit provenance root protocol path mismatch")
    resolved_gates = Path(gates_path).resolve() if gates_path else rooted_gates
    if resolved_gates != rooted_gates:
        raise ValueError("audit provenance root frozen-gates path mismatch")
    if campaign._gate_path(manifest, manifest_path) != rooted_gates:
        raise ValueError("manifest frozen-gates path differs from provenance root")
    manifest_hash = _sha256(manifest_path)
    gates_hash = _sha256(rooted_gates)
    protocol_hash = _sha256(rooted_protocol)
    expected_fields = {
        "audit_manifest_sha256": manifest_hash,
        "frozen_gates_sha256": gates_hash,
        "protocol_sha256": protocol_hash,
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "job_count": manifest["job_count"],
    }
    for key, expected in expected_fields.items():
        if root.get(key) != expected:
            raise ValueError(f"audit provenance root {key} mismatch")
    if manifest.get("frozen_gates_sha256") != gates_hash:
        raise ValueError("audit manifest does not bind the frozen gates")
    if manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("audit manifest protocol hash mismatch")
    return {
        "path": root_path,
        "file_sha256": _sha256(root_path),
        "content_sha256": root["content_sha256"],
        "payload": root,
    }


def create_source_equivalence(
    manifest_path: Path,
    output_path: Path,
    provenance_root_path: Path,
) -> dict[str, Any]:
    """Attest revision-only source identity between a manifest and current tree."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    _validate_source_manifest(manifest)
    trusted_root = validate_provenance_root(manifest_path, provenance_root_path)
    state = _repo_state()
    if state.get("paper_dirty") is not False:
        raise RuntimeError("source equivalence requires a clean paper repository")
    current_files, current_snapshot = campaign._source_snapshot()
    current_revision = state.get("library_revision")
    if not isinstance(current_revision, str) or not current_revision:
        raise RuntimeError("current library revision is unavailable")
    if manifest["source_file_sha256"] != current_files:
        raise RuntimeError("source file hashes differ from the reference manifest")
    if manifest.get("protocol_sha256") != campaign._sha256(campaign.PROTOCOL_PATH):
        raise RuntimeError("protocol differs from the reference manifest")
    if current_snapshot != _snapshot_digest(current_files, current_revision):
        raise RuntimeError("current source snapshot digest is inconsistent")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "reference_manifest": {
            "path": manifest_path.name,
            "sha256": _sha256(manifest_path),
        },
        "provenance_root": {
            "path": Path(provenance_root_path).name,
            "file_sha256": trusted_root["file_sha256"],
            "content_sha256": trusted_root["content_sha256"],
        },
        "protocol_sha256": manifest["protocol_sha256"],
        "original": {
            "source_file_sha256": manifest["source_file_sha256"],
            "source_snapshot_sha256": manifest["source_snapshot_sha256"],
            "library_revision": manifest["library_revision"],
            "repository_state": manifest["state"],
        },
        "current": {
            "source_file_sha256": current_files,
            "source_snapshot_sha256": current_snapshot,
            "library_revision": current_revision,
            "repository_state": state,
        },
    }
    payload["content_sha256"] = _content_digest(payload)
    _atomic_json(Path(output_path), payload)
    return payload


def validate_source_equivalence(
    manifest_path: Path,
    bridge_path: Path,
    provenance_root_path: Path,
) -> dict[str, Any]:
    """Validate a revision-only bridge against reference and current provenance."""
    manifest_path = Path(manifest_path)
    bridge = json.loads(Path(bridge_path).read_text())
    manifest = json.loads(manifest_path.read_text())
    _validate_source_manifest(manifest)
    trusted_root = validate_provenance_root(manifest_path, provenance_root_path)
    if not isinstance(bridge, dict) or bridge.get("schema_version") != 1:
        raise ValueError("source-equivalence bridge has an unsupported schema")
    if bridge.get("content_sha256") != _content_digest(bridge):
        raise ValueError("source-equivalence bridge content digest is invalid")
    reference = bridge.get("reference_manifest")
    if not isinstance(reference, dict) or reference.get("sha256") != _sha256(
        manifest_path
    ):
        raise ValueError("source-equivalence bridge references a different manifest")
    original = bridge.get("original")
    current = bridge.get("current")
    if not isinstance(original, dict) or not isinstance(current, dict):
        raise ValueError("source-equivalence bridge is malformed")
    expected_original = {
        "source_file_sha256": manifest["source_file_sha256"],
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "library_revision": manifest["library_revision"],
        "repository_state": manifest["state"],
    }
    if original != expected_original:
        raise ValueError("source-equivalence bridge original provenance mismatch")
    bridge_root = bridge.get("provenance_root")
    expected_root = {
        "path": Path(provenance_root_path).name,
        "file_sha256": trusted_root["file_sha256"],
        "content_sha256": trusted_root["content_sha256"],
    }
    if bridge_root != expected_root:
        raise ValueError("source-equivalence bridge provenance root mismatch")
    captured_files = current.get("source_file_sha256")
    captured_revision = current.get("library_revision")
    captured_snapshot = current.get("source_snapshot_sha256")
    captured_state = current.get("repository_state")
    if (
        not isinstance(captured_state, dict)
        or captured_state.get("paper_dirty") is not False
        or not isinstance(captured_state.get("paper_revision"), str)
        or not captured_state.get("paper_revision")
        or not isinstance(captured_revision, str)
        or not captured_revision
        or captured_state.get("library_revision") != captured_revision
        or not isinstance(captured_files, dict)
        or captured_snapshot != _snapshot_digest(captured_files, captured_revision)
    ):
        raise ValueError("bridge captured current provenance is inconsistent")
    if bridge.get("protocol_sha256") != campaign._sha256(campaign.PROTOCOL_PATH):
        raise RuntimeError("source-equivalence bridge protocol differs")
    state = _repo_state()
    if state.get("paper_dirty") is not False:
        raise RuntimeError(
            "source-equivalence validation requires a clean paper repository"
        )
    current_files, current_snapshot = campaign._source_snapshot()
    current_revision = state.get("library_revision")
    if (
        captured_files != current_files
        or captured_snapshot != current_snapshot
        or captured_revision != current_revision
    ):
        raise RuntimeError("current source provenance differs from the bridge")
    if original["source_file_sha256"] != current_files:
        raise RuntimeError("source-equivalence bridge contains changed file hashes")
    if current_snapshot != _snapshot_digest(current_files, str(current_revision)):
        raise RuntimeError("current source snapshot digest is inconsistent")
    return bridge


def load_records(
    manifest_path: Path,
    records_dir: Path | None = None,
    source_equivalence_path: Path | None = None,
    provenance_root_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a complete and internally consistent job-record set."""
    manifest_path = Path(manifest_path)
    if source_equivalence_path is None:
        manifest, loaded_jobs = campaign._load_manifest(manifest_path)
        if provenance_root_path is not None:
            validate_provenance_root(manifest_path, provenance_root_path)
    else:
        if provenance_root_path is None:
            raise ValueError("source-equivalence bridge requires --provenance-root")
        validate_source_equivalence(
            manifest_path, source_equivalence_path, provenance_root_path
        )
        manifest = json.loads(manifest_path.read_text())
        try:
            loaded_jobs = [_job_dict(job) for job in manifest["jobs"]]
        except (KeyError, TypeError) as error:
            raise ValueError("manifest campaign jobs are malformed") from error
    jobs = [_job_dict(campaign.asdict(job)) for job in loaded_jobs]
    records_dir = records_dir or manifest_path.parent / "job_records"
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("manifest contains duplicate job ids")
    found: list[dict[str, Any]] = []
    missing: list[str] = []
    expected_path = {f"{job.job_id}.json": job for job in jobs}
    for name, job in expected_path.items():
        path = records_dir / name
        if not path.is_file():
            missing.append(job.job_id)
            continue
        record = json.loads(path.read_text())
        recorded_job = _job_dict(record["job"])
        if recorded_job != job:
            raise ValueError(f"job record content mismatch for {job.job_id}")
        run_record = record["run_record"]
        # Remote shards record the manifest's absolute path on the worker host.
        # The job payload and frozen manifest hashes are the portable identity;
        # retain a basename check here to catch records copied from another phase.
        if (
            record.get("manifest")
            and Path(record["manifest"]).name != manifest_path.name
        ):
            raise ValueError(f"job record manifest path mismatch for {job.job_id}")
        run_dir = Path(run_record.get("directory", ""))
        if not run_dir.is_dir():
            run_dir = records_dir.parent / str(run_record["run_id"])
        checkpoint = run_dir / "checkpoint.npz"
        if not run_dir.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError(
                f"missing run directory/checkpoint for {job.job_id}"
            )
        found.append({"job": job, "record": record, "checkpoint": checkpoint})
    if missing:
        raise ValueError(f"missing job records: {', '.join(missing)}")
    if manifest.get("phase") == "audit":
        gate_path = campaign._gate_path(manifest, manifest_path)
        if not gate_path.is_file() or _sha256(gate_path) != manifest.get(
            "frozen_gates_sha256"
        ):
            raise RuntimeError("frozen audit gates are missing or changed")
    return manifest, found


def _moments(prob: np.ndarray) -> tuple[float, float]:
    profile = metrics.hamming_profile(prob, N)
    weights = np.arange(N + 1, dtype=float)
    mean = float(profile @ weights)
    std = float(np.sqrt(np.sum(profile * (weights - mean) ** 2)))
    return mean, std


def metric_row(generated: np.ndarray, target: np.ndarray) -> dict[str, float]:
    gm, gs = _moments(generated)
    tm, ts = _moments(target)
    return {
        "bitstring_tvd": float(metrics.bitstring_tvd(generated, target)),
        "hamming_profile_tvd": float(metrics.profile_tvd(generated, target, N)),
        "generated_mean": gm,
        "target_mean": tm,
        "absolute_mean_error": abs(gm - tm),
        "generated_std": gs,
        "target_std": ts,
        "absolute_std_error": abs(gs - ts),
        "within_shell_residual": float(metrics.within_shell_residual(generated, N)),
    }


def _condition_rows(
    entry: dict[str, Any],
    phase: str,
    split_name: str,
    conditions: np.ndarray,
    generated: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, Any]]:
    job: campaign.CampaignJob = entry["job"]
    rows = []
    for condition, model, target in zip(conditions, generated, targets, strict=True):
        category = protocol.classify_condition(
            float(condition), job.protocol, np.asarray(job.anchors)
        )
        row = {
            "phase": phase,
            "split": split_name,
            "encoder": job.encoder,
            "protocol": job.protocol,
            "anchor_count": job.anchor_count,
            "placement": job.placement,
            "layout_seed": job.layout_seed,
            "optimizer_seed": job.optimizer_seed,
            "budget_mode": job.budget_mode,
            "hyperparameter_id": job.hyperparameter_id,
            "model_kind": job.model_kind,
            "source": "training" if category == "training" else split_name,
            "condition": float(condition),
            "category": category,
        }
        row.update(metric_row(model, target))
        rows.append(row)
    pairs = {round(float(c), 12): i for i, c in enumerate(conditions)}
    for row in rows:
        complement = round(1.0 - row["condition"], 12)
        if row["condition"] <= 0.5 + 1e-12 and complement in pairs:
            row["complement_covariance_error"] = float(
                metrics.complement_tvd(
                    generated[pairs[round(row["condition"], 12)]],
                    generated[pairs[complement]],
                )
            )
        else:
            row["complement_covariance_error"] = None
    return rows


def extract_rows(
    entries: list[dict[str, Any]], phase: str, split_name: str
) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        with np.load(entry["checkpoint"]) as data:
            rows.extend(
                _condition_rows(
                    entry,
                    phase,
                    split_name,
                    data["training_conditions"],
                    data["train_probabilities"],
                    data["target_probabilities"],
                )
            )
            if entry["job"].model_kind != "independent_direct":
                rows.extend(
                    _condition_rows(
                        entry,
                        phase,
                        split_name,
                        data["evaluation_conditions"],
                        data["held_out_probabilities"],
                        data["held_out_target_probabilities"],
                    )
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write(path, _csv_payload(rows))


def _csv_payload(rows: list[dict[str, Any]]) -> str:
    import csv
    import io

    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    q25, median, q75 = np.quantile(arr, [0.25, 0.5, 0.75])
    return {
        "median": float(median),
        "mean": float(arr.mean()),
        "iqr": float(q75 - q25),
        "max": float(arr.max()),
    }


def _record_gate_check(
    checks: dict[str, bool | None],
    details: dict[str, dict[str, Any]],
    name: str,
    observed: Any,
    threshold: Any,
    passed: Any,
) -> None:
    native_pass = None if passed is None else bool(passed)
    checks[name] = native_pass
    details[name] = {
        "observed": observed,
        "threshold": threshold,
        "pass": native_pass,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = (
        "encoder",
        "protocol",
        "anchor_count",
        "placement",
        "layout_seed",
        "budget_mode",
        "category",
    )
    for row in rows:
        if row["category"] == "training":
            continue
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for key, group in sorted(
        groups.items(), key=lambda item: tuple(str(x) for x in item[0])
    ):
        item = dict(zip(keys, key, strict=True))
        item["seeds"] = sorted({r["optimizer_seed"] for r in group})
        item["metrics"] = {
            metric: _summary([r[metric] for r in group]) for metric in METRIC_KEYS
        }
        # This group fixes the layout but still spans condition rows and
        # optimizer seeds, so its undifferentiated variance is not a variance
        # component attributable to either source.  The marginal decomposition
        # below reports those two components separately.
        item["within_layout_row_variance"] = float(
            np.var([r["bitstring_tvd"] for r in group])
        )
        result.append(item)
    variance_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    variance_keys = tuple(key for key in keys if key != "layout_seed")
    for row in rows:
        if row["category"] == "training":
            continue
        variance_groups.setdefault(tuple(row[k] for k in variance_keys), []).append(row)
    decomposition = []
    for key, group in variance_groups.items():
        layout_means = {}
        seed_means = {}
        for layout in {r["layout_seed"] for r in group}:
            values = [r["bitstring_tvd"] for r in group if r["layout_seed"] == layout]
            layout_means[layout] = float(np.mean(values))
        for seed in {r["optimizer_seed"] for r in group}:
            values = [r["bitstring_tvd"] for r in group if r["optimizer_seed"] == seed]
            seed_means[seed] = float(np.mean(values))
        decomposition.append(
            {
                **dict(zip(variance_keys, key, strict=True)),
                "layout_marginal_means": {str(k): v for k, v in layout_means.items()},
                "optimizer_seed_marginal_means": {
                    str(k): v for k, v in seed_means.items()
                },
                "layout_variance": float(np.var(list(layout_means.values())))
                if len(layout_means) > 1
                else 0.0,
                "optimizer_seed_variance": float(np.var(list(seed_means.values())))
                if len(seed_means) > 1
                else 0.0,
            }
        )
    return {"groups": result, "variance_decomposition": decomposition}


def encoder_only_rows(
    conditions: np.ndarray,
    encoder: str,
    split: str,
    name: str = "interpolation",
    anchors: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Evaluate the frozen product encoder with a zero-parameter IQP circuit."""
    from .study import build_study

    values = np.asarray(conditions, dtype=float)
    if anchors is None:
        anchors = np.asarray(protocol.anchor_conditions(name, 3))
    config = RunConfig(
        study="scalar_gaussian_translation",
        encoder=encoder,
        condition_protocol=name,
        n_visible=N,
        n_ancillas=0,
        width_factor=protocol.WIDTH_FACTOR,
    )
    study = build_study(name, len(anchors), "even", None, encoder, split)
    ansatz = study.make_ansatz(config)
    model = ciqp.ConditionalModel(ansatz, study.make_encoder(values, config))
    probabilities = model.probabilities(np.zeros(ansatz.n_parameters))
    targets = protocol.target_probabilities(values)
    return _condition_rows(
        {
            "job": campaign.CampaignJob(
                "analysis",
                split,
                encoder,
                name,
                tuple(float(x) for x in anchors),
                len(anchors),
                "even",
                None,
                0,
                "analysis",
                "encoder-only",
                {},
                "encoder_only",
            ),
            "checkpoint": None,
        },
        "encoder_only",
        split,
        values,
        probabilities,
        targets,
    )


def interpolation_rows(
    anchor_conditions: np.ndarray,
    anchor_distributions: np.ndarray,
    conditions: np.ndarray,
    protocol_name: str,
    split: str,
) -> list[dict[str, Any]]:
    """Exact target anchors, convex linear interpolation, then endpoint hold."""
    anchors = np.asarray(anchor_conditions, dtype=float)
    distributions = np.asarray(anchor_distributions, dtype=float)
    if (
        anchors.ndim != 1
        or distributions.shape != (anchors.size, 2**N)
        or np.any(np.diff(anchors) <= 0)
    ):
        raise ValueError("anchors and distributions have incompatible shapes/order")
    values = np.asarray(conditions, dtype=float)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("conditions must lie in [0, 1]")
    out = []
    for c in values:
        if c <= anchors[0]:
            p = distributions[0]
        elif c >= anchors[-1]:
            p = distributions[-1]
        else:
            right = int(np.searchsorted(anchors, c))
            fraction = (c - anchors[right - 1]) / (anchors[right] - anchors[right - 1])
            p = (1.0 - fraction) * distributions[right - 1] + fraction * distributions[
                right
            ]
        out.append(p)
    job = campaign.CampaignJob(
        "analysis",
        split,
        "classical_anchor_interpolation",
        protocol_name,
        tuple(anchors),
        len(anchors),
        "specified",
        None,
        0,
        "analysis",
        "linear-convex-anchor-mixture",
        {},
        "classical_baseline",
    )
    return _condition_rows(
        {"job": job},
        "classical_baseline",
        split,
        values,
        np.asarray(out),
        protocol.target_probabilities(values),
    )


def _dev_rows(
    manifest_path: Path,
    records_dir: Path | None = None,
    source_equivalence_path: Path | None = None,
    provenance_root_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if source_equivalence_path is None:
        if provenance_root_path is None:
            manifest, entries = load_records(manifest_path, records_dir)
        else:
            manifest, entries = load_records(
                manifest_path,
                records_dir,
                provenance_root_path=provenance_root_path,
            )
    else:
        if provenance_root_path is None:
            raise ValueError("source-equivalence bridge requires --provenance-root")
        manifest, entries = load_records(
            manifest_path,
            records_dir,
            source_equivalence_path,
            provenance_root_path,
        )
    return manifest, extract_rows(entries, manifest["phase"], manifest["split"])


def stage1_summary(
    manifest_path: Path, records_dir: Path | None = None
) -> dict[str, Any]:
    manifest, rows = _dev_rows(manifest_path, records_dir)
    if manifest["phase"] != "stage1" or manifest["split"] != "dev":
        raise ValueError("stage1 summary requires development Stage 1 records")
    gate = json.loads(campaign.PROTOCOL_PATH.read_text())["stage1_development_gate"][
        "advance_ry_encoder_if_all"
    ]
    summaries = {}
    for encoder in campaign.STAGE1_ENCODERS:
        group = [r for r in rows if r["encoder"] == encoder and r["source"] == "dev"]
        seeds = sorted({r["optimizer_seed"] for r in group})
        if len(seeds) != 3:
            raise ValueError(f"Stage 1 requires three dev seeds for {encoder}")
        held = [r for r in group if r["category"] == "held_out_interpolation"]
        train = [
            r for r in rows if r["encoder"] == encoder and r["category"] == "training"
        ]
        by_seed = {s: [r for r in held if r["optimizer_seed"] == s] for s in seeds}
        seed_means = {
            s: float(np.mean([r["bitstring_tvd"] for r in by_seed[s]])) for s in seeds
        }
        train_seed = {
            s: float(
                np.mean([r["bitstring_tvd"] for r in train if r["optimizer_seed"] == s])
            )
            for s in seeds
        }
        baseline = encoder_only_rows(
            protocol.evaluation_grid("dev"),
            encoder,
            "dev",
            "interpolation",
            np.asarray([0.0, 0.5, 1.0]),
        )
        base_by_seed_mean = float(
            np.mean(
                [
                    r["bitstring_tvd"]
                    for r in baseline
                    if r["category"] == "held_out_interpolation"
                ]
            )
        )
        improvements = [base_by_seed_mean - seed_means[s] for s in seeds]
        movement = []
        for seed in seeds:
            seed_rows = sorted(
                [
                    r
                    for r in rows
                    if r["encoder"] == encoder
                    and r["optimizer_seed"] == seed
                    and r["category"] == "training"
                ],
                key=lambda r: r["condition"],
            )
            model_movement = (
                seed_rows[-1]["generated_mean"] - seed_rows[0]["generated_mean"]
            )
            target_movement = seed_rows[-1]["target_mean"] - seed_rows[0]["target_mean"]
            movement.append((model_movement, target_movement))
        checks = {
            "median_train_tvd_at_most": float(np.median(list(train_seed.values())))
            <= gate["median_train_tvd_at_most"],
            "median_held_interpolation_tvd_at_most": float(
                np.median(list(seed_means.values()))
            )
            <= gate["median_held_interpolation_tvd_at_most"],
            "worst_seed_held_mean_tvd_at_most": max(seed_means.values())
            <= gate["worst_seed_held_mean_tvd_at_most"],
            "median_improvement_at_least": (
                float(np.median(improvements))
                >= gate[
                    "median_absolute_improvement_over_own_encoder_only_held_mean_tvd_at_least"
                ]
            ),
            "generated_mean_movement_c0_to_c1": all(
                np.sign(m) == np.sign(t) and abs(m) >= 0.4 * abs(t) for m, t in movement
            ),
        }
        summaries[encoder] = {
            "pass": all(checks.values()),
            "checks": checks,
            "reasons": [k for k, v in checks.items() if not v],
            "seed_held_mean_tvd": seed_means,
            "seed_train_mean_tvd": train_seed,
            "encoder_only_held_mean_tvd": base_by_seed_mean,
            "seed_absolute_improvement": dict(zip(seeds, improvements, strict=True)),
            "mean_movement": {
                str(s): {"generated": m, "target": t}
                for s, (m, t) in zip(seeds, movement, strict=True)
            },
        }
    return {
        "manifest_sha256": _sha256(Path(manifest_path)),
        "protocol_sha256": manifest["protocol_sha256"],
        "gate": gate,
        "encoders": summaries,
    }


def select_stage2(
    manifest_path: Path, output_path: Path, records_dir: Path | None = None
) -> dict[str, Any]:
    manifest, rows = _dev_rows(manifest_path, records_dir)
    if manifest["phase"] != "stage2" or manifest["split"] != "dev":
        raise ValueError("Stage 2 selection requires dev Stage 2 records")
    candidates = sorted({r["hyperparameter_id"] for r in rows})
    if len(candidates) != 8:
        raise ValueError("Stage 2 requires all eight frozen candidates")
    ranking = []
    entries = load_records(manifest_path, records_dir)[1]
    for hp_id in candidates:
        subset = [
            r
            for r in rows
            if r["hyperparameter_id"] == hp_id
            and r["source"] == "dev"
            and r["category"] == "held_out_interpolation"
        ]
        seeds = sorted({r["optimizer_seed"] for r in subset})
        encoders = sorted({r["encoder"] for r in subset})
        if set(encoders) != set(campaign.STAGE1_ENCODERS) or len(seeds) != 3:
            raise ValueError(f"candidate {hp_id} lacks both encoders or three seeds")
        seed_means = {
            s: float(
                np.mean(
                    [r["bitstring_tvd"] for r in subset if r["optimizer_seed"] == s]
                )
            )
            for s in seeds
        }
        job = next(e["job"] for e in entries if e["job"].hyperparameter_id == hp_id)
        hp = job.hyperparameters
        ranking.append(
            {
                "id": hp_id,
                "hyperparameters": hp,
                "median_held_interpolation_tvd": float(
                    np.median(list(seed_means.values()))
                ),
                "worst_seed_held_mean_tvd": max(seed_means.values()),
                "compute": int(hp["updates"]) * int(hp["restarts"]),
                "seed_means": seed_means,
                "rejection_reasons": [],
            }
        )
    ranking.sort(
        key=lambda x: (
            x["median_held_interpolation_tvd"],
            x["worst_seed_held_mean_tvd"],
            x["compute"],
            x["id"],
        )
    )
    labels = (
        "median held interpolation TVD",
        "worst-seed mean TVD",
        "compute",
        "candidate id",
    )
    for index, candidate in enumerate(ranking[1:], 1):
        previous = ranking[index - 1]
        candidate_key = (
            candidate["median_held_interpolation_tvd"],
            candidate["worst_seed_held_mean_tvd"],
            candidate["compute"],
            candidate["id"],
        )
        previous_key = (
            previous["median_held_interpolation_tvd"],
            previous["worst_seed_held_mean_tvd"],
            previous["compute"],
            previous["id"],
        )
        position = next(
            i
            for i, (left, right) in enumerate(
                zip(candidate_key, previous_key, strict=True)
            )
            if left != right
        )
        candidate["rejection_reasons"] = [
            f"ranked below {previous['id']} by {labels[position]}"
        ]
    selected = ranking[0]
    result = {
        "hyperparameter_id": selected["id"],
        "hyperparameters": selected["hyperparameters"],
        "ranking": ranking,
        "manifest_sha256": _sha256(Path(manifest_path)),
        "protocol_sha256": manifest["protocol_sha256"],
        "source_hashes": {
            "protocol.json": manifest["protocol_sha256"],
            "source_snapshot": manifest.get("source_snapshot_sha256"),
        },
        "source_snapshot_sha256": manifest.get("source_snapshot_sha256"),
        "library_revision": manifest.get("state", {}).get("library_revision"),
        "rationale": (
            "One common configuration is selected jointly across both general "
            "RY encoders and all development seeds, preserving a controlled "
            "encoder comparison."
        ),
    }
    _atomic_json(output_path, result)
    return result


def evaluate_gates(rows: list[dict[str, Any]], gates_path: Path) -> dict[str, Any]:
    """Apply a supplied frozen_gates file without changing any threshold."""
    gates = json.loads(Path(gates_path).read_text())
    if gates.get("frozen") is not True:
        raise ValueError("frozen_gates must set frozen=true")
    # Mirrored protocols are directional diagnostics only. Keep the filter at
    # the evaluator boundary so no aggregate or reliability check can consume
    # them accidentally.
    qualifying_rows = [
        row for row in rows if row.get("protocol") not in campaign.MIRROR_PROTOCOLS
    ]
    results = {}
    criteria = gates["criteria"]
    for encoder in ("broadcast_affine", "qubit_affine", "anti_block"):
        encoder_rows = [r for r in qualifying_rows if r["encoder"] == encoder]
        checks: dict[str, bool] = {}
        check_details: dict[str, dict[str, Any]] = {}
        reasons: list[str] = []
        record_check = partial(_record_gate_check, checks, check_details)

        seed_means: dict[str, dict[int, float]] = {}
        for category, caps in criteria["categories"].items():
            subset = [r for r in encoder_rows if r["category"] == category]
            by_seed = {
                seed: [r for r in subset if r["optimizer_seed"] == seed]
                for seed in {r["optimizer_seed"] for r in subset}
            }
            means = {
                seed: float(np.mean([r["bitstring_tvd"] for r in values]))
                for seed, values in by_seed.items()
            }
            seed_means[category] = means
            if not subset:
                reasons.append(f"missing {category} data")
                record_check(f"{category}.present", False, True, False)
                continue
            observed = float(np.median(list(means.values())))
            threshold = caps["median_seed_mean_tvd_at_most"]
            record_check(
                f"{category}.median_seed_mean",
                observed,
                threshold,
                observed <= threshold,
            )
            observed = max(means.values())
            threshold = caps["worst_seed_mean_tvd_at_most"]
            record_check(
                f"{category}.worst_seed_mean",
                observed,
                threshold,
                observed <= threshold,
            )
            observed = max(r["bitstring_tvd"] for r in subset)
            threshold = caps["worst_condition_tvd_at_most"]
            record_check(
                f"{category}.worst_condition",
                observed,
                threshold,
                observed <= threshold,
            )
        general = encoder in {"broadcast_affine", "qubit_affine"}
        if general:
            conditions = np.asarray(
                sorted({float(r["condition"]) for r in encoder_rows}), dtype=float
            )
            baseline = encoder_only_rows(conditions, encoder, "audit")
            baseline_tvd = {
                float(r["condition"]): float(r["bitstring_tvd"]) for r in baseline
            }
            improvement_gate = criteria["encoder_only_improvement"]
            improvements = []
            for category in criteria["categories"]:
                category_rows = [r for r in encoder_rows if r["category"] == category]
                for seed in seed_means.get(category, {}):
                    selected = [r for r in category_rows if r["optimizer_seed"] == seed]
                    improvements.append(
                        float(
                            np.mean(
                                [baseline_tvd[float(r["condition"])] for r in selected]
                            )
                        )
                        - seed_means[category][seed]
                    )
            observed = float(np.median(improvements)) if improvements else None
            threshold = improvement_gate["median_seed_mean_tvd_improvement_at_least"]
            record_check(
                "encoder_only_improvement.median",
                observed,
                threshold,
                observed is not None and observed >= threshold,
            )
            observed = min(improvements) if improvements else None
            threshold = improvement_gate["worst_seed_mean_tvd_improvement_at_least"]
            record_check(
                "encoder_only_improvement.worst_group",
                observed,
                threshold,
                observed is not None and observed >= threshold,
            )
            movement = [r for r in encoder_rows if r["category"] == "training"]
            movement_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            group_fields = (
                "protocol",
                "anchor_count",
                "placement",
                "layout_seed",
                "budget_mode",
                "optimizer_seed",
            )
            for row in movement:
                movement_groups.setdefault(
                    tuple(row[field] for field in group_fields), []
                ).append(row)
            ratios: list[float] = []
            signs: list[bool] = []
            ratios_by_seed: dict[int, list[float]] = {}
            for key, group in movement_groups.items():
                ordered = sorted(group, key=lambda r: r["condition"])
                gm = ordered[-1]["generated_mean"] - ordered[0]["generated_mean"]
                tm = ordered[-1]["target_mean"] - ordered[0]["target_mean"]
                ratio = float(gm / tm) if tm else 0.0
                ratios.append(ratio)
                signs.append(bool(np.sign(gm) == np.sign(tm)))
                ratios_by_seed.setdefault(int(key[-1]), []).append(ratio)
            movement_gate = criteria["mean_movement"]
            observed = float(np.median(ratios)) if ratios else None
            threshold = movement_gate["median_generated_to_target_ratio_at_least"]
            record_check(
                "mean_movement.median_ratio",
                observed,
                threshold,
                observed is not None and observed >= threshold,
            )
            observed = (
                min(float(np.median(values)) for values in ratios_by_seed.values())
                if ratios_by_seed
                else None
            )
            threshold = movement_gate["worst_seed_median_ratio_at_least"]
            record_check(
                "mean_movement.worst_seed_median_ratio",
                observed,
                threshold,
                observed is not None and observed >= threshold,
            )
            observed = float(np.mean(signs)) if signs else None
            threshold = movement_gate["correct_sign_fraction_at_least"]
            record_check(
                "mean_movement.correct_sign_fraction",
                observed,
                threshold,
                observed is not None and observed >= threshold,
            )
            width = criteria["width"]
            width_errors = [r["absolute_std_error"] for r in encoder_rows]
            by_seed_width = [
                np.mean(
                    [
                        r["absolute_std_error"]
                        for r in encoder_rows
                        if r["optimizer_seed"] == seed
                    ]
                )
                for seed in {r["optimizer_seed"] for r in encoder_rows}
            ]
            observed = float(np.median(width_errors)) if width_errors else None
            threshold = width["median_absolute_std_error_at_most"]
            record_check(
                "width.median",
                observed,
                threshold,
                observed is not None and observed <= threshold,
            )
            observed = max(by_seed_width) if by_seed_width else None
            threshold = width["worst_seed_mean_absolute_std_error_at_most"]
            record_check(
                "width.worst_seed_mean",
                observed,
                threshold,
                observed is not None and observed <= threshold,
            )
            structure = criteria["structure"]
            profile = [r["hamming_profile_tvd"] for r in encoder_rows]
            shell = [r["within_shell_residual"] for r in encoder_rows]
            seeds = {r["optimizer_seed"] for r in encoder_rows}
            profile_by_seed = [
                float(
                    np.mean(
                        [
                            r["hamming_profile_tvd"]
                            for r in encoder_rows
                            if r["optimizer_seed"] == seed
                        ]
                    )
                )
                for seed in seeds
            ]
            shell_by_seed = [
                float(
                    np.mean(
                        [
                            r["within_shell_residual"]
                            for r in encoder_rows
                            if r["optimizer_seed"] == seed
                        ]
                    )
                )
                for seed in seeds
            ]
            for name, values, threshold_key, reducer in (
                (
                    "structure.profile",
                    profile,
                    "median_hamming_profile_tvd_at_most",
                    np.median,
                ),
                (
                    "structure.shell",
                    shell,
                    "median_within_shell_residual_at_most",
                    np.median,
                ),
                (
                    "structure.profile_worst_seed_mean",
                    profile_by_seed,
                    "worst_seed_mean_hamming_profile_tvd_at_most",
                    max,
                ),
                (
                    "structure.shell_worst_seed_mean",
                    shell_by_seed,
                    "worst_seed_mean_within_shell_residual_at_most",
                    max,
                ),
            ):
                observed = float(reducer(values)) if values else None
                threshold = structure[threshold_key]
                record_check(
                    name,
                    observed,
                    threshold,
                    observed is not None and observed <= threshold,
                )
            reliability = criteria["reliability"]
            seed_tvd = [
                np.mean(
                    [
                        r["bitstring_tvd"]
                        for r in encoder_rows
                        if r["optimizer_seed"] == seed
                    ]
                )
                for seed in {r["optimizer_seed"] for r in encoder_rows}
            ]
            observed = max(seed_tvd) - min(seed_tvd) if seed_tvd else None
            threshold = reliability["max_optimizer_seed_range_at_most"]
            record_check(
                "reliability.optimizer_seed_range",
                observed,
                threshold,
                observed is not None and observed <= threshold,
            )
            for category, cap in reliability[
                "max_random_layout_median_tvd_at_most_by_category"
            ].items():
                random_rows = [
                    r
                    for r in encoder_rows
                    if r["placement"] == "random" and r["category"] == category
                ]
                by_layout: dict[int, list[float]] = {}
                for row in random_rows:
                    # A frozen random layout is identified by its dedicated
                    # layout seed.  Each seed recurs across protocols, anchor
                    # counts, budgets, and optimizer seeds by design; those
                    # repetitions form the evidence for that layout rather
                    # than separate layouts that can change the estimand.
                    identity = int(row["layout_seed"])
                    by_layout.setdefault(identity, []).append(row["bitstring_tvd"])
                layout_medians = {
                    str(identity): float(np.median(values))
                    for identity, values in by_layout.items()
                }
                observed = max(layout_medians.values()) if layout_medians else None
                record_check(
                    f"reliability.random_layout.{category}",
                    {"per_layout_median_tvd": layout_medians, "maximum": observed},
                    cap,
                    observed is not None and observed <= cap,
                )
        results[encoder] = {
            "pass": bool(all(value is True for value in checks.values()))
            if general
            else None,
            "checks": checks,
            "check_details": check_details,
            "reasons": [k for k, v in checks.items() if v is not True],
            "seed_mean_tvd": seed_means,
        }
    return {"frozen_gates_sha256": _sha256(Path(gates_path)), "encoders": results}


def _read_rows_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for source_row in csv.DictReader(stream):
            row: dict[str, Any] = {}
            for key, raw in source_row.items():
                value = raw.strip() if raw is not None else ""
                if not value:
                    row[key] = None
                    continue
                try:
                    row[key] = int(value)
                    continue
                except ValueError:
                    pass
                try:
                    numeric = float(value)
                except ValueError:
                    row[key] = value
                else:
                    row[key] = numeric if np.isfinite(numeric) else None
            rows.append(row)
    return rows


def attest_audit(
    manifest_path: Path,
    records_dir: Path,
    gates_path: Path,
    rows_csv: Path,
    gate_result_path: Path,
    output_path: Path,
    source_equivalence_path: Path | None = None,
    provenance_root_path: Path | None = None,
) -> dict[str, Any]:
    """Write a deterministic content-addressed attestation for an audit."""
    manifest_path = Path(manifest_path)
    records_dir = Path(records_dir)
    gates_path = Path(gates_path)
    rows_csv = Path(rows_csv)
    gate_result_path = Path(gate_result_path)
    if source_equivalence_path is None:
        manifest, entries = load_records(manifest_path, records_dir)
    else:
        manifest, entries = load_records(
            manifest_path,
            records_dir,
            source_equivalence_path,
            provenance_root_path,
        )
    if manifest.get("phase") != "audit" or manifest.get("split") != "audit":
        raise ValueError("audit attestation requires an audit manifest")
    gates = json.loads(gates_path.read_text())
    expected_seeds = sorted(int(seed) for seed in gates.get("audit_seeds", []))
    observed_seeds = sorted({entry["job"].optimizer_seed for entry in entries})
    if not expected_seeds or observed_seeds != expected_seeds:
        raise ValueError(
            "audit seed set mismatch: "
            f"expected {expected_seeds}, observed {observed_seeds}"
        )
    expected_count = int(
        manifest["job_count"] if "job_count" in manifest else len(manifest["jobs"])
    )
    observed_count = len(entries)
    if observed_count != expected_count:
        raise ValueError(
            "audit job count mismatch: "
            f"expected {expected_count}, observed {observed_count}"
        )
    regenerated_rows = extract_rows(entries, manifest["phase"], manifest["split"])
    regenerated_csv = _csv_payload(regenerated_rows).encode("utf-8")
    supplied_csv = rows_csv.read_bytes()
    if regenerated_csv != supplied_csv:
        raise ValueError("supplied audit rows differ from checkpoint regeneration")
    row_verification = {
        "matches_regenerated_rows": True,
        "row_count": len(regenerated_rows),
        "canonical_csv_sha256": hashlib.sha256(regenerated_csv).hexdigest(),
    }
    gates_hash = _sha256(gates_path)
    if manifest.get("frozen_gates_sha256") != gates_hash:
        raise ValueError("audit manifest does not bind the supplied frozen gates")
    trusted_root = (
        validate_provenance_root(manifest_path, provenance_root_path, gates_path)
        if provenance_root_path is not None
        else None
    )
    if source_equivalence_path is not None and trusted_root is None:
        raise ValueError("source-equivalence bridge requires --provenance-root")
    gate_result = json.loads(gate_result_path.read_text())
    if gate_result.get("frozen_gates_sha256") != gates_hash:
        raise ValueError("gate result does not bind the supplied frozen gates")
    rows = _read_rows_csv(rows_csv)
    recomputed_gates = evaluate_gates(rows, gates_path)
    normalized_supplied = json.loads(
        json.dumps(gate_result, sort_keys=True, separators=(",", ":"))
    )
    normalized_recomputed = json.loads(
        json.dumps(recomputed_gates, sort_keys=True, separators=(",", ":"))
    )
    if normalized_supplied != normalized_recomputed:
        raise ValueError("supplied gate result differs from recomputation")
    gate_status: dict[str, dict[str, Any]] = {}
    for encoder in ("broadcast_affine", "qubit_affine"):
        encoder_result = gate_result.get("encoders", {}).get(encoder, {})
        checks = encoder_result.get("checks", {})
        check_count = len(checks)
        all_true = check_count == 28 and all(value is True for value in checks.values())
        gate_status[encoder] = {
            "check_count": check_count,
            "all_28_checks_native_true": bool(all_true),
            "reported_pass": encoder_result.get("pass") is True,
        }
    general_gates_valid = all(
        status["all_28_checks_native_true"] and status["reported_pass"]
        for status in gate_status.values()
    )
    if not general_gates_valid:
        raise ValueError(
            "both general encoders must report pass=true and exactly "
            "28 native-true checks"
        )
    gate_verification = {
        "recomputed_result_matches_supplied": True,
        "general_encoders": gate_status,
        "valid": bool(general_gates_valid),
    }
    evaluator_path = Path(__file__).resolve()
    repository_state = campaign.runner.repo_state()
    clean_repository = repository_state.get("paper_dirty") is False
    if source_equivalence_path is not None:
        validated_bridge = validate_source_equivalence(
            manifest_path,
            source_equivalence_path,
            provenance_root_path,
        )
        source_provenance_valid = True
        bridge_hash: str | None = validated_bridge["content_sha256"]
    else:
        current_files, current_snapshot = campaign._source_snapshot()
        source_provenance_valid = (
            manifest.get("source_file_sha256") == current_files
            and manifest.get("source_snapshot_sha256") == current_snapshot
            and manifest.get("protocol_sha256")
            == campaign._sha256(campaign.PROTOCOL_PATH)
        )
        bridge_hash = None
    payload: dict[str, Any] = {
        "schema_version": 1,
        "launch_valid": bool(
            clean_repository
            and source_provenance_valid
            and trusted_root is not None
            and general_gates_valid
        ),
        "repository_state": repository_state,
        "source_provenance": {
            "validated": bool(source_provenance_valid),
            "method": "source_equivalence_bridge"
            if bridge_hash
            else "direct_manifest_snapshot",
            "bridge_file_sha256": (
                _sha256(Path(source_equivalence_path))
                if source_equivalence_path is not None
                else None
            ),
            "bridge_content_sha256": bridge_hash,
        },
        "gate_verification": gate_verification,
        "row_verification": row_verification,
        "provenance_root": (
            {
                "path": Path(provenance_root_path).name,
                "file_sha256": trusted_root["file_sha256"],
                "content_sha256": trusted_root["content_sha256"],
            }
            if trusted_root is not None
            else None
        ),
        "evaluator": {"path": evaluator_path.name, "sha256": _sha256(evaluator_path)},
        "audit_manifest": {
            "path": manifest_path.name,
            "sha256": _sha256(manifest_path),
        },
        "frozen_gates": {"path": gates_path.name, "sha256": gates_hash},
        "long_form_rows_csv": {"path": rows_csv.name, "sha256": _sha256(rows_csv)},
        "gate_result_json": {
            "path": gate_result_path.name,
            "sha256": _sha256(gate_result_path),
        },
        "job_counts": {"expected": expected_count, "observed": observed_count},
        "audit_seeds": {"expected": expected_seeds, "observed": observed_seeds},
        "job_records": {
            entry["job"].job_id: {
                "sha256": _sha256(records_dir / f"{entry['job'].job_id}.json"),
                "checkpoint_sha256": _sha256(entry["checkpoint"]),
            }
            for entry in sorted(entries, key=lambda item: item["job"].job_id)
        },
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    payload["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    _atomic_json(Path(output_path), payload)
    return payload


def target_diagnostics(conditions: np.ndarray) -> dict[str, Any]:
    rows = []
    for c in np.asarray(conditions, dtype=float):
        row = {"condition": float(c), **protocol.target_diagnostics(float(c))}
        rows.append(row)

    def moments(condition: float) -> dict[str, float]:
        return protocol.target_diagnostics(condition)

    def ratio(low: float, high: float) -> float:
        at_low, at_high = moments(low), moments(high)
        mean_translation = abs(at_high["actual_mean"] - at_low["actual_mean"])
        std_contraction = at_low["actual_std"] - at_high["actual_std"]
        return (
            float(mean_translation / std_contraction)
            if std_contraction
            else float("inf")
        )

    return {
        "rows": rows,
        "translation_to_std_contraction_ratio": {
            "near_extrapolation_0.8_to_1.0": ratio(0.8, 1.0),
            "far_extrapolation_0.5_to_1.0": ratio(0.5, 1.0),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (
        "stage1",
        "select-stage2",
        "extract",
        "aggregate",
        "evaluate-gates",
        "attest-audit",
        "attest-source-equivalence",
        "target-diagnostics",
    ):
        p = sub.add_parser(command)
        p.add_argument(
            "--manifest",
            type=Path,
            required=command in {"attest-audit", "attest-source-equivalence"},
        )
        p.add_argument("--records-dir", type=Path, required=command == "attest-audit")
        p.add_argument("--output", type=Path, required=True)
        if command == "evaluate-gates":
            p.add_argument("--gates", type=Path, required=True)
        if command == "attest-audit":
            p.add_argument("--gates", type=Path, required=True)
            p.add_argument("--rows-csv", type=Path, required=True)
            p.add_argument("--gate-result", type=Path, required=True)
            p.add_argument("--source-equivalence", type=Path)
            p.add_argument("--provenance-root", type=Path)
        if command == "attest-source-equivalence":
            p.add_argument("--provenance-root", type=Path, required=True)
        if command in {"extract", "aggregate", "evaluate-gates"}:
            p.add_argument("--source-equivalence", type=Path)
            p.add_argument("--provenance-root", type=Path)
        if command == "target-diagnostics":
            p.add_argument("--split", choices=tuple(protocol.EVAL_SEEDS), default="dev")
    args = parser.parse_args(argv)
    if args.command == "target-diagnostics":
        _atomic_json(
            args.output, target_diagnostics(protocol.evaluation_grid(args.split))
        )
        return
    if args.command == "attest-audit":
        attest_audit(
            args.manifest,
            args.records_dir,
            args.gates,
            args.rows_csv,
            args.gate_result,
            args.output,
            args.source_equivalence,
            args.provenance_root,
        )
        return
    if args.command == "attest-source-equivalence":
        create_source_equivalence(args.manifest, args.output, args.provenance_root)
        return
    if args.command == "select-stage2":
        _atomic_json(
            args.output, select_stage2(args.manifest, args.output, args.records_dir)
        )
        return
    _, rows = _dev_rows(
        args.manifest,
        args.records_dir,
        getattr(args, "source_equivalence", None),
        getattr(args, "provenance_root", None),
    )
    if args.command == "stage1":
        result = stage1_summary(args.manifest, args.records_dir)
    elif args.command == "extract":
        _write_csv(args.output, rows)
        return
    elif args.command == "aggregate":
        result = aggregate_rows(rows)
    else:
        result = evaluate_gates(rows, args.gates)
    _atomic_json(args.output, result)


if __name__ == "__main__":
    main()

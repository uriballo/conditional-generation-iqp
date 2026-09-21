"""Frozen, content-addressed campaigns for the scalar Gaussian benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiments.common import runner, tracking
from experiments.common.config import RunConfig

from . import protocol
from .study import build_study

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")
DEFAULT_OUTPUT = runner.RESULTS / "scalar_gaussian_translation"
DEFAULT_MANIFEST = DEFAULT_OUTPUT / "manifest.json"
STAGE1_ENCODERS = ("broadcast_affine", "qubit_affine")
MAIN_ENCODERS = ("broadcast_affine", "qubit_affine", "anti_block")
ORIGINAL_PROTOCOLS = ("interpolation", "near", "far")
MIRROR_PROTOCOLS = ("mirror_near", "mirror_far")
LAYOUT_SEEDS = protocol.LAYOUT_SEEDS


def _protocol_data() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_snapshot() -> tuple[dict[str, str], str]:
    library_sources = sorted((ROOT / "iqp-cond/src/iqp_cond").rglob("*.py"))
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        *(
            ROOT / "experiments/common" / f"{name}.py"
            for name in (
                "config",
                "datasets",
                "encoders",
                "metrics",
                "runner",
                "tracking",
            )
        ),
        *(
            Path(__file__).parent / f"{name}.py"
            for name in ("__init__", "protocol", "study", "campaign")
        ),
        PROTOCOL_PATH,
        *library_sources,
    ]
    hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in paths}
    library_revision = runner.repo_state().get("library_revision")
    digest = hashlib.sha256(
        _canonical({"files": hashes, "library_revision": library_revision})
    ).hexdigest()
    return hashes, digest


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()


@dataclass(frozen=True)
class CampaignJob:
    """One independently resumable fit with all scientific choices addressed."""

    phase: str
    split: str
    encoder: str
    protocol: str
    anchors: tuple[float, ...]
    anchor_count: int
    placement: str
    layout_seed: int | None
    optimizer_seed: int
    budget_mode: str
    hyperparameter_id: str
    hyperparameters: dict[str, Any]
    model_kind: str = "conditional"
    fit_condition: tuple[float, ...] | None = None
    job_id: str = ""

    def __post_init__(self) -> None:
        if self.anchor_count != len(self.anchors):
            raise ValueError("anchor_count does not match anchors")
        if any(len(a) != 1 for a in (self.fit_condition,) if a is not None):
            raise ValueError("fit_condition must contain one scalar condition")
        identity = asdict(self)
        identity["job_id"] = ""
        digest = hashlib.sha256(_canonical(identity)).hexdigest()[:20]
        if self.job_id and self.job_id != digest:
            raise ValueError("job_id does not match job content")
        object.__setattr__(self, "job_id", digest)


def _seed_list(split: str) -> tuple[int, ...]:
    try:
        return tuple(protocol.EVAL_SEEDS[split])
    except KeyError as error:
        raise ValueError(f"unknown split {split!r}") from error


def _hyperparameter_sets() -> list[tuple[str, dict[str, Any]]]:
    data = _protocol_data()
    return [
        (f"stage2-{index:02d}", dict(candidate))
        for index, candidate in enumerate(data["stage2"]["candidates"], 1)
    ]


def _selected(path: Path) -> tuple[str, dict[str, Any]]:
    value = json.loads(path.read_text())
    if "hyperparameters" in value:
        settings = dict(value["hyperparameters"])
    else:
        settings = dict(value)
    identifier = str(value.get("hyperparameter_id", value.get("id", "selected")))
    settings.pop("hyperparameter_id", None)
    settings.pop("id", None)
    return identifier, settings


def _job(
    *,
    phase: str,
    split: str,
    encoder: str,
    name: str,
    count: int,
    placement: str,
    layout_seed: int | None,
    optimizer_seed: int,
    budget_mode: str,
    hp_id: str,
    hp: dict[str, Any],
    model_kind: str = "conditional",
    fit_condition: tuple[float, ...] | None = None,
) -> CampaignJob:
    anchors = tuple(
        float(c)
        for c in protocol.anchor_conditions(name, count, placement, layout_seed)
    )
    return CampaignJob(
        phase,
        split,
        encoder,
        name,
        anchors,
        count,
        placement,
        layout_seed,
        optimizer_seed,
        budget_mode,
        hp_id,
        dict(hp),
        model_kind,
        fit_condition,
    )


def build_jobs(
    phase: str,
    split: str = "dev",
    *,
    viable_encoders: tuple[str, ...] = STAGE1_ENCODERS,
    selected_config: Path | None = None,
) -> list[CampaignJob]:
    """Build a deterministic job set for a frozen phase and seed split."""
    if split not in protocol.EVAL_SEEDS:
        raise ValueError(f"unknown split {split!r}")
    if phase == "stage1" and split != "dev":
        raise ValueError("Stage 1 is restricted to development seeds")
    if phase == "stage2" and split != "dev":
        raise ValueError("Stage 2 is restricted to development seeds")
    if phase == "validation" and split != "validation":
        raise ValueError("validation jobs require validation seeds")
    if phase == "audit" and split != "audit":
        raise ValueError("audit jobs require locked audit seeds")
    if phase == "direct" and split not in {"validation", "audit"}:
        raise ValueError("direct baseline jobs require validation or audit seeds")
    data = _protocol_data()
    if phase == "stage1":
        hp_id, hp = "stage1-frozen", dict(data["stage1"])
        return [
            _job(
                phase=phase,
                split=split,
                encoder=encoder,
                name="interpolation",
                count=3,
                placement="even",
                layout_seed=None,
                optimizer_seed=seed,
                budget_mode="fixed_total",
                hp_id=hp_id,
                hp=hp,
            )
            for encoder in STAGE1_ENCODERS
            for seed in _seed_list(split)
        ]
    if phase == "stage2":
        invalid = set(viable_encoders) - set(STAGE1_ENCODERS)
        if invalid:
            raise ValueError(
                f"Stage 2 accepts viable RY encoders only: {sorted(invalid)}"
            )
        candidates = _hyperparameter_sets()
        return [
            _job(
                phase=phase,
                split=split,
                encoder=encoder,
                name="interpolation",
                count=3,
                placement="even",
                layout_seed=None,
                optimizer_seed=seed,
                budget_mode="fixed_total",
                hp_id=hp_id,
                hp=hp,
            )
            for encoder in viable_encoders
            for hp_id, hp in candidates
            for seed in _seed_list(split)
        ]
    if phase not in {"validation", "audit", "direct", "production"}:
        raise ValueError(f"unknown phase {phase!r}")
    if selected_config is None:
        raise ValueError(f"{phase} requires --selected-config")
    hp_id, hp = _selected(selected_config)
    seeds = _seed_list(split)
    if phase == "production":
        seeds = tuple(range(501, 511))
    jobs: list[CampaignJob] = []
    if phase in {"validation", "audit", "production"}:
        for encoder in MAIN_ENCODERS:
            for name in ORIGINAL_PROTOCOLS:
                for count in protocol.ANCHOR_COUNTS:
                    for placement, layout_seed in [
                        ("even", None),
                        *(("random", s) for s in LAYOUT_SEEDS),
                    ]:
                        for budget in ("fixed_total", "matched_exposure"):
                            for seed in seeds:
                                jobs.append(
                                    _job(
                                        phase=phase,
                                        split=split,
                                        encoder=encoder,
                                        name=name,
                                        count=count,
                                        placement=placement,
                                        layout_seed=layout_seed,
                                        optimizer_seed=seed,
                                        budget_mode=budget,
                                        hp_id=hp_id,
                                        hp=hp,
                                    )
                                )
        for encoder in MAIN_ENCODERS:
            for name in MIRROR_PROTOCOLS:
                for count in (3, 5):
                    for seed in seeds:
                        jobs.append(
                            _job(
                                phase=phase,
                                split=split,
                                encoder=encoder,
                                name=name,
                                count=count,
                                placement="even",
                                layout_seed=None,
                                optimizer_seed=seed,
                                budget_mode="fixed_total",
                                hp_id=hp_id,
                                hp=hp,
                            )
                        )
        for name in ORIGINAL_PROTOCOLS:
            for count in protocol.ANCHOR_COUNTS:
                for seed in seeds:
                    jobs.append(
                        _job(
                            phase=phase,
                            split=split,
                            encoder="condition_blind",
                            name=name,
                            count=count,
                            placement="even",
                            layout_seed=None,
                            optimizer_seed=seed,
                            budget_mode="fixed_total",
                            hp_id=hp_id,
                            hp=hp,
                        )
                    )
        return jobs
    if phase == "direct":
        conditions = set(float(c) for c in protocol.evaluation_grid(split))
        for name in ORIGINAL_PROTOCOLS:
            for count in protocol.ANCHOR_COUNTS:
                for placement, layout_seed in [
                    ("even", None),
                    *(("random", s) for s in LAYOUT_SEEDS),
                ]:
                    conditions.update(
                        protocol.anchor_conditions(
                            name, count, placement, layout_seed
                        ).tolist()
                    )
        return [
            _job(
                phase=phase,
                split=split,
                encoder="condition_blind",
                name="interpolation",
                count=2,
                placement="even",
                layout_seed=None,
                optimizer_seed=seed,
                budget_mode="direct",
                hp_id=hp_id,
                hp=hp,
                model_kind="independent_direct",
                fit_condition=(condition,),
            )
            for condition in sorted(conditions)
            for seed in seeds
        ]
    raise AssertionError("unreachable phase")


def _run_config(job: CampaignJob) -> RunConfig:
    if job.phase == "stage1":
        hp = job.hyperparameters
        updates = int(hp["updates"])
        learning_rate = float(hp["learning_rate"])
        restarts = int(hp["restarts"])
        batch_size = int(hp["batch_size"])
        n_ops = int(hp["n_ops"])
        mc_samples = int(hp["mc_samples"])
        interval = int(hp["validation_interval"])
        bodyness = tuple(float(v) for v in hp["kernel_bodyness"])
    else:
        updates = int(job.hyperparameters["updates"])
        learning_rate = float(job.hyperparameters["learning_rate"])
        restarts = int(job.hyperparameters["restarts"])
        batch_size, n_ops, mc_samples, interval = 64, 64, 256, 10
        bodyness = tuple(float(v) for v in job.hyperparameters["kernel_bodyness"])
    if job.budget_mode == "matched_exposure":
        updates = protocol.matched_exposure_steps(updates, job.anchor_count)
    config = RunConfig(
        study="scalar_gaussian_translation",
        encoder=job.encoder,
        condition_protocol=protocol.training_set_label(
            job.protocol,
            job.anchor_count,
            job.placement,
            job.layout_seed,
        ),
        model_kind=job.model_kind,
        selection_mode="exact_training_tvd",
        budget_mode="direct"
        if job.budget_mode == "direct"
        else (
            "condition_matched"
            if job.budget_mode == "matched_exposure"
            else "fixed_total"
        ),
        fit_condition=job.fit_condition,
        n_visible=8,
        n_ancillas=0,
        width_factor=0.5,
        n_train=512,
        n_validation=512,
        n_test=512,
        steps=updates,
        batch_size=batch_size,
        n_ops=n_ops,
        bodyness_targets=bodyness,
        mc_samples=mc_samples,
        learning_rate=learning_rate,
        final_learning_rate=float(
            job.hyperparameters.get("final_learning_rate", learning_rate / 10)
        ),
        validation_interval=interval,
        gemm_dtype="float32",
        sample_chunk=128,
        validation_n_ops=64,
        validation_mc_samples=256,
        initialization="random",
        init_scale=0.5,
        restarts=restarts,
        seeds=(job.optimizer_seed,),
    )
    _validate_config(config)
    return config


def _validate_config(config: RunConfig) -> None:
    if (
        config.n_visible != protocol.N_VISIBLE
        or config.n_ancillas != protocol.N_ANCILLAS
    ):
        raise ValueError(
            "scalar Gaussian campaign requires n_visible=8 and n_ancillas=0"
        )


def _manifest(
    jobs: list[CampaignJob],
    phase: str,
    split: str,
    *,
    enabled: bool,
    audit_gate_path: Path | None = None,
    manifest_parent: Path | None = None,
) -> dict[str, Any]:
    gate_sha = None
    if split == "audit":
        if audit_gate_path is None or not audit_gate_path.is_file():
            raise FileNotFoundError("audit-split manifests require frozen_gates.json")
        gate_sha = _sha256(audit_gate_path)
    groups: dict[str, int] = {}
    for job in jobs:
        key = f"{job.encoder}:{job.budget_mode}"
        groups[key] = groups.get(key, 0) + 1
    source_hashes, source_sha = _source_snapshot()
    gates_path = None
    if audit_gate_path is not None:
        base = Path.cwd() if manifest_parent is None else manifest_parent
        gates_path = os.path.relpath(audit_gate_path.resolve(), base.resolve())
    return {
        "study": "scalar_gaussian_translation",
        "phase": phase,
        "split": split,
        "created_utc": datetime.now(UTC).isoformat(),
        "enabled": enabled,
        "production": phase == "production",
        "production_enabled": False,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "frozen_gates_sha256": gate_sha,
        "frozen_gates_path": gates_path,
        "source_file_sha256": source_hashes,
        "source_snapshot_sha256": source_sha,
        "library_revision": runner.repo_state().get("library_revision"),
        "state": runner.repo_state(),
        "job_count": len(jobs),
        "groups": groups,
        "encoder_only_baseline": "deterministic analysis; no training job",
        "classical_anchor_interpolation_baseline": (
            "deterministic analysis; no training job"
        ),
        "jobs": [asdict(job) for job in jobs],
    }


def write_manifest(
    path: Path,
    phase: str,
    split: str,
    *,
    viable_encoders: tuple[str, ...] = STAGE1_ENCODERS,
    selected_config: Path | None = None,
    audit_gate_path: Path | None = None,
) -> dict[str, Any]:
    enabled = phase != "production"
    jobs = build_jobs(
        phase, split, viable_encoders=viable_encoders, selected_config=selected_config
    )
    payload = _manifest(
        jobs,
        phase,
        split,
        enabled=enabled,
        audit_gate_path=audit_gate_path,
        manifest_parent=path.parent,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, payload)
    return payload


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[CampaignJob]]:
    payload = json.loads(path.read_text())
    jobs = [CampaignJob(**item) for item in payload["jobs"]]
    if payload.get("protocol_sha256") != _sha256(PROTOCOL_PATH):
        raise RuntimeError("protocol.json changed since manifest generation")
    actual_hashes, actual_snapshot = _source_snapshot()
    if (
        payload.get("source_file_sha256") != actual_hashes
        or payload.get("source_snapshot_sha256") != actual_snapshot
    ):
        raise RuntimeError("source snapshot changed since manifest generation")
    return payload, jobs


def _gate_path(payload: dict[str, Any], manifest_path: Path) -> Path:
    value = payload.get("frozen_gates_path")
    path = Path(value) if value else Path("frozen_gates.json")
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _execute(
    job: CampaignJob, output_root: Path | None, manifest_path: Path
) -> dict[str, Any]:
    config = _run_config(job)
    payload = json.loads(manifest_path.read_text())
    if job.split == "audit":
        gate = _gate_path(payload, manifest_path)
        if not gate.is_file() or _sha256(gate) != payload.get("frozen_gates_sha256"):
            raise RuntimeError("frozen audit gates are missing or changed")
    study = build_study(
        job.protocol,
        job.anchor_count,
        job.placement,
        job.layout_seed or 1101,
        job.encoder,
        job.split,
    )
    record = runner.run(config, study, job.optimizer_seed, output_root=output_root)
    if output_root is not None:
        index_root = output_root
    else:
        index_root = runner.RESULTS
    tracking.append_index(record, index_root)
    target = Path(record["directory"]).parent / "job_records" / f"{job.job_id}.json"
    audit_sha = payload.get("frozen_gates_sha256") if job.split == "audit" else None
    _atomic_json(
        target,
        {
            "job": asdict(job),
            "run_record": record,
            "manifest": str(manifest_path),
            "audit_gates_sha256": audit_sha,
        },
    )
    return record


def _assert_runnable(payload: dict[str, Any], manifest_path: Path) -> None:
    if payload.get("production") or payload.get("phase") == "production":
        raise RuntimeError("production manifests are proposals and cannot be run")
    if not payload.get("enabled"):
        raise RuntimeError("manifest is disabled")
    if payload.get("split") == "audit":
        gates = _gate_path(payload, manifest_path)
        if not gates.is_file() or _sha256(gates) != payload.get("frozen_gates_sha256"):
            raise RuntimeError("frozen audit gates are missing or changed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument(
        "phase",
        choices=("stage1", "stage2", "validation", "audit", "direct", "production"),
    )
    generate.add_argument("--split", default="dev", choices=tuple(protocol.EVAL_SEEDS))
    generate.add_argument("--viable-encoders", default=",".join(STAGE1_ENCODERS))
    generate.add_argument("--selected-config", type=Path)
    generate.add_argument("--frozen-gates", type=Path)
    listing = sub.add_parser("list-shard")
    listing.add_argument("shard_index", type=int)
    listing.add_argument("shard_count", type=int)
    one = sub.add_parser("run-job")
    one.add_argument("job_id")
    shard = sub.add_parser("run-shard")
    shard.add_argument("shard_index", type=int)
    shard.add_argument("shard_count", type=int)
    for child in (one, shard):
        child.add_argument("--workers", type=int, default=1)
        child.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        encoders = tuple(
            x.strip() for x in args.viable_encoders.split(",") if x.strip()
        )
        payload = write_manifest(
            args.manifest,
            args.phase,
            args.split,
            viable_encoders=encoders,
            selected_config=args.selected_config,
            audit_gate_path=args.frozen_gates,
        )
        print(
            f"wrote {payload['job_count']} jobs to {args.manifest} "
            f"(enabled={payload['enabled']})"
        )
        return
    payload, jobs = _load_manifest(args.manifest)
    if args.command == "list-shard":
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("invalid shard index/count")
        for index, job in enumerate(jobs):
            if index % args.shard_count == args.shard_index:
                print(job.job_id)
        return
    _assert_runnable(payload, args.manifest)
    if args.workers != 1:
        raise ValueError("run-shard worker parallelism is managed by launch_strix.sh")
    if args.command == "run-job":
        selected = next((job for job in jobs if job.job_id == args.job_id), None)
        if selected is None:
            raise ValueError(f"unknown job id {args.job_id!r}")
        _execute(selected, args.output_root, args.manifest)
    else:
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("invalid shard index/count")
        for index, job in enumerate(jobs):
            if index % args.shard_count == args.shard_index:
                _execute(job, args.output_root, args.manifest)


if __name__ == "__main__":
    main()

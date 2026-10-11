"""Execute frozen paper jobs with public library operations and local training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .config import RunConfig

SCALING_PROTOCOL = json.loads(
    (
        Path(__file__).parents[1]
        / "paper"
        / "data"
        / "scaling"
        / "scaling_protocol.json"
    ).read_text()
)

GAUSSIAN_SEEDS = tuple(range(17100, 17105))
SCALING_SIZES = (64, 128, 256, 512, 1024, 2048)
SCALING_SEEDS = (100, 101, 102)
LIBRARY_COMMIT = "f90875da72b3379c1ea38b844a0a908a20eacf24"
SELECTED = {
    "gaussian": dict(
        name="adam",
        learning_rate=0.005,
        final_learning_rate_ratio=0.1,
        init_scale=0.25,
        b1=0.9,
    ),
    "scaling": dict(
        name="adam",
        learning_rate=SCALING_PROTOCOL["candidate"]["learning_rate"],
        final_learning_rate_ratio=SCALING_PROTOCOL["candidate"]["final_ratio"],
        init_scale=SCALING_PROTOCOL["candidate"]["init_scale"],
        encoder_scale=SCALING_PROTOCOL["candidate"]["encoder_scale"],
        b1=SCALING_PROTOCOL["candidate"]["b1"],
        b2=SCALING_PROTOCOL["candidate"]["b2"],
    ),
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_job(
    experiment, seed, *, size=None, optimizer=None, stage="reported", beta=None
):
    if experiment not in SELECTED or stage != "reported":
        raise ValueError("unsupported experiment or stage")
    if beta is not None:
        raise ValueError("reported jobs do not accept beta")
    if experiment == "scaling" and size not in SCALING_SIZES:
        raise ValueError("unsupported scaling size")
    settings = dict(optimizer or SELECTED.get(experiment, {}))
    if "optimizer" in settings:
        settings["name"] = settings.pop("optimizer")
    job = dict(
        experiment=experiment,
        seed=int(seed),
        size=size,
        optimizer=settings,
        stage=stage,
        beta=beta,
    )
    job["config"] = config_for(job).as_dict()
    job["job_id"] = digest(job)[:16]
    return json.loads(json.dumps(job))


def config_for(job, *, smoke=False):
    kind = job["experiment"]
    seed = job["seed"]
    p = job["optimizer"]
    lr = p["learning_rate"]
    ratio = p.get("final_learning_rate_ratio", 1.0)
    base = dict(
        study=f"paper_{kind}_{job['stage']}_{digest(p)[:12]}",
        seeds=(seed,),
        encoder="parity" if kind == "scaling" else "phase_075",
        encoder_scale=float(p.get("encoder_scale", 0.75)),
        n_visible=16,
        n_ancillas=0,
        n_train=10000,
        n_validation=2000,
        n_test=2000,
        steps=2000,
        batch_size=450,
        n_ops=2048,
        mc_samples=2048,
        validation_interval=25,
        validation_n_ops=256,
        validation_mc_samples=16384,
        gemm_dtype="float32",
        sample_chunk=256,
        learning_rate=lr,
        final_learning_rate=lr * ratio,
        bodyness_targets=(2.0, 4.0, 6.0),
        mask_refresh="step",
        restarts=1 if job["stage"] == "screen" else 3,
        init_scale=p["init_scale"],
        selection_mode="exact_development_profile_tvd",
    )
    if kind == "gaussian":
        base.update(
            encoder="broadcast_ryrz",
            condition_protocol="frozen5",
            n_train=512,
            n_validation=1000,
            n_test=1000,
            steps=1024,
            batch_size=128,
            n_ops=1024,
            mc_samples=1024,
            validation_interval=32,
            validation_n_ops=1024,
            validation_mc_samples=1024,
            bodyness_targets=(1.0,),
            restarts=1,
            selection_mode="exact_development_profile_tvd",
        )
    elif kind == "scaling":
        base.update(
            encoder="parity",
            n_visible=job["size"],
            steps=5000,
            n_ops=512,
            mc_samples=1024,
            validation_n_ops=512,
            validation_mc_samples=1024,
            bodyness_targets=(2.0, 4.0),
            selection_mode="validation_mmd",
            validation_banks=4,
            validation_interval=100,
            validation_scales=tuple(SCALING_PROTOCOL["scales"][str(job["size"])]),
            shared_restart_streams=True,
        )
    config = RunConfig(**base)
    if smoke:
        config = replace(
            config,
            steps=4,
            restarts=1,
            n_train=32,
            n_validation=32,
            n_test=32,
            batch_size=8,
            n_ops=8,
            mc_samples=16,
            validation_n_ops=8,
            validation_mc_samples=16,
            validation_interval=2,
            sample_chunk=8,
        )
    return config


def _source_identity():
    import iqp_cond

    from . import engine

    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    library_root = Path(iqp_cond.__file__).parent
    library_sources = {
        str(p.relative_to(library_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(library_root.rglob("*.py"))
    }
    return dict(
        state=engine.repo_state(),
        source_sha256=sources,
        library_source_sha256=library_sources,
    )


def execute_job(job, output_root, *, smoke=False):
    from . import engine
    from .studies import build_study
    from .training import make_trainer

    canonical = build_job(
        job["experiment"],
        job["seed"],
        size=job.get("size"),
        optimizer=job.get("optimizer"),
        stage=job["stage"],
        beta=job.get("beta"),
    )
    if job != canonical:
        raise ValueError("job contents differ from its frozen identity")
    root = Path(output_root)
    receipt_root = root / "receipts"
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_root / (job["job_id"] + ".json")
    source = _source_identity()
    # Revision/dirty flags are informational; content hashes govern compatible resume.
    identity = dict(
        job=job,
        smoke=bool(smoke),
        source_sha256=source["source_sha256"],
        library_commit=LIBRARY_COMMIT,
    )
    if job["experiment"] == "scaling":
        identity["library_source_sha256"] = source["library_source_sha256"]
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old["identity"] != identity:
            raise ValueError("existing job source or configuration differs")
        if not (Path(old["record"]["directory"]) / "metrics.json").exists():
            raise ValueError("completed receipt has missing artifacts")
        return old["record"]
    config = config_for(job, smoke=smoke)
    study = build_study(job, config)
    settings = {
        k: v
        for k, v in job["optimizer"].items()
        if k not in {"init_scale", "encoder_scale"}
    }
    if job["experiment"] == "scaling":
        settings["compute_validation_mmd"] = False
    trainer = make_trainer(settings)
    if job["experiment"] == "scaling":
        from .scaling import _evaluate_mmd_trajectory

        def evaluate(model, snapshots, conditions, seed, best_step):
            return _evaluate_mmd_trajectory(
                model,
                snapshots,
                conditions,
                seed,
                best_step,
                config=config,
                replicate_banks=2 if smoke else 5,
                target_draws=32 if smoke else 2000,
                evaluation_interval=2 if smoke else 500,
            )

        record = engine.run_large(
            config,
            study,
            job["seed"],
            evaluate=evaluate,
            output_root=root,
            training_fn=trainer,
        )
    else:
        record = engine.run(
            config,
            study,
            job["seed"],
            output_root=root,
            training_fn=trainer,
            capture_checkpoints=True,
        )
    record["mean_selection_score"] = record["best_selection_score"]
    record.update(
        experiment=job["experiment"], job_id=job["job_id"], job=job, smoke=bool(smoke)
    )
    engine._write_json(
        receipt_path, dict(identity=identity, source=source, record=record)
    )
    return record

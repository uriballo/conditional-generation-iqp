"""Execute frozen paper jobs with public library operations and local training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import RunConfig

GAUSSIAN_SEEDS = tuple(range(17100, 17110))
SHARED_SEEDS = BASELINE_SEEDS = tuple(range(5400, 5405))
SCALING_SIZES = (64, 128, 256, 512, 1024, 2048)
SCALING_SEEDS = (100, 101, 102)
REPORT_BETAS = (0.4, 1.0, 1.6)
LIBRARY_COMMIT = "f90875da72b3379c1ea38b844a0a908a20eacf24"
SELECTED = {
    "shared": dict(
        name="adam",
        learning_rate=0.04014691865370211,
        final_learning_rate_ratio=0.08111685641025783,
        init_scale=0.43008014083790563,
        b1=0.967827235881252,
    ),
    "conditional": dict(
        name="adam",
        learning_rate=0.040042143801868595,
        final_learning_rate_ratio=0.01954909530299359,
        init_scale=0.3494510602968495,
        b1=0.8120567898119135,
    ),
    "unconditional": dict(
        name="adam",
        learning_rate=0.032402102895288054,
        final_learning_rate_ratio=0.011269232004298215,
        init_scale=0.07580759858905509,
        b1=0.9750900060714265,
    ),
    "gaussian": dict(
        name="adam",
        learning_rate=0.005,
        final_learning_rate_ratio=0.1,
        init_scale=0.05,
        b1=0.9,
    ),
    "scaling": dict(
        name="adam",
        learning_rate=0.01,
        final_learning_rate_ratio=0.1,
        init_scale=0.1,
        b1=0.9,
    ),
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_job(
    experiment, seed, *, size=None, optimizer=None, stage="reported", beta=None
):
    if experiment not in (*SELECTED, "symmetry") or stage not in (
        "reported",
        "screen",
        "confirm",
        "final",
    ):
        raise ValueError("unsupported experiment or stage")
    if experiment == "scaling" and size not in SCALING_SIZES:
        raise ValueError("unsupported scaling size")
    if experiment in ("conditional", "unconditional") and beta not in REPORT_BETAS:
        raise ValueError("baseline needs one reporting temperature")
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
    if experiment != "symmetry":
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
        encoder="phase_075",
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
            n_train=1000,
            n_validation=1000,
            n_test=1000,
            steps=300,
            batch_size=100,
            n_ops=1024,
            mc_samples=1024,
            validation_interval=30,
            validation_n_ops=1024,
            validation_mc_samples=1024,
            bodyness_targets=(1.0,),
            restarts=1,
        )
    elif kind == "scaling":
        base.update(
            n_visible=job["size"],
            steps=5000,
            n_ops=512,
            mc_samples=1024,
            validation_n_ops=512,
            validation_mc_samples=1024,
            bodyness_targets=(2.0, 4.0),
            selection_mode="validation_mmd",
        )
    elif kind in ("conditional", "unconditional"):
        base.update(
            fit_condition=(job["beta"],), selection_mode="exact_training_profile_tvd"
        )
        if kind == "unconditional":
            base.update(
                encoder="blind", model_kind="independent_direct", budget_mode="direct"
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
    from . import engine

    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    return dict(state=engine.repo_state(), source_sha256=sources)


def _symmetry(directory):
    # U=H D H evaluated directly on the two stated input states.
    phases = np.asarray([0.70, -1.25, 0.06, 1.34, -0.08, 0.77])
    codes = np.arange(8)
    bits = (codes[:, None] >> np.arange(3)) & 1
    generators = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    )
    h = (-1.0) ** (bits @ bits.T) / np.sqrt(8)
    unitary = h @ np.diag(np.exp(1j * ((-1.0) ** (bits @ generators.T)) @ phases)) @ h
    ordinary = np.zeros(8)
    ordinary[0] = 1
    encoded = ordinary.copy() / np.sqrt(2)
    encoded[3] = 1 / np.sqrt(2)
    record = dict(
        encoded=(np.abs(unitary @ encoded) ** 2).tolist(),
        ordinary=(np.abs(unitary @ ordinary) ** 2).tolist(),
        phase_vector=phases.tolist(),
        flip_mask_little_endian=3,
    )
    directory.mkdir(parents=True, exist_ok=True)
    return record


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
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old["identity"] != identity:
            raise ValueError("existing job source or configuration differs")
        if not (Path(old["record"]["directory"]) / "metrics.json").exists():
            raise ValueError("completed receipt has missing artifacts")
        return old["record"]
    if job["experiment"] == "symmetry":
        directory = root / "symmetry" / job["job_id"]
        record = _symmetry(directory)
        record["directory"] = str(directory)
        engine._write_json(directory / "metrics.json", record)
        engine._write_json(directory / "run.json", source)
    else:
        config = config_for(job, smoke=smoke)
        study = build_study(job, config)
        settings = {k: v for k, v in job["optimizer"].items() if k != "init_scale"}
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

"""Frozen sampled campaigns for binomial-centered Gaussian-shaped families."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import iqp_cond as ciqp
import jax
import jax.numpy as jnp
import numpy as np

from .ansatz import square_lattice_ansatz
from .gaussian2d_lattice import CollectivePhaseEncoder, metrics, sample_targets
from .training import make_trainer

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path(__file__).with_name("gaussian2d_lattice_protocol.json")
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
DESIGNS = ("grid", "axes")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_conditions(design):
    return {
        k: np.asarray(v, dtype=float) for k, v in PROTOCOL["protocols"][design].items()
    }


def initial_parameters(seed, candidate):
    values = np.random.default_rng(seed).normal(0, candidate["init_scale"], 40)
    values[:16] += candidate["local_mean"]
    return values


def source_files():
    names = (
        "__init__.py",
        "ansatz.py",
        "training.py",
        "gaussian2d_lattice.py",
        "gaussian2d_lattice_run.py",
        "gaussian2d_lattice_protocol.json",
    )
    files = {f"paper_experiments/{n}": ROOT / "paper_experiments" / n for n in names}
    library = Path(ciqp.__file__).parent
    files.update(
        {
            f"iqp-cond/src/iqp_cond/{p.relative_to(library)}": p
            for p in sorted(library.rglob("*.py"))
        }
    )
    return files


def identity():
    return {
        "source_sha256": {name: sha(path) for name, path in source_files().items()},
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("iqp-cond", "jax", "jaxlib", "numpy", "optax", "pennylane")
        },
        "python": sys.version,
        "platform": platform.platform(),
        "backend": jax.default_backend(),
    }


def _settings(design, stage, candidate, smoke=False):
    cfg = dict(PROTOCOL["budgets"])
    if design == "translation":
        cfg.update(PROTOCOL["translation"])
    else:
        cfg.update(candidate)
        cfg["steps"] = cfg[
            {
                "screen": "screen_steps",
                "confirm": "confirmation_steps",
                "final": "final_steps",
            }[stage]
        ]
        cfg["validation_interval"] = 64 if stage == "screen" else 100
    if smoke:
        cfg.update(
            steps=2,
            n_train=32,
            batch_size=8,
            n_ops=16,
            mc_samples=32,
            sample_chunk=16,
            validation_interval=1,
        )
    return cfg


def _compatible_fit(path, expected):
    receipt = path / "fit.json"
    if not receipt.exists():
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(
                f"unfinished fit directory cannot be overwritten: {path}"
            )
        return None
    old = json.loads(receipt.read_text())
    if old.get("status") != "complete" or old.get("request") != expected:
        raise RuntimeError(f"incompatible or failed fit: {path}")
    if not (path / "fit.npz").exists() or sha(path / "fit.npz") != old.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError(f"checkpoint integrity failure: {path}")
    return old


def run_fit(output, design, stage, candidate_id, seed, *, smoke=False, ident=None):
    ident = identity() if ident is None else ident
    candidate = (
        PROTOCOL["translation"]
        if design == "translation"
        else PROTOCOL["candidates"][candidate_id]
    )
    cfg = _settings(design, stage, candidate, smoke)
    expected = {
        "design": design,
        "stage": stage,
        "candidate": candidate_id,
        "seed": seed,
        "smoke": smoke,
        "config": cfg,
        "identity": ident,
    }
    output = Path(output)
    old = _compatible_fit(output, expected)
    if old is not None:
        return old
    output.mkdir(parents=True, exist_ok=True)
    write(output / "config.json", expected)
    pairs = load_conditions(design)
    pool = sample_targets(
        pairs["fitting"], cfg["n_train"], np.random.default_rng(seed + 101)
    )
    ansatz = square_lattice_ansatz(16)
    encoder = CollectivePhaseEncoder.from_conditions(pairs["fitting"])
    model = ciqp.ConditionalModel(ansatz, encoder)
    initial = initial_parameters(seed, candidate)
    sigmas = tuple(
        (-2 * np.log(1 - 2 * w / 16)) ** -0.5 for w in cfg["expected_weights"]
    )
    config = ciqp.TrainingConfig(
        steps=cfg["steps"],
        batch_size=cfg["batch_size"],
        n_ops=cfg["n_ops"],
        mc_samples=cfg["mc_samples"],
        sample_chunk=cfg["sample_chunk"],
        gemm_dtype=jnp.float32,
        validation_n_ops=1,
        validation_mc_samples=2,
        sigmas=sigmas,
        learning_rate=cfg["learning_rate"],
        final_learning_rate=cfg["learning_rate"] * 0.1,
        validation_interval=cfg["validation_interval"],
        mask_refresh="step",
    )
    settings = dict(
        PROTOCOL["optimizer"],
        learning_rate=cfg["learning_rate"],
        compute_validation_mmd=False,
        rng_seeds={
            k: seed + v for k, v in PROTOCOL["rng_offsets"].items() if k != "pool"
        },
    )
    trainer = make_trainer(settings)
    checkpoints = []

    def select(theta):
        return float(np.mean(metrics(theta, pairs["development"])["full_tvd"]))

    def checkpoint(step, theta):
        checkpoints.append((step, np.asarray(theta).copy()))

    started = time.perf_counter()
    print(f"START {design}/{stage}/{candidate_id}/{seed}", flush=True)
    try:
        result = trainer(
            model,
            pool,
            pool,
            seed=seed,
            config=config,
            initial_params=initial,
            checkpoint_callback=checkpoint,
            selection_metric=select,
        )
        saved = dict(
            initial_params=initial,
            best_params=result.best_params,
            final_params=result.final_params,
            checkpoint_steps=np.asarray([p[0] for p in checkpoints]),
            checkpoint_params=np.asarray([p[1] for p in checkpoints]),
            train_losses=result.train_losses,
            selection_scores=result.selection_scores,
            pool=pool,
            generators=np.asarray(ansatz.generators),
            **{f"{k}_conditions": v for k, v in pairs.items()},
        )
        record = dict(
            request=expected,
            status="complete",
            best_step=result.best_step,
            best_score=float(result.best_selection_score),
            seconds=time.perf_counter() - started,
            pool_sha256=hashlib.sha256(pool.tobytes()).hexdigest(),
            development_metrics=metrics(result.best_params, pairs["development"]),
        )
        if stage == "final" and not smoke:
            record["reporting_metrics"] = metrics(
                result.best_params, pairs["reporting"]
            )
        np.savez_compressed(output / "fit.npz", **saved)
        record["checkpoint_sha256"] = sha(output / "fit.npz")
        write(output / "fit.json", record)
        print(
            f"DONE {design}/{stage}/{candidate_id}/{seed}: "
            f"dev TVD={record['best_score']:.5f}, step={record['best_step']}, "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )
        return record
    except Exception as error:
        write(
            output / "fit.json",
            dict(
                request=expected,
                status="failed",
                error=repr(error),
                traceback=traceback.format_exc(),
            ),
        )
        raise


def run_campaign(output, *, fit_runner=run_fit, smoke=False):
    output = Path(output)
    ident = identity()
    receipt = output / "campaign_receipt.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["identity"] != ident or previous["smoke"] != smoke:
            raise RuntimeError("campaign source or configuration changed")
    state = dict(status="running", identity=ident, smoke=smoke, records=[], winners={})
    write(receipt, state)

    def fit(design, stage, candidate, seed):
        destination = output / design / stage / candidate / str(seed)
        row = fit_runner(
            destination, design, stage, candidate, seed, smoke=smoke, ident=ident
        )
        if row["status"] != "complete":
            raise RuntimeError("incomplete fit")
        state["records"].append(
            dict(
                design=design,
                stage=stage,
                candidate=candidate,
                seed=seed,
                receipt=str(destination / "fit.json"),
                checkpoint_sha256=row["checkpoint_sha256"],
                best_score=row["best_score"],
            )
        )
        write(receipt, state)
        return row

    try:
        if smoke:
            for design in (*DESIGNS, "translation"):
                fit(
                    design,
                    "final",
                    "m0_lr0.01" if design != "translation" else "fixed",
                    9,
                )
        else:
            shortlists = {}
            for design in DESIGNS:
                rows = [
                    (c, fit(design, "screen", c, PROTOCOL["seeds"]["screen"][0]))
                    for c in PROTOCOL["candidates"]
                ]
                shortlists[design] = [
                    c
                    for c, _ in sorted(rows, key=lambda x: (x[1]["best_score"], x[0]))[
                        :2
                    ]
                ]
            write(output / "shortlists.json", shortlists)
            for design in DESIGNS:
                scores = {
                    c: float(
                        np.mean(
                            [
                                fit(design, "confirm", c, s)["best_score"]
                                for s in PROTOCOL["seeds"]["confirm"]
                            ]
                        )
                    )
                    for c in shortlists[design]
                }
                state["winners"][design] = min(scores, key=lambda c: (scores[c], c))
                write(receipt, state)
            # Both geometries finish selection before any final reporting evaluation.
            write(
                output / "frozen_winners.json",
                dict(
                    winners=state["winners"],
                    selection="confirmation development FULL TVD only",
                    identity=ident,
                ),
            )
            for design in DESIGNS:
                for seed in PROTOCOL["seeds"]["final"]:
                    fit(design, "final", state["winners"][design], seed)
            for seed in PROTOCOL["translation"]["seeds"]:
                fit("translation", "final", "fixed", seed)
        state["status"] = "complete"
        write(receipt, state)
        return state
    except Exception as error:
        state.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
        write(receipt, state)
        raise


def freeze(output):
    output = Path(output)
    manifest = output / "source-manifest.json"
    active = {name: sha(path) for name, path in source_files().items()}
    snapshot = output / "source-workspace"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["sha256"] != active:
            raise RuntimeError("active sources differ from frozen campaign")
        for name, digest in saved["sha256"].items():
            if sha(snapshot / name) != digest:
                raise RuntimeError(f"frozen source changed: {name}")
        return snapshot
    if snapshot.exists():
        raise RuntimeError("unmanifested frozen source directory")
    for name, path in source_files().items():
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    origins = {}
    for label, path in (("paper", ROOT), ("library", Path(ciqp.__file__).parents[2])):
        origins[label] = {}
        for key, args in (
            ("revision", ["rev-parse", "HEAD"]),
            ("dirty", ["status", "--porcelain"]),
        ):
            result = subprocess.run(
                ["git", "-c", "core.fsmonitor=false", *args],
                cwd=path,
                capture_output=True,
                text=True,
                check=False,
            )
            origins[label][key] = result.stdout.strip()
    write(
        output / "source-manifest.json",
        dict(sha256=active, origins=origins, snapshot_root=str(snapshot)),
    )
    return snapshot


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "campaign"), required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.worker:
        run_campaign(output, smoke=args.stage == "smoke")
        return
    snapshot = freeze(output)
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(snapshot), str(snapshot / "iqp-cond/src"))),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "paper_experiments.gaussian2d_lattice_run",
            "--output",
            str(output),
            "--stage",
            args.stage,
            "--worker",
        ],
        cwd=snapshot,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()

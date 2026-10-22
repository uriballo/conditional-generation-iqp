"""Frozen campaign for the two-coordinate diagonal-conditioning baseline.

Mirrors :mod:`paper_experiments.gaussian2d_lattice_run` for the "axes" design:
same conditions, budgets, seeds, target pools and random streams, with a fixed
|0>^16 input and the affine phase map of :mod:`gaussian2d_diagonal`.
"""

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
from datetime import UTC, datetime
from pathlib import Path

import iqp_cond as ciqp
import jax
import jax.numpy as jnp
import numpy as np

from .gaussian2d_diagonal import (
    ansatz_of,
    initial_weights,
    make_trainer,
    metrics,
    zero_encoder,
)
from .gaussian2d_lattice import sample_targets

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_NAMES = {
    "affine": "gaussian2d_diagonal_protocol.json",
    "quadratic": "gaussian2d_diagonal_quadratic_protocol.json",
}
COMPLETE_PROTOCOL_NAMES = {
    "affine": "gaussian2d_diagonal_complete_protocol.json",
    "quadratic": "gaussian2d_diagonal_complete_quadratic_protocol.json",
}
ANSATZ_PROTOCOL_NAMES = {
    "lattice": PROTOCOL_NAMES,
    "complete": COMPLETE_PROTOCOL_NAMES,
}
PROTOCOL_PATH = Path(__file__).with_name(PROTOCOL_NAMES["affine"])
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
PARENT_PATH = ROOT / PROTOCOL["parent_protocol"]["path"]
DESIGN = "axes"
SMOKE_SETTINGS = dict(
    steps=2,
    n_train=32,
    batch_size=8,
    n_ops=16,
    mc_samples=32,
    sample_chunk=16,
    validation_interval=1,
)


def use_protocol(variant, ansatz="lattice"):
    """Select the basis (affine default) and diagonal layer (lattice default)."""
    global PROTOCOL, PROTOCOL_PATH, PARENT_PATH
    PROTOCOL_PATH = Path(__file__).with_name(ANSATZ_PROTOCOL_NAMES[ansatz][variant])
    PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
    PARENT_PATH = ROOT / PROTOCOL["parent_protocol"]["path"]
    if basis_name() != variant:
        raise RuntimeError("protocol basis differs from requested variant")
    if ansatz_name() != ansatz:
        raise RuntimeError("protocol ansatz differs from requested variant")
    return PROTOCOL


def ansatz_name():
    """Protocols without an ansatz field use the open 4x4 lattice."""
    return PROTOCOL.get("ansatz", "lattice")


def model_name():
    if ansatz_name() == "lattice":
        return f"diagonal_{basis_name()}_zero_input"
    return f"diagonal_{ansatz_name()}_{basis_name()}_zero_input"


def basis_name():
    """Protocols without a basis field are the original affine campaign."""
    return PROTOCOL.get("basis", "affine")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_parent():
    """The inherited fields must equal the hashed parent protocol."""
    if sha(PARENT_PATH) != PROTOCOL["parent_protocol"]["sha256"]:
        raise RuntimeError("parent protocol changed")
    parent = json.loads(PARENT_PATH.read_text())
    for key in ("target_formula", "selection", "budgets", "seeds", "optimizer"):
        if PROTOCOL[key] != parent[key]:
            raise RuntimeError(f"protocol field differs from parent: {key}")
    if PROTOCOL["rng_offsets"] != parent["rng_offsets"]:
        raise RuntimeError("protocol field differs from parent: rng_offsets")
    if PROTOCOL["protocols"][DESIGN] != parent["protocols"][DESIGN]:
        raise RuntimeError("axes conditions differ from parent")
    return parent


def load_conditions():
    return {
        k: np.asarray(v, dtype=float) for k, v in PROTOCOL["protocols"][DESIGN].items()
    }


def source_files():
    names = (
        "__init__.py",
        "ansatz.py",
        "gaussian2d_lattice.py",
        "gaussian2d_lattice_protocol.json",
        "gaussian2d_diagonal.py",
        "gaussian2d_diagonal_run.py",
        "gaussian2d_diagonal_protocol.json",
        "gaussian2d_diagonal_quadratic_protocol.json",
        "gaussian2d_diagonal_complete_protocol.json",
        "gaussian2d_diagonal_complete_quadratic_protocol.json",
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


def settings(stage, candidate, smoke=False):
    cfg = dict(PROTOCOL["budgets"])
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
        cfg.update(SMOKE_SETTINGS)
    return cfg


def target_pool(seed, n_train):
    """Identical to the parent runner's pool for the same seed."""
    pairs = load_conditions()
    return sample_targets(
        pairs["fitting"],
        n_train,
        np.random.default_rng(seed + PROTOCOL["rng_offsets"]["pool"]),
    )


def rng_seeds(seed):
    return {k: seed + v for k, v in PROTOCOL["rng_offsets"].items() if k != "pool"}


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


def run_fit(output, stage, candidate_id, seed, *, smoke=False, ident=None):
    ident = identity() if ident is None else ident
    candidate = PROTOCOL["candidates"][candidate_id]
    cfg = settings(stage, candidate, smoke)
    expected = {
        "design": DESIGN,
        "model": model_name(),
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
    pairs = load_conditions()
    pool = target_pool(seed, cfg["n_train"])
    ansatz = ansatz_of(ansatz_name())
    encoder = zero_encoder(len(pairs["fitting"]))
    initial = initial_weights(seed, candidate, basis_name(), ansatz_name())
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
    trainer = make_trainer(
        dict(
            PROTOCOL["optimizer"],
            learning_rate=cfg["learning_rate"],
            rng_seeds=rng_seeds(seed),
        )
    )
    checkpoints = []

    def select(weights):
        return float(np.mean(metrics(weights, pairs["development"])["full_tvd"]))

    def checkpoint(step, weights):
        checkpoints.append((step, np.asarray(weights).copy()))

    started = time.perf_counter()
    print(f"START {DESIGN}/{stage}/{candidate_id}/{seed}", flush=True)
    try:
        result = trainer(
            ansatz,
            encoder,
            pairs["fitting"],
            pool,
            seed=seed,
            config=config,
            initial_params=initial,
            selection_metric=select,
            checkpoint_callback=checkpoint,
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
        if stage == "final":
            record["reporting_metrics"] = metrics(
                result.best_params, pairs["reporting"]
            )
        np.savez_compressed(output / "fit.npz", **saved)
        record["checkpoint_sha256"] = sha(output / "fit.npz")
        write(output / "fit.json", record)
        print(
            f"DONE {DESIGN}/{stage}/{candidate_id}/{seed}: "
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


def run_campaign(output, *, fit_runner=run_fit, smoke=False, stop_after=None):
    """Screen -> confirm -> frozen winner -> final, as in the parent protocol.

    ``stop_after`` may be "screen" or "confirm" to pause the campaign; a later
    call with the same sources resumes from the completed fit receipts.
    """
    output = Path(output)
    check_parent()
    ident = identity()
    receipt = output / "campaign_receipt.json"
    started_at = datetime.now(UTC).isoformat()
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous["identity"] != ident or previous["smoke"] != smoke:
            raise RuntimeError("campaign source or configuration changed")
        started_at = previous.get("started_at", started_at)
    state = dict(
        status="running",
        basis=basis_name(),
        **({} if ansatz_name() == "lattice" else {"ansatz": ansatz_name()}),
        protocol=PROTOCOL_PATH.name,
        protocol_sha256=sha(PROTOCOL_PATH),
        parent_protocol_sha256=PROTOCOL["parent_protocol"]["sha256"],
        identity=ident,
        smoke=smoke,
        started_at=started_at,
        records=[],
        shortlist=[],
        winner=None,
    )
    write(receipt, state)
    clock = time.perf_counter()

    def fit(stage, candidate, seed):
        destination = output / DESIGN / stage / candidate / str(seed)
        row = fit_runner(destination, stage, candidate, seed, smoke=smoke, ident=ident)
        if row["status"] != "complete":
            raise RuntimeError("incomplete fit")
        state["records"].append(
            dict(
                design=DESIGN,
                stage=stage,
                candidate=candidate,
                seed=seed,
                receipt=str(destination / "fit.json"),
                checkpoint_sha256=row["checkpoint_sha256"],
                best_score=row["best_score"],
                seconds=row.get("seconds"),
            )
        )
        write(receipt, state)
        return row

    def pause(stage):
        if stop_after == stage:
            state.update(
                status=f"paused_after_{stage}",
                session_seconds=time.perf_counter() - clock,
            )
            write(receipt, state)
            return True
        return False

    try:
        rows = [
            (c, fit("screen", c, PROTOCOL["seeds"]["screen"][0]))
            for c in PROTOCOL["candidates"]
        ]
        state["shortlist"] = [
            c for c, _ in sorted(rows, key=lambda x: (x[1]["best_score"], x[0]))[:2]
        ]
        write(output / "shortlists.json", {DESIGN: state["shortlist"]})
        if pause("screen"):
            return state
        scores = {
            c: float(
                np.mean(
                    [
                        fit("confirm", c, s)["best_score"]
                        for s in PROTOCOL["seeds"]["confirm"]
                    ]
                )
            )
            for c in state["shortlist"]
        }
        state["confirmation_scores"] = scores
        state["winner"] = min(scores, key=lambda c: (scores[c], c))
        write(receipt, state)
        write(
            output / "frozen_winners.json",
            dict(
                winners={DESIGN: state["winner"]},
                selection="confirmation development FULL TVD only",
                identity=ident,
            ),
        )
        if pause("confirm"):
            return state
        for seed in PROTOCOL["seeds"]["final"]:
            fit("final", state["winner"], seed)
        state.update(
            status="complete",
            finished_at=datetime.now(UTC).isoformat(),
            session_seconds=time.perf_counter() - clock,
            fit_seconds_total=float(sum(r["seconds"] or 0.0 for r in state["records"])),
        )
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
        manifest,
        dict(
            sha256=active,
            origins=origins,
            snapshot_root=str(snapshot),
        ),
    )
    return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "campaign"), required=True)
    parser.add_argument("--stop-after", choices=("screen", "confirm"))
    parser.add_argument("--basis", choices=tuple(PROTOCOL_NAMES), default="affine")
    parser.add_argument(
        "--ansatz", choices=tuple(ANSATZ_PROTOCOL_NAMES), default="lattice"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    use_protocol(args.basis, args.ansatz)
    if args.worker:
        run_campaign(output, smoke=args.stage == "smoke", stop_after=args.stop_after)
        return
    snapshot = freeze(output)
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(snapshot), str(snapshot / "iqp-cond/src"))),
    )
    command = [
        sys.executable,
        "-m",
        "paper_experiments.gaussian2d_diagonal_run",
        "--output",
        str(output),
        "--stage",
        args.stage,
        "--worker",
        "--basis",
        args.basis,
        "--ansatz",
        args.ansatz,
    ]
    if args.stop_after:
        command += ["--stop-after", args.stop_after]
    subprocess.run(command, cwd=snapshot, env=env, check=True)


if __name__ == "__main__":
    main()

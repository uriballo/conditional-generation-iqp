"""Local, resumable reproduction of the manuscript's staged Optuna searches."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .runtime import SELECTED, build_job, digest, execute_job

SCREEN = (5200, 5201)
CONFIRM = (5300, 5301, 5302)
FINAL = (5400, 5401, 5402, 5403, 5404)
BETAS = (0.4, 1.0, 1.6)
STUDIES = (
    "shared_adam",
    "shared_schedule_free",
    "independent_conditional",
    "independent_unconditional",
)


def protocol(smoke=False):
    return dict(
        trials=2 if smoke else 48,
        screen_seeds=list(SCREEN[:1] if smoke else SCREEN),
        confirmation_seeds=list(CONFIRM[:1] if smoke else CONFIRM),
        final_seeds=list(FINAL[:1] if smoke else FINAL),
        batch_size=2,
        n_startup_trials=12,
        pruning=False,
        confirmation_candidates=2,
        smoke=bool(smoke),
        ranges=dict(
            adam_learning_rate=[0.0003, 0.05],
            schedule_free_learning_rate=[0.0003, 0.1],
            init_scale=[0.01, 0.7],
            b1=[0.8, 0.98],
            final_learning_rate_ratio=[0.01, 0.3],
            warmup_steps=[0, 50, 100, 200, 400],
        ),
        source_sha256=_source_hashes(),
    )


def _source_hashes():
    import hashlib

    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _reference(name="adam"):
    p = dict(name=name, learning_rate=0.01, init_scale=0.5, b1=0.9)
    p.update(final_learning_rate_ratio=0.1) if name == "adam" else p.update(
        warmup_steps=100
    )
    return p


def _candidate(trial, name):
    p = dict(
        name=name,
        learning_rate=trial.suggest_float(
            "learning_rate", 0.0003, 0.05 if name == "adam" else 0.1, log=True
        ),
        init_scale=trial.suggest_float("init_scale", 0.01, 0.7, log=True),
        b1=trial.suggest_float("b1", 0.8, 0.98),
    )
    if name == "adam":
        p["final_learning_rate_ratio"] = trial.suggest_float(
            "final_learning_rate_ratio", 0.01, 0.3, log=True
        )
    else:
        p["warmup_steps"] = trial.suggest_categorical(
            "warmup_steps", [0, 50, 100, 200, 400]
        )
    return p


def _jobs(family, candidate, stage, seeds):
    kind = (
        "shared"
        if family.startswith("shared")
        else "conditional"
        if family == "independent_conditional"
        else "unconditional"
    )
    return [
        build_job(kind, seed, optimizer=candidate, stage=stage, beta=beta)
        for seed in seeds
        for beta in ((None,) if kind == "shared" else BETAS)
    ]


def _worker(arguments):
    job, root, smoke = arguments
    return execute_job(job, root, smoke=smoke)


def _score(records, family):
    field = (
        "development_profile_tvd_mean"
        if family.startswith("shared")
        else "train_profile_tvd_mean"
    )
    values = [float(r[field]) for r in records]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite HPO selection metric")
    return float(np.mean(values))


def _summarize(records):
    result = dict(records=records)
    for key in ("held_out_profile_tvd_per_condition", "held_out_tvd_per_condition"):
        if all(key in r for r in records):
            values = np.asarray([r[key] for r in records])
            name = "profile" if "profile" in key else "full"
            result[name + "_per_case"] = values.tolist()
            # Baseline records are seed-major, with one temperature per record.
            if records[0]["job"]["experiment"] in ("conditional", "unconditional"):
                values = values.reshape(-1, len(BETAS))
            result[name + "_mean"] = values.mean(0).tolist()
            result[name + "_sd"] = (
                values.std(0, ddof=1).tolist() if len(values) > 1 else None
            )
    return result


def run_all(root, *, smoke=False, workers=1, executor=None):
    """Run four studies, confirmation and fresh-seed evaluation.

    ``executor`` is an optional test injection; production always fits the
    sampled model through ``execute_job``. A checkpointed sampler is a trusted
    local file generated here, not a downloaded pickle.
    """
    import optuna

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if workers < 1:
        raise ValueError("workers must be positive")
    spec = protocol(smoke)
    manifest = root / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError("existing HPO protocol or source differs")
    _write(manifest, spec)
    lock = root / "coordinator.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            "HPO coordinator lock exists; inspect before resuming"
        ) from error
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    pool = None
    try:
        if workers > 1 and executor is None:
            pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
        cache = root / "fit-cache"
        cache.mkdir(exist_ok=True)

        def fit(jobs):
            records = {}
            pending = []
            for j in jobs:
                path = cache / (j["job_id"] + ".json")
                if path.exists():
                    record = json.loads(path.read_text())
                    if record["job"] != j:
                        raise ValueError("cached HPO fit differs")
                    if executor is None and not Path(record["directory"]).is_dir():
                        raise ValueError("cached HPO fit has missing artifacts")
                    records[j["job_id"]] = record
                else:
                    pending.append(j)
            if pool:
                fresh = list(
                    pool.map(_worker, [(j, root / "fits", smoke) for j in pending])
                )
            else:
                fresh = [
                    (executor or execute_job)(j, root / "fits", smoke=smoke)
                    for j in pending
                ]
            for j, r in zip(pending, fresh, strict=True):
                r = dict(r, job=j)
                _write(cache / (j["job_id"] + ".json"), r)
                records[j["job_id"]] = r
            return [records[j["job_id"]] for j in jobs]

        confirmations = []
        for index, family in enumerate(STUDIES):
            name = "schedule_free" if family.endswith("schedule_free") else "adam"
            path = root / (family + "-sampler.pkl")
            sampler = (
                pickle.loads(path.read_bytes())
                if path.exists()
                else optuna.samplers.TPESampler(
                    seed=(20261004 + index if index < 2 else 20261014 + index - 2),
                    n_startup_trials=12,
                )
            )
            study = optuna.create_study(
                study_name=family,
                storage="sqlite:///" + str(root / "optuna.sqlite3"),
                direction="minimize",
                sampler=sampler,
                pruner=optuna.pruners.NopPruner(),
                load_if_exists=True,
            )
            if not study.trials:
                study.enqueue_trial(
                    {k: v for k, v in _reference(name).items() if k != "name"}
                )
                if index >= 2:
                    study.enqueue_trial(
                        {k: v for k, v in SELECTED["shared"].items() if k != "name"}
                    )
            batch_path = root / (family + "-pending.json")
            while (
                sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
                < spec["trials"]
            ):
                if batch_path.exists():
                    batch = json.loads(batch_path.read_text())
                else:
                    completed = sum(
                        t.state == optuna.trial.TrialState.COMPLETE
                        for t in study.trials
                    )
                    if any(
                        t.state == optuna.trial.TrialState.RUNNING for t in study.trials
                    ):
                        raise RuntimeError(
                            "running trial without pending batch; cannot safely resume"
                        )
                    batch = []
                    for _ in range(min(2, spec["trials"] - completed)):
                        trial = study.ask()
                        p = _candidate(trial, name)
                        trial.set_user_attr("candidate", p)
                        batch.append(dict(number=trial.number, candidate=p))
                    _write(batch_path, batch)
                    path.write_bytes(pickle.dumps(sampler))
                groups = [
                    _jobs(family, row["candidate"], "screen", spec["screen_seeds"])
                    for row in batch
                ]
                all_records = fit([j for group in groups for j in group])
                offset = 0
                for row, group in zip(batch, groups, strict=True):
                    records = all_records[offset : offset + len(group)]
                    offset += len(group)
                    if (
                        study.trials[row["number"]].state
                        != optuna.trial.TrialState.COMPLETE
                    ):
                        study.tell(row["number"], _score(records, family))
                path.write_bytes(pickle.dumps(sampler))
                batch_path.unlink()
            ranked = sorted(
                [
                    t
                    for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ],
                key=lambda t: (t.value, t.number),
            )
            candidates = []
            seen = set()
            for trial in ranked:
                p = trial.user_attrs["candidate"]
                identity = digest(p)
                if identity not in seen:
                    candidates.append(p)
                    seen.add(identity)
                if len(candidates) == 2:
                    break
            if len(candidates) != 2:
                raise ValueError("need two distinct configurations to confirm")
            rows = []
            for p in candidates:
                records = fit(_jobs(family, p, "confirm", spec["confirmation_seeds"]))
                rows.append(
                    dict(
                        family=family,
                        candidate=p,
                        score=_score(records, family),
                        records=records,
                    )
                )
            confirmations.extend(rows)
        _write(root / "confirmation.json", confirmations)
        shared = min(
            (r for r in confirmations if r["family"].startswith("shared")),
            key=lambda r: r["score"],
        )
        winners = [shared] + [
            min(
                (r for r in confirmations if r["family"] == family),
                key=lambda r: r["score"],
            )
            for family in STUDIES[2:]
        ]
        _write(
            root / "winner.json",
            [{k: r[k] for k in ("family", "candidate", "score")} for r in winners],
        )
        final = []
        for r in winners:
            records = fit(
                _jobs(r["family"], r["candidate"], "final", spec["final_seeds"])
            )
            final.append(
                dict(
                    family=r["family"], candidate=r["candidate"], **_summarize(records)
                )
            )
        records = fit(_jobs("shared_adam", _reference(), "final", spec["final_seeds"]))
        final.append(
            dict(
                family="shared_original_adam",
                candidate=_reference(),
                **_summarize(records),
            )
        )
        _write(root / "final-results.json", final)
        return dict(
            winners=[
                {k: r[k] for k in ("family", "candidate", "score")} for r in winners
            ],
            final=final,
        )
    finally:
        if pool:
            pool.shutdown()
        lock.unlink(missing_ok=True)

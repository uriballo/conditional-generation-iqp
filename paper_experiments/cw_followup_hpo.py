"""Random-proposal stage of the Curie--Weiss search, run jointly for four studies."""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import optuna

from .cw_followup_config import (
    FAMILIES,
    MARKERS,
    candidate,
    limits,
    protocol,
    stream_record,
    structured_seed,
)
from .cw_followup_fit import digest, run_fit, streams, write_json
from .cw_followup_metrics import curve_scores
from .provenance import source_identity as _source_identity


def source_identity():
    return _source_identity(
        Path(__file__).with_name("cw_followup_protocol.json"),
        protocol()["protocol_sha256"],
    )


def seed_manifest(smoke=False):
    spec = protocol(smoke)
    budget = limits(spec)
    records = {}

    def add(record):
        records[record["sha256"]] = record

    for stage in ("screen", "confirmation", "final"):
        for repeat in range(budget["repeats"][stage]):
            for beta in (*spec["training"]["shared_anchors"], *MARKERS):
                add(stream_record(stage, repeat, str(float(beta)), "target_pool"))
            for case in ("shared", *(str(float(b)) for b in MARKERS)):
                for start in range(budget["starts"][stage]):
                    for record in streams(stage, repeat, case, start).values():
                        add(record)
    for family in FAMILIES:
        for phase in ("random", "tpe"):
            add(stream_record("sampler", family, phase))
    occupied = {}
    for record in records.values():
        if record["seed"] in occupied:
            raise ValueError("unintended uint32 seed collision")
        occupied[record["seed"]] = record["sha256"]
    return dict(
        streams=list(records.values()),
        backend_seed_bits=32,
        intentional_reuse=(
            "same stage/repeat/case/start/purpose across candidates "
            "and matched families"
        ),
        purpose_mapping={
            "batches": "target minibatches",
            "bands": "kernel slots",
            "conditions": "training anchors",
            "masks": "operator masks",
            "reference": "encoder reference samples",
        },
    )


def jobs(context, proposal, stage="screen"):
    budget = limits(protocol(context["smoke"]))
    for repeat in range(budget["repeats"][stage]):
        for beta in (
            MARKERS if context["family"].startswith("independent_") else (None,)
        ):
            yield dict(
                family=context["family"],
                candidate=proposal["config"],
                trial_id=proposal["trial_id"],
                stage=stage,
                repeat=repeat,
                beta=beta,
                smoke=context["smoke"],
                identity=context["identity"],
            )


def score_records(records, family, expected_repeats):
    direct = family.startswith("independent_")
    expected_cases = set(MARKERS) if direct else {None}
    repeat_scores = []
    for repeat in range(expected_repeats):
        cases = [r for r in records if r["job"]["repeat"] == repeat]
        if (
            len(cases) != len(expected_cases)
            or {r["job"]["beta"] for r in cases} != expected_cases
        ):
            raise ValueError("missing or duplicate repeat cases")
        if any(r["status"] != "ok" for r in cases):
            return dict(
                J=1.0, P=None, valid=False, search_penalty_not_measured_tvd=True
            )
        cases.sort(key=lambda r: r["job"]["beta"] or 0)
        curves = {
            k: np.concatenate([np.asarray(r["metrics_curves"][k]) for r in cases])
            for k in ("beta", "tvd", "magnetization_error", "heat_error")
        }
        repeat_scores.append(
            curve_scores(
                curves["tvd"],
                curves["magnetization_error"],
                curves["heat_error"],
                curves["beta"],
            )
        )
    if len(records) != expected_repeats * len(expected_cases):
        raise ValueError("unexpected repeat records")
    return dict(
        J=float(np.mean([r["J"] for r in repeat_scores])),
        P=float(np.mean([r["P"] for r in repeat_scores])),
        valid=True,
        repeats=repeat_scores,
    )


def _fit_with_retry(job, root):
    path = root / "retry-state" / (digest(job) + ".json")
    receipt = root / "fits" / digest(job) / "metrics.json"
    if receipt.exists():
        return run_fit(job, root)
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {"attempts": 0, "interruptions": []}
    )
    while state["attempts"] < 2:
        state["attempts"] += 1
        write_json(path, state)
        try:
            return run_fit(job, root)
        except (OSError, RuntimeError) as error:
            state["interruptions"].append(
                dict(type=type(error).__name__, error=str(error))
            )
            write_json(path, state)
    raise RuntimeError("repeated infrastructure interruption requires diagnosis")


def _trial_worker(arguments):
    context, proposal, root = arguments
    root = Path(root)
    if source_identity() != context["identity"]:
        raise ValueError("worker environment differs from frozen campaign identity")
    path = root / "proposals" / f"{proposal['trial_id']:03d}.json"
    records = [_fit_with_retry(job, root) for job in jobs(context, proposal)]
    score = score_records(
        records,
        context["family"],
        limits(protocol(context["smoke"]))["repeats"]["screen"],
    )
    result = dict(**proposal, **score, fits=[r["directory"] for r in records])
    write_json(path, result)
    print(
        f"{context['family']} random {proposal['trial_id'] + 1}: "
        f"J={score['J']:.6f} P={score['P']}",
        flush=True,
    )
    return context["family"], result


def _open_study(root, family):
    root.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=family,
        storage="sqlite:///" + str((root / "trials.sqlite3").resolve()),
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(
            seed=structured_seed("sampler", family, "random")
        ),
        pruner=optuna.pruners.NopPruner(),
    )


def freeze_proposals(root, smoke=False):
    """Freeze all random draws before any fit, independent of completion order."""
    path = root / "random-proposals.json"
    if path.exists():
        return json.loads(path.read_text())
    if any((root / "studies" / f / "trials.sqlite3").exists() for f in FAMILIES):
        raise RuntimeError("partial proposal freeze requires diagnosis; no resampling")
    result = {}
    count = limits(protocol(smoke))["trials"]
    for family in FAMILIES:
        study = _open_study(root / "studies" / family, family)
        rows = []
        for _ in range(count):
            trial = study.ask()
            config = candidate(trial, family)
            rows.append(
                dict(
                    trial_id=trial.number,
                    config=json.loads(json.dumps(config)),
                    optuna_params=trial.params,
                )
            )
        result[family] = rows
    write_json(path, result)
    return result


def _run_random_locked(root, smoke=False, workers=1):
    spec = protocol(smoke)
    identity = source_identity()
    manifest = dict(
        protocol=spec,
        identity=identity,
        limits=limits(spec),
        phase="random",
        review_barrier="all four random stages; no B/F or TPE",
    )
    path = root / "campaign.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("campaign configuration/source/environment identity differs")
    write_json(path, manifest)
    write_json(root / "seed-manifest.json", seed_manifest(smoke))
    proposals = freeze_proposals(root, smoke)
    studies = {f: _open_study(root / "studies" / f, f) for f in FAMILIES}
    arguments = []
    for trial_id in range(limits(spec)["trials"]):
        for family in FAMILIES:
            proposal = proposals[family][trial_id]
            study = studies[family]
            trial = study.trials[trial_id]
            if trial.params != proposal["optuna_params"]:
                raise ValueError("frozen Optuna proposal differs")
            if trial.state == optuna.trial.TrialState.COMPLETE:
                receipt = (
                    root / "studies" / family / "proposals" / f"{trial_id:03d}.json"
                )
                if (
                    not receipt.exists()
                    or json.loads(receipt.read_text())["J"] != trial.value
                ):
                    raise ValueError("completed trial receipt differs")
                continue
            if trial.state != optuna.trial.TrialState.RUNNING:
                raise ValueError("unexpected frozen trial state")
            context = dict(family=family, smoke=smoke, identity=identity)
            arguments.append((context, proposal, root / "studies" / family))

    def record(result):
        family, row = result
        study = studies[family]
        trial = optuna.trial.Trial(study, study.trials[row["trial_id"]]._trial_id)
        trial.set_user_attr("phase", "random")
        trial.set_user_attr("P", row["P"])
        trial.set_user_attr("valid", row["valid"])
        trial.set_user_attr("config", row["config"])
        study.tell(trial, row["J"])
        write_json(
            root / "progress.json",
            dict(
                phase="random",
                counts={
                    f: sum(
                        t.state == optuna.trial.TrialState.COMPLETE for t in s.trials
                    )
                    for f, s in studies.items()
                },
                target_per_study=limits(spec)["trials"],
            ),
        )

    if workers == 1:
        for argument in arguments:
            record(_trial_worker(argument))
    elif arguments:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(_trial_worker, a) for a in arguments]
            for future in as_completed(futures):
                record(future.result())
    counts = {f: len(s.trials) for f, s in studies.items()}
    if any(
        n != limits(spec)["trials"]
        or any(t.state != optuna.trial.TrialState.COMPLETE for t in studies[f].trials)
        for f, n in counts.items()
    ):
        raise RuntimeError("four-study random barrier incomplete")
    status = dict(
        status="paused_for_random_review",
        counts=counts,
        tpe_started=False,
        confirmation_started=False,
        final_started=False,
        model_evaluation_scope="shared A and independent direct markers only",
    )
    write_json(root / "random-review-ready.json", status)
    return status


def run_all(root, *, smoke=False, workers=1):
    """Run the random phase and stop for review."""
    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / "campaign.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another campaign is running in this directory"
            ) from error
        return _run_random_locked(root, smoke=smoke, workers=workers)
    finally:
        os.close(descriptor)

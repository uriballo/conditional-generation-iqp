"""Explicit TPE continuation of the reviewed, frozen random CW studies.

The random campaign receipt stays immutable. Only this orchestration module and
its CLI dispatch may differ from that receipt; every fit source, dependency,
seed schedule, and numerical backend must match. Each study asks one proposal
at a time and tells its outcome before asking the next.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import multiprocessing
import os
import pickle
import sqlite3
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import optuna

from . import cw_followup_hpo as parent
from .cw_followup_config import candidate, limits, protocol, structured_seed
from .cw_followup_fit import _verify_artifacts, digest, write_json

_ALLOWED_ORCHESTRATION = {"run_experiments.py", "paper_experiments/cw_followup_tpe.py"}


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_transition(before, after):
    old = dict(before)
    new = dict(after)
    old_files = old.pop("paper_sources")
    new_files = new.pop("paper_sources")
    if old != new:
        raise ValueError("continuation numerical environment/library identity differs")
    changes = {}
    for name in old_files.keys() | new_files.keys():
        previous, current = old_files.get(name), new_files.get(name)
        if previous != current:
            if name not in _ALLOWED_ORCHESTRATION or current is None:
                raise ValueError(f"frozen fit/source identity differs: {name}")
            changes[name] = dict(before=previous, after=current)
    return changes


def _load_study(root, family, sampler=None):
    path = root / "studies" / family / "trials.sqlite3"
    if not path.is_file():
        raise ValueError(f"missing existing study: {family}")
    return optuna.load_study(
        study_name=family,
        storage="sqlite:///" + str(path.resolve()),
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )


def _validate_result(root, context, proposal, trial, row):
    """Verify stored scores against exact fit receipts and artifact hashes."""
    if (
        row.get("trial_id") != trial.number
        or row.get("config") != proposal["config"]
        or row.get("optuna_params") != proposal["optuna_params"]
        or trial.params != proposal["optuna_params"]
    ):
        raise ValueError("trial/proposal configuration differs")
    jobs = list(parent.jobs(context, proposal))
    directories = row.get("fits", [])
    if len(directories) != len(jobs):
        raise ValueError("trial has missing fit receipts")
    records = []
    for job, stored_directory in zip(jobs, directories, strict=True):
        directory = root / "studies" / context["family"] / "fits" / digest(job)
        if Path(stored_directory).name != directory.name:
            raise ValueError("fit receipt directory differs")
        record = json.loads((directory / "metrics.json").read_text())
        if record["job"] != job:
            raise ValueError("fit receipt identity differs")
        _verify_artifacts(directory, record["artifact_sha256"])
        records.append(record)
    score = parent.score_records(
        records,
        context["family"],
        limits(protocol(context["smoke"]))["repeats"]["screen"],
    )
    for key in ("J", "P", "valid"):
        if row.get(key) != score[key]:
            raise ValueError(f"stored trial score differs: {key}")
    if trial.state == optuna.trial.TrialState.COMPLETE:
        if trial.value != score["J"]:
            raise ValueError("SQLite objective differs from receipt")
        for key in ("P", "valid", "config"):
            if trial.user_attrs.get(key) != row[key]:
                raise ValueError(f"SQLite metadata differs: {key}")
    return score


def _audit_random(root, manifest, studies, smoke):
    budget = limits(protocol(smoke))
    count = budget["trials"]
    ready = json.loads((root / "random-review-ready.json").read_text())
    if ready.get("status") != "paused_for_random_review" or ready.get(
        "counts"
    ) != dict.fromkeys(parent.FAMILIES, count):
        raise ValueError("four-study random review barrier not complete")
    if any(
        ready.get(k) is not False
        for k in ("tpe_started", "confirmation_started", "final_started")
    ):
        raise ValueError("unexpected phases in random review receipt")
    proposals = json.loads((root / "random-proposals.json").read_text())
    audit = {}
    for family, study in studies.items():
        trials = study.trials
        if (
            len(proposals[family]) != count
            or len(trials) < count
            or len(trials) > 2 * count
        ):
            raise ValueError("random proposal/history budget differs")
        rows = []
        for number, proposal in enumerate(proposals[family]):
            trial = trials[number]
            if (
                trial.number != number
                or trial.state != optuna.trial.TrialState.COMPLETE
                or trial.user_attrs.get("phase") != "random"
            ):
                raise ValueError("random trial history incomplete or altered")
            path = root / "studies" / family / "proposals" / f"{number:03d}.json"
            row = json.loads(path.read_text())
            context = dict(family=family, smoke=smoke, identity=manifest["identity"])
            _validate_result(root, context, proposal, trial, row)
            rows.append(
                dict(
                    trial_id=number,
                    sha256=_file_hash(path),
                    J=row["J"],
                    P=row["P"],
                    valid=row["valid"],
                )
            )
        audit[family] = rows
    return audit


def _backup_random(root):
    """Keep an independent SQLite backup before the first adaptive ask."""
    backup = root / "random-stage-backup"
    backup.mkdir(exist_ok=True)
    for family in parent.FAMILIES:
        destination = backup / family / "trials.sqlite3"
        if destination.exists():
            raise ValueError("partial random backup requires diagnosis")
        destination.parent.mkdir(exist_ok=True)
        source = sqlite3.connect(
            (root / "studies" / family / "trials.sqlite3").resolve().as_uri()
            + "?mode=ro",
            uri=True,
        )
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
        finally:
            source.close()
            target.close()
    for name in (
        "campaign.json",
        "random-proposals.json",
        "random-review-ready.json",
        "seed-manifest.json",
    ):
        (backup / name).write_bytes((root / name).read_bytes())
    return {
        str(p.relative_to(backup)): _file_hash(p)
        for p in sorted(backup.rglob("*"))
        if p.is_file()
    }


def _new_sampler(family, smoke):
    return optuna.samplers.TPESampler(
        seed=structured_seed("sampler", family, "tpe"),
        n_startup_trials=limits(protocol(smoke))["startup_trials"],
        n_ei_candidates=24,
        multivariate=True,
        group=False,
        constant_liar=True,
    )


def _save_state(path, sampler, proposal, identity):
    payload = pickle.dumps(sampler, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(
        path,
        dict(
            identity=identity,
            proposal=proposal,
            sampler_sha256=hashlib.sha256(payload).hexdigest(),
            sampler_pickle_base64=base64.b64encode(payload).decode("ascii"),
        ),
    )


def _load_state(path, identity):
    state = json.loads(path.read_text())
    if state["identity"] != identity:
        raise ValueError("sampler continuation identity differs")
    payload = base64.b64decode(state["sampler_pickle_base64"], validate=True)
    if hashlib.sha256(payload).hexdigest() != state["sampler_sha256"]:
        raise ValueError("sampler checkpoint corrupt")
    # Only checkpoints generated by this local runner are accepted, with the
    # source/dependency identity and byte checksum validated before decoding.
    sampler = pickle.loads(payload)
    if not isinstance(sampler, optuna.samplers.TPESampler):
        raise ValueError("unexpected stored sampler")
    return sampler, state["proposal"]


def _evaluate_trial(arguments):
    context, proposal, root = arguments
    root = Path(root)
    if parent.source_identity() != context["identity"]:
        raise ValueError("worker source/environment identity differs")
    records = [
        parent._fit_with_retry(job, root) for job in parent.jobs(context, proposal)
    ]
    score = parent.score_records(
        records,
        context["family"],
        limits(protocol(context["smoke"]))["repeats"]["screen"],
    )
    row = dict(**proposal, **score, fits=[r["directory"] for r in records])
    write_json(root / "proposals" / f"{proposal['trial_id']:03d}.json", row)
    print(
        f"{context['family']} TPE trial #{proposal['trial_id']}: "
        f"J={row['J']:.6f} P={row['P']}",
        flush=True,
    )
    return context["family"], row


def _run_locked(root, smoke, workers):
    budget = limits(protocol(smoke))
    count = budget["trials"]
    baseline = json.loads((root / "campaign.json").read_text())
    if (
        baseline["protocol"] != protocol(smoke)
        or baseline["limits"] != budget
        or baseline["phase"] != "random"
    ):
        raise ValueError("parent scientific protocol differs")
    identity = parent.source_identity()
    transition = _source_transition(baseline["identity"], identity)
    studies = {f: _load_study(root, f) for f in parent.FAMILIES}
    audit = _audit_random(root, baseline, studies, smoke)
    manifest_path = root / "tpe-continuation.json"
    manifest = dict(
        identity=identity,
        parent_identity=baseline["identity"],
        parent_campaign_sha256=_file_hash(root / "campaign.json"),
        protocol_sha256=protocol()["protocol_sha256"],
        smoke=smoke,
        source_transition=transition,
        random_audit=audit,
        random_manifest_sha256=_file_hash(root / "random-proposals.json"),
        random_trials_per_study=count,
        additional_tpe_trials_per_study=count,
        sampler=dict(
            n_startup_trials=budget["startup_trials"],
            n_ei_candidates=24,
            multivariate=True,
            group=False,
            constant_liar=True,
            pruner="NopPruner",
        ),
        authorization=(
            "Author authorized 60 additional trials after completed "
            "random review on 2026-10-09"
        ),
        stop="120 screening trials per study; no automatic confirmation/final fits",
    )
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("TPE continuation source/history identity differs")
        hashes = json.loads((root / "random-stage-backup" / "hashes.json").read_text())
        for name, checksum in hashes.items():
            if _file_hash(root / "random-stage-backup" / name) != checksum:
                raise ValueError("random-stage backup differs")
    else:
        if any(len(s.trials) != count for s in studies.values()):
            raise ValueError("adaptive history exists without continuation receipt")
        hashes = _backup_random(root)
        write_json(root / "random-stage-backup" / "hashes.json", hashes)
        write_json(manifest_path, manifest)
    state_identity = digest(manifest)
    state_paths = {f: root / "tpe-samplers" / f"{f}.json" for f in parent.FAMILIES}
    pending, samplers = {}, {}
    for family in parent.FAMILIES:
        path = state_paths[family]
        if path.exists():
            sampler, proposal = _load_state(path, state_identity)
        else:
            if len(studies[family].trials) != count:
                raise ValueError(
                    "adaptive history without sampler checkpoint; no redraw"
                )
            sampler, proposal = _new_sampler(family, smoke), None
            _save_state(path, sampler, proposal, state_identity)
        studies[family] = _load_study(root, family, sampler)
        samplers[family], pending[family] = sampler, proposal
        trials = studies[family].trials
        for number, trial in enumerate(trials[count:], start=count):
            if trial.number != number or trial.state not in (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.RUNNING,
            ):
                raise ValueError("nonsequential adaptive trial history")
            if trial.state == optuna.trial.TrialState.COMPLETE:
                row = json.loads(
                    (
                        root / "studies" / family / "proposals" / f"{number:03d}.json"
                    ).read_text()
                )
                context = dict(family=family, smoke=smoke, identity=identity)
                _validate_result(root, context, row, trial, row)
                if trial.user_attrs.get("phase") != "tpe":
                    raise ValueError("adaptive trial phase differs")
            elif (
                proposal is None
                or proposal["trial_id"] != number
                or number != len(trials) - 1
            ):
                raise ValueError(
                    "orphan adaptive proposal; requires diagnosis, no redraw"
                )
        if proposal is not None:
            trial = trials[proposal["trial_id"]]
            if trial.params != proposal["optuna_params"]:
                raise ValueError("pending sampler proposal differs")
            if trial.state == optuna.trial.TrialState.COMPLETE:
                pending[family] = None
                _save_state(path, sampler, None, state_identity)

    def progress():
        counts = {
            f: sum(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials)
            for f, s in studies.items()
        }
        write_json(
            root / "progress.json",
            dict(
                phase="tpe",
                counts=counts,
                target_per_study=2 * count,
                random_completed_per_study=count,
                tpe_counts={f: n - count for f, n in counts.items()},
            ),
        )
        return counts

    def next_job(family):
        study = studies[family]
        proposal = pending[family]
        if proposal is None:
            if len(study.trials) >= 2 * count:
                return None
            trial = study.ask()
            trial.set_user_attr("phase", "tpe")
            config = candidate(trial, family)
            proposal = dict(
                trial_id=trial.number,
                config=json.loads(json.dumps(config)),
                optuna_params=trial.params,
            )
            # Persist RNG and pending proposal together before dispatch. A crash
            # inside ask/configuration is diagnosed as orphaned, never rerolled.
            _save_state(state_paths[family], samplers[family], proposal, state_identity)
            pending[family] = proposal
        context = dict(family=family, smoke=smoke, identity=identity)
        return context, proposal, root / "studies" / family

    def record(arguments, result):
        context, proposal, _ = arguments
        family, row = result
        if family != context["family"]:
            raise ValueError("worker family differs")
        study = studies[family]
        frozen = study.trials[proposal["trial_id"]]
        _validate_result(root, context, proposal, frozen, row)
        trial = optuna.trial.Trial(study, frozen._trial_id)
        for key in ("P", "valid", "config"):
            trial.set_user_attr(key, row[key])
        study.tell(trial, row["J"])
        pending[family] = None
        _save_state(state_paths[family], samplers[family], None, state_identity)
        progress()

    progress()
    if workers == 1:
        while True:
            any_work = False
            for family in parent.FAMILIES:
                arguments = next_job(family)
                if arguments is not None:
                    any_work = True
                    record(arguments, _evaluate_trial(arguments))
            if not any_work:
                break
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {}
            for family in parent.FAMILIES:
                arguments = next_job(family)
                if arguments is not None:
                    futures[pool.submit(_evaluate_trial, arguments)] = arguments
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    arguments = futures.pop(future)
                    record(arguments, future.result())
                    arguments = next_job(arguments[0]["family"])
                    if arguments is not None:
                        futures[pool.submit(_evaluate_trial, arguments)] = arguments
    counts = progress()
    if any(n != 2 * count for n in counts.values()):
        raise ValueError("adaptive phase budget incomplete")
    status = dict(
        status="paused_for_tpe_review",
        counts=counts,
        random_trials_per_study=count,
        tpe_trials_per_study=count,
        confirmation_started=False,
        final_started=False,
    )
    write_json(root / "tpe-review-ready.json", status)
    return status


def run_all(root, *, smoke=False, workers=1):
    """Continue the existing studies only through the authorized TPE budget."""
    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(root).resolve()
    if not (root / "random-review-ready.json").is_file():
        raise ValueError("completed random review checkpoint required for TPE")
    descriptor = os.open(root / "campaign.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another campaign is running") from error
        return _run_locked(root, smoke, workers)
    finally:
        os.close(descriptor)

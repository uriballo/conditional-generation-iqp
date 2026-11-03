"""Explicit, resumable HPO stages for the three classical CW models."""

from __future__ import annotations

import fcntl
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import optuna

from .cw_classical_config import (
    CAPACITIES,
    FAMILIES,
    betas,
    candidate,
    classical_stream,
    digest,
    limits,
    protocol,
    seed_manifest,
)
from .cw_classical_fit import _hash, run_fit, source_identity
from .cw_followup_fit import _verify_artifacts, write_json
from .cw_followup_metrics import choose_confirmed, shortlist

RESULTS_ROOT = Path(__file__).parents[1] / "results"


def _freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("frozen classical campaign identity/configuration differs")
    else:
        write_json(path, value)


def _validate_proposal(row):
    p = row["optuna_params"]
    expected = dict(
        learning_rate=p["learning_rate"],
        init_scale=p["init_scale"],
        b1=p["b1"],
        b2=1 - p["one_minus_beta2"],
        schedule=p["schedule"],
    )
    if row["config"] != expected:
        raise ValueError("proposal configuration differs from Optuna parameters")


def _study(root, family, capacity, phase="random", number=0, *, startup_trials=60):
    seed = classical_stream("sampler", family, capacity, phase, number)["seed"]
    sampler = (
        optuna.samplers.RandomSampler(seed=seed)
        if phase == "random"
        else optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=startup_trials,
            n_ei_candidates=24,
            multivariate=True,
            group=False,
            constant_liar=True,
        )
    )
    return optuna.create_study(
        study_name=family,
        direction="minimize",
        load_if_exists=True,
        storage="sqlite:///" + str((root / f"{family}.sqlite3").resolve()),
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )


def _jobs(manifest, proposal, family, stage):
    for repeat in range(limits(manifest["smoke"])["repeats"][stage]):
        yield dict(
            family=family,
            capacity=manifest["capacity"],
            smoke=manifest["smoke"],
            candidate=proposal["config"],
            trial_id=proposal["trial_id"],
            stage=stage,
            repeat=repeat,
            identity=manifest["identity"],
        )


def _fit_worker(arguments):
    return run_fit(*arguments)


def _aggregate(records, expected):
    if len(records) != expected or sorted(r["job"]["repeat"] for r in records) != list(
        range(expected)
    ):
        raise ValueError("missing or duplicate classical fit repeats")
    if any(row["status"] != "ok" for row in records):
        return dict(J=1.0, P=None, valid=False, search_penalty_not_measured_tvd=True)
    return dict(
        J=float(np.mean([row["scores"]["J"] for row in records])),
        P=float(np.mean([row["scores"]["P"] for row in records])),
        valid=True,
    )


def _fit_proposal(root, manifest, family, proposal, stage, executor):
    jobs = list(_jobs(manifest, proposal, family, stage))
    arguments = [(job, root) for job in jobs]
    records = list(
        map(_fit_worker, arguments)
        if executor is None
        else executor.map(_fit_worker, arguments)
    )
    result = dict(
        **proposal,
        **_aggregate(records, len(jobs)),
        fits=[
            dict(
                directory=r["directory"],
                receipt_sha256=_hash(root / r["directory"] / "metrics.json"),
            )
            for r in records
        ],
    )
    write_json(root / stage / family / f"{proposal['trial_id']:03d}.json", result)
    return result


def _audit_result(root, manifest, family, row, stage):
    _validate_proposal(row)
    jobs = list(_jobs(manifest, row, family, stage))
    if len(row["fits"]) != len(jobs):
        raise ValueError("missing classical fit receipts")
    records = []
    for job, reference in zip(jobs, row["fits"], strict=True):
        directory = root / "fits" / digest(job)
        if reference["directory"] != str(directory.relative_to(root)):
            raise ValueError("classical fit directory differs")
        if _hash(directory / "metrics.json") != reference["receipt_sha256"]:
            raise ValueError("classical fit receipt differs")
        record = json.loads((directory / "metrics.json").read_text())
        if record["job"] != job:
            raise ValueError("classical fit job differs")
        _verify_artifacts(directory, record["artifact_sha256"])
        for pool in record["target_pools"]:
            if _hash(root / pool["path"]) != pool["artifact_sha256"]:
                raise ValueError("classical target pool artifact differs")
        records.append(record)
    expected = _aggregate(records, len(jobs))
    if any(row.get(key) != expected[key] for key in ("J", "P", "valid")):
        raise ValueError("classical proposal score differs from receipts")
    return records


def _random(root, manifest, executor):
    count = limits(manifest["smoke"])["trials"]
    freeze = root / "random-proposals.json"
    if not freeze.exists():
        if any(root.glob("*.sqlite3")):
            raise ValueError(
                "interrupted proposal freeze requires diagnosis; no resampling"
            )
        proposals = {}
        for family in FAMILIES:
            study = _study(root, family, manifest["capacity"])
            proposals[family] = []
            for _ in range(count):
                trial = study.ask()
                config = candidate(trial)
                trial.set_user_attr("phase", "random")
                proposals[family].append(
                    dict(
                        trial_id=trial.number,
                        config=config,
                        optuna_params=trial.params,
                    )
                )
        write_json(freeze, proposals)
    proposals = json.loads(freeze.read_text())
    if set(proposals) != set(FAMILIES) or any(
        len(rows) != count for rows in proposals.values()
    ):
        raise ValueError("random proposal budget differs")
    for family in FAMILIES:
        study = _study(root, family, manifest["capacity"])
        for row in proposals[family]:
            _validate_proposal(row)
            trial = study.trials[row["trial_id"]]
            if (
                trial.params != row["optuna_params"]
                or trial.user_attrs.get("phase") != "random"
            ):
                raise ValueError("frozen random proposal differs")
            path = root / "screen" / family / f"{row['trial_id']:03d}.json"
            if path.exists():
                result = json.loads(path.read_text())
                if any(
                    result[key] != row[key]
                    for key in ("trial_id", "config", "optuna_params")
                ):
                    raise ValueError("frozen random result differs from proposal")
                _audit_result(root, manifest, family, result, "screen")
            else:
                if trial.state != optuna.trial.TrialState.RUNNING:
                    raise ValueError("completed random trial has missing result")
                result = _fit_proposal(root, manifest, family, row, "screen", executor)
            _tell(study, trial, result)
    ready = dict(
        status="paused_for_random_review", counts=dict.fromkeys(FAMILIES, count)
    )
    _freeze(root / "random-review-ready.json", ready)
    return ready


def _tell(study, trial, result):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        if trial.value != result["J"] or trial.user_attrs.get(
            "result_sha256"
        ) != digest(result):
            raise ValueError("Optuna result differs from recorded proposal")
        return
    if trial.state != optuna.trial.TrialState.RUNNING:
        raise ValueError("unexpected Optuna trial state")
    live = optuna.trial.Trial(study, trial._trial_id)
    live.set_user_attr("result_sha256", digest(result))
    live.set_user_attr("valid", result["valid"])
    study.tell(live, result["J"])


def _audit_screen(root, manifest, *, complete=False):
    count = limits(manifest["smoke"])["trials"]
    ready_path = root / (
        "tpe-review-ready.json" if complete else "random-review-ready.json"
    )
    ready = json.loads(ready_path.read_text())
    total = count * (2 if complete else 1)
    if ready["counts"] != dict.fromkeys(FAMILIES, total):
        raise ValueError("classical screening review barrier incomplete")
    rows = {}
    for family in FAMILIES:
        study = _study(root, family, manifest["capacity"])
        if len(study.trials) < total or len(study.trials) > 2 * count:
            raise ValueError("classical proposal history budget differs")
        rows[family] = []
        for number in range(total):
            trial = study.trials[number]
            row = json.loads(
                (root / "screen" / family / f"{number:03d}.json").read_text()
            )
            if row["trial_id"] != number or trial.params != row["optuna_params"]:
                raise ValueError("classical trial parameters differ")
            if trial.user_attrs.get("phase") != ("random" if number < count else "tpe"):
                raise ValueError("classical trial phase differs")
            _audit_result(root, manifest, family, row, "screen")
            if trial.state != optuna.trial.TrialState.COMPLETE:
                raise ValueError("classical screening trial incomplete")
            _tell(study, trial, row)
            rows[family].append(row)
    return rows


def _tpe(root, manifest, executor):
    if (root / "confirmation.json").exists():
        raise ValueError("no HPO proposals after confirmation inspection")
    _audit_screen(root, manifest)
    count = limits(manifest["smoke"])["trials"]
    for family in FAMILIES:
        for number in range(count, 2 * count):
            study = _study(
                root, family, manifest["capacity"], "tpe", number, startup_trials=count
            )
            pending = root / "tpe-proposals" / family / f"{number:03d}.json"
            if pending.exists():
                proposal = json.loads(pending.read_text())
                _validate_proposal(proposal)
                trial = study.trials[number]
                if trial.params != proposal["optuna_params"]:
                    raise ValueError("pending TPE proposal differs")
            else:
                if len(study.trials) != number:
                    raise ValueError(
                        "interrupted TPE ask requires diagnosis; no resampling"
                    )
                trial = study.ask()
                config = candidate(trial)
                trial.set_user_attr("phase", "tpe")
                proposal = dict(
                    trial_id=trial.number, config=config, optuna_params=trial.params
                )
                write_json(pending, proposal)
                trial = study.trials[number]
            path = root / "screen" / family / f"{number:03d}.json"
            if path.exists():
                result = json.loads(path.read_text())
                if any(
                    result[key] != proposal[key]
                    for key in ("trial_id", "config", "optuna_params")
                ):
                    raise ValueError("frozen TPE result differs from proposal")
                _audit_result(root, manifest, family, result, "screen")
            else:
                result = _fit_proposal(
                    root, manifest, family, proposal, "screen", executor
                )
            _tell(study, trial, result)
    ready = dict(
        status="paused_for_tpe_review", counts=dict.fromkeys(FAMILIES, 2 * count)
    )
    _freeze(root / "tpe-review-ready.json", ready)
    return ready


def _confirmation(root, manifest, executor):
    screen = _audit_screen(root, manifest, complete=True)
    choices = {family: shortlist(rows) for family, rows in screen.items()}
    frozen = {
        family: [
            {key: row[key] for key in ("trial_id", "config", "optuna_params")}
            for row in rows
        ]
        for family, rows in choices.items()
    }
    _freeze(root / "shortlist.json", frozen)
    results = {}
    for family, rows in frozen.items():
        results[family] = []
        for proposal in rows:
            path = root / "confirmation" / family / f"{proposal['trial_id']:03d}.json"
            if path.exists():
                result = json.loads(path.read_text())
                _audit_result(root, manifest, family, result, "confirmation")
            else:
                result = _fit_proposal(
                    root, manifest, family, proposal, "confirmation", executor
                )
            if not result["valid"]:
                raise FloatingPointError("confirmation failed; no replacement seeds")
            results[family].append(result)
    winners = {family: choose_confirmed(rows) for family, rows in results.items()}
    result = dict(status="confirmation_complete", candidates=results, winners=winners)
    _freeze(root / "confirmation.json", result)
    return result


def _final(root, manifest, executor, confirmation_path=None):
    """Final fits of the confirmed winners; ``confirmation_path`` names a stored
    confirmation record whose winners are trained without the search."""
    stored = confirmation_path is not None
    confirmation_path = confirmation_path or root / "confirmation.json"
    confirmation = json.loads(confirmation_path.read_text())
    if confirmation["status"] != "confirmation_complete":
        raise ValueError("completed confirmation required")
    if not stored:
        frozen = json.loads((root / "shortlist.json").read_text())
        _audit_screen(root, manifest, complete=True)
        for family, rows in confirmation["candidates"].items():
            if [r["config"] for r in rows] != [r["config"] for r in frozen[family]]:
                raise ValueError("confirmed candidate shortlist differs")
            for row in rows:
                _audit_result(root, manifest, family, row, "confirmation")
            if choose_confirmed(rows) != confirmation["winners"][family]:
                raise ValueError("confirmed winner differs")
    families = {}
    for family in FAMILIES:
        winner = confirmation["winners"][family]
        proposal = {key: winner[key] for key in ("trial_id", "config", "optuna_params")}
        path = root / "final" / family / f"{proposal['trial_id']:03d}.json"
        if path.exists():
            result = json.loads(path.read_text())
            records = _audit_result(root, manifest, family, result, "final")
        else:
            result = _fit_proposal(root, manifest, family, proposal, "final", executor)
            records = _audit_result(root, manifest, family, result, "final")
        if not result["valid"]:
            raise FloatingPointError("final fit failed; all fixed repeats required")
        records.sort(key=lambda row: row["job"]["repeat"])
        families[family] = dict(
            trial_id=winner["trial_id"],
            config=winner["config"],
            architecture=records[0]["architecture"],
            beta=betas("final", manifest["smoke"]),
            curves={
                key: [row["metrics_curves"][key] for row in records]
                for key in records[0]["metrics_curves"]
                if key != "beta"
            },
            marker_beta=(0.4, 1.0, 2.0),
            marker_curves={
                key: [row["marker_curves"][key] for row in records]
                for key in records[0]["marker_curves"]
                if key != "beta"
            },
            repeat_scores=[row["scores"] for row in records],
            repeat_ids=[row["job"]["repeat"] for row in records],
            probability_evaluation=[row["probability_evaluation"] for row in records],
            marker_probability_evaluation=[
                row["marker_probability_evaluation"] for row in records
            ],
            target_pools=[row["target_pools"] for row in records],
            training_seconds_including_compile=[
                sum(s["training_seconds_including_compile"] for s in row["starts"])
                for row in records
            ],
            sampling=[row["sampling"] for row in records],
            fits=result["fits"],
        )
    bundle = dict(
        kind="cw_classical_final",
        stage="final",
        smoke=manifest["smoke"],
        capacity=manifest["capacity"],
        repeats=limits(manifest["smoke"])["repeats"]["final"],
        starts=limits(manifest["smoke"])["starts"]["final"],
        families=families,
        provenance=dict(
            manifest=manifest,
            confirmation_sha256=_hash(confirmation_path),
            pairing="parent target-pool namespace and final repeat IDs",
        ),
    )
    write_json(root / "figure_data.json", bundle)
    return dict(status="final_complete", bundle=str(root / "figure_data.json"))


def run_all(
    root,
    *,
    phase="random",
    capacity="small",
    smoke=False,
    workers=1,
    confirmation=None,
):
    """Run one phase; ``confirmation`` (a stored confirmation record) makes the
    final phase train its winners without the search stages."""
    if capacity not in CAPACITIES or workers < 1:
        raise ValueError("invalid capacity or worker count")
    if phase not in ("random", "tpe", "confirmation", "final", "reporting"):
        raise ValueError("unknown classical campaign phase")
    root = Path(root)
    if not root.resolve().is_relative_to(RESULTS_ROOT.resolve()):
        raise ValueError(
            "new classical campaign outputs must be under ignored results/"
        )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = dict(
            protocol=protocol(smoke),
            capacity=capacity,
            smoke=bool(smoke),
            limits=limits(smoke),
            identity=source_identity(),
        )
        _freeze(root / "campaign.json", manifest)
        _freeze(root / "seed-manifest.json", seed_manifest(smoke))
        prerequisite = {
            "tpe": "random-review-ready.json",
            "confirmation": "tpe-review-ready.json",
            "reporting": "tpe-review-ready.json",
            "final": None if confirmation else "confirmation.json",
        }.get(phase)
        if prerequisite and not (root / prerequisite).is_file():
            raise ValueError(f"classical {phase} requires completed {prerequisite}")
        context = (
            nullcontext(None)
            if workers == 1
            else ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
        )
        with context as executor:
            if phase == "random":
                return _random(root, manifest, executor)
            if phase == "tpe":
                return _tpe(root, manifest, executor)
            if phase == "confirmation":
                return _confirmation(root, manifest, executor)
            if phase == "final":
                return _final(root, manifest, executor, confirmation)
            _confirmation(root, manifest, executor)
            return _final(root, manifest, executor)

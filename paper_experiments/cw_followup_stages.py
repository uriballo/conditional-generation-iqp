"""Explicit confirmation and fresh final fitting of the frozen CW search.

Only orchestration and CLI dispatch may change from completed screening.
Every numerical fit source, dependency, library and seed schedule is preserved.
Screening artifacts and study databases are read, never retold or resampled.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import zip_longest
from pathlib import Path

import numpy as np
import optuna

from . import cw_followup_hpo as parent
from . import cw_followup_tpe as tpe
from .cw_followup_config import C_STAR, FAMILIES, MARKERS, limits, protocol, stage_betas
from .cw_followup_fit import _verify_artifacts, digest, write_json
from .cw_followup_metrics import choose_confirmed, curve_scores, shortlist

_ALLOWED = {"run_experiments.py", "paper_experiments/cw_followup_stages.py"}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _transition(before, after):
    old, new = dict(before), dict(after)
    old_files, new_files = old.pop("paper_sources"), new.pop("paper_sources")
    if old != new:
        raise ValueError("confirmation numerical environment/library differs")
    changes = {}
    for name in old_files.keys() | new_files.keys():
        if old_files.get(name) != new_files.get(name):
            if name not in _ALLOWED or name not in new_files:
                raise ValueError(f"frozen numerical fit source differs: {name}")
            changes[name] = dict(before=old_files.get(name), after=new_files[name])
    return changes


def _screening(root, smoke):
    """Audit completed original study records before any B evaluation."""
    continuation = json.loads((root / "tpe-continuation.json").read_text())
    baseline = json.loads((root / "campaign.json").read_text())
    count = 2 * limits(protocol(smoke))["trials"]
    ready = json.loads((root / "tpe-review-ready.json").read_text())
    if (
        ready.get("status") != "paused_for_tpe_review"
        or ready.get("counts") != dict.fromkeys(FAMILIES, count)
        or ready.get("confirmation_started") is not False
        or ready.get("final_started") is not False
        or continuation["smoke"] != smoke
        or baseline["protocol"] != protocol(smoke)
        or _hash(root / "campaign.json") != continuation["parent_campaign_sha256"]
        or _hash(root / "random-proposals.json")
        != continuation["random_manifest_sha256"]
    ):
        raise ValueError("completed frozen TPE screening barrier required")
    selections, hashes = {}, {}
    for family in FAMILIES:
        study = tpe._load_study(root, family)
        if len(study.trials) != count:
            raise ValueError("screening proposal count differs")
        rows = []
        for number, trial in enumerate(study.trials):
            phase = "random" if number < count // 2 else "tpe"
            if (
                trial.number != number
                or trial.state != optuna.trial.TrialState.COMPLETE
                or trial.user_attrs.get("phase") != phase
            ):
                raise ValueError("screening history is incomplete or differs")
            path = root / "studies" / family / "proposals" / f"{number:03d}.json"
            row = json.loads(path.read_text())
            context = dict(
                family=family,
                smoke=smoke,
                identity=baseline["identity"]
                if phase == "random"
                else continuation["identity"],
            )
            tpe._validate_result(root, context, row, trial, row)
            rows.append(row)
            hashes[str(path.relative_to(root))] = _hash(path)
        selections[family] = [
            {k: r[k] for k in ("trial_id", "config", "J", "P", "valid")}
            for r in shortlist(rows)
        ]
    return continuation, selections, hashes


def _freeze(path, manifest):
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("frozen stage manifest differs; no restart/resampling")
    else:
        write_json(path, manifest)


def _worker(arguments):
    job, root = arguments
    if parent.source_identity() != job["identity"]:
        raise ValueError("worker differs from frozen stage identity")
    return parent._fit_with_retry(job, Path(root))


def _jobs(selections, stage, identity, smoke):
    queues = []
    for family in FAMILIES:
        contexts = dict(family=family, identity=identity, smoke=smoke)
        queues.append(
            [
                job
                for proposal in selections[family]
                for job in parent.jobs(contexts, proposal, stage=stage)
            ]
        )
    # Interleave families. There is no adaptive proposal or feedback in these fits.
    return [j for batch in zip_longest(*queues) for j in batch if j is not None]


def _collect(directory, selections, stage, identity, smoke):
    rows = {}
    repeats = limits(protocol(smoke))["repeats"][stage]
    for family in FAMILIES:
        rows[family] = []
        context = dict(family=family, identity=identity, smoke=smoke)
        for proposal in selections[family]:
            records = []
            for job in parent.jobs(context, proposal, stage=stage):
                fit_dir = directory / "fits" / digest(job)
                record = json.loads((fit_dir / "metrics.json").read_text())
                if record["job"] != job:
                    raise ValueError("stage fit identity differs")
                _verify_artifacts(fit_dir, record["artifact_sha256"])
                records.append(record)
            rows[family].append(
                dict(
                    **{k: proposal[k] for k in ("trial_id", "config")},
                    **parent.score_records(records, family, repeats),
                    fits=[r["directory"] for r in records],
                )
            )
    return rows


def _export_final(directory, winners, identity, smoke):
    spec = protocol(smoke)
    repeats = limits(spec)["repeats"]["final"]
    data = dict(
        kind="cw_followup_final",
        stage="final",
        smoke=smoke,
        repeats=repeats,
        starts=limits(spec)["starts"]["final"],
        C_star=C_STAR,
        families={},
        provenance=dict(identity=identity, protocol=spec, fit_receipt_sha256={}),
    )
    for family in FAMILIES:
        winner = winners[family]
        context = dict(family=family, identity=identity, smoke=smoke)
        all_curves, all_markers, scores = [], [], []
        for repeat in range(repeats):
            curves, markers = [], []
            for job in parent.jobs(context, winner, stage="final"):
                if job["repeat"] != repeat:
                    continue
                fit_dir = directory / "fits" / digest(job)
                receipt = fit_dir / "metrics.json"
                record = json.loads(receipt.read_text())
                if record["job"] != job or record["status"] != "ok":
                    raise ValueError("final fit missing/invalid; no repeat omission")
                _verify_artifacts(fit_dir, record["artifact_sha256"])
                curves.append(record["metrics_curves"])
                if family.startswith("shared_"):
                    with np.load(
                        fit_dir / "markers.npz", allow_pickle=False
                    ) as archive:
                        markers.append({k: archive[k].tolist() for k in archive.files})
                else:
                    markers.append(record["metrics_curves"])
                data["provenance"]["fit_receipt_sha256"][
                    str(receipt.relative_to(directory))
                ] = _hash(receipt)
            joined = {
                k: np.concatenate([np.asarray(c[k]) for c in curves]) for k in curves[0]
            }
            joined_markers = {
                k: np.concatenate([np.asarray(c[k]) for c in markers])
                for k in markers[0]
            }
            expected = stage_betas(
                spec, "final", None if family.startswith("shared_") else "independent"
            )
            # Independent reports comprise the three cases, not a shared F sweep.
            if family.startswith("independent_"):
                expected = MARKERS
            if not np.array_equal(joined["beta"], expected) or not np.array_equal(
                joined_markers["beta"], MARKERS
            ):
                raise ValueError("final reporting temperature grid differs")
            scores.append(
                curve_scores(
                    joined["tvd"],
                    joined["magnetization_error"],
                    joined["heat_error"],
                    joined["beta"],
                )
            )
            all_curves.append(joined)
            all_markers.append(joined_markers)
        data["families"][family] = dict(
            trial_id=winner["trial_id"],
            config=winner["config"],
            beta=all_curves[0]["beta"].tolist(),
            curves={
                k: np.stack([c[k] for c in all_curves]).tolist()
                for k in all_curves[0]
                if k != "beta"
            },
            marker_beta=list(MARKERS),
            marker_curves={
                k: np.stack([c[k] for c in all_markers]).tolist()
                for k in all_markers[0]
                if k != "beta"
            },
            repeat_scores=scores,
        )
    write_json(directory / "figure_data.json", data)
    return data


def _run_locked(root, phase, smoke, workers):
    continuation, shortlists, hashes = _screening(root, smoke)
    identity = parent.source_identity()
    changes = _transition(continuation["identity"], identity)
    confirmation = root.parent / "confirmation"
    final = root.parent / "final"
    manifest = dict(
        identity=identity,
        parent_identity=continuation["identity"],
        source_transition=changes,
        tpe_manifest_sha256=_hash(root / "tpe-continuation.json"),
        screening_proposal_sha256=hashes,
        shortlists=shortlists,
        smoke=smoke,
        authorization="Author requested confirmation and final fitting 2026-10-10",
        protocol=protocol(smoke),
    )
    # Freeze all shortlists before the first confirmation evaluation of B.
    _freeze(confirmation / "campaign.json", manifest)

    def run_stage(stage, selections, destination):
        planned = _jobs(selections, stage, identity, smoke)
        done = []

        def progress(record):
            done.append(record)
            counts = {f: sum(r["job"]["family"] == f for r in done) for f in FAMILIES}
            write_json(
                root.parent / "reporting-progress.json",
                dict(
                    stage=stage,
                    fits_completed=len(done),
                    fits_total=len(planned),
                    counts=counts,
                ),
            )
            print(f"{stage}: {len(done)}/{len(planned)} fits; {counts}", flush=True)

        if workers == 1:
            for job in planned:
                progress(_worker((job, destination)))
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                futures = [pool.submit(_worker, (job, destination)) for job in planned]
                for future in as_completed(futures):
                    progress(future.result())
        rows = _collect(destination, selections, stage, identity, smoke)
        write_json(destination / "results.json", rows)
        return rows

    if phase in ("confirmation", "reporting"):
        results = run_stage("confirmation", shortlists, confirmation)
        winners = {f: choose_confirmed(results[f]) for f in FAMILIES}
        _freeze(confirmation / "winners.json", winners)
        write_json(
            confirmation / "review-ready.json",
            dict(status="confirmation_complete", winners=winners),
        )
        if phase == "confirmation":
            return winners
    else:
        if not (confirmation / "review-ready.json").is_file():
            raise ValueError("completed confirmation required before final fitting")
        results = _collect(confirmation, shortlists, "confirmation", identity, smoke)
        winners = {f: choose_confirmed(results[f]) for f in FAMILIES}
        _freeze(confirmation / "winners.json", winners)
    # Winner configurations are fixed before any final-grid evaluation.
    selections = {f: [winners[f]] for f in FAMILIES}
    _freeze(
        final / "campaign.json",
        dict(
            identity=identity,
            winners=winners,
            confirmation_manifest_sha256=_hash(confirmation / "campaign.json"),
            smoke=smoke,
        ),
    )
    run_stage("final", selections, final)
    data = _export_final(final, winners, identity, smoke)
    status = dict(
        status="final_complete",
        repeats=data["repeats"],
        winners={f: winners[f]["trial_id"] for f in FAMILIES},
        figure_data_sha256=_hash(final / "figure_data.json"),
    )
    write_json(final / "review-ready.json", status)
    return status


def run_all(root, *, phase="reporting", smoke=False, workers=1):
    """Run explicitly authorized confirmation, final, or both with fixed seeds."""
    if phase not in ("confirmation", "final", "reporting") or workers < 1:
        raise ValueError("invalid phase/workers")
    root = Path(root).resolve()
    if not (root / "tpe-review-ready.json").is_file():
        raise ValueError("completed TPE screening required")
    descriptor = os.open(root / "campaign.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another campaign is running") from error
        return _run_locked(root, phase, smoke, workers)
    finally:
        os.close(descriptor)

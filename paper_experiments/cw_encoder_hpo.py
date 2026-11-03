"""Single-family HPO campaigns for non-IQP Curie--Weiss encoders.

Each campaign follows the Curie--Weiss follow-up protocol with one family named
by its protocol file, whose encoder options are fixed at the protocol values.
Fits, scoring, shortlist, confirmation rule and TPE sampler settings reuse the
follow-up modules. Fit streams use the follow-up namespace, so every fit pairs
with the shared-parity fit of the same stage, repeat and start. Each phase stops
when it completes. Fits can be sharded across machines with identical source
identity; recording and selection run once.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import optuna

from . import cw_diagonal_fit
from . import cw_followup_hpo as audited
from . import cw_followup_tpe as audited_tpe
from .cw_followup_config import (
    C_STAR,
    MARKERS,
    limits,
    stage_betas,
    structured_seed,
)
from .cw_followup_config import (
    candidate as audited_candidate,
)
from .cw_followup_config import (
    protocol as audited_protocol,
)
from .cw_followup_fit import ENCODER_OPTIONS, _verify_artifacts, digest, write_json
from .cw_followup_metrics import choose_confirmed, curve_scores, shortlist
from .provenance import source_identity as _source_identity

CAMPAIGNS = {
    "twisted": "cw_twisted_protocol.json",
    "twisted_independent": "cw_twisted_independent_protocol.json",
    "diagonal_k4": "cw_diagonal_k4_protocol.json",
}
# Campaigns whose model is not an input encoder; every other campaign uses the
# follow-up fit.
FITS = {
    "diagonal_k4": cw_diagonal_fit.fit_with_retry,
}
PHASES = ("random", "tpe", "confirmation", "final")
CAMPAIGN = "twisted"
PROTOCOL = Path(__file__).with_name(CAMPAIGNS[CAMPAIGN])
FAMILY = "shared_twisted_parity"


def configure(name):
    """Select the campaign; spawned workers call this before any fit."""
    global CAMPAIGN, PROTOCOL, FAMILY
    if name not in CAMPAIGNS:
        raise ValueError(f"unknown campaign: {name}")
    CAMPAIGN, PROTOCOL = name, Path(__file__).with_name(CAMPAIGNS[name])
    FAMILY = protocol()["family"]["name"]


def protocol():
    value = json.loads(PROTOCOL.read_text())
    stored = value.pop("protocol_sha256")
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(body).hexdigest() != stored:
        raise ValueError(f"{CAMPAIGN} protocol hash mismatch")
    parent = audited_protocol()["protocol_sha256"]
    if value["parent_protocol"]["protocol_sha256"] != parent:
        raise ValueError("parent protocol differs")
    value["protocol_sha256"] = stored
    return value


def source_identity():
    return _source_identity(PROTOCOL, protocol()["protocol_sha256"])


def candidate(trial):
    """Shared-parity search space plus the fixed protocol options.

    A protocol may replace the beta_scale range, because the scale of a new
    encoder schedule need not share the parity encoder's units. A protocol
    without an encoder hyperparameter names its parent search family and the
    fixed model constants it adds.
    """
    family = protocol()["family"]
    if "search_family" in family:
        config = audited_candidate(trial, family["search_family"])
    elif "beta_scale_range" not in family:
        config = audited_candidate(trial, "shared_parity")
    else:
        config = audited_candidate(trial, "independent_unconditional")
        low, high = family["beta_scale_range"]
        config["beta_scale"] = trial.suggest_float("beta_scale", low, high, log=True)
    for key in ENCODER_OPTIONS:
        if key in family:
            config[key] = float(family[key])
    config.update(family.get("fixed_config", {}))
    return config


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise ValueError(f"frozen manifest differs: {path.name}")
    else:
        write_json(path, value)


def _context(smoke, identity):
    return dict(family=FAMILY, smoke=smoke, identity=identity)


def _fit_job(job, root):
    return FITS.get(CAMPAIGN, audited._fit_with_retry)(job, Path(root))


def _fit(arguments):
    name, job, root = arguments
    configure(name)
    if source_identity() != job["identity"]:
        raise ValueError("worker source identity differs from the campaign")
    return _fit_job(job, root)


def _run_fits(jobs, root, workers):
    if workers == 1:
        return [_fit((CAMPAIGN, job, root)) for job in jobs]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        return list(pool.map(_fit, [(CAMPAIGN, job, root) for job in jobs]))


def _shard(jobs, shard):
    if shard is None:
        return jobs
    index, count = shard
    return jobs[index::count]


def _study(root, sampler):
    path = root / "study" / "trials.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=FAMILY,
        storage="sqlite:///" + str(path.resolve()),
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )


def _score(proposal, records, smoke):
    score = audited.score_records(
        records, FAMILY, limits(audited_protocol(smoke))["repeats"]["screen"]
    )
    return dict(**proposal, **score, fits=[r["directory"] for r in records])


def _campaign(root, smoke, identity):
    manifest = dict(
        protocol=protocol(),
        parent_limits=limits(audited_protocol(smoke)),
        identity=identity,
        smoke=smoke,
    )
    _freeze(root / "campaign.json", manifest)
    return manifest


def _shard_identity(root):
    """Shards reuse the primary's frozen campaign and must match its identity."""
    identity = source_identity()
    campaign = root / "campaign.json"
    if not campaign.is_file() or json.loads(campaign.read_text())["identity"] != (
        json.loads(json.dumps(identity))
    ):
        raise ValueError("shard requires the primary campaign with identical identity")
    return identity


def _random(root, smoke, workers, shard, prepare=False):
    path = root / "random-proposals.json"
    if shard is not None:
        identity = _shard_identity(root)
        if not path.is_file():
            raise ValueError("shards require proposals frozen by the primary")
    else:
        identity = source_identity()
        _campaign(root, smoke, identity)
    count = limits(audited_protocol(smoke))["trials"]
    if not path.exists():
        if (root / "study" / "trials.sqlite3").exists():
            raise RuntimeError("partial proposal freeze requires diagnosis")
        sampler = optuna.samplers.RandomSampler(
            seed=structured_seed("sampler", FAMILY, "random")
        )
        study = _study(root, sampler)
        rows = []
        for _ in range(count):
            trial = study.ask()
            rows.append(
                dict(
                    trial_id=trial.number,
                    config=json.loads(json.dumps(candidate(trial))),
                    optuna_params=trial.params,
                )
            )
        write_json(path, rows)
    proposals = json.loads(path.read_text())
    context = _context(smoke, identity)
    jobs = [j for p in proposals for j in audited.jobs(context, p)]
    if prepare:
        return dict(status="prepared", fits=len(jobs))
    _run_fits(_shard(jobs, shard), root / "study", workers)
    if shard is not None:
        return dict(status="shard_complete", shard=list(shard))
    study = _study(root, None)
    for proposal in proposals:
        trial = study.trials[proposal["trial_id"]]
        if trial.params != proposal["optuna_params"]:
            raise ValueError("frozen Optuna proposal differs")
        if trial.state == optuna.trial.TrialState.COMPLETE:
            continue
        records = [
            _fit_job(job, root / "study") for job in audited.jobs(context, proposal)
        ]
        row = _score(proposal, records, smoke)
        write_json(
            root / "study" / "proposals" / f"{proposal['trial_id']:03d}.json", row
        )
        live = optuna.trial.Trial(study, trial._trial_id)
        live.set_user_attr("phase", "random")
        for key in ("P", "valid", "config"):
            live.set_user_attr(key, row[key])
        study.tell(live, row["J"])
    status = dict(status="paused_for_random_review", trials=len(study.trials))
    write_json(root / "random-review-ready.json", status)
    return status


def _tpe(root, smoke, workers):
    if not (root / "random-review-ready.json").is_file():
        raise ValueError("completed random phase required")
    identity = source_identity()
    baseline = json.loads((root / "campaign.json").read_text())
    if baseline["identity"] != identity:
        raise ValueError("TPE source identity differs from the random phase")
    count = limits(audited_protocol(smoke))["trials"]
    manifest = dict(
        parent_campaign_sha256=_hash(root / "campaign.json"),
        random_proposals_sha256=_hash(root / "random-proposals.json"),
    )
    _freeze(root / "tpe-continuation.json", manifest)
    state_path = root / "tpe-sampler.json"
    state_identity = digest(manifest)
    if state_path.exists():
        sampler, pending = audited_tpe._load_state(state_path, state_identity)
    else:
        sampler, pending = audited_tpe._new_sampler(FAMILY, smoke), None
        audited_tpe._save_state(state_path, sampler, None, state_identity)
    study = _study(root, sampler)
    context = _context(smoke, identity)
    while len(study.trials) < 2 * count or pending is not None:
        if pending is None:
            trial = study.ask()
            trial.set_user_attr("phase", "tpe")
            pending = dict(
                trial_id=trial.number,
                config=json.loads(json.dumps(candidate(trial))),
                optuna_params=trial.params,
            )
            # Persist the sampler state with the proposal before any fit.
            audited_tpe._save_state(state_path, sampler, pending, state_identity)
        frozen = study.trials[pending["trial_id"]]
        if frozen.params != pending["optuna_params"]:
            raise ValueError("pending TPE proposal differs")
        if frozen.state != optuna.trial.TrialState.COMPLETE:
            jobs = list(audited.jobs(context, pending))
            records = _run_fits(jobs, root / "study", workers)
            row = _score(pending, records, smoke)
            write_json(
                root / "study" / "proposals" / f"{pending['trial_id']:03d}.json", row
            )
            live = optuna.trial.Trial(study, frozen._trial_id)
            for key in ("P", "valid", "config"):
                live.set_user_attr(key, row[key])
            study.tell(live, row["J"])
            print(f"TPE trial {pending['trial_id']}: J={row['J']:.6f}", flush=True)
        pending = None
        audited_tpe._save_state(state_path, sampler, None, state_identity)
    status = dict(status="paused_for_tpe_review", trials=len(study.trials))
    write_json(root / "tpe-review-ready.json", status)
    return status


def _screening_rows(root, smoke):
    count = 2 * limits(audited_protocol(smoke))["trials"]
    study = _study(root, None)
    if len(study.trials) != count or any(
        t.state != optuna.trial.TrialState.COMPLETE for t in study.trials
    ):
        raise ValueError("completed screening required")
    rows = []
    for trial in study.trials:
        path = root / "study" / "proposals" / f"{trial.number:03d}.json"
        row = json.loads(path.read_text())
        if row["J"] != trial.value or row["trial_id"] != trial.number:
            raise ValueError("screening receipt differs from the study")
        rows.append(row)
    return rows


def _collect(directory, selections, stage, identity, smoke):
    repeats = limits(audited_protocol(smoke))["repeats"][stage]
    rows = []
    for proposal in selections:
        records = []
        for job in audited.jobs(_context(smoke, identity), proposal, stage=stage):
            fit_dir = directory / "fits" / digest(job)
            record = json.loads((fit_dir / "metrics.json").read_text())
            if record["job"] != job:
                raise ValueError("stage fit identity differs")
            _verify_artifacts(fit_dir, record["artifact_sha256"])
            records.append(record)
        rows.append(
            dict(
                trial_id=proposal["trial_id"],
                config=proposal["config"],
                **audited.score_records(records, FAMILY, repeats),
                fits=[r["directory"] for r in records],
            )
        )
    return rows


def _confirmation(root, smoke, workers, shard, prepare=False):
    if not (root / "tpe-review-ready.json").is_file():
        raise ValueError("completed TPE phase required")
    directory = root / "confirmation"
    if shard is not None:
        # Shards take the shortlist the primary froze before any confirmation fit.
        identity = _shard_identity(root)
        selections = json.loads((directory / "campaign.json").read_text())["shortlist"]
    else:
        identity = source_identity()
        rows = _screening_rows(root, smoke)
        selections = [
            {k: r[k] for k in ("trial_id", "config", "J", "P", "valid")}
            for r in shortlist(rows)
        ]
        _freeze(
            directory / "campaign.json",
            dict(identity=identity, shortlist=selections),
        )
    context = _context(smoke, identity)
    jobs = [j for p in selections for j in audited.jobs(context, p, "confirmation")]
    if prepare:
        return dict(status="prepared", fits=len(jobs))
    _run_fits(_shard(jobs, shard), directory, workers)
    if shard is not None:
        return dict(status="shard_complete", shard=list(shard))
    results = _collect(directory, selections, "confirmation", identity, smoke)
    write_json(directory / "results.json", results)
    winner = choose_confirmed(results)
    _freeze(directory / "winner.json", winner)
    status = dict(status="confirmation_complete", winner=winner["trial_id"])
    write_json(directory / "review-ready.json", status)
    return status


def _final(root, smoke, workers, shard, prepare=False, winner=None):
    """Final fits of the confirmed winner, or of a stored ``winner`` without
    the search and confirmation stages."""
    confirmation = root / "confirmation"
    identity = source_identity()
    if shard is not None:
        _shard_identity(root)
    if winner is None:
        if not (confirmation / "review-ready.json").is_file():
            raise ValueError("completed confirmation required")
        winner = json.loads((confirmation / "winner.json").read_text())
        manifest = dict(
            identity=identity,
            winner=winner,
            confirmation_campaign_sha256=_hash(confirmation / "campaign.json"),
        )
    else:
        _campaign(root, smoke, identity)
        manifest = dict(identity=identity, winner=winner, stored_winner=True)
    directory = root / "final"
    _freeze(directory / "campaign.json", manifest)
    jobs = list(audited.jobs(_context(smoke, identity), winner, stage="final"))
    if prepare:
        return dict(status="prepared", fits=len(jobs))
    _run_fits(_shard(jobs, shard), directory, workers)
    if shard is not None:
        return dict(status="shard_complete", shard=list(shard))
    data = _export_final(directory, winner, jobs, smoke)
    status = dict(
        status="final_complete",
        winner=winner["trial_id"],
        figure_data_sha256=_hash(directory / "figure_data.json"),
        mean_J=float(np.mean([s["J"] for s in data["repeat_scores"]])),
    )
    write_json(directory / "review-ready.json", status)
    return status


def _export_final(directory, winner, jobs, smoke):
    """Store per-repeat final curves and markers in the audited figure layout."""
    spec = audited_protocol(smoke)
    if FAMILY.startswith("independent_"):
        return _export_direct_final(directory, winner, jobs, smoke)
    curves, markers, scores, receipts = [], [], [], {}
    for job in jobs:
        fit_dir = directory / "fits" / digest(job)
        record = _final_record(directory, fit_dir, job, receipts)
        curve = {k: np.asarray(v) for k, v in record["metrics_curves"].items()}
        if not np.array_equal(curve["beta"], stage_betas(spec, "final")):
            raise ValueError("final reporting temperature grid differs")
        with np.load(fit_dir / "markers.npz", allow_pickle=False) as archive:
            marker = {k: archive[k] for k in archive.files}
        if not np.array_equal(marker["beta"], MARKERS):
            raise ValueError("final marker temperatures differ")
        scores.append(
            curve_scores(
                curve["tvd"],
                curve["magnetization_error"],
                curve["heat_error"],
                curve["beta"],
            )
        )
        curves.append(curve)
        markers.append(marker)
    data = dict(
        kind=f"cw_{CAMPAIGN}_final",
        family=FAMILY,
        trial_id=winner["trial_id"],
        config=winner["config"],
        C_star=C_STAR,
        beta=curves[0]["beta"].tolist(),
        curves={
            k: np.stack([c[k] for c in curves]).tolist()
            for k in curves[0]
            if k != "beta"
        },
        marker_beta=list(MARKERS),
        marker_curves={
            k: np.stack([m[k] for m in markers]).tolist()
            for k in markers[0]
            if k != "beta"
        },
        repeat_scores=scores,
        provenance=dict(protocol=protocol(), fit_receipt_sha256=receipts),
    )
    write_json(directory / "figure_data.json", data)
    return data


def _final_record(directory, fit_dir, job, receipts):
    record = json.loads((fit_dir / "metrics.json").read_text())
    if record["job"] != job or record["status"] != "ok":
        raise ValueError("final fit missing or invalid; no repeat omission")
    _verify_artifacts(fit_dir, record["artifact_sha256"])
    receipts[str((fit_dir / "metrics.json").relative_to(directory))] = _hash(
        fit_dir / "metrics.json"
    )
    return record


def _export_direct_final(directory, winner, jobs, smoke):
    """Temperature-specific fits: one fit per marker, stacked by repeat."""
    repeats = limits(audited_protocol(smoke))["repeats"]["final"]
    receipts, by_repeat = {}, {}
    for job in jobs:
        record = _final_record(
            directory, directory / "fits" / digest(job), job, receipts
        )
        by_repeat.setdefault(job["repeat"], []).append(record)
    if sorted(by_repeat) != list(range(repeats)):
        raise ValueError("final repeats are incomplete")
    curves, scores = [], []
    for repeat in range(repeats):
        cases = sorted(by_repeat[repeat], key=lambda r: r["job"]["beta"])
        if [r["job"]["beta"] for r in cases] != list(MARKERS):
            raise ValueError("final marker temperatures differ")
        curve = {
            k: np.concatenate([np.asarray(r["metrics_curves"][k]) for r in cases])
            for k in cases[0]["metrics_curves"]
        }
        if not np.array_equal(curve["beta"], MARKERS):
            raise ValueError("final marker temperatures differ")
        curves.append(curve)
        scores.append(
            curve_scores(
                curve["tvd"],
                curve["magnetization_error"],
                curve["heat_error"],
                curve["beta"],
            )
        )
    stacked = {
        k: np.stack([c[k] for c in curves]).tolist() for k in curves[0] if k != "beta"
    }
    data = dict(
        kind=f"cw_{CAMPAIGN}_final",
        family=FAMILY,
        trial_id=winner["trial_id"],
        config=winner["config"],
        C_star=C_STAR,
        beta=list(MARKERS),
        curves=stacked,
        marker_beta=list(MARKERS),
        marker_curves=stacked,
        repeat_scores=scores,
        provenance=dict(protocol=protocol(), fit_receipt_sha256=receipts),
    )
    write_json(directory / "figure_data.json", data)
    return data


def run(
    root,
    phase,
    *,
    smoke=False,
    workers=1,
    shard=None,
    prepare=False,
    campaign="twisted",
    winner=None,
):
    """Run one phase; ``prepare`` freezes its manifest for shards and stops.

    ``winner`` (trial_id and config) makes the final phase train a stored
    selection instead of the confirmation winner.
    """
    configure(campaign)
    if phase not in PHASES or workers < 1:
        raise ValueError("invalid phase or workers")
    if prepare and (shard is not None or phase == "tpe"):
        raise ValueError("prepare freezes random/confirmation/final on the primary")
    if shard is not None and phase == "tpe":
        raise ValueError(
            "TPE proposals are sequential; shard random/confirmation/final"
        )
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = "campaign.lock" if shard is None else f"shard-{shard[0]}.lock"
    descriptor = os.open(root / lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another process holds this campaign lock") from error
        if phase == "random":
            return _random(root, smoke, workers, shard, prepare)
        if phase == "tpe":
            return _tpe(root, smoke, workers)
        if phase == "confirmation":
            return _confirmation(root, smoke, workers, shard, prepare)
        return _final(root, smoke, workers, shard, prepare, winner)
    finally:
        os.close(descriptor)


def _parse_shard(text):
    if text is None:
        return None
    index, count = (int(x) for x in text.split("/"))
    if not 0 <= index < count:
        raise argparse.ArgumentTypeError("shard must be k/K with 0 <= k < K")
    return index, count


def main(argv=None, campaign=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    if campaign is None:
        parser.add_argument("--campaign", choices=CAMPAIGNS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard", type=_parse_shard, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args(argv)
    status = run(
        args.root,
        args.phase,
        smoke=args.smoke,
        workers=args.workers,
        shard=args.shard,
        prepare=args.prepare,
        campaign=campaign or args.campaign,
    )
    print(json.dumps(status, indent=1), flush=True)


if __name__ == "__main__":
    main()

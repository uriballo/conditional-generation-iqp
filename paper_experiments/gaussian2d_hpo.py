"""Random-search campaigns for the Hamming-weight family.

Five campaigns share one protocol (``gaussian2d_lattice_protocol.json``): input
conditioning on the 4x4 lattice and four diagonal-conditioning baselines
(affine or quadratic condition map, lattice or complete diagonal layer). The
search space, sampler settings and stage structure are those of the 16-qubit
Curie--Weiss campaigns, plus a one-body initialization mean. Fit streams and
target pools depend only on stage, repeat and start, so all trials and all five
campaigns are paired. Each phase stops when it completes and resumes from the
fit receipts.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import iqp_cond as ciqp
import jax.numpy as jnp
import numpy as np
import optuna

from . import gaussian2d_diagonal as diagonal
from . import gaussian2d_lattice as lattice
from .ansatz import square_lattice_ansatz
from .cw_followup_config import candidate as cw_candidate
from .cw_followup_fit import _verify_artifacts, digest, write_json, write_npz
from .cw_followup_tpe import _load_state, _save_state
from .provenance import source_identity as _source_identity
from .training import make_trainer

HERE = Path(__file__).parent
PARENT = HERE / "gaussian2d_lattice_protocol.json"
CAMPAIGNS = {
    "input": "gaussian2d_lattice_protocol.json",
    "diagonal": "gaussian2d_diagonal_protocol.json",
    "diagonal_quadratic": "gaussian2d_diagonal_quadratic_protocol.json",
    "diagonal_complete": "gaussian2d_diagonal_complete_protocol.json",
    "diagonal_complete_quadratic": (
        "gaussian2d_diagonal_complete_quadratic_protocol.json"
    ),
}
PHASES = ("random", "tpe", "confirmation", "final")
SMOKE = dict(
    steps=2,
    validation_interval=1,
    n_train=32,
    batch_size=8,
    n_ops=16,
    mc_samples=32,
    sample_chunk=16,
    trials=3,
    shortlist=2,
    repeats={"screen": 1, "confirmation": 1, "final": 2},
    starts={"screen": 1, "confirmation": 2, "final": 2},
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol(campaign="input"):
    parent = json.loads(PARENT.read_text())
    if campaign == "input":
        return parent
    value = json.loads((HERE / CAMPAIGNS[campaign]).read_text())
    if value["parent_protocol"]["sha256"] != _sha(PARENT):
        raise ValueError("diagonal protocol names a different parent protocol")
    return value


def limits(smoke=False):
    if smoke:
        return json.loads(json.dumps(SMOKE))
    spec = protocol()
    training = spec["training"]
    return dict(
        steps=training["updates_per_start"],
        validation_interval=100,
        n_train=training["target_pool_draws_per_fitting_condition"],
        batch_size=training["target_minibatch"],
        n_ops=training["operator_masks_per_update"],
        mc_samples=training["reference_draws_per_update"],
        sample_chunk=training["sample_chunk"],
        trials=spec["search"]["random_trials"],
        shortlist=spec["confirmation"]["candidates"],
        repeats=dict(
            screen=spec["search"]["screening"]["repeats"],
            confirmation=spec["confirmation"]["repeats_per_candidate"],
            final=spec["final_fitting"]["repeats"],
        ),
        starts=dict(
            screen=spec["search"]["screening"]["starts_per_repeat"],
            confirmation=spec["confirmation"]["starts_per_repeat"],
            final=spec["final_fitting"]["starts_per_repeat"],
        ),
    )


def seed(*keys):
    spec = protocol()["randomness"]
    payload = [spec["master_seed"], spec["namespace"], *keys]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(canonical).digest()[:4], "big")


def conditions():
    return {
        k: np.asarray(v, dtype=float)
        for k, v in protocol()["protocols"]["axes"].items()
    }


def candidate(trial):
    """The Curie--Weiss search space plus the one-body initialization mean."""
    config = cw_candidate(trial, "independent_unconditional")
    config["one_body_mean"] = trial.suggest_float("one_body_mean", 0, math.pi / 2)
    return config


def source_identity(campaign):
    path = HERE / CAMPAIGNS[campaign]
    return _source_identity(path, _sha(path))


def _model(campaign):
    """Initialization, trainer and exact metrics of one campaign model."""
    pairs = conditions()
    if campaign == "input":
        ansatz = square_lattice_ansatz(16)
        model = ciqp.ConditionalModel(
            ansatz, lattice.CollectivePhaseEncoder.from_conditions(pairs["fitting"])
        )

        def initial(config, rng):
            values = rng.normal(0, 1, 40) * config["init_scale"]
            values[:16] += config["one_body_mean"]
            return values

        def train(settings, training, pool, init, select, checkpoint, reference):
            return make_trainer(dict(settings, compute_validation_mmd=False))(
                model,
                pool,
                pool,
                seed=reference,
                config=training,
                initial_params=init,
                checkpoint_callback=checkpoint,
                selection_metric=select,
            )

        return initial, train, lattice.metrics
    spec = protocol(campaign)
    basis, ansatz_name = spec["basis"], spec["ansatz"]
    phases = diagonal.n_phases(ansatz_name)
    ansatz = diagonal.ansatz_of(ansatz_name)
    encoder = diagonal.zero_encoder(len(pairs["fitting"]))

    def initial(config, rng):
        w0 = rng.normal(0, 1, phases) * config["init_scale"]
        w0[:16] += config["one_body_mean"]
        return np.concatenate([w0, np.zeros((diagonal.n_terms(basis) - 1) * phases)])

    def train(settings, training, pool, init, select, checkpoint, reference):
        return diagonal.make_trainer(settings)(
            ansatz,
            encoder,
            pairs["fitting"],
            pool,
            seed=reference,
            config=training,
            initial_params=init,
            selection_metric=select,
            checkpoint_callback=checkpoint,
        )

    return initial, train, diagonal.metrics


def run_fit(job, root):
    """One repeat: every start, checkpoint selection on J over all starts."""
    root = Path(root)
    directory = root / "fits" / digest(job)
    receipt = directory / "metrics.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["job"] != job:
            raise ValueError("existing fit identity differs")
        _verify_artifacts(directory, saved["artifact_sha256"])
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "job.json", job)
    budget = limits(job["smoke"])
    stage, repeat, config = job["stage"], job["repeat"], job["config"]
    pairs = conditions()
    pool_seed = seed(stage, repeat, "target_pool")
    pool = lattice.sample_targets(
        pairs["fitting"], budget["n_train"], np.random.default_rng(pool_seed)
    )
    initial, train, metrics = _model(job["campaign"])
    ratio = 1.0 if config["schedule"] == "constant" else 0.1
    sigmas = tuple(
        (-2 * np.log(1 - 2 * w / 16)) ** -0.5 for w in config["expected_weights"]
    )
    training = ciqp.TrainingConfig(
        steps=budget["steps"],
        batch_size=budget["batch_size"],
        n_ops=budget["n_ops"],
        mc_samples=budget["mc_samples"],
        sample_chunk=budget["sample_chunk"],
        gemm_dtype=jnp.float32,
        validation_n_ops=1,
        validation_mc_samples=2,
        sigmas=sigmas,
        learning_rate=config["learning_rate"],
        final_learning_rate=config["learning_rate"] * ratio,
        validation_interval=budget["validation_interval"],
        mask_refresh="step",
    )

    def select(theta):
        return float(np.mean(metrics(theta, pairs["development"])["full_tvd"]))

    started = time.perf_counter()
    rows, params, durations, streams = [], [], [], []
    try:
        for start in range(budget["starts"][stage]):
            stream = {
                purpose: seed(stage, repeat, start, purpose)
                for purpose in protocol()["randomness"]["purposes"]
            }
            streams.append(stream)
            archive = directory / f"start-{start}.npz"
            start_receipt = directory / f"start-{start}.json"
            if start_receipt.exists():
                saved = json.loads(start_receipt.read_text())
                _verify_artifacts(directory, {archive.name: saved["sha256"]})
                with np.load(archive) as data:
                    params.extend(data["checkpoint_params"])
                rows.extend(saved["checkpoints"])
                durations.append(saved["seconds"])
                continue
            init = initial(config, np.random.default_rng(stream["initialization"]))
            local_rows, local_params = [], []

            def checkpoint(step, theta, rows=local_rows, kept=local_params, s=start):
                kept.append(np.asarray(theta).copy())
                rows.append(dict(step=int(step), start=s))

            settings = dict(
                name="adam",
                learning_rate=config["learning_rate"],
                b1=config["b1"],
                b2=config["b2"],
                eps=1e-8,
                final_learning_rate_ratio=ratio,
                rng_seeds={k: v for k, v in stream.items() if k != "initialization"},
            )
            tick = time.perf_counter()
            result = train(
                settings,
                training,
                pool,
                init,
                select,
                checkpoint,
                stream["reference"],
            )
            duration = time.perf_counter() - tick
            for row, score in zip(local_rows, result.selection_scores, strict=True):
                row["J"] = float(score)
            write_npz(
                archive,
                initial_params=init,
                checkpoint_params=np.stack(local_params),
                checkpoint_steps=np.array([r["step"] for r in local_rows]),
                final_params=result.final_params,
                train_losses=result.train_losses,
            )
            write_json(
                start_receipt,
                dict(checkpoints=local_rows, seconds=duration, sha256=_sha(archive)),
            )
            rows.extend(local_rows)
            params.extend(local_params)
            durations.append(duration)
        index = min(
            range(len(rows)),
            key=lambda i: (rows[i]["J"], rows[i]["step"], rows[i]["start"]),
        )
        best = params[index]
        write_npz(directory / "parameters.npz", best_params=best)
        summary = dict(
            status="ok",
            J=rows[index]["J"],
            selected_step=rows[index]["step"],
            selected_start=rows[index]["start"],
            checkpoints=rows,
            development_metrics=metrics(best, pairs["development"]),
        )
        if stage == "final":
            summary["reporting_metrics"] = metrics(best, pairs["reporting"])
    except FloatingPointError as error:
        summary = dict(status="numerical_failure", J=1.0, error=str(error))
    summary.update(
        job=job,
        target_pool=dict(
            seed=pool_seed, sha256=hashlib.sha256(pool.tobytes()).hexdigest()
        ),
        streams=streams,
        start_seconds=durations,
        wall_seconds=time.perf_counter() - started,
        directory=str(directory),
        artifact_sha256={p.name: _sha(p) for p in sorted(directory.glob("*.npz"))},
    )
    write_json(receipt, json.loads(json.dumps(summary, allow_nan=False)))
    print(
        f"{job['campaign']} {stage} trial {job['trial_id']} repeat {repeat}: "
        f"J={summary['J']:.5f} ({summary['wall_seconds']:.0f}s)",
        flush=True,
    )
    return summary


def jobs(campaign, proposal, stage, smoke, identity):
    for repeat in range(limits(smoke)["repeats"][stage]):
        yield dict(
            campaign=campaign,
            trial_id=proposal["trial_id"],
            config=proposal["config"],
            stage=stage,
            repeat=repeat,
            smoke=smoke,
            identity=identity,
        )


def _fit(arguments):
    job, root = arguments
    if source_identity(job["campaign"]) != job["identity"]:
        raise ValueError("worker source identity differs from the campaign")
    return run_fit(job, root)


def _run_fits(jobs_, root, workers):
    if workers == 1:
        return [_fit((job, root)) for job in jobs_]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        return list(pool.map(_fit, [(job, root) for job in jobs_]))


def _score(proposal, records):
    valid = all(r["status"] == "ok" for r in records)
    return dict(
        trial_id=proposal["trial_id"],
        config=proposal["config"],
        J=float(np.mean([r["J"] for r in records])) if valid else 1.0,
        valid=valid,
        repeat_J=[r["J"] for r in records],
        fits=[r["directory"] for r in records],
    )


def _study(root, campaign, sampler):
    path = root / "study" / "trials.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=campaign,
        storage="sqlite:///" + str(path.resolve()),
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )


def _freeze(path, value):
    value = json.loads(json.dumps(value))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"frozen manifest differs: {path}")
    else:
        write_json(path, value)


def _tell(study, row, phase):
    trial = study.trials[row["trial_id"]]
    if trial.state == optuna.trial.TrialState.COMPLETE:
        return
    live = optuna.trial.Trial(study, trial._trial_id)
    live.set_user_attr("phase", phase)
    live.set_user_attr("valid", row["valid"])
    study.tell(live, row["J"])


def _random(root, campaign, smoke, workers, identity):
    _freeze(
        root / "campaign.json",
        dict(
            campaign=campaign,
            protocol=protocol(campaign),
            identity=identity,
            smoke=smoke,
        ),
    )
    path = root / "random-proposals.json"
    if not path.exists():
        if (root / "study" / "trials.sqlite3").exists():
            raise RuntimeError("partial proposal freeze requires diagnosis")
        study = _study(
            root,
            campaign,
            optuna.samplers.RandomSampler(seed=seed("sampler", campaign, "random")),
        )
        rows = []
        for _ in range(limits(smoke)["trials"]):
            trial = study.ask()
            rows.append(
                dict(
                    trial_id=trial.number,
                    config=candidate(trial),
                    optuna_params=trial.params,
                )
            )
        write_json(path, rows)
    proposals = json.loads(path.read_text())
    all_jobs = [
        j for p in proposals for j in jobs(campaign, p, "screen", smoke, identity)
    ]
    _run_fits(all_jobs, root / "study", workers)
    study = _study(root, campaign, None)
    for proposal in proposals:
        if study.trials[proposal["trial_id"]].params != proposal["optuna_params"]:
            raise ValueError("frozen Optuna proposal differs")
        records = [
            run_fit(job, root / "study")
            for job in jobs(campaign, proposal, "screen", smoke, identity)
        ]
        row = _score(proposal, records)
        write_json(
            root / "study" / "proposals" / f"{proposal['trial_id']:03d}.json", row
        )
        _tell(study, row, "random")
    status = dict(status="random_complete", trials=len(study.trials))
    write_json(root / "random-complete.json", status)
    return status


def _tpe(root, campaign, smoke, workers, identity):
    if not (root / "random-complete.json").is_file():
        raise ValueError("completed random phase required")
    spec = protocol()["search"]["tpe_sampler"]
    count = limits(smoke)["trials"]
    state_path = root / "tpe-sampler.json"
    state_identity = digest(dict(campaign=campaign, identity=identity))
    if state_path.exists():
        sampler, pending = _load_state(state_path, state_identity)
    else:
        sampler = optuna.samplers.TPESampler(
            seed=seed("sampler", campaign, "tpe"),
            n_startup_trials=count,
            n_ei_candidates=spec["n_ei_candidates"],
            multivariate=spec["multivariate"],
            group=spec["group"],
            constant_liar=spec["constant_liar"],
        )
        pending = None
        _save_state(state_path, sampler, None, state_identity)
    study = _study(root, campaign, sampler)
    while len(study.trials) < 2 * count or pending is not None:
        if pending is None:
            trial = study.ask()
            pending = dict(
                trial_id=trial.number,
                config=candidate(trial),
                optuna_params=trial.params,
            )
            _save_state(state_path, sampler, pending, state_identity)
        records = _run_fits(
            list(jobs(campaign, pending, "screen", smoke, identity)),
            root / "study",
            workers,
        )
        row = _score(pending, records)
        write_json(
            root / "study" / "proposals" / f"{pending['trial_id']:03d}.json", row
        )
        _tell(study, row, "tpe")
        pending = None
        _save_state(state_path, sampler, None, state_identity)
    status = dict(status="tpe_complete", trials=len(study.trials))
    write_json(root / "tpe-complete.json", status)
    return status


def shortlist(rows, size):
    chosen, seen = [], []
    for row in sorted(rows, key=lambda r: (r["J"], r["trial_id"])):
        if row["valid"] and row["config"] not in seen:
            chosen.append(row)
            seen.append(row["config"])
        if len(chosen) == size:
            break
    return chosen


def _confirmation(root, campaign, smoke, workers, identity):
    if not (root / "random-complete.json").is_file():
        raise ValueError("completed screening required")
    study = _study(root, campaign, None)
    if any(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise ValueError("incomplete screening trial")
    rows = [
        json.loads((root / "study" / "proposals" / f"{t.number:03d}.json").read_text())
        for t in study.trials
    ]
    directory = root / "confirmation"
    selections = [
        dict(trial_id=r["trial_id"], config=r["config"], screening_J=r["J"])
        for r in shortlist(rows, limits(smoke)["shortlist"])
    ]
    _freeze(
        directory / "campaign.json",
        dict(identity=identity, screening_trials=len(rows), shortlist=selections),
    )
    all_jobs = [
        j
        for p in selections
        for j in jobs(campaign, p, "confirmation", smoke, identity)
    ]
    records = _run_fits(all_jobs, directory, workers)
    results = []
    for proposal in selections:
        mine = [r for r in records if r["job"]["trial_id"] == proposal["trial_id"]]
        results.append(
            dict(**_score(proposal, mine), screening_J=proposal["screening_J"])
        )
    write_json(directory / "results.json", results)
    winner = min(results, key=lambda r: (r["J"], r["trial_id"]))
    _freeze(directory / "winner.json", winner)
    return dict(
        status="confirmation_complete", winner=winner["trial_id"], J=winner["J"]
    )


def _final(root, campaign, smoke, workers, identity):
    winner = json.loads((root / "confirmation" / "winner.json").read_text())
    directory = root / "final"
    _freeze(directory / "campaign.json", dict(identity=identity, winner=winner))
    records = _run_fits(
        list(jobs(campaign, winner, "final", smoke, identity)), directory, workers
    )
    if any(r["status"] != "ok" for r in records):
        raise RuntimeError("final fit failed; no repeat omission")
    status = dict(
        status="final_complete",
        winner=winner["trial_id"],
        mean_J=float(np.mean([r["J"] for r in records])),
        fits=[r["directory"] for r in records],
    )
    write_json(directory / "complete.json", status)
    return status


def load_campaign(root):
    """Search record and verified final fits of a completed campaign.

    Fits are located by job digest, so campaigns can be copied between machines.
    Returns the search summary and a list of (receipt, best parameters).
    """
    root = Path(root)
    manifest = json.loads((root / "campaign.json").read_text())
    winner = json.loads((root / "final" / "campaign.json").read_text())["winner"]
    if not (root / "final" / "complete.json").is_file():
        raise ValueError("completed final phase required")
    fits = []
    for job in jobs(
        manifest["campaign"], winner, "final", manifest["smoke"], manifest["identity"]
    ):
        directory = root / "final" / "fits" / digest(job)
        record = json.loads((directory / "metrics.json").read_text())
        if record["job"] != job or record["status"] != "ok":
            raise ValueError(f"final fit identity or status differs: {directory}")
        _verify_artifacts(directory, record["artifact_sha256"])
        with np.load(directory / "parameters.npz") as data:
            fits.append((record, data["best_params"]))
    screening = [
        json.loads(path.read_text())
        for path in sorted((root / "study" / "proposals").glob("*.json"))
    ]
    search = dict(
        campaign=manifest["campaign"],
        protocol=manifest["protocol"],
        identity=manifest["identity"],
        screening_trials=len(screening),
        tpe_trials=len(screening) - limits(manifest["smoke"])["trials"],
        screening_top=[
            {k: r[k] for k in ("trial_id", "J", "config")}
            for r in shortlist(screening, 10)
        ],
        confirmation=[
            {k: r[k] for k in ("trial_id", "J", "repeat_J", "screening_J", "config")}
            for r in json.loads((root / "confirmation" / "results.json").read_text())
        ],
        winner={k: winner[k] for k in ("trial_id", "J", "config")},
        final_J=[r["J"] for r, _ in fits],
        final_seconds=[r["wall_seconds"] for r, _ in fits],
    )
    return search, fits


def run(root, campaign, phase, *, smoke=False, workers=1):
    if campaign not in CAMPAIGNS or phase not in PHASES or workers < 1:
        raise ValueError("invalid campaign, phase or workers")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = source_identity(campaign)
    if (root / "campaign.json").exists():
        frozen = json.loads((root / "campaign.json").read_text())
        if frozen["identity"] != json.loads(json.dumps(identity)):
            raise ValueError("source identity differs from the frozen campaign")
    descriptor = os.open(root / "campaign.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another process holds this campaign lock") from error
        phase_runner = dict(
            random=_random, tpe=_tpe, confirmation=_confirmation, final=_final
        )[phase]
        return phase_runner(root, campaign, smoke, workers, identity)
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", choices=CAMPAIGNS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    status = run(
        args.root, args.campaign, args.phase, smoke=args.smoke, workers=args.workers
    )
    print(json.dumps(status, indent=1), flush=True)


if __name__ == "__main__":
    main()

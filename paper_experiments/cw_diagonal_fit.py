"""Recorded Curie--Weiss fits for diagonal-layer conditioning.

Mirrors ``cw_followup_fit.run_fit`` step for step: the same target pools,
streams, checkpoints, global checkpoint selection, evaluation and artifacts.
The optimized vector is the flattened Chebyshev weights w, and every loss and
evaluation uses theta(beta). The trainer copies ``training.make_trainer`` with
identical batch, band, condition, mask and reference schedules, so updates
pair with the shared-parity fits.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import iqp_cond as ciqp
import jax
import jax.numpy as jnp
import numpy as np
import optax
from iqp_cond import ConditionalModel
from iqp_cond.operators import OperatorBank, sample_nonidentity_masks_jax
from iqp_cond.training import TrainingResult

from .config import RunConfig
from .cw_followup_config import limits as followup_limits
from .cw_followup_config import protocol, selection_betas, stage_betas, stream_record
from .cw_followup_fit import (
    PROBABILITY_TOLERANCE,
    _file_digest,
    _verify_artifacts,
    digest,
    streams,
    write_json,
    write_npz,
)
from .cw_followup_metrics import curve_scores, select_checkpoint
from .diagonal_conditioning import (
    N_PHASES,
    evaluate,
    ghz_encoder,
    initial_weights,
    theta_of,
)
from .target import curie_weiss_target

_SEED_KEYS = {"batches", "bands", "conditions", "masks", "reference"}


def make_trainer(settings, anchors, terms):
    """``training.make_trainer`` with theta(beta_c) at the sampled anchor c."""
    learning_rate = float(settings["learning_rate"])
    b1, b2 = float(settings["b1"]), float(settings["b2"])
    eps = float(settings.get("eps", 1e-8))
    ratio = float(settings["final_learning_rate_ratio"])
    rng_seeds = settings["rng_seeds"]
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not (0 < b1 < 1 and 0 < b2 < 1 and eps > 0 and 0 < ratio <= 1):
        raise ValueError("invalid Adam settings")
    if set(rng_seeds) != _SEED_KEYS:
        raise ValueError("rng_seeds must have exactly the required keys")
    anchors = jnp.asarray(anchors, dtype=jnp.float64)

    def train(
        model,
        training_targets,
        *,
        config,
        initial_params,
        selection_metric,
        checkpoint_callback,
    ):
        if config.mask_refresh != "step":
            raise ValueError("sampled optimizer adapter requires step mask refresh")
        training_targets = np.asarray(training_targets, dtype=np.float64)
        initial = np.asarray(initial_params, dtype=np.float64)
        if (
            initial.shape != (terms * model.ansatz.n_parameters,)
            or not np.isfinite(initial).all()
        ):
            raise ValueError("invalid initial parameters")
        if (
            training_targets.ndim != 3
            or training_targets.shape[0] != model.encoder.n_conditions
            or training_targets.shape[0] != len(anchors)
            or training_targets.shape[2] != model.visible
            or training_targets.shape[1] < 2
            or not np.isin(training_targets, [0, 1]).all()
        ):
            raise ValueError("invalid target dimensions")
        sigmas = tuple(config.resolved_sigmas(model.visible))
        batches = np.random.default_rng(int(rng_seeds["batches"])).integers(
            0,
            training_targets.shape[1],
            size=(config.steps, training_targets.shape[0], config.batch_size),
        )
        bands = np.random.default_rng(int(rng_seeds["bands"])).integers(
            0, len(sigmas), size=config.steps
        )
        conditions = np.random.default_rng(int(rng_seeds["conditions"])).integers(
            0, model.encoder.n_conditions, size=config.steps
        )
        schedule = optax.cosine_decay_schedule(learning_rate, config.steps, alpha=ratio)
        optimizer = optax.adam(schedule, b1=b1, b2=b2, eps=eps)
        params = jnp.asarray(initial)
        state = optimizer.init(params)
        mask_key = jax.random.PRNGKey(int(rng_seeds["masks"]))
        reference_key = jax.random.PRNGKey(int(rng_seeds["reference"]))

        def compile_step(sigma):
            def loss(weights, beta, step_key, target_batch, encoder, bank):
                return (
                    ConditionalModel(model.ansatz, encoder, n_visible=model.n_visible)
                    .mmd(
                        theta_of(weights, beta, terms),
                        target_batch,
                        step_key,
                        n_samples=config.mc_samples,
                        shared_samples=True,
                        gemm_dtype=config.gemm_dtype,
                        estimator_dtype=config.estimator_dtype,
                        sample_chunk=config.sample_chunk,
                        bank=bank,
                    )
                    .loss
                )

            @jax.jit
            def step(
                weights, opt_state, beta, step_key, target_batch, encoder, mask_step_key
            ):
                masks, mask_weights = sample_nonidentity_masks_jax(
                    mask_step_key, config.n_ops, model.visible, sigma
                )
                bank = OperatorBank(
                    sigmas=(sigma,),
                    masks=masks[None, None],
                    weights=mask_weights[None, None],
                    n_ops=config.n_ops,
                    n_visible=model.visible,
                    n_masks=config.n_ops,
                )
                value, grad = jax.value_and_grad(loss)(
                    weights, beta, step_key, target_batch, encoder, bank
                )
                updates, opt_state = optimizer.update(grad, opt_state, weights)
                return value, optax.apply_updates(weights, updates), opt_state

            return step

        steps = tuple(compile_step(sigma) for sigma in sigmas)
        train_losses, checkpoint_steps, scores = [], [], []
        best = dict(params=params, step=0, score=float("inf"))
        started = time.perf_counter()

        def checkpoint(step, weights):
            score = float(selection_metric(np.asarray(weights)))
            if not np.isfinite(score) or not np.isfinite(np.asarray(weights)).all():
                raise FloatingPointError("nonfinite training checkpoint")
            checkpoint_steps.append(step)
            scores.append(score)
            if score < best["score"]:
                best.update(params=np.asarray(weights).copy(), step=step, score=score)
            checkpoint_callback(step, np.asarray(weights).copy())
            if step % 500 == 0:
                print(
                    f"sampled diagonal adam step {step}: "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

        checkpoint(0, params)
        for index in range(config.steps):
            condition = int(conditions[index])
            batch = jnp.asarray(
                training_targets[condition : condition + 1, batches[index, condition]]
            )
            value, params, state = steps[int(bands[index])](
                params,
                state,
                anchors[condition],
                jax.random.fold_in(reference_key, index),
                batch,
                model.encoder.select(condition),
                jax.random.fold_in(mask_key, index),
            )
            train_losses.append(value)
            if (
                index + 1
            ) % config.validation_interval == 0 or index + 1 == config.steps:
                checkpoint(index + 1, params)
        final = params
        if (
            not np.isfinite(np.asarray(final)).all()
            or not np.isfinite(np.asarray(train_losses)).all()
        ):
            raise FloatingPointError("nonfinite final optimization")
        return TrainingResult(
            seed=int(rng_seeds["reference"]),
            best_params=np.asarray(best["params"]),
            final_params=np.asarray(final),
            best_step=best["step"],
            best_validation_loss=float("nan"),
            best_selection_score=best["score"],
            train_steps=np.arange(1, config.steps + 1),
            train_losses=np.asarray(jnp.stack(train_losses)),
            validation_steps=np.asarray(checkpoint_steps),
            validation_losses=np.full(len(checkpoint_steps), np.nan),
            selection_scores=np.asarray(scores),
        )

    return train


def run_fit(job, root):
    """Fit one repeat, select once across every checkpoint and start."""
    root = Path(root)
    job = json.loads(json.dumps(job))
    directory = root / "fits" / digest(job)
    receipt = directory / "metrics.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["job"] != job:
            raise ValueError("existing fit identity differs")
        _verify_artifacts(directory, saved["artifact_sha256"])
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    job_path = directory / "job.json"
    if job_path.exists() and json.loads(job_path.read_text()) != job:
        raise ValueError("interrupted fit identity differs")
    write_json(job_path, job)
    spec = protocol(job["smoke"])
    budget = followup_limits(spec)
    stage, repeat = job["stage"], job["repeat"]
    rng_stage = job.get("seed_stage", stage)
    candidate = job["candidate"]
    terms = int(candidate["chebyshev_terms"])
    if job["family"].startswith("independent_") or job["beta"] is not None or terms < 1:
        raise ValueError("diagonal conditioning is one shared circuit")
    betas = spec["training"]["shared_anchors"]
    selecting_betas = selection_betas(spec)
    case = "shared"
    pool_records, pool_arrays = [], []
    for beta in betas:
        stream = stream_record(rng_stage, repeat, str(float(beta)), "target_pool")
        path = root / "pools" / (stream["sha256"] + ".npz")
        draws = curie_weiss_target(16, beta).sample(
            budget["n_train"], np.random.default_rng(stream["seed"])
        )
        pool_hash = hashlib.sha256(draws.tobytes()).hexdigest()
        if path.exists():
            with np.load(path) as previous:
                if not np.array_equal(previous["targets"], draws):
                    raise ValueError("paired pool differs from recorded draws")
        else:
            write_npz(path, targets=draws)
        pool_arrays.append(draws)
        pool_records.append(
            dict(
                beta=beta,
                stream=stream,
                sha256=pool_hash,
                draws=len(draws),
                path=str(path),
            )
        )
    targets = np.stack(pool_arrays).astype(np.float64)
    ratio = 1.0 if candidate["schedule"] == "constant" else 0.1
    if candidate["schedule"] not in ("constant", "cosine"):
        raise ValueError("unknown learning-rate schedule")
    config = RunConfig(
        study="cw_followup",
        encoder="parity",
        n_visible=16,
        n_train=budget["n_train"],
        n_validation=budget["n_train"],
        n_test=2,
        steps=budget["steps"],
        batch_size=budget["batch_size"],
        n_ops=budget["n_ops"],
        mc_samples=budget["mc_samples"],
        validation_n_ops=2,
        validation_mc_samples=2,
        bodyness_targets=tuple(candidate["expected_weights"]),
        learning_rate=candidate["learning_rate"],
        final_learning_rate=candidate["learning_rate"] * ratio,
        validation_interval=budget["validation_interval"],
        gemm_dtype="float32",
        sample_chunk=256,
        init_scale=candidate["init_scale"],
        restarts=budget["starts"][stage],
    )
    model = ciqp.ConditionalModel(ciqp.Ansatz.complete(16), ghz_encoder(len(betas)))
    if model.ansatz.n_parameters != N_PHASES:
        raise ValueError("unexpected ansatz size")
    checkpoints, parameters, stream_records, durations = [], [], [], []
    started = time.perf_counter()
    initial = None
    start = 0
    try:
        for start in range(config.restarts):
            stream = streams(rng_stage, repeat, case, start)
            stream_records.append(stream)
            start_receipt = directory / f"restart-{start}.json"
            start_archive = directory / f"restart-{start}.npz"
            if start_receipt.exists():
                saved = json.loads(start_receipt.read_text())
                if saved["stream"] != stream:
                    raise ValueError("saved restart stream differs")
                _verify_artifacts(directory, {start_archive.name: saved["sha256"]})
                with np.load(start_archive) as archive:
                    parameters.extend(archive["checkpoint_params"])
                checkpoints.extend(saved["checkpoints"])
                durations.append(saved["seconds"])
                continue
            initial = initial_weights(
                np.random.default_rng(stream["initialization"]["seed"]).normal(
                    0, 1, N_PHASES
                ),
                candidate["init_scale"],
                terms,
            )
            trainer = make_trainer(
                dict(
                    learning_rate=candidate["learning_rate"],
                    b1=candidate["b1"],
                    b2=candidate["b2"],
                    eps=1e-8,
                    final_learning_rate_ratio=ratio,
                    rng_seeds={
                        k: v["seed"] for k, v in stream.items() if k != "initialization"
                    },
                ),
                betas,
                terms,
            )
            cache = {}
            local_records, local_params = [], []

            def score(weights, cache=cache):
                report = evaluate(selecting_betas, weights, terms)
                metrics = curve_scores(
                    report["tvd"],
                    report["magnetization_error"],
                    report["heat_error"],
                    report["beta"],
                )
                if not np.isfinite([metrics["J"], metrics["P"]]).all():
                    raise FloatingPointError("nonfinite checkpoint objective")
                cache.update(
                    weights=np.asarray(weights).copy(), report=report, scores=metrics
                )
                return metrics["J"]

            def checkpoint(
                step,
                weights,
                cache=cache,
                start=start,
                score=score,
                local_records=local_records,
                local_params=local_params,
            ):
                if "weights" not in cache or not np.array_equal(
                    cache["weights"], weights
                ):
                    score(weights)
                record = dict(
                    step=int(step),
                    start=start,
                    **cache["scores"],
                    beta=cache["report"]["beta"].tolist(),
                    tvd=cache["report"]["tvd"].tolist(),
                    magnetization_error=cache["report"]["magnetization_error"].tolist(),
                    heat_error=cache["report"]["heat_error"].tolist(),
                )
                local_records.append(record)
                local_params.append(np.asarray(weights).copy())
                # Preserve partial states even if later updates fail or interrupt.
                write_npz(
                    directory / f"checkpoint-{start}.npz",
                    params=np.stack(local_params),
                    step=np.array([r["step"] for r in local_records]),
                    J=np.array([r["J"] for r in local_records]),
                    P=np.array([r["P"] for r in local_records]),
                )
                write_json(directory / f"checkpoint-{start}.json", local_records)

            tick = time.perf_counter()
            result = trainer(
                model,
                targets,
                config=config.training_config(),
                initial_params=initial,
                checkpoint_callback=checkpoint,
                selection_metric=score,
            )
            duration = time.perf_counter() - tick
            expected = list(
                range(0, budget["steps"] + 1, budget["validation_interval"])
            )
            if expected[-1] != budget["steps"]:
                expected.append(budget["steps"])
            if [r["step"] for r in local_records] != expected:
                raise ValueError("missing or duplicate training checkpoints")
            write_npz(
                start_archive,
                initial_params=initial,
                checkpoint_params=np.stack(local_params),
                final_params=result.final_params,
                train_losses=result.train_losses,
                train_steps=result.train_steps,
                checkpoint_steps=np.array(expected),
            )
            write_json(
                start_receipt,
                dict(
                    stream=stream,
                    checkpoints=local_records,
                    seconds=duration,
                    sha256=_file_digest(start_archive),
                ),
            )
            parameters.extend(local_params)
            checkpoints.extend(local_records)
            durations.append(duration)
        selected = select_checkpoint(checkpoints)
        index = next(i for i, r in enumerate(checkpoints) if r is selected)
        params = parameters[index]
        report = evaluate(stage_betas(spec, stage), params, terms)
        scores = curve_scores(
            report["tvd"],
            report["magnetization_error"],
            report["heat_error"],
            report["beta"],
        )
        write_npz(directory / "evaluation.npz", **report)
        if stage == "final":
            write_npz(
                directory / "markers.npz", **evaluate((0.4, 1.0, 2.0), params, terms)
            )
        write_npz(
            directory / "parameters.npz",
            best_params=params,
            checkpoint_params=np.stack(parameters),
            checkpoint_step=np.array([r["step"] for r in checkpoints]),
            checkpoint_start=np.array([r["start"] for r in checkpoints]),
            checkpoint_J=np.array([r["J"] for r in checkpoints]),
            checkpoint_P=np.array([r["P"] for r in checkpoints]),
        )
        summary = dict(
            status="ok",
            **scores,
            ranking_score=scores["J"],
            selection_J=selected["J"],
            selection_P=selected["P"],
            selected_start=selected["start"],
            selected_step=selected["step"],
            checkpoints=checkpoints,
            metrics_curves={k: np.asarray(v).tolist() for k, v in report.items()},
        )
    except FloatingPointError as error:
        if initial is not None:
            write_npz(directory / "failed-start.npz", initial_params=initial)
        summary = dict(
            status="numerical_failure",
            J=1.0,
            P=None,
            ranking_score=1.0,
            search_penalty_not_measured_tvd=True,
            error=str(error),
            failed_restart=start,
        )
    summary.update(
        job=job,
        config=config.as_dict(),
        model=dict(
            conditioning="diagonal, theta(beta)=sum_k T_k(beta-1) w_k",
            chebyshev_terms=terms,
            n_parameters=terms * N_PHASES,
            input="parity encoder at fixed t=pi/2, condition independent",
        ),
        target_pools=pool_records,
        streams=stream_records,
        restart_seconds=durations,
        wall_seconds=time.perf_counter() - started,
        backend=jax.default_backend(),
        devices=[str(x) for x in jax.devices()],
        probability_tolerance=PROBABILITY_TOLERANCE,
        directory=str(directory),
        auxiliary_validation_mmd="disabled; no validation or test pools generated",
        artifact_sha256={
            p.name: _file_digest(p) for p in sorted(directory.glob("*.npz"))
        },
    )
    summary = json.loads(json.dumps(summary, allow_nan=False))
    write_json(receipt, summary)
    return summary


def fit_with_retry(job, root):
    """``cw_followup_hpo._fit_with_retry`` around the diagonal ``run_fit``."""
    root = Path(root)
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

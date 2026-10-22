"""Recorded likelihood/ELBO fits for the accepted classical CW comparison."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

from .cw_classical_config import (
    CAPACITIES,
    PROTOCOL_PATH,
    betas,
    classical_stream,
    limits,
    model,
    protocol,
    target_stream,
)
from .cw_classical_evaluation import optimized_probabilities
from .cw_followup_config import MARKERS
from .cw_followup_fit import _target_arrays, _verify_artifacts, write_json, write_npz
from .cw_followup_metrics import curve_scores, select_checkpoint
from .provenance import source_identity as recorded_identity
from .target import curie_weiss_target


def source_identity():
    jax.config.update("jax_enable_x64", True)
    identity = recorded_identity(PROTOCOL_PATH, protocol()["protocol_sha256"])
    identity["actual_jax_enable_x64"] = bool(jax.config.jax_enable_x64)
    return identity


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probabilities(family, params, beta, spec):
    n = spec["n"]
    settings = protocol()["evaluation"]["vae_quadrature"]
    q, metadata = optimized_probabilities(family, params, beta, n, settings)
    q = np.asarray(q)
    tolerance = protocol()["evaluation"]["probability_tolerance"]
    if not np.isfinite(q).all() or np.any(q < 0) or abs(q.sum() - 1) > tolerance:
        raise FloatingPointError("invalid classical probability normalization")
    if np.max(abs(q - q[::-1])) > tolerance:
        raise FloatingPointError("classical global spin-flip symmetry differs")
    return q, metadata


def evaluate(family, params, grid, budget):
    grid = tuple(float(x) for x in grid)
    p, k, target_shells = _target_arrays(grid, budget["n"])
    laws, quadrature = zip(
        *(probabilities(family, params, beta, budget) for beta in grid), strict=True
    )
    q = np.stack(laws)
    n = budget["n"]
    shells = np.stack([np.bincount(k, weights=row, minlength=n + 1) for row in q])
    m = (2 * np.arange(n + 1) - n) / n
    energies = -n * m**2 / 2
    magnetization = shells @ abs(m)
    heat = np.asarray(grid) ** 2 / n * (shells @ energies**2 - (shells @ energies) ** 2)
    target_heat = (
        np.asarray(grid) ** 2
        / n
        * (target_shells @ energies**2 - (target_shells @ energies) ** 2)
    )
    curves = dict(
        beta=grid,
        tvd=abs(q - p).sum(1) / 2,
        profile_tvd=abs(shells - target_shells).sum(1) / 2,
        shells=shells,
        target_shells=target_shells,
        magnetization=magnetization,
        magnetization_error=abs(magnetization - target_shells @ abs(m)),
        heat=heat,
        target_heat=target_heat,
        heat_error=abs(heat - target_heat),
        signed_magnetization=shells @ m,
        flip_defect=abs(q - q[:, ::-1]).sum(1) / 2,
    )
    # Tiny cancellation can make a zero fluctuation negative at machine precision.
    for name in ("heat", "target_heat"):
        curves[name] = np.maximum(curves[name], 0)
    curves["heat_error"] = abs(curves["heat"] - curves["target_heat"])
    return {key: np.asarray(value).tolist() for key, value in curves.items()}, list(
        quadrature
    )


def score(curves):
    return curve_scores(
        curves["tvd"],
        curves["magnetization_error"],
        curves["heat_error"],
        curves["beta"],
    )


def _pools(root, stage, repeat, budget):
    arrays, records = [], []
    for beta in protocol()["training"]["shared_anchors"]:
        stream = target_stream(stage, repeat, beta)
        draws = curie_weiss_target(budget["n"], beta).sample(
            budget["n_train"], np.random.default_rng(stream["seed"])
        )
        path = root / "pools" / (stream["sha256"] + ".npz")
        if path.exists():
            with np.load(path) as archive:
                if not np.array_equal(archive["targets"], draws):
                    raise ValueError("paired target pool differs")
        else:
            write_npz(path, targets=draws)
        arrays.append(draws)
        records.append(
            dict(
                beta=beta,
                stream=stream,
                draws=len(draws),
                array_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),
                path=str(path.relative_to(root)),
                artifact_sha256=_hash(path),
            )
        )
    return np.stack(arrays), records


def _updater(module, optimizer):
    @jax.jit
    def update(parameters, optimizer_state, x, beta, key):
        value, gradient = jax.value_and_grad(module.loss)(parameters, x, beta, key)
        updates, next_state = optimizer.update(gradient, optimizer_state, parameters)
        next_params = optax.apply_updates(parameters, updates)
        finite = jnp.isfinite(value)
        for leaf in jax.tree_util.tree_leaves((gradient, next_params)):
            finite = finite & jnp.all(jnp.isfinite(leaf))
        return next_params, next_state, value, finite

    return update


def run_fit(job, root):
    """Select once over every scheduled checkpoint and start; never mix fits."""
    from .cw_classical_config import digest

    job = json.loads(json.dumps(job))
    if job["capacity"] not in CAPACITIES:
        raise ValueError("unknown classical capacity")
    if source_identity() != job["identity"]:
        raise ValueError("classical fit source/environment identity differs")
    root = Path(root)
    directory = root / "fits" / digest(job)
    receipt = directory / "metrics.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["job"] != job:
            raise ValueError("completed fit identity differs")
        _verify_artifacts(directory, saved["artifact_sha256"])
        for pool in saved.get("target_pools", []):
            if _hash(root / pool["path"]) != pool["artifact_sha256"]:
                raise ValueError("completed paired target pool differs")
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    job_path = directory / "job.json"
    if job_path.exists() and json.loads(job_path.read_text()) != job:
        raise ValueError("interrupted fit identity differs")
    write_json(job_path, job)
    module = model(job["family"])
    budget = limits(job["smoke"])
    stage, repeat = job["stage"], job["repeat"]
    pools, pool_records = _pools(root, stage, repeat, budget)
    config = job["candidate"]
    architecture = module.architecture(budget["n"], job["capacity"])
    steps = list(range(0, budget["steps"] + 1, budget["checkpoint_interval"]))
    all_checkpoints, all_parameters, start_records = [], [], []
    selecting = betas(stage, job["smoke"], selection=True)
    anchors = protocol()["training"]["shared_anchors"]
    initialized = module.init(jax.random.key(0), budget["n"], job["capacity"])
    _, unflatten = ravel_pytree(initialized)
    try:
        for start in range(budget["starts"][stage]):
            basename = f"start-{start}"
            archive_path, manifest_path = (
                directory / (basename + ".npz"),
                directory / (basename + ".json"),
            )
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest["job"] != job or _hash(archive_path) != manifest["sha256"]:
                    raise ValueError("completed start artifact differs")
                with np.load(archive_path) as archive:
                    all_parameters.extend(archive["parameters"])
                all_checkpoints.extend(manifest["checkpoints"])
                start_records.append(manifest)
                continue
            streams = {
                purpose: classical_stream(stage, repeat, start, purpose)
                for purpose in ("initialization", "batches", "conditions", "latent")
            }
            params = module.init(
                jax.random.key(streams["initialization"]["seed"]),
                budget["n"],
                job["capacity"],
                config["init_scale"],
            )
            if module.parameter_count(params) != architecture["parameter_count"]:
                raise ValueError("architecture parameter count differs")
            learning_rate = config["learning_rate"]
            if config["schedule"] == "cosine":
                learning_rate = optax.cosine_decay_schedule(
                    learning_rate, budget["steps"], alpha=0.1
                )
            elif config["schedule"] != "constant":
                raise ValueError("unknown optimizer schedule")
            optimizer = optax.adam(learning_rate, b1=config["b1"], b2=config["b2"])
            state = optimizer.init(params)

            update = _updater(module, optimizer)
            batch_rng = np.random.default_rng(streams["batches"]["seed"])
            condition_rng = np.random.default_rng(streams["conditions"]["seed"])
            latent_key = jax.random.key(streams["latent"]["seed"])
            snapshots, checkpoints, losses = [], [], []
            training_seconds, evaluation_seconds = 0.0, 0.0
            for step in range(budget["steps"] + 1):
                if step in steps:
                    timer = time.perf_counter()
                    curves, quadrature = evaluate(
                        job["family"], params, selecting, budget
                    )
                    evaluation_seconds += time.perf_counter() - timer
                    values = score(curves)
                    if not all(np.isfinite(v) for v in values.values()):
                        raise FloatingPointError("nonfinite classical checkpoint score")
                    flat, _ = ravel_pytree(params)
                    snapshots.append(np.asarray(flat))
                    checkpoints.append(
                        dict(
                            step=step,
                            start=start,
                            **values,
                            curves=curves,
                            probability_evaluation=quadrature,
                        )
                    )
                if step == budget["steps"]:
                    break
                anchor = int(condition_rng.integers(len(anchors)))
                indices = batch_rng.integers(
                    budget["n_train"], size=budget["batch_size"]
                )
                timer = time.perf_counter()
                params, state, loss_value, finite = update(
                    params,
                    state,
                    pools[anchor, indices].astype(np.float64),
                    jnp.asarray(anchors[anchor]),
                    jax.random.fold_in(latent_key, step),
                )
                if not bool(finite):
                    raise FloatingPointError(
                        "nonfinite classical loss/gradient/parameters"
                    )
                losses.append(float(loss_value))
                training_seconds += time.perf_counter() - timer
            write_npz(
                archive_path, parameters=np.stack(snapshots), losses=np.array(losses)
            )
            manifest = dict(
                job=job,
                checkpoints=checkpoints,
                streams=streams,
                sha256=_hash(archive_path),
                training_seconds_including_compile=training_seconds,
                checkpoint_evaluation_seconds=evaluation_seconds,
            )
            write_json(manifest_path, manifest)
            all_checkpoints.extend(checkpoints)
            all_parameters.extend(snapshots)
            start_records.append(manifest)
        selected = select_checkpoint(all_checkpoints)
        index = next(i for i, row in enumerate(all_checkpoints) if row is selected)
        selected_flat = all_parameters[index]
        params = unflatten(jnp.asarray(selected_flat))
        curves, diagnostics = evaluate(
            job["family"], params, betas(stage, job["smoke"]), budget
        )
        markers, marker_diagnostics = (None, None)
        timing = None
        if stage == "final":
            markers, marker_diagnostics = evaluate(
                job["family"], params, MARKERS, budget
            )
            sample_stream = classical_stream(
                "final", repeat, job["family"], job["capacity"], "samples"
            )
            sampling = jax.jit(
                lambda p, key: module.sample(p, key, 1.0, budget["sampling_draws"])
            )
            timer = time.perf_counter()
            sampling(params, jax.random.key(sample_stream["seed"])).block_until_ready()
            cold = time.perf_counter() - timer
            timer = time.perf_counter()
            draws = sampling(
                params, jax.random.fold_in(jax.random.key(sample_stream["seed"]), 1)
            )
            draws.block_until_ready()
            elapsed = time.perf_counter() - timer
            write_npz(directory / "samples.npz", samples=np.asarray(draws))
            timing = dict(
                beta=1.0,
                draws=budget["sampling_draws"],
                warm_seconds=elapsed,
                samples_per_second=budget["sampling_draws"] / elapsed,
                first_batch_seconds_including_compile=cold,
                stream=sample_stream,
            )
        write_npz(directory / "selected.npz", parameters=selected_flat)
        if source_identity() != job["identity"]:
            raise ValueError("classical source/environment changed during fit")
        result = dict(
            status="ok",
            job=job,
            directory=str(directory.relative_to(root)),
            architecture=architecture,
            target_pools=pool_records,
            selected_start=selected["start"],
            selected_step=selected["step"],
            selection_J=selected["J"],
            selection_P=selected["P"],
            metrics_curves=curves,
            scores=score(curves),
            probability_evaluation=diagnostics,
            marker_curves=markers,
            marker_probability_evaluation=marker_diagnostics,
            sampling=timing,
            starts=start_records,
        )
    except FloatingPointError as error:
        result = dict(
            status="numerical_failure",
            job=job,
            directory=str(directory.relative_to(root)),
            error=str(error),
            target_pools=pool_records,
            architecture=architecture,
            search_penalty_not_measured_tvd=True,
        )
    result["artifact_sha256"] = {
        path.name: _hash(path)
        for path in sorted(directory.iterdir())
        if path.suffix in (".json", ".npz") and path.name != "metrics.json"
    }
    write_json(receipt, result)
    return result

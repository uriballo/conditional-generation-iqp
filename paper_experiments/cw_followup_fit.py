"""Recorded fits with recoverable global checkpoint selection for CW follow-up."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from functools import lru_cache
from pathlib import Path

import iqp_cond as ciqp
import jax
import jax.numpy as jnp
import numpy as np

from .config import RunConfig
from .cw_followup_config import limits as followup_limits
from .cw_followup_config import (
    protocol,
    selection_betas,
    stage_betas,
    stream_record,
)
from .cw_followup_metrics import curve_scores, select_checkpoint
from .parity_encoder import ParityEncoder
from .phase_encoder import BroadcastPhaseEncoder
from .target import curie_weiss_target
from .training import make_trainer

PROBABILITY_TOLERANCE = 1e-8


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def limits(smoke=False):
    spec = protocol(smoke)
    return spec["runtime"] if smoke else followup_limits(spec)


def encoder(family, betas, scale=0.75, n=16):
    if family in {"shared_parity", "independent_parity"}:
        return ParityEncoder.from_betas(betas, n, scale=scale)
    if family == "independent_unconditional":
        return ciqp.AngleEncoder(np.zeros((len(betas), 1, n)), axes=("Y" * n,))
    if family != "shared_product":
        raise ValueError("unknown follow-up family")
    return BroadcastPhaseEncoder.from_betas(betas, n, beta_scale=scale)


@lru_cache(maxsize=3)
def _exact_kernel(n):
    """Diagonal phases followed by an orthonormal Walsh--Hadamard transform.

    Uses the public ansatz generator bank and encoder reference amplitudes.
    Little-endian bit order agrees with the public probability simulator.
    """
    bits = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int32)
    generators = np.asarray(ciqp.Ansatz.complete(n).generators, dtype=np.int32)
    characters = jnp.asarray(1 - 2 * ((bits @ generators.T) % 2), dtype=jnp.float64)

    @jax.jit
    def evaluate(theta, reference):
        state = reference * jnp.exp(1j * (characters @ theta))[None, :]
        batch = reference.shape[0]
        for bit in range(n):
            width = 1 << bit
            blocks = state.reshape(batch, -1, 2, width)
            left, right = blocks[:, :, 0, :], blocks[:, :, 1, :]
            state = jnp.stack((left + right, left - right), axis=2).reshape(batch, -1)
        return jnp.abs(state) ** 2 / (1 << n)

    return evaluate


def reference_amplitudes(enc):
    if isinstance(enc, BroadcastPhaseEncoder):
        n = enc.n_qubits
        k = np.array([int(x).bit_count() for x in range(1 << n)])
        return enc.amp_zero[:, None] ** (n - k) * enc.amp_one[:, None] ** k
    return enc.post_hadamard_amplitudes()


def exact_probabilities(family, betas, params, scale=0.75, n=16):
    rows = []
    for start in range(0, len(betas), 4):
        enc = encoder(family, betas[start : start + 4], scale, n)
        value = np.asarray(
            _exact_kernel(n)(jnp.asarray(params), reference_amplitudes(enc))
        )
        if (
            not np.isfinite(value).all()
            or np.any(value < -PROBABILITY_TOLERANCE)
            or not np.allclose(value.sum(1), 1, atol=PROBABILITY_TOLERANCE, rtol=0)
        ):
            raise FloatingPointError("invalid exact model probabilities")
        rows.append(value)
    return np.concatenate(rows)


@lru_cache(maxsize=12)
def _target_arrays(betas, n=16):
    bits = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    k = bits.sum(1)
    laws = [curie_weiss_target(n, beta) for beta in betas]
    p = np.stack([law.bitstring_probabilities(bits) for law in laws])
    return p, k, np.stack([law.probabilities for law in laws])


def evaluate(family, betas, params, scale=0.75, n=16):
    betas = tuple(float(b) for b in betas)
    q = exact_probabilities(family, betas, params, scale, n)
    p, k, target_shells = _target_arrays(betas, n)
    tvd = np.abs(q - p).sum(1) / 2
    shells = np.stack([np.bincount(k, weights=row, minlength=n + 1) for row in q])
    absolute_m = np.abs((2 * np.arange(n + 1) - n) / n)
    magnetization = shells @ absolute_m
    # H=-n m^2/2, so C=beta^2 Var(H)/n.  Compute both model and exact target.
    m2 = ((2 * np.arange(n + 1) - n) / n) ** 2
    energies = -n * m2 / 2
    heat = (
        np.asarray(betas) ** 2 * (shells @ (energies**2) - (shells @ energies) ** 2) / n
    )
    target_heat = (
        np.asarray(betas) ** 2
        * (target_shells @ (energies**2) - (target_shells @ energies) ** 2)
        / n
    )
    return dict(
        beta=np.asarray(betas),
        tvd=tvd,
        shells=shells,
        target_shells=target_shells,
        magnetization=magnetization,
        magnetization_error=abs(magnetization - target_shells @ absolute_m),
        heat=heat,
        target_heat=target_heat,
        heat_error=abs(heat - target_heat),
        signed_magnetization=shells @ ((2 * np.arange(n + 1) - n) / n),
        flip_defect=np.abs(q - q[:, ::-1]).sum(1) / 2,
    )


def streams(stage, repeat, case, restart):
    base = (stage, int(repeat), case, int(restart))
    return {
        purpose: stream_record(*base, purpose)
        for purpose in (
            "initialization",
            "batches",
            "bands",
            "conditions",
            "masks",
            "reference",
        )
    }


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_artifacts(directory, hashes):
    for name, expected in hashes.items():
        path = directory / name
        if not path.exists() or _file_digest(path) != expected:
            raise ValueError("recorded fit artifact is missing or differs: " + name)


def run_fit(job, root):
    """Fit one repeat/circuit, select once across every checkpoint and start."""
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
    family, stage, repeat = job["family"], job["stage"], job["repeat"]
    rng_stage = job.get("seed_stage", stage)
    candidate = job["candidate"]
    scale = candidate.get("beta_scale", 0.75)
    direct = family.startswith("independent_")
    case_beta = float(job["beta"]) if direct else None
    betas = [case_beta] if direct else spec["training"]["shared_anchors"]
    selecting_betas = selection_betas(spec, case_beta)
    case = str(case_beta) if direct else "shared"
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
        encoder="parity"
        if "parity" in family
        else ("unconditional" if direct else "broadcast"),
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
    model = ciqp.ConditionalModel(
        ciqp.Ansatz.complete(16), encoder(family, betas, scale)
    )
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
            initial = (
                np.random.default_rng(stream["initialization"]["seed"]).normal(
                    0, 1, 136
                )
                * candidate["init_scale"]
            )
            trainer = make_trainer(
                dict(
                    name="adam",
                    learning_rate=candidate["learning_rate"],
                    b1=candidate["b1"],
                    b2=candidate["b2"],
                    eps=1e-8,
                    final_learning_rate_ratio=ratio,
                    compute_validation_mmd=False,
                    rng_seeds={
                        k: v["seed"] for k, v in stream.items() if k != "initialization"
                    },
                )
            )
            cache = {}
            local_records, local_params = [], []

            def score(theta, cache=cache):
                report = evaluate(family, selecting_betas, theta, scale)
                metrics = curve_scores(
                    report["tvd"],
                    report["magnetization_error"],
                    report["heat_error"],
                    report["beta"],
                )
                if not np.isfinite([metrics["J"], metrics["P"]]).all():
                    raise FloatingPointError("nonfinite checkpoint objective")
                cache.update(
                    theta=np.asarray(theta).copy(), report=report, scores=metrics
                )
                return metrics["J"]

            def checkpoint(
                step,
                theta,
                cache=cache,
                start=start,
                score=score,
                local_records=local_records,
                local_params=local_params,
            ):
                if "theta" not in cache or not np.array_equal(cache["theta"], theta):
                    score(theta)
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
                local_params.append(np.asarray(theta).copy())
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
                targets,
                seed=stream["reference"]["seed"],
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
        report = evaluate(family, stage_betas(spec, stage, case_beta), params, scale)
        scores = curve_scores(
            report["tvd"],
            report["magnetization_error"],
            report["heat_error"],
            report["beta"],
        )
        write_npz(directory / "evaluation.npz", **report)
        if stage == "final" and not direct:
            write_npz(
                directory / "markers.npz",
                **evaluate(family, (0.4, 1.0, 2.0), params, scale),
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

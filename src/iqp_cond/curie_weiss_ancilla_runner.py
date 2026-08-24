"""Training and exact visible-marginal evaluation for the ancilla campaign."""

from __future__ import annotations

import hashlib
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ancilla import (
    ancilla_ansatz,
    exact_ancilla_marginal_probabilities,
    exact_ancilla_marginal_probabilities_from_full_angles,
    pad_observables_for_ancillas,
)
from .artifacts import sha256_file
from .curie_weiss_ancilla_campaign import (
    CurieWeissAncillaCampaignStore,
    CurieWeissAncillaResultBundle,
)
from .curie_weiss_ancilla_protocol import (
    ENCODING_COMPARISON_VERSION,
    CurieWeissAncillaFamily,
    CurieWeissAncillaManifest,
    CurieWeissAncillaRunIdentity,
    broadcast_angle,
    plan_curie_weiss_ancilla_runs,
    role_separated_angle,
)
from .curie_weiss_data import exact_curie_weiss_target
from .curie_weiss_statistics import thermodynamic_metrics
from .encoding import ANGLE_ENCODING
from .exact import enumerate_bitstrings, shell_probabilities
from .gaussian_data import write_immutable_npz
from .gaussian_runner import exact_mmd_squared
from .ising_training import runtime_environment
from .mmd import _target_terms, mmd_mask_weights, mmd_operator_probability
from .simulator import _estimate_pauli_z_unchecked
from .training import TrainingConfig, _batch_schedule

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True, slots=True)
class AncillaTrainingResult:
    generators: np.ndarray
    best_params: np.ndarray
    final_params: np.ndarray
    best_step: int
    best_validation_loss: float
    validation_steps: np.ndarray
    validation_losses: np.ndarray


@dataclass(frozen=True, slots=True)
class ReusedDataset:
    training: np.ndarray
    validation: np.ndarray
    path: Path
    sha256: str


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_digest(repository: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "configs/curie_weiss_ancilla_foundations_v1.json",
        "configs/curie_weiss_m5_foundations_v1.json",
        "configs/curie_weiss_m5_foundations_v2.json",
        "configs/curie_weiss_m5_foundations_v3.json",
        "configs/curie_weiss_encoding_comparison_v1.json",
        "src/iqp_cond/curie_weiss_ancilla_protocol.py",
        "src/iqp_cond/curie_weiss_ancilla_campaign.py",
        "src/iqp_cond/curie_weiss_ancilla_runner.py",
        "src/iqp_cond/curie_weiss_ancilla_report.py",
        "src/iqp_cond/ancilla.py",
        "src/iqp_cond/mmd.py",
        "src/iqp_cond/simulator.py",
        "src/iqp_cond/exact.py",
    ):
        path = repository / relative
        if path.exists():
            digest.update(relative.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def load_reused_dataset(
    repository: Path,
    manifest: CurieWeissAncillaManifest,
    run: CurieWeissAncillaRunIdentity,
) -> ReusedDataset:
    source = manifest.source_for_seed(run.optimizer_seed)
    path = repository / source.path
    if sha256_file(path) != source.sha256:
        raise ValueError("declared dataset SHA-256 differs")
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            "manifest_digest",
            "n_qubits",
            "optimizer_seed",
            "training_betas",
            "training_seeds",
            "validation_seeds",
            "training",
            "validation",
        }
        if set(archive.files) != expected:
            raise ValueError("dataset archive contains undeclared roles")
        if (
            int(archive["n_qubits"]) != 8
            or int(archive["optimizer_seed"]) != run.optimizer_seed
        ):
            raise ValueError("dataset coordinates differ")
        if not np.array_equal(archive["training_betas"], manifest.training_betas):
            raise ValueError("dataset temperature grid differs")
        training = np.asarray(archive["training"], dtype=np.int8)
        validation = np.asarray(archive["validation"], dtype=np.int8)
    if run.family is CurieWeissAncillaFamily.TEMPERATURE_REFERENCE:
        index = manifest.training_betas.index(float(run.reference_beta))
        training, validation = training[[index]], validation[[index]]
    return ReusedDataset(training, validation, path, source.sha256)


def _generator_identity_initialization(
    seed: int, generators: np.ndarray
) -> jnp.ndarray:
    values = []
    scale = 0.1 / np.sqrt(153.0)
    for generator in np.asarray(generators, dtype=np.int8):
        support = tuple(int(index) for index in np.flatnonzero(generator))
        payload = f"{seed}|{support}".encode()
        derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        values.append(np.random.default_rng(derived).normal(0.0, scale))
    return jnp.asarray(values, dtype=jnp.float64)


def estimator_seed(run: CurieWeissAncillaRunIdentity) -> int:
    if run.family is not CurieWeissAncillaFamily.TEMPERATURE_REFERENCE:
        return run.optimizer_seed
    payload = f"{run.optimizer_seed}|reference|{run.reference_beta:.1f}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def training_config(
    manifest: CurieWeissAncillaManifest,
    run: CurieWeissAncillaRunIdentity,
    *,
    updates: int | None = None,
    lightweight: bool = False,
) -> TrainingConfig:
    settings = manifest.raw["training"]
    steps = int(settings["updates"] if updates is None else updates)
    return TrainingConfig(
        steps=steps,
        batch_size=450
        if run.family is CurieWeissAncillaFamily.TEMPERATURE_REFERENCE
        else 75,
        learning_rate=float(settings["initial_learning_rate"]),
        final_learning_rate=float(settings["final_learning_rate"]),
        validation_interval=min(int(settings["validation_interval"]), steps),
        sigmas=tuple(float(v) for v in settings["bandwidths"]),
        n_ops=256,
        n_iqp_samples=16 if lightweight else int(settings["training_model_samples"]),
        validation_n_ops=256,
        validation_n_iqp_samples=16
        if lightweight
        else int(settings["validation_model_samples"]),
    )


def condition_angles(
    manifest: CurieWeissAncillaManifest,
    run: CurieWeissAncillaRunIdentity,
    *,
    evaluation: bool,
) -> np.ndarray:
    if evaluation:
        betas = (
            (float(run.reference_beta),)
            if run.family is CurieWeissAncillaFamily.TEMPERATURE_REFERENCE
            else manifest.evaluation_betas
        )
    else:
        betas = (
            (float(run.reference_beta),)
            if run.family is CurieWeissAncillaFamily.TEMPERATURE_REFERENCE
            else manifest.training_betas
        )
    if run.family is CurieWeissAncillaFamily.NO_CONDITION:
        betas = (1.0,) * len(betas)
    if manifest.campaign_version == ENCODING_COMPARISON_VERSION:
        return np.asarray(
            [role_separated_angle(manifest, beta, run.family) for beta in betas]
        )
    return np.asarray(
        [broadcast_angle(manifest, beta, run.total_qubits) for beta in betas]
    )


def _full_mask_mmd(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    angles: jnp.ndarray,
    targets: jnp.ndarray,
    sigmas: tuple[float, ...],
    n_samples: int,
    key: jax.Array,
    data_masks: jnp.ndarray,
    model_masks: jnp.ndarray,
    weights: jnp.ndarray,
    encoding: str = ANGLE_ENCODING,
) -> jnp.ndarray:
    losses = []
    for sigma, model_key in zip(
        sigmas, jax.random.split(key, len(sigmas)), strict=True
    ):
        model = _estimate_pauli_z_unchecked(
            params, generators, model_masks, angles, model_key, n_samples, encoding
        )
        target_means, target_squares = _target_terms(targets, data_masks)
        sums = jnp.sum(model.sample_values, axis=1)
        square_sums = jnp.sum(model.sample_values**2, axis=1)
        model_squares = (sums**2 - square_sums) / (n_samples * (n_samples - 1))
        per_mask = model_squares - 2.0 * model.values * target_means + target_squares
        probability = mmd_operator_probability(float(sigma))
        law = probability**weights * (1.0 - probability) ** (8 - weights)
        losses.append(jnp.mean(jnp.sum(law * per_mask, axis=-1)))
    return jnp.mean(jnp.stack(losses))


def train_model(
    manifest: CurieWeissAncillaManifest,
    run: CurieWeissAncillaRunIdentity,
    dataset: ReusedDataset,
    config: TrainingConfig,
    encoding: str = ANGLE_ENCODING,
) -> AncillaTrainingResult:
    ansatz = ancilla_ansatz(8, run.ancillas)
    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    data_masks = jnp.asarray(enumerate_bitstrings(8), dtype=jnp.float64)
    model_masks = jnp.asarray(pad_observables_for_ancillas(data_masks, run.ancillas))
    weights = mmd_mask_weights(8)
    angles = jnp.asarray(condition_angles(manifest, run, evaluation=False))
    params = _generator_identity_initialization(run.optimizer_seed, ansatz.generators)
    schedule = optax.cosine_decay_schedule(
        config.learning_rate,
        config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    state = optimizer.init(params)
    stream_seed = estimator_seed(run)
    batches = _batch_schedule(
        dataset.training.shape[0], dataset.training.shape[1], config, stream_seed
    )

    def loss(current, targets, key, samples):
        return _full_mask_mmd(
            current,
            generators,
            angles,
            targets,
            config.sigmas,
            samples,
            key,
            data_masks,
            model_masks,
            weights,
            encoding,
        )

    value_gradient = jax.jit(jax.value_and_grad(loss), static_argnums=(3,))
    validate = jax.jit(
        lambda current: loss(
            current,
            jnp.asarray(dataset.validation),
            jax.random.PRNGKey(stream_seed + 1_000_000),
            config.validation_n_iqp_samples,
        )
    )
    best, final = params, params
    best_step = 0
    best_validation = float(validate(params))
    validation_steps, validation_losses = [0], [best_validation]
    base_key = jax.random.PRNGKey(stream_seed)
    for step in range(config.steps):
        indices = batches[step]
        batch = np.stack(
            [
                dataset.training[index, indices[index]]
                for index in range(dataset.training.shape[0])
            ]
        )
        _, gradient = value_gradient(
            final,
            jnp.asarray(batch),
            jax.random.fold_in(base_key, step),
            config.n_iqp_samples,
        )
        updates, state = optimizer.update(gradient, state, final)
        final = optax.apply_updates(final, updates)
        completed = step + 1
        if completed % config.validation_interval == 0:
            current = float(validate(final))
            validation_steps.append(completed)
            validation_losses.append(current)
            if current < best_validation:
                best, best_step, best_validation = final, completed, current
    return AncillaTrainingResult(
        np.asarray(ansatz.generators, dtype=np.int8),
        np.asarray(best),
        np.asarray(final),
        best_step,
        best_validation,
        np.asarray(validation_steps),
        np.asarray(validation_losses),
    )


def evaluate_run(
    manifest: CurieWeissAncillaManifest,
    run: CurieWeissAncillaRunIdentity,
    params: np.ndarray,
    generators: np.ndarray,
    encoding: str = ANGLE_ENCODING,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    betas = (
        np.asarray([run.reference_beta], dtype=np.float64)
        if run.family is CurieWeissAncillaFamily.TEMPERATURE_REFERENCE
        else np.asarray(manifest.evaluation_betas)
    )
    full_angles = condition_angles(manifest, run, evaluation=True)
    if manifest.campaign_version == ENCODING_COMPARISON_VERSION:
        probabilities = exact_ancilla_marginal_probabilities_from_full_angles(
            params,
            generators,
            full_angles,
            n_data_qubits=8,
            n_ancillas=run.ancillas,
            encoding=encoding,
        )
    else:
        data_angles = full_angles[:, :8]
        probabilities = exact_ancilla_marginal_probabilities(
            params,
            generators,
            data_angles,
            n_data_qubits=8,
            n_ancillas=run.ancillas,
            scheme="broadcast_all",
            encoding=encoding,
        )
    bitstrings = enumerate_bitstrings(8)
    rows, targets = [], []
    for beta, model in zip(betas, probabilities, strict=True):
        target = exact_curie_weiss_target(8, float(beta))
        metrics = thermodynamic_metrics(target, model).to_dict()
        metrics.update(
            {
                "beta": float(beta),
                "role": "training" if beta in manifest.training_betas else "held_out",
                "exact_mmd": exact_mmd_squared(model, target.probabilities),
                "exact_mmd_by_bandwidth": {
                    str(sigma): exact_mmd_squared(model, target.probabilities, (sigma,))
                    for sigma in (1.0, 2.0, 4.0)
                },
                "target_profile": target.profile.tolist(),
                "model_profile": shell_probabilities(model, bitstrings).tolist(),
            }
        )
        rows.append(metrics)
        targets.append(target.probabilities)
    return (
        {"evaluation_rows": rows, "performance_thresholds_applied": False},
        {
            "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
            "run_id": np.asarray(run.run_id, dtype="S64"),
            "betas": betas,
            "condition_angles": full_angles,
            "model_probabilities": probabilities,
            "target_probabilities": np.asarray(targets),
        },
    )


def _portable(path: Path, repository: Path) -> str:
    return str(path.resolve().relative_to(repository.resolve()))


def execute_run(
    repository: Path,
    manifest: CurieWeissAncillaManifest,
    store: CurieWeissAncillaCampaignStore,
    run: CurieWeissAncillaRunIdentity,
    *,
    invocation: tuple[str, ...],
    updates: int | None = None,
    require_preflight: bool = True,
    lightweight: bool = False,
) -> CurieWeissAncillaResultBundle:
    repository = repository.resolve()
    store.validate_declared(run)
    if store.has_complete(run):
        return store.load_result(run)
    if require_preflight:
        evidence = store.require_preflight()
        if evidence.get("source_tree_digest") != source_digest(repository):
            raise ValueError("preflight does not bind the current source")
    store.reserve(run, require_preflight=require_preflight)
    started, timer = datetime.now(UTC), time.perf_counter()
    dataset = load_reused_dataset(repository, manifest, run)
    config = training_config(manifest, run, updates=updates, lightweight=lightweight)
    result = train_model(manifest, run, dataset, config, encoding=manifest.encoding)
    metrics, arrays = evaluate_run(
        manifest, run, result.best_params, result.generators, encoding=manifest.encoding
    )
    metrics.update(
        {
            "best_validation_conditional_mmd": result.best_validation_loss,
            "best_step": result.best_step,
            "optimizer_updates": config.steps,
            "source_tree_digest": source_digest(repository),
            "learned_parameter_count": run.parameter_count,
            "compiled_generator_count": len(result.generators),
            "visible_character_masks": 256,
        }
    )
    checkpoint = store.checkpoint_path(run)
    checkpoint_sha = write_immutable_npz(
        checkpoint,
        {
            "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
            "run_id": np.asarray(run.run_id, dtype="S64"),
            "generators": result.generators,
            "best_params": result.best_params,
            "final_params": result.final_params,
            "best_step": np.asarray(result.best_step),
            "best_validation_loss": np.asarray(result.best_validation_loss),
            "validation_steps": result.validation_steps,
            "validation_losses": result.validation_losses,
        },
    )
    probability_path = store.probabilities_path(run)
    probability_sha = write_immutable_npz(probability_path, arrays)
    completed = datetime.now(UTC)
    bundle = CurieWeissAncillaResultBundle(
        run=run,
        manifest_digest=manifest.digest,
        code_revision=_git_revision(repository),
        environment={
            **runtime_environment(repository / "uv.lock"),
            "platform": platform.platform(),
            "source_tree_digest": source_digest(repository),
        },
        invocation=invocation,
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        runtime_seconds=time.perf_counter() - timer,
        peak_memory_bytes=_peak_rss_bytes(),
        dataset_path=_portable(dataset.path, repository),
        dataset_sha256=dataset.sha256,
        checkpoint_path=_portable(checkpoint, repository),
        checkpoint_sha256=checkpoint_sha,
        probabilities_path=_portable(probability_path, repository),
        probabilities_sha256=probability_sha,
        estimator_randomness={
            "optimizer_seed": run.optimizer_seed,
            "training_estimator_seed": estimator_seed(run),
            "validation_estimator_seed": estimator_seed(run) + 1_000_000,
        },
        metrics=metrics,
    )
    store.record_result(bundle)
    return bundle


def runtime_preflight(
    repository: Path, manifest: CurieWeissAncillaManifest
) -> dict[str, object]:
    planned = plan_curie_weiss_ancilla_runs(manifest)
    if manifest.campaign_version == ENCODING_COMPARISON_VERSION:
        runs = [run for run in planned if run.optimizer_seed == 0]
    else:
        runs = [
            next(
                run
                for run in planned
                if run.ancillas == manifest.headline_ancillas
                and run.optimizer_seed == 0
                and run.family is CurieWeissAncillaFamily.CONDITIONED
            )
        ]
    arm_evidence = []
    with TemporaryDirectory(prefix="iqp-jax-preflight-") as cache_directory:
        jax.config.update("jax_compilation_cache_dir", cache_directory)
        jax.clear_caches()
        for run in runs:
            dataset = load_reused_dataset(repository.resolve(), manifest, run)
            config = training_config(manifest, run, updates=100)
            start = time.perf_counter()
            result = train_model(
                manifest, run, dataset, config, encoding=manifest.encoding
            )
            elapsed = time.perf_counter() - start
            arm_evidence.append(
                {
                    "family": run.family.value,
                    "elapsed_seconds": elapsed,
                    "projected_seconds_per_model": elapsed * 20.0,
                    "finite": bool(np.isfinite(result.validation_losses).all()),
                }
            )
    peak = _peak_rss_bytes()
    limits = manifest.raw["runtime_preflight"]
    finite = all(bool(row["finite"]) for row in arm_evidence)
    projected = max(float(row["projected_seconds_per_model"]) for row in arm_evidence)
    return {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "updates": 100,
        "compilation_cache": "fresh_temporary_directory",
        "arms": arm_evidence,
        "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in arm_evidence),
        "projected_seconds_per_model": projected,
        "peak_rss_bytes": peak,
        "finite": finite,
        "passed": bool(
            finite
            and projected <= limits["max_projected_seconds_per_model"]
            and peak <= limits["max_peak_rss_bytes"]
        ),
        "performance_gate": False,
        "source_tree_digest": source_digest(repository.resolve()),
    }

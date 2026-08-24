"""Training and exact-evaluation seam for the compact foundations campaign."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from .exact import exact_iqp_probabilities, exact_iqp_probabilities_many
from .foundations_campaign import FoundationsCampaignStore
from .foundations_data import (
    FoundationsExactArchive,
    load_foundations_exact_archive,
    load_foundations_training_validation,
    write_immutable_npz,
)
from .foundations_estimator import EstimatorSelection
from .foundations_protocol import FoundationsManifest
from .ising import BetaGrid as IsingBetaGrid
from .ising import beta_to_broadcast_angles
from .ising_training import (
    DatasetRole,
    EstimatorProtocol,
    EstimatorStreams,
    ExactTargetSet,
    ExactTVDRecord,
    IsingTrainingResult,
    OptimizationProtocol,
    build_model_design,
    evaluate_exact_target_set,
    result_bundle_from_training,
    runtime_environment,
    train_ising_model,
    utc_now,
)
from .phase_conditioning import exact_effective_phases, normalize_conditions
from .protocol import ModelFamily, RunIdentity, Stage, canonical_json_bytes


def foundations_grid(manifest: FoundationsManifest) -> IsingBetaGrid:
    return IsingBetaGrid(
        training=manifest.beta_grid.training,
        heldout=manifest.beta_grid.held_out,
        provenance=manifest.campaign_version,
    )


def _run_betas(manifest: FoundationsManifest, run: RunIdentity) -> tuple[float, ...]:
    if run.condition_scope in {"all_training", "all_training_smoke"}:
        return manifest.beta_grid.training
    if run.condition_scope.startswith("train_beta_"):
        try:
            index = int(run.condition_scope.removeprefix("train_beta_"))
            return (manifest.beta_grid.training[index],)
        except (ValueError, IndexError) as error:
            raise ValueError("run has an invalid Training-beta scope") from error
    raise ValueError("run has an unsupported foundations condition scope")


def optimization_for_run(
    manifest: FoundationsManifest,
    run: RunIdentity,
    *,
    max_updates: int | None = None,
) -> OptimizationProtocol:
    """Resolve the preregistered LR schedule and foundations update count."""
    if run.tuning_candidate_id is None:
        candidates = manifest.training.candidates_for(
            ModelFamily.FIXED_INPUT
            if run.model_family is ModelFamily.SEPARATE
            else run.model_family
        )
        candidate = candidates[1]
    else:
        candidate = manifest.training.candidate(run.tuning_candidate_id)
    updates = (
        manifest.runtime.smoke_updates
        if run.stage is Stage.RUNTIME_FEASIBILITY
        else manifest.training.max_updates
    )
    if max_updates is not None:
        updates = max_updates
    return OptimizationProtocol(
        max_updates=updates,
        batch_size_per_condition=manifest.training.batch_size,
        learning_rate=candidate.initial_learning_rate,
        final_learning_rate=candidate.final_learning_rate,
        validation_interval=min(manifest.training.validation_interval, updates),
        production=False,
    )


def model_probabilities_at_all_conditions(
    result: IsingTrainingResult,
    manifest: FoundationsManifest,
) -> np.ndarray:
    """Return one frozen checkpoint's complete 65,536-state law at nine betas."""
    betas = manifest.beta_grid.training + manifest.beta_grid.held_out
    grid = foundations_grid(manifest)
    parameters = result.best_parameters
    generators = result.generators
    if result.design.family is ModelFamily.FIXED_INPUT:
        angles = beta_to_broadcast_angles(np.asarray(betas), grid, 16)
        return np.asarray(
            exact_iqp_probabilities_many(parameters, generators, angles),
            dtype=np.float64,
        )
    if result.design.family is ModelFamily.CONDITION_MODULATED_PHASE:
        normalized = normalize_conditions(
            np.asarray(betas), grid.training[0], grid.training[-1]
        )
        phases = exact_effective_phases(
            parameters, normalized, result.design.phase_feature
        )
        fixed_input = result.design.condition_angles[0]
        return np.stack(
            [
                np.asarray(
                    exact_iqp_probabilities(phase, generators, fixed_input),
                    dtype=np.float64,
                )
                for phase in phases
            ]
        )
    if result.design.family is ModelFamily.NO_CONDITION:
        fixed_input = result.design.condition_angles[0]
        probability = np.asarray(
            exact_iqp_probabilities(parameters, generators, fixed_input),
            dtype=np.float64,
        )
        return np.broadcast_to(probability, (len(betas), probability.size)).copy()
    if result.design.family is ModelFamily.SEPARATE:
        probability = np.asarray(
            exact_iqp_probabilities(
                parameters, generators, result.design.condition_angles[0]
            ),
            dtype=np.float64,
        )
        return np.broadcast_to(probability, (len(betas), probability.size)).copy()
    raise ValueError("foundations exact evaluation received an unsupported family")


def physical_diagnostics(
    model_probabilities: np.ndarray,
    exact: FoundationsExactArchive,
) -> tuple[dict[str, float], ...]:
    """Compute energy and coupled-edge correlation errors at all nine betas."""
    targets = np.concatenate(
        (exact.training.probabilities, exact.held_out.probabilities), axis=0
    )
    if model_probabilities.shape != targets.shape:
        raise ValueError("model and target laws do not align for physical diagnostics")
    spins = 1.0 - 2.0 * exact.bitstrings
    edges = np.argwhere(np.triu(exact.couplings != 0.0, k=1))
    if edges.size == 0:
        raise ValueError("exact Ising archive contains no coupled edges")
    characters = np.stack(
        [spins[:, left] * spins[:, right] for left, right in edges], axis=1
    )
    result = []
    for model, target in zip(model_probabilities, targets, strict=True):
        result.append(
            {
                "energy_absolute_error": abs(
                    float(model @ exact.energies - target @ exact.energies)
                ),
                "correlation_rmse": float(
                    np.sqrt(np.mean((model @ characters - target @ characters) ** 2))
                ),
            }
        )
    return tuple(result)


def _revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def execute_foundations_run(
    store: FoundationsCampaignStore,
    run: RunIdentity,
    *,
    datasets_directory: Path,
    targets_directory: Path,
    selection: EstimatorSelection,
    repository: Path,
    invocation: Sequence[str] | None = None,
    max_updates: int | None = None,
) -> tuple[IsingTrainingResult, tuple[ExactTVDRecord, ...]]:
    """Reserve, train, checkpoint, evaluate, and record one declared run."""
    if selection.manifest_digest != store.manifest.digest:
        raise ValueError("estimator selection belongs to another manifest")
    if store.reservation_path(run).exists():
        bundle = store.require_complete(run)
        raise ValueError(
            f"run {bundle.run.run_id} is already complete; immutable replay refused"
        )
    store.reserve(run)
    manifest = store.manifest
    betas = _run_betas(manifest, run)
    data = load_foundations_training_validation(
        datasets_directory, manifest, run.coupling_seed, betas
    )
    design = build_model_design(
        run.model_family,
        betas,
        foundations_grid(manifest),
        phase_feature=run.phase_feature_assignment,
        n_qubits=16,
    )
    estimator = EstimatorProtocol(
        bandwidths=manifest.estimator.bandwidths,
        operators_per_bandwidth=selection.operator_budget,
        reference_samples_per_operator_condition=selection.reference_budget,
        production=False,
    )
    optimization = optimization_for_run(manifest, run, max_updates=max_updates)
    started_at = utc_now()
    result = train_ising_model(
        design,
        data,
        run.optimizer_seed,
        estimator=estimator,
        optimization=optimization,
        streams=EstimatorStreams.for_run(run),
        run=run,
    )
    completed_at = utc_now()

    evaluation: tuple[ExactTVDRecord, ...] = ()
    exact_sha: str | None = None
    probability_sha: str | None = None
    diagnostics: tuple[dict[str, float], ...] = ()
    if run.stage is Stage.REPRESENTABILITY_SCREEN:
        exact = load_foundations_exact_archive(
            targets_directory / f"coupling_{run.coupling_seed}.npz",
            manifest,
            run.coupling_seed,
        )
        index = manifest.beta_grid.training.index(betas[0])
        target = ExactTargetSet(
            DatasetRole.TRAINING,
            run.coupling_seed,
            betas,
            exact.training.probabilities[[index]],
            exact.source_sha256,
        )
        evaluation = evaluate_exact_target_set(
            result, target, foundations_grid(manifest), production=True
        )
        exact_sha = exact.source_sha256
    elif run.stage is Stage.PRIMARY:
        exact = load_foundations_exact_archive(
            targets_directory / f"coupling_{run.coupling_seed}.npz",
            manifest,
            run.coupling_seed,
        )
        evaluation = evaluate_exact_target_set(
            result, exact.training, foundations_grid(manifest), production=True
        ) + evaluate_exact_target_set(
            result, exact.held_out, foundations_grid(manifest), production=True
        )
        exact_sha = exact.source_sha256
        probabilities = model_probabilities_at_all_conditions(result, manifest)
        diagnostics = physical_diagnostics(probabilities, exact)
        probability_path = store.root / "probabilities" / f"{run.run_id}.npz"
        probability_sha = write_immutable_npz(
            probability_path,
            {
                "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
                "run_id": np.asarray(run.run_id, dtype="S64"),
                "betas": np.asarray(
                    manifest.beta_grid.training + manifest.beta_grid.held_out
                ),
                "probabilities": probabilities,
            },
        )

    checkpoint_path = store.root / "checkpoints" / f"{run.run_id}.npz"
    bundle = result_bundle_from_training(
        run,
        result,
        data,
        exact_sha,
        evaluation,
        checkpoint_path,
        checkpoint_artifact_path=str(checkpoint_path),
        code_revision=_revision(repository),
        environment=runtime_environment(repository / "uv.lock"),
        invocation=tuple(invocation or (sys.executable, "foundations_runner")),
        started_at=started_at,
        completed_at=completed_at,
    )
    if diagnostics:
        metrics = bundle.raw_metrics
        metrics["probabilities_sha256"] = probability_sha
        metrics["physical_diagnostics"] = list(diagnostics)
        bundle = replace(
            bundle,
            raw_metrics_json=canonical_json_bytes(metrics).decode("utf-8"),
        )
    store.record_result(bundle)
    return result, evaluation

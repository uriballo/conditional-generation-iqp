#!/usr/bin/env python3
"""Gate-aware production runner for the frozen 16-qubit Ising campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from iqp_cond.artifacts import CompletionStatus, ParameterAccounting, ResultBundle
from iqp_cond.campaign import (
    CampaignError,
    CampaignStore,
    Gate,
    GateBlockedError,
    GateDecision,
    PlanEntry,
    PlanStatus,
)
from iqp_cond.campaign_ancilla import resolve_reused_m0_checkpoint
from iqp_cond.campaign_finalizers import (
    finalize_ancilla_result,
    finalize_primary,
    finalize_representability,
    finalize_representability_diagnostic,
    finalize_runtime_feasibility,
    finalize_tuning,
    release_ancilla_extension,
)
from iqp_cond.ising import BetaGrid as IsingBetaGrid
from iqp_cond.ising_training import (
    DatasetRole,
    EstimatorProtocol,
    EstimatorStreams,
    FeasibilityTelemetry,
    OptimizationProtocol,
    ansatz_for_run,
    build_model_design,
    build_model_design_for_run,
    evaluate_exact_target_set,
    evaluate_exact_tvd,
    exact_training_target_set,
    load_exact_target_sets,
    load_training_validation_data,
    process_peak_rss_bytes,
    result_bundle_from_training,
    runtime_environment,
    train_exact_objective_diagnostic,
    train_ising_model,
    utc_now,
)
from iqp_cond.protocol import CampaignManifest, ModelFamily, RunIdentity, Stage

STAGE_ALIASES = {
    "feasibility": Stage.RUNTIME_FEASIBILITY,
    "representability": Stage.REPRESENTABILITY_SCREEN,
    "tuning": Stage.TUNING,
    "primary": Stage.PRIMARY,
    "ancilla": Stage.ANCILLA,
    "diagnostic": Stage.REPRESENTABILITY_DIAGNOSTIC,
}


def _entry_payload(entry: PlanEntry) -> dict[str, object]:
    return {
        "run_id": entry.run.run_id,
        "stage": entry.run.stage.value,
        "coupling_seed": entry.run.coupling_seed,
        "condition_scope": entry.run.condition_scope,
        "model_family": entry.run.model_family.value,
        "phase_feature_assignment": entry.run.phase_feature_assignment,
        "tuning_candidate_id": entry.run.tuning_candidate_id,
        "optimizer_seed": entry.run.optimizer_seed,
        "status": entry.status.value,
        "reason": entry.reason,
    }


def executable_entries(
    store: CampaignStore,
    stage: Stage,
    *,
    run_id: str | None = None,
    resume: bool = False,
) -> tuple[PlanEntry, ...]:
    """Resolve executable ledger entries, rejecting every blocked state first."""
    entries = store.dry_run(stage)
    if run_id is not None:
        entries = tuple(entry for entry in entries if entry.run.run_id == run_id)
        if not entries:
            raise ValueError(f"run {run_id} is not declared in stage {stage.value}")
    selected = []
    for entry in entries:
        if entry.status is PlanStatus.BLOCKED:
            raise GateBlockedError(entry.reason or f"stage {stage.value} is blocked")
        if entry.status is PlanStatus.FAILED:
            raise CampaignError(
                f"run {entry.run.run_id} already has an immutable failed result",
            )
        if entry.status is PlanStatus.RESERVED:
            if not resume:
                raise CampaignError(
                    f"run {entry.run.run_id} is reserved; pass --resume explicitly",
                )
            raise CampaignError(
                f"run {entry.run.run_id} is reserved, but no complete optimizer "
                "state is archived; refusing to retrain from initialization",
            )
        elif entry.status is PlanStatus.ELIGIBLE:
            selected.append(entry)
    return tuple(selected)


def execute_stage(
    store: CampaignStore,
    stage: Stage,
    executor: Callable[[RunIdentity], ResultBundle],
    *,
    run_id: str | None = None,
    resume: bool = False,
) -> tuple[ResultBundle, ...]:
    """Execute selected declared runs only after the campaign store releases them."""
    entries = executable_entries(store, stage, run_id=run_id, resume=resume)
    return tuple(
        store.execute_one(entry.run, executor, resume=resume) for entry in entries
    )


def stage_for_declared_run(store: CampaignStore, run_id: str) -> Stage:
    """Resolve a single immutable run ID without bypassing stage planning."""
    for stage in STAGE_ALIASES.values():
        try:
            entries = store.dry_run(stage)
        except GateBlockedError:
            continue
        if any(entry.run.run_id == run_id for entry in entries):
            return stage
    raise ValueError(f"run {run_id} is not declared by the available campaign plan")


def _code_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if not commit:
        raise RuntimeError("git returned an empty code revision")
    source_paths = sorted((repository / "src" / "iqp_cond").rglob("*.py"))
    source_paths.append(Path(__file__).resolve())
    digest = hashlib.sha256()
    for path in source_paths:
        try:
            relative = path.relative_to(repository)
        except ValueError:
            relative = path
        digest.update(str(relative).encode("utf-8"))
        digest.update(path.read_bytes())
    return f"git:{commit};source_sha256:{digest.hexdigest()}"


def _diagnostic_beta(
    store: CampaignStore,
    manifest: CampaignManifest,
    run: RunIdentity,
) -> float:
    """Select the worst canonical screened cell for the one exact diagnostic."""
    candidates = []
    for entry in store.dry_run(Stage.REPRESENTABILITY_SCREEN):
        if entry.run.coupling_seed != run.coupling_seed:
            continue
        bundle = store.result(entry.run.run_id)
        if bundle is None or bundle.completion_status is not CompletionStatus.COMPLETED:
            raise CampaignError(
                "exact diagnostic requires every canonical screen result",
            )
        beta = _run_beta(entry.run, manifest)
        candidates.append((_training_tvd(bundle, beta), beta))
    if len(candidates) != 3:
        raise CampaignError("exact diagnostic requires canonical low/anchor/high cells")
    return max(candidates, key=lambda item: (item[0], -item[1]))[1]


def _training_executor(
    manifest: CampaignManifest,
    store: CampaignStore,
    campaign_root: Path,
    dataset_directory: Path,
    target_directory: Path,
    repository: Path,
    invocation: Sequence[str],
) -> Callable[[RunIdentity], ResultBundle]:
    estimator = EstimatorProtocol.from_manifest(manifest)
    environment = runtime_environment(repository / "uv.lock")
    code_revision = _code_revision(repository)
    grid = IsingBetaGrid(
        training=manifest.beta_grid.training,
        heldout=manifest.beta_grid.held_out,
        provenance=manifest.calibration.method,
    )

    def execute(run: RunIdentity) -> ResultBundle:
        if run.model_family not in {
            ModelFamily.FIXED_INPUT,
            ModelFamily.CONDITION_MODULATED_PHASE,
            ModelFamily.NO_CONDITION,
            ModelFamily.SEPARATE,
            ModelFamily.MATCHED_CAPACITY_CONTROL,
        }:
            raise ValueError("this runner does not execute the declared model family")
        objective_name = (
            "exact_conditional_mmd_diagnostic"
            if run.stage is Stage.REPRESENTABILITY_DIAGNOSTIC
            else "sampled_conditional_mmd"
        )
        reused_m0 = None
        if run.stage is Stage.REPRESENTABILITY_DIAGNOSTIC:
            diagnostic_beta = _diagnostic_beta(store, manifest, run)
            design = build_model_design(
                ModelFamily.SEPARATE,
                (diagnostic_beta,),
                grid,
            )
        else:
            design = build_model_design_for_run(manifest, run)
        data = load_training_validation_data(
            dataset_directory,
            run.coupling_seed,
            design.betas,
            expected_base_data_seed=run.data_seed,
            training_grid=manifest.beta_grid.training,
        )
        started_at = utc_now()
        started = time.perf_counter()
        streams = EstimatorStreams.for_run(run)
        ansatz = ansatz_for_run(run)
        try:
            if run.stage is Stage.REPRESENTABILITY_DIAGNOSTIC:
                target = exact_training_target_set(
                    run.coupling_seed,
                    design.betas[0],
                )
                result = train_exact_objective_diagnostic(
                    design,
                    data,
                    target,
                    run.optimizer_seed,
                    estimator=estimator,
                    optimization=OptimizationProtocol.for_run(manifest, run),
                    streams=streams,
                    run=run,
                )
            else:
                result = train_ising_model(
                    design,
                    data,
                    run.optimizer_seed,
                    estimator=estimator,
                    optimization=OptimizationProtocol.for_run(manifest, run),
                    streams=streams,
                    ansatz=ansatz,
                    run=run,
                )
            evaluation = ()
            exact_targets_sha256 = None
            if run.stage in {Stage.PRIMARY, Stage.ANCILLA}:
                if run.stage is Stage.ANCILLA:
                    reused_m0 = resolve_reused_m0_checkpoint(
                        store,
                        manifest,
                        run,
                    ).checkpoint
                targets = load_exact_target_sets(
                    target_directory / f"coupling_{run.coupling_seed}.npz",
                    manifest,
                    run.coupling_seed,
                )
                evaluation = evaluate_exact_tvd(
                    result,
                    *targets,
                    grid,
                    production=True,
                )
                exact_targets_sha256 = targets[0].source_sha256
            elif run.stage is Stage.REPRESENTABILITY_SCREEN:
                target = exact_training_target_set(
                    run.coupling_seed,
                    design.betas[0],
                )
                evaluation = evaluate_exact_target_set(
                    result,
                    target,
                    grid,
                    production=True,
                )
                exact_targets_sha256 = target.source_sha256
            elif run.stage is Stage.REPRESENTABILITY_DIAGNOSTIC:
                evaluation = evaluate_exact_target_set(
                    result,
                    target,
                    grid,
                    production=True,
                )
                exact_targets_sha256 = target.source_sha256
            result = replace(
                result,
                telemetry=FeasibilityTelemetry(
                    runtime_seconds=time.perf_counter() - started,
                    peak_rss_bytes=process_peak_rss_bytes(),
                ),
            )
            completed_at = utc_now()
            relative_checkpoint = Path("checkpoints") / f"{run.run_id}.npz"
            return result_bundle_from_training(
                run,
                result,
                data,
                exact_targets_sha256,
                evaluation,
                campaign_root / relative_checkpoint,
                checkpoint_artifact_path=str(relative_checkpoint),
                code_revision=code_revision,
                environment=environment,
                invocation=invocation,
                started_at=started_at,
                completed_at=completed_at,
                manifest=manifest,
                objective_name=objective_name,
                reused_m0_checkpoint=reused_m0,
            )
        except Exception as error:
            n_qubits = int(ansatz.generators.shape[1])
            n_parameters = int(ansatz.n_parameters)
            return ResultBundle.create(
                run=run,
                manifest_digest=manifest.digest,
                code_revision=code_revision,
                environment=environment,
                invocation=tuple(invocation),
                started_at=started_at,
                completed_at=utc_now(),
                runtime_seconds=time.perf_counter() - started,
                peak_memory_bytes=process_peak_rss_bytes(),
                parameter_accounting=ParameterAccounting(
                    n_qubits=n_qubits,
                    n_generators=n_parameters,
                    n_trainable_parameters=n_parameters,
                    n_ancillas=(
                        run.ancilla_rung
                        if run.stage is Stage.ANCILLA
                        and run.model_family is ModelFamily.FIXED_INPUT
                        and run.ancilla_rung is not None
                        else 0
                    ),
                ),
                dataset_hashes=data.dataset_hashes(),
                estimator_randomness=streams.to_dict(),
                checkpoint=None,
                raw_metrics={
                    "model_family": run.model_family.value,
                    "objective": objective_name,
                    "failure_phase": "training_evaluation_or_checkpoint",
                    "exception_type": type(error).__name__,
                    "phase_feature": run.phase_feature_assignment,
                    "tuning_candidate_id": run.tuning_candidate_id,
                    "ancilla_rung": run.ancilla_rung,
                },
                completion_status=CompletionStatus.FAILED,
                error=f"{type(error).__name__}: {error}",
            )

    return execute


def _run_beta(run: RunIdentity, manifest: CampaignManifest) -> float:
    if not run.condition_scope.startswith("train_beta_"):
        raise ValueError("representability run does not identify one Training beta")
    index = int(run.condition_scope.removeprefix("train_beta_"))
    return manifest.beta_grid.training[index]


def _training_tvd(bundle: ResultBundle, beta: float) -> float:
    matches = [
        float(record["tvd"])
        for record in bundle.raw_metrics["exact_tvd"]
        if record["role"] == DatasetRole.TRAINING.value
        and np.isclose(float(record["beta"]), beta, atol=1e-14, rtol=0.0)
    ]
    if len(matches) != 1:
        raise RuntimeError("result bundle lacks its exact representability TVD")
    return matches[0]


def finalize_completed_stage(
    store: CampaignStore,
    manifest: CampaignManifest,
    stage: Stage,
) -> None:
    """Release only the gate whose full preregistered evidence is complete."""
    del manifest
    entries = store.dry_run(stage)
    terminal = {PlanStatus.COMPLETED, PlanStatus.FAILED}
    if not entries or any(entry.status not in terminal for entry in entries):
        return
    if stage is Stage.RUNTIME_FEASIBILITY:
        if store.gate_record(Gate.RUNTIME_FEASIBILITY) is None:
            finalize_runtime_feasibility(store)
    elif stage is Stage.REPRESENTABILITY_SCREEN:
        if store.gate_record(Gate.REPRESENTABILITY) is None:
            finalize_representability(store)
    elif stage is Stage.REPRESENTABILITY_DIAGNOSTIC:
        finalize_representability_diagnostic(store)
    elif stage is Stage.TUNING:
        if store.gate_record(Gate.TUNING_PROTOCOL) is None:
            finalize_tuning(store)
    elif stage is Stage.PRIMARY:
        retention = store.gate_record(Gate.BENEFIT_RETENTION)
        interpolation = store.gate_record(Gate.INTERPOLATION)
        if retention is None and interpolation is None:
            finalized = finalize_primary(store)
            retention = finalized.benefit_retention.record
            interpolation = finalized.interpolation.record
        elif retention is None or interpolation is None:
            raise CampaignError("primary gate state is only partially finalized")
        if (
            retention.decision is GateDecision.PASSED
            and interpolation.decision is GateDecision.PASSED
            and store.gate_record(Gate.ANCILLA_EXTENSION) is None
        ):
            release_ancilla_extension(store)
    elif stage is Stage.ANCILLA:
        finalize_ancilla_result(store)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/quantum_campaign_v1.json"),
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path("results/quantum-campaign-v1/campaign"),
    )
    parser.add_argument(
        "--dataset-directory",
        type=Path,
        default=Path("results/quantum-campaign-v1/datasets"),
    )
    parser.add_argument(
        "--target-directory",
        type=Path,
        default=Path("results/quantum-campaign-v1/targets"),
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--stage", choices=tuple(STAGE_ALIASES))
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage is None and args.run_id is None:
        raise SystemExit("either --stage or --run-id is required")
    manifest = CampaignManifest.from_json(args.manifest)
    store = CampaignStore(args.campaign_root, manifest)
    try:
        stage = (
            STAGE_ALIASES[args.stage]
            if args.stage is not None
            else stage_for_declared_run(store, args.run_id)
        )
        if args.dry_run:
            entries = store.dry_run(stage)
            if args.run_id is not None:
                entries = tuple(
                    entry for entry in entries if entry.run.run_id == args.run_id
                )
                if not entries:
                    raise ValueError("run-id is not declared in the selected stage")
            print(json.dumps([_entry_payload(entry) for entry in entries], indent=2))
            return 0
        entries = executable_entries(
            store,
            stage,
            run_id=args.run_id,
            resume=args.resume,
        )
        executor = _training_executor(
            manifest,
            store,
            args.campaign_root,
            args.dataset_directory,
            args.target_directory,
            args.repository,
            tuple(sys.argv if argv is None else (Path(__file__).name, *argv)),
        )
        bundles = tuple(
            store.execute_one(entry.run, executor, resume=args.resume)
            for entry in entries
        )
        finalize_completed_stage(store, manifest, stage)
        print(
            json.dumps(
                [
                    {
                        "run_id": bundle.run.run_id,
                        "result_digest": bundle.digest,
                        "runtime_seconds": bundle.runtime_seconds,
                        "peak_memory_bytes": bundle.peak_memory_bytes,
                    }
                    for bundle in bundles
                ],
                indent=2,
            ),
        )
        return 0
    except (CampaignError, ValueError, OSError, RuntimeError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

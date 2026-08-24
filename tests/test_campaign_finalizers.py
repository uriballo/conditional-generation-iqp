"""Scientific acceptance tests for evidence-derived campaign finalizers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import iqp_cond.campaign_finalizers as finalizers
from iqp_cond.artifacts import (
    CheckpointProvenance,
    CompletionStatus,
    ParameterAccounting,
    ResultBundle,
)
from iqp_cond.campaign import (
    CampaignStore,
    Gate,
    GateDecision,
    TuningDecision,
    plan_ancilla,
    plan_primary,
    plan_representability_screen,
    plan_runtime_feasibility,
    plan_tuning,
)
from iqp_cond.campaign_finalizers import (
    AcceptanceEvidenceError,
    finalize_ancilla_result,
    finalize_primary,
    finalize_representability,
    finalize_runtime_feasibility,
    finalize_sampling_quality,
    finalize_target_calibration,
    finalize_tuning,
    release_ancilla_extension,
)
from iqp_cond.protocol import (
    CampaignManifest,
    ModelFamily,
    ProtocolValidationError,
    RunIdentity,
    Stage,
    content_digest,
)

REPOSITORY = Path(__file__).parents[1]
MANIFEST_PATH = REPOSITORY / "configs" / "quantum_campaign_v1.json"
TIMESTAMP = "2026-08-10T10:00:00+00:00"


@pytest.fixture
def manifest() -> CampaignManifest:
    return CampaignManifest.from_json(MANIFEST_PATH)


def _digest(name: str) -> str:
    return content_digest({"finalizer_fixture": name})


def _pass_gate(
    store: CampaignStore,
    gate: Gate,
    *,
    tuning_decision: TuningDecision | None = None,
) -> None:
    evidence = (
        tuning_decision.evidence_digest if tuning_decision else _digest(gate.value)
    )
    if gate is Gate.THEOREM_NOVELTY:
        store.release_gate(
            gate,
            GateDecision.PASSED,
            evidence_digest=evidence,
            reason="fixture predecessor",
            decided_at=TIMESTAMP,
        )
    else:
        store._release_finalized_gate(
            gate,
            GateDecision.PASSED,
            evidence_digest=evidence,
            reason="fixture predecessor",
            decided_at=TIMESTAMP,
            tuning_decision=tuning_decision,
        )


def _decision(manifest: CampaignManifest) -> TuningDecision:
    selections = {
        family: manifest.training_protocol.candidates_for(family)[0].candidate_id
        for family in (
            ModelFamily.FIXED_INPUT,
            ModelFamily.CONDITION_MODULATED_PHASE,
            ModelFamily.NO_CONDITION,
        )
    }
    return TuningDecision.create(
        manifest,
        selections,
        evidence_digest=_digest("tuning-decision"),
    )


def _release_through_sampling(store: CampaignStore) -> None:
    for gate in (
        Gate.THEOREM_NOVELTY,
        Gate.TARGET_CALIBRATION,
        Gate.SAMPLING_QUALITY,
    ):
        _pass_gate(store, gate)


def _release_through_tuning(store: CampaignStore) -> TuningDecision:
    _release_through_sampling(store)
    for gate in (Gate.RUNTIME_FEASIBILITY, Gate.REPRESENTABILITY):
        _pass_gate(store, gate)
    decision = _decision(store.manifest)
    _pass_gate(store, Gate.TUNING_PROTOCOL, tuning_decision=decision)
    return decision


def _training_hashes(manifest: CampaignManifest, seed: int) -> dict[str, str]:
    return {
        f"{role}_beta_{index}": _digest(f"{role}-{seed}-{index}")
        for role in ("training", "validation")
        for index in range(len(manifest.beta_grid.training))
    }


def _exact_rows(
    manifest: CampaignManifest,
    *,
    training_tvd: float,
    heldout_tvd: float,
) -> list[dict[str, object]]:
    return [
        {"role": "training", "beta": beta, "tvd": training_tvd}
        for beta in manifest.beta_grid.training
    ] + [
        {"role": "held_out", "beta": beta, "tvd": heldout_tvd}
        for beta in manifest.beta_grid.held_out
    ]


def _bundle(
    run: RunIdentity,
    manifest: CampaignManifest,
    *,
    score: float = 0.1,
    exact_rows: list[dict[str, object]] | None = None,
    reused_m0: CheckpointProvenance | None = None,
    extra_dataset_hashes: dict[str, str] | None = None,
) -> ResultBundle:
    checkpoint = CheckpointProvenance(
        source_run_id=run.run_id,
        manifest_digest=manifest.digest,
        selection_rule="lowest frozen Validation Conditional MMD",
        selected_step=25,
        validation_metric="conditional_mmd",
        validation_value=score,
        artifact_path=f"checkpoints/{run.run_id}.npz",
        artifact_sha256=_digest(f"checkpoint-{run.run_id}"),
    )
    dataset_hashes = _training_hashes(manifest, run.data_seed)
    if exact_rows is not None:
        dataset_hashes = {
            "training": _digest(f"training-{run.coupling_seed}"),
            "validation": _digest(f"validation-{run.coupling_seed}"),
            "exact_targets": _digest(f"targets-{run.coupling_seed}"),
        }
    if extra_dataset_hashes:
        dataset_hashes.update(extra_dataset_hashes)
    metrics = {
        "model_family": run.model_family.value,
        "objective": "sampled_conditional_mmd",
        "phase_feature": run.phase_feature_assignment,
        "tuning_candidate_id": run.tuning_candidate_id,
        "best_validation_conditional_mmd": score,
        "max_updates": manifest.training_protocol.max_updates,
        "batch_size_per_condition": manifest.training_protocol.batch_size,
        "bandwidths": list(manifest.estimator_budget.bandwidths),
        "operators_per_bandwidth": (manifest.estimator_budget.operators_per_bandwidth),
        "reference_samples_per_operator_condition": (
            manifest.estimator_budget.reference_samples_per_operator_condition
        ),
        "exact_tvd": exact_rows or [],
        "reused_m0_checkpoint": reused_m0.to_dict() if reused_m0 else None,
    }
    return ResultBundle.create(
        run=run,
        manifest_digest=manifest.digest,
        code_revision="finalizer-fixture-revision",
        environment={"python": "test", "backend": "exact"},
        invocation=("fixture", "--run-id", run.run_id),
        started_at=TIMESTAMP,
        completed_at="2026-08-10T10:00:01+00:00",
        runtime_seconds=1.0,
        peak_memory_bytes=4_096,
        parameter_accounting=ParameterAccounting(16, 136, 136, 0),
        dataset_hashes=dataset_hashes,
        estimator_randomness={"operators": 11, "reference": 13},
        checkpoint=checkpoint,
        raw_metrics=metrics,
        completion_status=CompletionStatus.COMPLETED,
    )


def test_raw_release_cannot_fabricate_a_downstream_pass(tmp_path, manifest):
    store = CampaignStore(tmp_path, manifest)
    _pass_gate(store, Gate.THEOREM_NOVELTY)

    with pytest.raises(ProtocolValidationError, match="campaign finalizer"):
        store.release_gate(
            Gate.TARGET_CALIBRATION,
            GateDecision.PASSED,
            evidence_digest=_digest("fabricated-pass"),
            reason="caller supplied a digest but no derivation",
        )


def test_real_calibration_archive_is_derived_and_byte_bound(tmp_path, manifest):
    store = CampaignStore(tmp_path, manifest)
    _pass_gate(store, Gate.THEOREM_NOVELTY)

    result = finalize_target_calibration(
        store,
        REPOSITORY,
        decided_at=TIMESTAMP,
    )

    assert result.record.decision is GateDecision.PASSED
    assert result.binding.evidence_sha256 == result.record.evidence_digest
    assert len(result.evidence["target_archives"]) == 8
    assert result.evidence["model_evidence_fields"] == []


def _sampler_audit(seed: int, beta: float, samples: int) -> dict[str, object]:
    return {
        "coupling_seed": seed,
        "beta": beta,
        "n_chains": 8,
        "samples_per_chain": samples,
        "min_energy_ess": 100.0,
        "max_chain_energy_z": 0.30,
        "energy_mean_z": 0.10,
        "magnetization_error": 0.08,
        "correlation_rmse": 0.08,
        "character_rmse": 0.08,
        "empirical_bitstring_tvd": 0.5,
        "passed": True,
        "failed_checks": [],
    }


def _sampler_summary(tmp_path: Path, manifest: CampaignManifest) -> Path:
    records = []
    data = tmp_path / "data"
    data.mkdir()
    for seed in manifest.coupling_seeds:
        for beta_index, beta in enumerate(manifest.beta_grid.training):
            training_seed = manifest.data_seed_for(seed) * 100 + 2 * beta_index
            validation_seed = training_seed + 1
            training = np.zeros((5_000, 16), dtype=np.int8)
            validation = np.ones((2_000, 16), dtype=np.int8)
            relative = Path("data") / f"{seed}-{beta_index}.npz"
            archive = tmp_path / relative
            np.savez_compressed(
                archive,
                training=training,
                validation=validation,
                training_seed=np.asarray(training_seed, dtype=np.int64),
                validation_seed=np.asarray(validation_seed, dtype=np.int64),
                beta=np.asarray(beta, dtype=np.float64),
            )
            records.append(
                {
                    "coupling_seed": seed,
                    "beta": beta,
                    "training_seed": training_seed,
                    "validation_seed": validation_seed,
                    "n_chains": 8,
                    "burn_in_sweeps": 4_000,
                    "thinning_sweeps": 32,
                    "single_spin_proposals_per_sweep": 16,
                    "global_spin_flip_on_retained_samples": True,
                    "training_count": 5_000,
                    "validation_count": 2_000,
                    "training_sha256": hashlib.sha256(training.tobytes()).hexdigest(),
                    "validation_sha256": hashlib.sha256(
                        validation.tobytes(),
                    ).hexdigest(),
                    "archive": str(relative),
                    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "training_audit": _sampler_audit(seed, beta, 625),
                    "validation_audit": _sampler_audit(seed, beta, 250),
                    "passed": True,
                },
            )
    schedule = {
        "n_chains": 8,
        "burn_in": 4_000,
        "thinning": 32,
        "symmetry_mix": True,
    }
    summary = tmp_path / "sampler-summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "campaign_version": manifest.campaign_version,
                "manifest_digest": manifest.digest,
                "sampler_protocol": manifest.sampler_protocol.to_dict(),
                "configuration_digest": content_digest(schedule),
                "expected_cells": 40,
                "records": records,
                "passed": True,
            },
        ),
    )
    return summary


def test_sampling_finalizer_verifies_all_40_split_archives(tmp_path, manifest):
    store = CampaignStore(tmp_path / "campaign", manifest)
    _pass_gate(store, Gate.THEOREM_NOVELTY)
    _pass_gate(store, Gate.TARGET_CALIBRATION)
    summary = _sampler_summary(tmp_path, manifest)

    result = finalize_sampling_quality(
        store,
        summary,
        tmp_path,
        decided_at=TIMESTAMP,
    )

    assert result.record.decision is GateDecision.PASSED
    assert result.evidence["expected_cells"] == 40
    assert len(result.evidence["records"]) == 40


def test_runtime_and_representability_are_computed_from_declared_bundles(
    tmp_path,
    manifest,
    monkeypatch,
):
    runtime_store = CampaignStore(tmp_path / "runtime", manifest)
    _release_through_sampling(runtime_store)
    runtime_run = plan_runtime_feasibility(manifest)[0]
    runtime_bundles = ((runtime_run, _bundle(runtime_run, manifest)),)
    monkeypatch.setattr(finalizers, "_stage_bundles", lambda *_: runtime_bundles)

    runtime = finalize_runtime_feasibility(runtime_store, decided_at=TIMESTAMP)
    assert runtime.record.decision is GateDecision.PASSED
    assert runtime.evidence["telemetry"]["operators_per_bandwidth"] == 2_000

    representability_store = CampaignStore(tmp_path / "representability", manifest)
    _release_through_sampling(representability_store)
    _pass_gate(representability_store, Gate.RUNTIME_FEASIBILITY)
    screen_bundles = []
    for index, run in enumerate(plan_representability_screen(manifest)):
        beta_index = int(run.condition_scope.removeprefix("train_beta_"))
        tvd = 0.35 if index == 0 else 0.25
        rows = [
            {
                "role": "training",
                "beta": manifest.beta_grid.training[beta_index],
                "tvd": tvd,
            },
        ]
        screen_bundles.append((run, _bundle(run, manifest, exact_rows=rows)))
    monkeypatch.setattr(
        finalizers,
        "_stage_bundles",
        lambda *_: tuple(screen_bundles),
    )

    screen = finalize_representability(
        representability_store,
        decided_at=TIMESTAMP,
    )
    assert screen.record.decision is GateDecision.PASSED
    assert screen.evidence["median_tvd"] == pytest.approx(0.25)
    assert screen.evidence["max_tvd"] == pytest.approx(0.35)
    assert screen.evidence["sole_diagnostic_eligible"] is False


def test_tuning_requires_36_training_only_runs_and_selects_three_winners(
    tmp_path,
    manifest,
    monkeypatch,
):
    store = CampaignStore(tmp_path, manifest)
    _release_through_sampling(store)
    _pass_gate(store, Gate.RUNTIME_FEASIBILITY)
    _pass_gate(store, Gate.REPRESENTABILITY)
    candidate_order = {
        candidate.candidate_id: index % 3
        for index, candidate in enumerate(manifest.training_protocol.tuning_candidates)
    }
    bundles = tuple(
        (
            run,
            _bundle(
                run,
                manifest,
                score=0.1 + 0.1 * candidate_order[run.tuning_candidate_id],
            ),
        )
        for run in plan_tuning(manifest)
    )
    monkeypatch.setattr(finalizers, "_stage_bundles", lambda *_: bundles)

    result = finalize_tuning(store, decided_at=TIMESTAMP)

    assert result.record.decision is GateDecision.PASSED
    assert result.record.tuning_decision is not None
    assert len(result.record.tuning_decision.selections) == 3
    assert result.evidence["held_out_access_detected"] is False


def test_tuning_rejects_heldout_or_exact_target_leakage(
    tmp_path,
    manifest,
    monkeypatch,
):
    store = CampaignStore(tmp_path, manifest)
    _release_through_sampling(store)
    _pass_gate(store, Gate.RUNTIME_FEASIBILITY)
    _pass_gate(store, Gate.REPRESENTABILITY)
    runs = plan_tuning(manifest)
    bundles = []
    for index, run in enumerate(runs):
        extra = {"exact_targets": _digest("forbidden")} if index == 0 else None
        bundles.append((run, _bundle(run, manifest, extra_dataset_hashes=extra)))
    monkeypatch.setattr(finalizers, "_stage_bundles", lambda *_: tuple(bundles))

    with pytest.raises(AcceptanceEvidenceError, match="Held-out/Target"):
        finalize_tuning(store, decided_at=TIMESTAMP)


def _primary_bundles(
    manifest: CampaignManifest,
    decision: TuningDecision,
) -> tuple[tuple[RunIdentity, ResultBundle], ...]:
    values = []
    heldout_tvd = {
        ModelFamily.FIXED_INPUT: 0.30,
        ModelFamily.CONDITION_MODULATED_PHASE: 0.20,
        ModelFamily.NO_CONDITION: 0.60,
        ModelFamily.SEPARATE: 0.50,
    }
    for run in plan_primary(manifest, decision):
        rows = _exact_rows(
            manifest,
            training_tvd=0.20,
            heldout_tvd=heldout_tvd[run.model_family],
        )
        values.append((run, _bundle(run, manifest, exact_rows=rows)))
    return tuple(values)


def _ancilla_bundles(
    manifest: CampaignManifest,
    decision: TuningDecision,
    primary: tuple[tuple[RunIdentity, ResultBundle], ...],
) -> tuple[tuple[RunIdentity, ResultBundle], ...]:
    m0 = {
        (run.coupling_seed, run.optimizer_seed): bundle
        for run, bundle in primary
        if run.model_family is ModelFamily.FIXED_INPUT
    }
    values = []
    for run in plan_ancilla(manifest, decision):
        paired = m0[(run.coupling_seed, run.optimizer_seed)]
        heldout = 0.10 if run.model_family is ModelFamily.FIXED_INPUT else 0.20
        rows = _exact_rows(manifest, training_tvd=0.15, heldout_tvd=heldout)
        values.append(
            (
                run,
                _bundle(
                    run,
                    manifest,
                    exact_rows=rows,
                    reused_m0=paired.checkpoint,
                ),
            ),
        )
    return tuple(values)


def test_primary_and_ancilla_finalizers_reconstruct_every_paired_cell(
    tmp_path,
    manifest,
    monkeypatch,
):
    store = CampaignStore(tmp_path, manifest)
    tuning = _release_through_tuning(store)
    primary_bundles = _primary_bundles(manifest, tuning)
    ancilla_bundles = _ancilla_bundles(manifest, tuning, primary_bundles)

    def stage_bundles(_store, stage):
        if stage is Stage.PRIMARY:
            return primary_bundles
        if stage is Stage.ANCILLA:
            return ancilla_bundles
        raise AssertionError(f"unexpected stage {stage}")

    monkeypatch.setattr(finalizers, "_stage_bundles", stage_bundles)
    primary = finalize_primary(store, decided_at=TIMESTAMP)

    assert primary.benefit_retention.record.decision is GateDecision.PASSED
    assert primary.interpolation.record.decision is GateDecision.PASSED
    assert len(primary.paired_records) == 8 * 5 * 4
    assert {
        record.method_tvds["oracle_adjacent"] for record in primary.paired_records
    } == {0.5}
    assert primary.benefit_retention.evidence["benefit_retention"][
        "median_retention"
    ] == pytest.approx(0.75)

    released = release_ancilla_extension(store, decided_at=TIMESTAMP)
    assert released.record.decision is GateDecision.PASSED
    ancilla = finalize_ancilla_result(store)

    assert ancilla.passed
    assert ancilla.evidence["declared_run_count"] == 24
    assert ancilla.evidence["paired_record_count"] == 2 * 3 * 2 * 4
    assert ancilla.evidence["both_rungs_reported"] is True
    assert ancilla.evidence["descriptive_only"] is True
    assert ancilla.evidence["result"]["population_claim_allowed"] is False

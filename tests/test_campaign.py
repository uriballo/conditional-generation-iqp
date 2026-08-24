import copy
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from iqp_cond.artifacts import (
    CheckpointProvenance,
    CompletionStatus,
    CorruptArtifactError,
    DuplicateArtifactError,
    IncompatibleArtifactError,
    ParameterAccounting,
    ResultBundle,
)
from iqp_cond.campaign import (
    CampaignStore,
    DuplicateGateError,
    Gate,
    GateBlockedError,
    GateDecision,
    PlanStatus,
    RunNotReservedError,
    TuningDecision,
    UndeclaredRunError,
    plan_ancilla,
    plan_primary,
    plan_representability_diagnostic,
    plan_representability_screen,
    plan_runtime_feasibility,
    plan_tuning,
)
from iqp_cond.protocol import (
    BudgetPool,
    CampaignManifest,
    ModelFamily,
    ProtocolValidationError,
    Stage,
    content_digest,
)

MANIFEST_PATH = Path(__file__).parents[1] / "configs" / "quantum_campaign_v1.json"
TIMESTAMP = "2026-08-10T10:00:00+00:00"


@pytest.fixture
def manifest() -> CampaignManifest:
    return CampaignManifest.from_json(MANIFEST_PATH)


def _evidence(name: str) -> str:
    return content_digest({"evidence": name})


def _tuning_decision(
    manifest: CampaignManifest,
    *,
    evidence_digest: str | None = None,
) -> TuningDecision:
    candidate_ids = {
        family: manifest.training_protocol.candidates_for(family)[1].candidate_id
        for family in (
            ModelFamily.FIXED_INPUT,
            ModelFamily.CONDITION_MODULATED_PHASE,
            ModelFamily.NO_CONDITION,
        )
    }
    return TuningDecision.create(
        manifest,
        candidate_ids,
        evidence_digest=evidence_digest or _evidence(Gate.TUNING_PROTOCOL.value),
    )


def _release_fixture_gate(
    store: CampaignStore,
    gate: Gate,
    decision: GateDecision,
    *,
    evidence_digest: str,
    reason: str,
    decided_at: str,
    tuning_decision: TuningDecision | None = None,
):
    """Exercise the record seam after fixture evidence has been pre-derived."""
    release = (
        store.release_gate
        if gate is Gate.THEOREM_NOVELTY or decision is GateDecision.FAILED
        else store._release_finalized_gate
    )
    return release(
        gate,
        decision,
        evidence_digest=evidence_digest,
        reason=reason,
        decided_at=decided_at,
        tuning_decision=tuning_decision,
    )


def _release_through(
    store: CampaignStore,
    terminal_gate: Gate,
    *,
    terminal_decision: GateDecision = GateDecision.PASSED,
) -> None:
    order = (
        Gate.THEOREM_NOVELTY,
        Gate.TARGET_CALIBRATION,
        Gate.SAMPLING_QUALITY,
        Gate.RUNTIME_FEASIBILITY,
        Gate.REPRESENTABILITY,
        Gate.TUNING_PROTOCOL,
    )
    for gate in order:
        decision = terminal_decision if gate is terminal_gate else GateDecision.PASSED
        evidence_digest = _evidence(gate.value)
        _release_fixture_gate(
            store,
            gate,
            decision,
            evidence_digest=evidence_digest,
            reason=f"fixture decision for {gate.value}",
            decided_at=TIMESTAMP,
            tuning_decision=(
                _tuning_decision(
                    store.manifest,
                    evidence_digest=evidence_digest,
                )
                if gate is Gate.TUNING_PROTOCOL and decision is GateDecision.PASSED
                else None
            ),
        )
        if gate is terminal_gate:
            break


def _accounting(run) -> ParameterAccounting:
    if run.stage is not Stage.ANCILLA:
        return ParameterAccounting(16, 136, 136, 0)
    total_qubits = 16 + run.ancilla_rung
    parameters = total_qubits * (total_qubits + 1) // 2
    if run.model_family is ModelFamily.FIXED_INPUT:
        return ParameterAccounting(
            total_qubits,
            parameters,
            parameters,
            run.ancilla_rung,
        )
    return ParameterAccounting(16, parameters, parameters, 0)


def _bundle(run, *, status=CompletionStatus.COMPLETED) -> ResultBundle:
    checkpoint = None
    error = "intentional fixture failure" if status is CompletionStatus.FAILED else None
    if status is CompletionStatus.COMPLETED:
        checkpoint = CheckpointProvenance(
            source_run_id=run.run_id,
            manifest_digest=run.configuration_digest,
            selection_rule="lowest frozen Validation Conditional MMD",
            selected_step=25,
            validation_metric="conditional_mmd",
            validation_value=0.125,
            artifact_path=f"checkpoints/{run.run_id}.npz",
            artifact_sha256=_evidence(f"checkpoint-{run.run_id}"),
        )
    return ResultBundle.create(
        run=run,
        manifest_digest=run.configuration_digest,
        code_revision="abc123def456",
        environment={
            "python": "3.13",
            "platform": "test",
            "lock_sha256": _evidence("uv.lock"),
        },
        invocation=("python", "-m", "iqp_cond.runner", "--run-id", run.run_id),
        started_at=TIMESTAMP,
        completed_at="2026-08-10T10:00:01+00:00",
        runtime_seconds=1.0,
        peak_memory_bytes=1_024,
        parameter_accounting=_accounting(run),
        dataset_hashes={
            "training": _evidence(f"training-{run.data_seed}"),
            "validation": _evidence(f"validation-{run.data_seed}"),
            "exact_targets": _evidence(f"targets-{run.coupling_seed}"),
        },
        estimator_randomness={"operators": 7, "reference": 11, "validation": 13},
        checkpoint=checkpoint,
        raw_metrics={"bitstring_tvd": 0.2, "completed": status.value},
        completion_status=status,
        error=error,
    )


def test_primary_plan_is_exactly_the_frozen_320_run_cartesian_product(manifest):
    decision = _tuning_decision(manifest)
    runs = plan_primary(manifest, decision)
    counts = Counter(run.model_family for run in runs)

    assert len(runs) == 320
    assert len({run.run_id for run in runs}) == 320
    assert counts == {
        ModelFamily.FIXED_INPUT: 40,
        ModelFamily.CONDITION_MODULATED_PHASE: 40,
        ModelFamily.NO_CONDITION: 40,
        ModelFamily.SEPARATE: 200,
    }
    assert all(run.stage is Stage.PRIMARY for run in runs)
    assert all(
        run.phase_feature_assignment
        == decision.selection_for(
            ModelFamily.CONDITION_MODULATED_PHASE,
        ).phase_feature_assignment
        for run in runs
        if run.model_family is ModelFamily.CONDITION_MODULATED_PHASE
    )
    assert all(
        run.data_seed == manifest.data_seed_for(run.coupling_seed) for run in runs
    )


def test_pre_primary_plans_are_exact_and_budget_partitioned(manifest):
    runtime = plan_runtime_feasibility(manifest)
    representability = plan_representability_screen(manifest)
    tuning = plan_tuning(manifest)

    assert len(runtime) == 1
    assert runtime[0].stage is Stage.RUNTIME_FEASIBILITY
    assert runtime[0].model_family is ModelFamily.FIXED_INPUT
    assert len(representability) == 6
    assert {run.coupling_seed for run in representability} == {666, 101}
    assert {run.condition_scope for run in representability} == {
        "train_beta_0",
        "train_beta_2",
        "train_beta_4",
    }
    assert {run.model_family for run in representability} == {ModelFamily.SEPARATE}
    assert len(tuning) == 36
    assert len({run.run_id for run in tuning}) == 36
    assert Counter(run.model_family for run in tuning) == {
        ModelFamily.FIXED_INPUT: 12,
        ModelFamily.CONDITION_MODULATED_PHASE: 12,
        ModelFamily.NO_CONDITION: 12,
    }
    assert set(
        run.phase_feature_assignment
        for run in tuning
        if run.model_family is ModelFamily.CONDITION_MODULATED_PHASE
    ) == set(manifest.model_protocol.phase_features)
    candidate_counts = Counter(run.tuning_candidate_id for run in tuning)
    assert set(candidate_counts.values()) == {4}


def test_ancilla_plan_is_exactly_24_new_runs_and_both_rungs_are_reported(manifest):
    runs = plan_ancilla(manifest, _tuning_decision(manifest))

    assert len(runs) == 24
    assert len({run.run_id for run in runs}) == 24
    assert {run.ancilla_rung for run in runs} == {3, 5}
    assert {run.model_family for run in runs} == {
        ModelFamily.FIXED_INPUT,
        ModelFamily.MATCHED_CAPACITY_CONTROL,
    }
    cells = Counter(
        (run.coupling_seed, run.optimizer_seed, run.ancilla_rung) for run in runs
    )
    assert set(cells.values()) == {2}


def test_store_rejects_a_different_manifest_at_the_same_campaign_root(
    tmp_path,
    manifest,
):
    CampaignStore(tmp_path, manifest)
    payload = copy.deepcopy(manifest.to_dict())
    payload["campaign_version"] = "quantum_campaign_v2"
    other = CampaignManifest.from_dict(payload)

    with pytest.raises(IncompatibleArtifactError, match="different manifest"):
        CampaignStore(tmp_path, other)


def test_failed_gate_is_terminal_and_blocks_every_descendant(tmp_path, manifest):
    store = CampaignStore(tmp_path, manifest)
    store.release_gate(
        Gate.THEOREM_NOVELTY,
        GateDecision.FAILED,
        evidence_digest=_evidence("novelty"),
        reason="central theorem did not clear the novelty gate",
        decided_at=TIMESTAMP,
    )

    with pytest.raises(DuplicateGateError):
        store.release_gate(
            Gate.THEOREM_NOVELTY,
            GateDecision.PASSED,
            evidence_digest=_evidence("revised"),
            reason="attempted rewrite",
            decided_at=TIMESTAMP,
        )
    with pytest.raises(GateBlockedError, match="terminally blocked"):
        _release_fixture_gate(
            store,
            Gate.TARGET_CALIBRATION,
            GateDecision.PASSED,
            evidence_digest=_evidence("calibration"),
            reason="must remain blocked",
            decided_at=TIMESTAMP,
        )

    decision = _tuning_decision(manifest)
    downstream_runs = (
        *plan_runtime_feasibility(manifest),
        *plan_representability_screen(manifest),
        *plan_tuning(manifest),
        *plan_primary(manifest, decision),
        *plan_ancilla(manifest, decision),
        *plan_representability_diagnostic(manifest),
    )
    for run in downstream_runs:
        with pytest.raises(GateBlockedError, match="theorem_novelty"):
            store.reserve_run(run, reserved_at=TIMESTAMP)
    for stage in (
        Stage.RUNTIME_FEASIBILITY,
        Stage.REPRESENTABILITY_SCREEN,
        Stage.TUNING,
    ):
        assert {entry.status for entry in store.dry_run(stage)} == {
            PlanStatus.BLOCKED,
        }
    for stage in (Stage.PRIMARY, Stage.ANCILLA):
        with pytest.raises(GateBlockedError, match="theorem_novelty"):
            store.dry_run(stage)


def test_gate_evidence_archives_and_restores_without_rewriting_decision(
    tmp_path,
    manifest,
):
    store = CampaignStore(tmp_path / "campaign", manifest)
    evidence = b"reviewed theorem novelty evidence\n"
    evidence_digest = hashlib.sha256(evidence).hexdigest()
    source = tmp_path / "theorem-review.md"
    source.write_bytes(evidence)
    store.release_gate(
        Gate.THEOREM_NOVELTY,
        GateDecision.FAILED,
        evidence_digest=evidence_digest,
        reason="fixture novelty failure",
        decided_at=TIMESTAMP,
    )
    decision_path = store.gates_directory / "theorem_novelty.json"
    original_decision = decision_path.read_bytes()

    binding = store.archive_gate_evidence(Gate.THEOREM_NOVELTY, source)
    assert binding.evidence_sha256 == evidence_digest
    assert store.gate_evidence_binding(Gate.THEOREM_NOVELTY) == binding
    assert decision_path.read_bytes() == original_decision

    source.unlink()
    restored = store.restore_gate_evidence(
        Gate.THEOREM_NOVELTY,
        tmp_path / "restored" / "theorem-review.md",
    )
    assert restored.read_bytes() == evidence
    assert decision_path.read_bytes() == original_decision


def test_gate_evidence_rejects_bytes_not_named_by_terminal_digest(
    tmp_path,
    manifest,
):
    store = CampaignStore(tmp_path / "campaign", manifest)
    store.release_gate(
        Gate.THEOREM_NOVELTY,
        GateDecision.FAILED,
        evidence_digest=_evidence("expected"),
        reason="fixture novelty failure",
        decided_at=TIMESTAMP,
    )
    source = tmp_path / "wrong.md"
    source.write_text("different evidence")

    with pytest.raises(IncompatibleArtifactError, match="evidence bytes"):
        store.archive_gate_evidence(Gate.THEOREM_NOVELTY, source)


def test_representability_failure_releases_exactly_one_diagnostic(tmp_path, manifest):
    store = CampaignStore(tmp_path, manifest)
    _release_through(
        store,
        Gate.REPRESENTABILITY,
        terminal_decision=GateDecision.FAILED,
    )
    diagnostic = plan_representability_diagnostic(manifest)[0]

    assert store.dry_run(Stage.REPRESENTABILITY_DIAGNOSTIC)[0].status is (
        PlanStatus.ELIGIBLE
    )
    store.reserve_run(diagnostic, reserved_at=TIMESTAMP)
    assert store.remaining_budget(BudgetPool.FEASIBILITY_REPRESENTABILITY) == 19
    with pytest.raises(DuplicateArtifactError):
        store.reserve_run(diagnostic, reserved_at=TIMESTAMP)
    with pytest.raises(UndeclaredRunError):
        store.reserve_run(
            replace(diagnostic, optimizer_seed=manifest.optimizer_seeds[1]),
            reserved_at=TIMESTAMP,
        )
    with pytest.raises(GateBlockedError, match="terminally blocked"):
        _release_fixture_gate(
            store,
            Gate.TUNING_PROTOCOL,
            GateDecision.PASSED,
            evidence_digest=_evidence("tuning"),
            reason="not allowed after representability failure",
            decided_at=TIMESTAMP,
        )


def test_feasibility_pool_declares_only_eight_nontransferable_identities(
    tmp_path,
    manifest,
):
    store = CampaignStore(tmp_path, manifest)
    _release_through(store, Gate.SAMPLING_QUALITY)
    runtime = plan_runtime_feasibility(manifest)[0]
    store.reserve_run(runtime, reserved_at=TIMESTAMP)
    _release_fixture_gate(
        store,
        Gate.RUNTIME_FEASIBILITY,
        GateDecision.PASSED,
        evidence_digest=_evidence("runtime"),
        reason="production estimator fits runtime and memory ceiling",
        decided_at=TIMESTAMP,
    )
    for run in plan_representability_screen(manifest):
        store.reserve_run(run, reserved_at=TIMESTAMP)
    store.release_gate(
        Gate.REPRESENTABILITY,
        GateDecision.FAILED,
        evidence_digest=_evidence("representability"),
        reason="fixture terminal representability failure",
        decided_at=TIMESTAMP,
    )
    diagnostic = plan_representability_diagnostic(manifest)[0]
    store.reserve_run(diagnostic, reserved_at=TIMESTAMP)

    assert store.used_budget(BudgetPool.FEASIBILITY_REPRESENTABILITY) == 8
    assert store.remaining_budget(BudgetPool.FEASIBILITY_REPRESENTABILITY) == 12
    assert store.remaining_budget(BudgetPool.TUNING) == 36
    assert store.remaining_budget(BudgetPool.PRIMARY) == 320
    with pytest.raises(UndeclaredRunError):
        store.reserve_run(
            replace(diagnostic, condition_scope="rescue_objective"),
            reserved_at=TIMESTAMP,
        )


def test_tuning_gate_embeds_manifest_bound_decision_before_primary_planning(
    tmp_path,
    manifest,
):
    store = CampaignStore(tmp_path, manifest)
    _release_through(store, Gate.REPRESENTABILITY)

    assert len(store.dry_run(Stage.TUNING)) == 36
    with pytest.raises(ProtocolValidationError, match="tuning decision"):
        _release_fixture_gate(
            store,
            Gate.TUNING_PROTOCOL,
            GateDecision.PASSED,
            evidence_digest=_evidence("tuning"),
            reason="missing terminal candidate selection",
            decided_at=TIMESTAMP,
        )
    evidence = _evidence("tuning-decision")
    decision = _tuning_decision(manifest, evidence_digest=evidence)
    record = _release_fixture_gate(
        store,
        Gate.TUNING_PROTOCOL,
        GateDecision.PASSED,
        evidence_digest=evidence,
        reason="deterministic Validation-only selection completed",
        decided_at=TIMESTAMP,
        tuning_decision=decision,
    )

    assert record.tuning_decision == decision
    assert store.tuning_decision() == decision
    phase_selection = decision.selection_for(ModelFamily.CONDITION_MODULATED_PHASE)
    primary = plan_primary(manifest, decision)
    assert all(
        run.phase_feature_assignment == phase_selection.phase_feature_assignment
        and run.tuning_candidate_id == phase_selection.candidate_id
        for run in primary
        if run.model_family is ModelFamily.CONDITION_MODULATED_PHASE
    )
    with pytest.raises(DuplicateGateError):
        _release_fixture_gate(
            store,
            Gate.TUNING_PROTOCOL,
            GateDecision.PASSED,
            evidence_digest=evidence,
            reason="cannot replace an immutable selection",
            decided_at=TIMESTAMP,
            tuning_decision=decision,
        )


def test_visible_pre_primary_results_use_16_qubit_parameter_accounting(
    tmp_path,
    manifest,
):
    store = CampaignStore(tmp_path, manifest)
    _release_through(store, Gate.SAMPLING_QUALITY)
    runtime = plan_runtime_feasibility(manifest)[0]
    store.execute_one(
        runtime,
        lambda run: _bundle(run),
        reserved_at=TIMESTAMP,
    )
    _release_fixture_gate(
        store,
        Gate.RUNTIME_FEASIBILITY,
        GateDecision.PASSED,
        evidence_digest=_evidence("runtime-accounting"),
        reason="fixture runtime gate",
        decided_at=TIMESTAMP,
    )
    representability = plan_representability_screen(manifest)[0]
    store.execute_one(
        representability,
        lambda run: _bundle(run),
        reserved_at=TIMESTAMP,
    )
    _release_fixture_gate(
        store,
        Gate.REPRESENTABILITY,
        GateDecision.PASSED,
        evidence_digest=_evidence("representability-accounting"),
        reason="fixture representability gate",
        decided_at=TIMESTAMP,
    )
    tuning = plan_tuning(manifest)[0]
    store.execute_one(
        tuning,
        lambda run: _bundle(run),
        reserved_at=TIMESTAMP,
    )

    assert store.result(runtime.run_id).parameter_accounting == _accounting(runtime)
    assert store.result(representability.run_id).parameter_accounting == _accounting(
        representability,
    )
    assert store.result(tuning.run_id).parameter_accounting == _accounting(tuning)
    assert store.remaining_budget(BudgetPool.FEASIBILITY_REPRESENTABILITY) == 18
    assert store.remaining_budget(BudgetPool.TUNING) == 35


def test_primary_runs_require_tuning_gate_and_reject_undeclared_identity(
    tmp_path,
    manifest,
):
    store = CampaignStore(tmp_path, manifest)
    run = plan_primary(manifest, _tuning_decision(manifest))[0]

    with pytest.raises(GateBlockedError):
        store.dry_run(Stage.PRIMARY)
    with pytest.raises(GateBlockedError):
        store.reserve_run(run, reserved_at=TIMESTAMP)
    _release_through(store, Gate.TUNING_PROTOCOL)
    assert len(store.dry_run(Stage.PRIMARY)) == 320
    with pytest.raises(UndeclaredRunError):
        store.reserve_run(
            replace(run, condition_scope="post_hoc_scope"),
            reserved_at=TIMESTAMP,
        )


def test_result_bundle_is_immutable_complete_and_independently_resumable(
    tmp_path,
    manifest,
):
    store = CampaignStore(tmp_path, manifest)
    _release_through(store, Gate.TUNING_PROTOCOL)
    run = plan_primary(manifest, store.tuning_decision())[0]
    bundle = _bundle(run)

    store.execute_one(run, lambda actual: bundle, reserved_at=TIMESTAMP)
    loaded = store.result(run.run_id)

    assert loaded == bundle
    assert loaded.digest == bundle.digest
    assert loaded.raw_metrics["bitstring_tvd"] == pytest.approx(0.2)
    assert store.remaining_budget(BudgetPool.PRIMARY) == 319
    assert store.remaining_budget(BudgetPool.TUNING) == 36
    assert store.remaining_budget(BudgetPool.ANCILLA) == 24
    assert store.dry_run(Stage.PRIMARY)[0].status is PlanStatus.COMPLETED
    assert not list(tmp_path.rglob("*.tmp"))
    with pytest.raises(DuplicateArtifactError):
        store.record_result(bundle)


def test_result_bundle_rejects_unknown_schema_version(manifest):
    bundle = _bundle(plan_primary(manifest, _tuning_decision(manifest))[0])
    payload = bundle.to_dict()
    payload["schema_version"] = 2

    with pytest.raises(ProtocolValidationError, match="schema_version"):
        ResultBundle.from_dict(payload)


def test_failed_result_is_terminal_and_visible_in_dry_run(tmp_path, manifest):
    store = CampaignStore(tmp_path, manifest)
    _release_through(store, Gate.TUNING_PROTOCOL)
    run = plan_primary(manifest, store.tuning_decision())[1]
    store.reserve_run(run, reserved_at=TIMESTAMP)
    store.record_result(_bundle(run, status=CompletionStatus.FAILED))

    status = {entry.run.run_id: entry.status for entry in store.dry_run(Stage.PRIMARY)}
    assert status[run.run_id] is PlanStatus.FAILED
    with pytest.raises(DuplicateArtifactError):
        store.execute_one(run, _bundle, resume=True)


def test_result_without_reservation_and_incompatible_checkpoint_are_rejected(
    tmp_path,
    manifest,
):
    store = CampaignStore(tmp_path, manifest)
    _release_through(store, Gate.TUNING_PROTOCOL)
    run = plan_primary(manifest, store.tuning_decision())[0]

    with pytest.raises(RunNotReservedError):
        store.record_result(_bundle(run))
    checkpoint = CheckpointProvenance(
        source_run_id=run.run_id,
        manifest_digest=_evidence("another-manifest"),
        selection_rule="validation",
        selected_step=1,
        validation_metric="loss",
        validation_value=0.1,
        artifact_path="checkpoint.npz",
        artifact_sha256=_evidence("checkpoint"),
    )
    with pytest.raises(IncompatibleArtifactError, match="different campaign"):
        ResultBundle.create(
            run=run,
            manifest_digest=manifest.digest,
            code_revision="abc",
            environment={"python": "3.13"},
            invocation=("run",),
            started_at=TIMESTAMP,
            completed_at=TIMESTAMP,
            runtime_seconds=0.0,
            peak_memory_bytes=0,
            parameter_accounting=_accounting(run),
            dataset_hashes={"training": _evidence("training")},
            estimator_randomness={"operators": 1},
            checkpoint=checkpoint,
            raw_metrics={},
            completion_status=CompletionStatus.COMPLETED,
        )


def test_tampering_with_a_result_bundle_is_detected(tmp_path, manifest):
    store = CampaignStore(tmp_path, manifest)
    _release_through(store, Gate.TUNING_PROTOCOL)
    run = plan_primary(manifest, store.tuning_decision())[0]
    store.reserve_run(run, reserved_at=TIMESTAMP)
    path = store.record_result(_bundle(run))
    envelope = json.loads(path.read_text())
    envelope["payload"]["raw_metrics"]["bitstring_tvd"] = 0.99
    path.write_text(json.dumps(envelope))

    with pytest.raises(CorruptArtifactError, match="digest mismatch"):
        store.result(run.run_id)


def test_ancilla_runs_require_both_primary_gates_and_explicit_release(
    tmp_path,
    manifest,
):
    store = CampaignStore(tmp_path, manifest)
    _release_through(store, Gate.TUNING_PROTOCOL)
    run = plan_ancilla(manifest, store.tuning_decision())[0]

    with pytest.raises(GateBlockedError):
        store.reserve_run(run, reserved_at=TIMESTAMP)
    _release_fixture_gate(
        store,
        Gate.BENEFIT_RETENTION,
        GateDecision.PASSED,
        evidence_digest=_evidence("benefit"),
        reason="binding retention rule passed",
        decided_at=TIMESTAMP,
    )
    _release_fixture_gate(
        store,
        Gate.INTERPOLATION,
        GateDecision.PASSED,
        evidence_digest=_evidence("interpolation"),
        reason="all three interpolation rules passed",
        decided_at=TIMESTAMP,
    )
    _release_fixture_gate(
        store,
        Gate.ANCILLA_EXTENSION,
        GateDecision.PASSED,
        evidence_digest=_evidence("ancilla-release"),
        reason="both visible-only gates passed",
        decided_at=TIMESTAMP,
    )
    store.reserve_run(run, reserved_at=TIMESTAMP)

    assert store.remaining_budget(BudgetPool.ANCILLA) == 23
    assert store.remaining_budget(BudgetPool.PRIMARY) == 320

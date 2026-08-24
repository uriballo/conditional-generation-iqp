"""Contract tests for leakage-safe Ising campaign training and evaluation."""

import json
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.artifacts import CompletionStatus
from iqp_cond.campaign import (
    CampaignStore,
    Gate,
    GateBlockedError,
    GateDecision,
    PlanStatus,
    TuningDecision,
    plan_ancilla,
    plan_primary,
    plan_representability_diagnostic,
    plan_representability_screen,
    plan_runtime_feasibility,
    plan_tuning,
)
from iqp_cond.campaign_ancilla import resolve_reused_m0_checkpoint
from iqp_cond.exact import exact_iqp_probabilities_many
from iqp_cond.ising import BetaGrid, beta_to_broadcast_angles
from iqp_cond.ising_training import (
    PRODUCTION_BANDWIDTHS,
    PRODUCTION_BATCH_SIZE_PER_CONDITION,
    PRODUCTION_N_PARAMETERS,
    PRODUCTION_N_QUBITS,
    PRODUCTION_N_STATES,
    PRODUCTION_OPERATORS_PER_BANDWIDTH,
    PRODUCTION_REFERENCE_SAMPLES_PER_OPERATOR_CONDITION,
    REPRESENTABILITY_CELL_TVD_MAX,
    REPRESENTABILITY_COUPLING_SEEDS,
    REPRESENTABILITY_MEDIAN_TVD_MAX,
    DatasetRole,
    EstimatorProtocol,
    EstimatorStreams,
    ExactTargetSet,
    ExactTVDRecord,
    FeasibilityTelemetry,
    IsingTrainingResult,
    ModelDesign,
    OptimizationProtocol,
    RoleTaggedDataset,
    TrainingHistory,
    TrainingValidationData,
    ansatz_for_run,
    balanced_minibatch_indices,
    build_model_design,
    build_model_design_for_run,
    condition_balanced_minibatch,
    evaluate_exact_tvd,
    matched_capacity_ansatz,
    representability_gate,
    result_bundle_from_training,
    run_representability_screen,
    sampled_training_objective,
    train_exact_objective_diagnostic,
    train_ising_model,
)
from iqp_cond.mmd import conditional_mmd
from iqp_cond.phase_conditioning import PHASE_FEATURE_MAPS
from iqp_cond.protocol import CampaignManifest, ModelFamily, Stage

MANIFEST_PATH = Path(__file__).parents[1] / "configs" / "quantum_campaign_v1.json"
RUNNER_PATH = Path(__file__).parents[1] / "scripts" / "run_ising_campaign.py"
ZERO_DIGEST = "0" * 64


@pytest.fixture
def manifest() -> CampaignManifest:
    return CampaignManifest.from_json(MANIFEST_PATH)


def _tuning_decision(manifest):
    return TuningDecision.create(
        manifest,
        {
            family: manifest.training_protocol.candidates_for(family)[0].candidate_id
            for family in (
                ModelFamily.FIXED_INPUT,
                ModelFamily.CONDITION_MODULATED_PHASE,
                ModelFamily.NO_CONDITION,
            )
        },
        evidence_digest=ZERO_DIGEST,
    )


@pytest.fixture(scope="module")
def tiny_training_fixture():
    grid = BetaGrid.from_interval(0.2, 0.5)
    betas = grid.training[:2]
    rng = np.random.default_rng(17)
    training = tuple(
        RoleTaggedDataset(
            DatasetRole.TRAINING,
            666,
            beta,
            100 + 2 * index,
            rng.integers(0, 2, size=(16, 2), dtype=np.int8),
        )
        for index, beta in enumerate(betas)
    )
    validation = tuple(
        RoleTaggedDataset(
            DatasetRole.VALIDATION,
            666,
            beta,
            101 + 2 * index,
            rng.integers(0, 2, size=(8, 2), dtype=np.int8),
        )
        for index, beta in enumerate(betas)
    )
    data = TrainingValidationData(training, validation)
    design = build_model_design(
        ModelFamily.FIXED_INPUT,
        betas,
        grid,
        n_qubits=2,
    )
    estimator = EstimatorProtocol.testing(
        bandwidths=(1.0, 2.0),
        operators_per_bandwidth=8,
        reference_samples_per_operator_condition=8,
    )
    optimization = OptimizationProtocol.testing(
        max_updates=2,
        batch_size_per_condition=4,
        validation_interval=1,
    )
    result = train_ising_model(
        design,
        data,
        0,
        estimator=estimator,
        optimization=optimization,
        streams=EstimatorStreams(11, 13, 17),
        ansatz=complete_ansatz(2),
    )
    return grid, data, design, estimator, optimization, result


def test_production_estimator_and_parameter_contracts_are_exact(manifest):
    estimator = EstimatorProtocol.from_manifest(manifest)

    assert estimator.bandwidths == PRODUCTION_BANDWIDTHS
    assert estimator.operators_per_bandwidth == (PRODUCTION_OPERATORS_PER_BANDWIDTH)
    assert estimator.reference_samples_per_operator_condition == (
        PRODUCTION_REFERENCE_SAMPLES_PER_OPERATOR_CONDITION
    )
    assert PRODUCTION_BATCH_SIZE_PER_CONDITION == 256
    assert PRODUCTION_N_QUBITS == 16
    assert PRODUCTION_N_PARAMETERS == 136
    assert PRODUCTION_N_STATES == 65_536

    with pytest.raises(ValueError, match="2,000-by-2,000"):
        EstimatorProtocol(operators_per_bandwidth=1_999)
    with pytest.raises(ValueError, match="256 samples"):
        OptimizationProtocol(batch_size_per_condition=255)


def test_production_runner_records_a_terminal_failed_bundle_after_training_error(
    tmp_path,
    manifest,
    monkeypatch,
):
    runner = runpy.run_path(str(RUNNER_PATH))
    training_executor = runner["_training_executor"]
    globals_ = training_executor.__globals__
    run = plan_runtime_feasibility(manifest)[0]
    training = tuple(
        RoleTaggedDataset(
            DatasetRole.TRAINING,
            run.coupling_seed,
            beta,
            100 + 2 * index,
            np.zeros((2, 16), dtype=np.int8),
        )
        for index, beta in enumerate(manifest.beta_grid.training)
    )
    validation = tuple(
        RoleTaggedDataset(
            DatasetRole.VALIDATION,
            run.coupling_seed,
            beta,
            101 + 2 * index,
            np.ones((2, 16), dtype=np.int8),
        )
        for index, beta in enumerate(manifest.beta_grid.training)
    )
    data = TrainingValidationData(training, validation)

    monkeypatch.setitem(
        globals_,
        "load_training_validation_data",
        lambda *args, **kwargs: data,
    )
    monkeypatch.setitem(
        globals_,
        "train_ising_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fixture crash")),
    )
    monkeypatch.setitem(globals_, "_code_revision", lambda repository: "test-revision")
    monkeypatch.setitem(
        globals_,
        "runtime_environment",
        lambda lockfile: {"python": "test", "backend": "cpu"},
    )

    store = CampaignStore(tmp_path / "campaign", manifest)
    store.release_gate(
        Gate.THEOREM_NOVELTY,
        GateDecision.PASSED,
        evidence_digest=ZERO_DIGEST,
        reason="fixture",
    )
    for gate in (Gate.TARGET_CALIBRATION, Gate.SAMPLING_QUALITY):
        store._release_finalized_gate(
            gate,
            GateDecision.PASSED,
            evidence_digest=ZERO_DIGEST,
            reason="fixture",
        )
    executor = training_executor(
        manifest,
        store,
        store.root,
        tmp_path / "datasets",
        tmp_path / "targets",
        Path(__file__).parents[1],
        ("run_ising_campaign.py", "--run-id", run.run_id),
    )

    bundle = store.execute_one(run, executor)

    assert bundle.completion_status is CompletionStatus.FAILED
    assert bundle.error == "RuntimeError: fixture crash"
    assert bundle.checkpoint is None
    assert bundle.raw_metrics["failure_phase"] == ("training_evaluation_or_checkpoint")
    entry = store.dry_run(Stage.RUNTIME_FEASIBILITY)[0]
    assert entry.status is PlanStatus.FAILED


def test_all_four_model_families_resolve_frozen_inputs_features_and_seeds(manifest):
    feasibility = plan_runtime_feasibility(manifest)[0]
    representability = plan_representability_screen(manifest)[0]
    tuning = plan_tuning(manifest)
    fixed = build_model_design_for_run(manifest, feasibility)
    separate = build_model_design_for_run(manifest, representability)
    phase_runs = [
        run
        for run in tuning
        if run.model_family is ModelFamily.CONDITION_MODULATED_PHASE
        and run.coupling_seed == 666
        and run.optimizer_seed == 0
    ]
    no_condition_run = next(
        run
        for run in tuning
        if run.model_family is ModelFamily.NO_CONDITION
        and run.coupling_seed == 666
        and run.optimizer_seed == 0
    )
    no_condition = build_model_design_for_run(manifest, no_condition_run)

    assert fixed.family is ModelFamily.FIXED_INPUT
    assert not np.array_equal(fixed.condition_angles[0], fixed.condition_angles[-1])
    assert separate.family is ModelFamily.SEPARATE
    assert len(separate.betas) == 1
    assert {run.phase_feature_assignment for run in phase_runs} == set(
        PHASE_FEATURE_MAPS,
    )
    assert all(
        np.array_equal(
            build_model_design_for_run(manifest, run).condition_angles[0],
            build_model_design_for_run(manifest, run).condition_angles[-1],
        )
        for run in phase_runs
    )
    assert np.all(no_condition.condition_angles == no_condition.condition_angles[0])
    assert (
        OptimizationProtocol.for_run(
            manifest,
            no_condition_run,
        ).learning_rate
        == manifest.training_protocol.candidate(
            no_condition_run.tuning_candidate_id,
        ).initial_learning_rate
    )

    matched = {(run.tuning_candidate_id, run.coupling_seed): set() for run in tuning}
    for run in tuning:
        matched[(run.tuning_candidate_id, run.coupling_seed)].add(run.optimizer_seed)
    assert set(map(frozenset, matched.values())) == {frozenset((0, 1))}


def test_ancilla_and_matched_capacity_run_identities_resolve_real_topologies(manifest):
    runs = plan_ancilla(manifest, _tuning_decision(manifest))
    fixed = next(
        run
        for run in runs
        if run.ancilla_rung == 3 and run.model_family is ModelFamily.FIXED_INPUT
    )
    control = next(
        run
        for run in runs
        if run.ancilla_rung == 3
        and run.model_family is ModelFamily.MATCHED_CAPACITY_CONTROL
    )

    fixed_design = build_model_design_for_run(manifest, fixed)
    control_design = build_model_design_for_run(manifest, control)
    fixed_ansatz = ansatz_for_run(fixed)
    control_ansatz = ansatz_for_run(control)

    assert fixed_design.n_qubits == fixed_ansatz.n_qubits == 19
    assert control_design.n_qubits == control_ansatz.n_qubits == 16
    assert fixed_ansatz.n_parameters == control_ansatz.n_parameters == 190
    assert np.all(np.asarray(control_ansatz.generators)[136:].sum(axis=1) == 3)
    assert len({tuple(row) for row in np.asarray(control_ansatz.generators)}) == 190
    assert matched_capacity_ansatz(16, 5).n_parameters == 231


def test_both_ancilla_families_roundtrip_real_bundles_through_campaign_store(
    tmp_path,
    manifest,
):
    decision = _tuning_decision(manifest)
    store = CampaignStore(tmp_path / "campaign", manifest)
    store.release_gate(
        Gate.THEOREM_NOVELTY,
        GateDecision.PASSED,
        evidence_digest=ZERO_DIGEST,
        reason="fixture novelty passed",
    )
    for gate in (
        Gate.TARGET_CALIBRATION,
        Gate.SAMPLING_QUALITY,
        Gate.RUNTIME_FEASIBILITY,
        Gate.REPRESENTABILITY,
    ):
        store._release_finalized_gate(
            gate,
            GateDecision.PASSED,
            evidence_digest=ZERO_DIGEST,
            reason="fixture predecessor passed",
        )
    store._release_finalized_gate(
        Gate.TUNING_PROTOCOL,
        GateDecision.PASSED,
        evidence_digest=ZERO_DIGEST,
        reason="fixture tuning passed",
        tuning_decision=decision,
    )
    for gate in (Gate.BENEFIT_RETENTION, Gate.INTERPOLATION, Gate.ANCILLA_EXTENSION):
        store._release_finalized_gate(
            gate,
            GateDecision.PASSED,
            evidence_digest=ZERO_DIGEST,
            reason="fixture empirical gate passed",
        )

    rng = np.random.default_rng(91)
    training = tuple(
        RoleTaggedDataset(
            DatasetRole.TRAINING,
            666,
            beta,
            10 + 2 * index,
            rng.integers(0, 2, size=(3, 16), dtype=np.int8),
        )
        for index, beta in enumerate(manifest.beta_grid.training)
    )
    validation = tuple(
        RoleTaggedDataset(
            DatasetRole.VALIDATION,
            666,
            beta,
            11 + 2 * index,
            rng.integers(0, 2, size=(2, 16), dtype=np.int8),
        )
        for index, beta in enumerate(manifest.beta_grid.training)
    )
    data = TrainingValidationData(training, validation)
    evaluation = tuple(
        ExactTVDRecord(role, beta, 0.2, 4)
        for role, betas in (
            (DatasetRole.TRAINING, manifest.beta_grid.training),
            (DatasetRole.HELD_OUT, manifest.beta_grid.held_out),
        )
        for beta in betas
    )
    primary = next(
        run
        for run in plan_primary(manifest, decision)
        if run.coupling_seed == 666
        and run.optimizer_seed == 0
        and run.model_family is ModelFamily.FIXED_INPUT
    )
    primary_design = build_model_design_for_run(manifest, primary)
    primary_ansatz = ansatz_for_run(primary)
    optimization = OptimizationProtocol.testing(
        max_updates=1,
        batch_size_per_condition=2,
        validation_interval=1,
    )
    history = TrainingHistory(
        train_steps=np.array([1]),
        train_losses=np.array([0.2]),
        validation_steps=np.array([0, 1]),
        validation_losses=np.array([0.1, 0.2]),
    )
    primary_result = IsingTrainingResult(
        design=primary_design,
        coupling_seed=666,
        optimizer_seed=0,
        generators=np.asarray(primary_ansatz.generators),
        best_parameters=np.zeros(primary_ansatz.n_parameters),
        final_parameters=np.zeros(primary_ansatz.n_parameters),
        best_step=0,
        best_validation_loss=0.1,
        history=history,
        estimator=EstimatorProtocol.testing(),
        optimization=optimization,
        streams=EstimatorStreams.for_run(primary),
        telemetry=FeasibilityTelemetry(0.0, 0),
    )
    primary_bundle = result_bundle_from_training(
        primary,
        primary_result,
        data,
        ZERO_DIGEST,
        evaluation,
        tmp_path / "campaign" / "checkpoints" / f"{primary.run_id}.npz",
        checkpoint_artifact_path=f"checkpoints/{primary.run_id}.npz",
        code_revision="fixture",
        environment={"python": "test"},
        invocation=("test-primary-runner", primary.run_id),
        started_at="2026-08-10T10:00:00+00:00",
        completed_at="2026-08-10T10:00:01+00:00",
    )
    store.execute_one(primary, lambda _run: primary_bundle)
    reused = primary_bundle.checkpoint
    assert reused is not None

    selected_runs = [
        run
        for run in plan_ancilla(manifest, decision)
        if run.coupling_seed == 666
        and run.optimizer_seed == 0
        and run.ancilla_rung == 3
    ]
    for run in selected_runs:
        assert resolve_reused_m0_checkpoint(store, manifest, run) == primary_bundle
        design = build_model_design_for_run(manifest, run)
        ansatz = ansatz_for_run(run)
        result = IsingTrainingResult(
            design=design,
            coupling_seed=666,
            optimizer_seed=0,
            generators=np.asarray(ansatz.generators),
            best_parameters=np.zeros(ansatz.n_parameters),
            final_parameters=np.zeros(ansatz.n_parameters),
            best_step=0,
            best_validation_loss=0.1,
            history=history,
            estimator=EstimatorProtocol.testing(),
            optimization=optimization,
            streams=EstimatorStreams.for_run(run),
            telemetry=FeasibilityTelemetry(0.0, 0),
        )
        bundle = result_bundle_from_training(
            run,
            result,
            data,
            ZERO_DIGEST,
            evaluation,
            tmp_path / "campaign" / "checkpoints" / f"{run.run_id}.npz",
            checkpoint_artifact_path=f"checkpoints/{run.run_id}.npz",
            code_revision="fixture",
            environment={"python": "test"},
            invocation=("test-ancilla-runner", run.run_id),
            started_at="2026-08-10T10:00:00+00:00",
            completed_at="2026-08-10T10:00:01+00:00",
            reused_m0_checkpoint=reused,
        )

        stored = store.execute_one(run, lambda _run, value=bundle: value)
        assert stored.raw_metrics["reused_m0_checkpoint"]["source_run_id"] == (
            primary.run_id
        )
        assert stored.raw_metrics["ancilla_rung"] == 3
        if run.model_family is ModelFamily.FIXED_INPUT:
            assert stored.raw_metrics["ancilla_conditioning_scheme"] == (
                "broadcast_all"
            )
        else:
            assert "3local" in stored.raw_metrics["matched_capacity_generator_protocol"]
        assert store.result(run.run_id) == stored

    assert {run.model_family for run in selected_runs} == {
        ModelFamily.FIXED_INPUT,
        ModelFamily.MATCHED_CAPACITY_CONTROL,
    }


def test_role_tagging_rejects_heldout_checkpoint_access(tiny_training_fixture):
    _, data, *_ = tiny_training_fixture
    leaked = tuple(replace(item, role=DatasetRole.HELD_OUT) for item in data.validation)

    with pytest.raises(ValueError, match="Held-out access is forbidden"):
        TrainingValidationData(data.training, leaked)


def test_minibatches_are_deterministic_and_balanced_per_condition(
    tiny_training_fixture,
):
    _, data, _, _, optimization, _ = tiny_training_fixture
    first = balanced_minibatch_indices(data, optimization, seed=23)
    second = balanced_minibatch_indices(data, optimization, seed=23)
    batch = condition_balanced_minibatch(data, first, step=0)

    assert batch.shape == (len(data.training), 4, data.n_qubits)
    assert all(schedule.shape == (2, 4) for schedule in first)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_sampled_objective_delegates_to_unbiased_mmd_and_is_differentiable(
    tiny_training_fixture,
):
    _, data, design, estimator, _, _ = tiny_training_fixture
    generators = jnp.asarray(complete_ansatz(2).generators)
    parameters = jnp.asarray([0.1, -0.2, 0.3])
    targets = jnp.asarray(data.stacked_training()[:, :4])
    key = jax.random.PRNGKey(41)

    actual = sampled_training_objective(
        parameters,
        generators,
        design,
        targets,
        estimator,
        key,
    )
    expected = conditional_mmd(
        parameters,
        generators,
        jnp.asarray(design.condition_angles),
        targets,
        estimator.bandwidths,
        estimator.operators_per_bandwidth,
        estimator.reference_samples_per_operator_condition,
        key,
    ).loss
    gradient = jax.grad(sampled_training_objective)(
        parameters,
        generators,
        design,
        targets,
        estimator,
        key,
    )

    assert actual == expected
    assert gradient.shape == parameters.shape
    assert jnp.all(jnp.isfinite(gradient))


@pytest.mark.parametrize("phase_feature", PHASE_FEATURE_MAPS)
def test_every_preregistered_phase_map_has_a_differentiable_objective(
    phase_feature,
):
    grid = BetaGrid.from_interval(0.2, 0.5)
    design = build_model_design(
        ModelFamily.CONDITION_MODULATED_PHASE,
        grid.training[:2],
        grid,
        phase_feature=phase_feature,
    )
    generators = jnp.asarray(complete_ansatz(16).generators)
    targets = jnp.asarray(
        np.random.default_rng(3).integers(0, 2, size=(2, 3, 16)),
    )
    estimator = EstimatorProtocol.testing(
        operators_per_bandwidth=2,
        reference_samples_per_operator_condition=2,
    )
    parameters = jnp.linspace(-0.1, 0.1, 136)

    def objective(values):
        return sampled_training_objective(
            values,
            generators,
            design,
            targets,
            estimator,
            jax.random.PRNGKey(7),
        )

    first = objective(parameters)
    second = objective(parameters)
    gradient = jax.grad(objective)(parameters)

    assert first == second
    assert gradient.shape == (136,)
    assert jnp.all(jnp.isfinite(gradient))


def test_training_uses_frozen_updates_and_validation_only_checkpoint_selection(
    tiny_training_fixture,
):
    _, _, _, _, optimization, result = tiny_training_fixture

    assert result.history.train_steps.tolist() == [1, 2]
    assert len(result.history.train_steps) == optimization.max_updates
    best_index = int(np.argmin(result.history.validation_losses))
    assert result.best_step == result.history.validation_steps[best_index]
    assert result.best_validation_loss == pytest.approx(
        result.history.validation_losses[best_index],
    )
    assert result.telemetry.runtime_seconds >= 0.0
    assert result.telemetry.peak_rss_bytes > 0


def test_exact_objective_diagnostic_is_a_real_training_path_without_heldout_access(
    tiny_training_fixture,
):
    grid, data, _, estimator, optimization, _ = tiny_training_fixture
    one_cell = TrainingValidationData(data.training[:1], data.validation[:1])
    design = build_model_design(
        ModelFamily.SEPARATE,
        (one_cell.betas[0],),
        grid,
        n_qubits=2,
    )
    target = ExactTargetSet(
        DatasetRole.TRAINING,
        666,
        design.betas,
        np.full((1, 4), 0.25),
        ZERO_DIGEST,
    )

    result = train_exact_objective_diagnostic(
        design,
        one_cell,
        target,
        0,
        estimator=estimator,
        optimization=optimization,
        streams=EstimatorStreams(31, 37, 41),
        ansatz=complete_ansatz(2),
    )

    assert result.design.family is ModelFamily.SEPARATE
    assert result.history.train_steps.tolist() == [1, 2]
    assert np.all(np.isfinite(result.history.validation_losses))
    with pytest.raises(ValueError, match="cannot access Held-out"):
        train_exact_objective_diagnostic(
            design,
            one_cell,
            replace(target, role=DatasetRole.HELD_OUT),
            0,
            estimator=estimator,
            optimization=optimization,
            streams=EstimatorStreams(31, 37, 41),
            ansatz=complete_ansatz(2),
        )


def test_production_training_rejects_non_16q_non_136_fixture(
    tiny_training_fixture,
    manifest,
):
    _, data, design, _, _, _ = tiny_training_fixture

    with pytest.raises(ValueError, match="16-qubit, 136-generator"):
        train_ising_model(
            design,
            data,
            0,
            estimator=EstimatorProtocol.from_manifest(manifest),
            optimization=OptimizationProtocol(),
            streams=EstimatorStreams(1, 2, 3),
            ansatz=complete_ansatz(2),
        )


def _exact_target_sets_for_result(grid, result):
    all_betas = grid.training + grid.heldout
    angles = beta_to_broadcast_angles(np.asarray(all_betas), grid, n_qubits=2)
    probabilities = exact_iqp_probabilities_many(
        result.best_parameters,
        result.generators,
        angles,
    )
    split = len(grid.training)
    return (
        ExactTargetSet(
            DatasetRole.TRAINING,
            666,
            grid.training,
            probabilities[:split],
            ZERO_DIGEST,
        ),
        ExactTargetSet(
            DatasetRole.HELD_OUT,
            666,
            grid.heldout,
            probabilities[split:],
            ZERO_DIGEST,
        ),
    )


def test_exact_tvd_evaluates_training_and_untouched_heldout_grids(
    tiny_training_fixture,
):
    grid, _, _, _, _, result = tiny_training_fixture
    training_targets, heldout_targets = _exact_target_sets_for_result(grid, result)

    records = evaluate_exact_tvd(
        result,
        training_targets,
        heldout_targets,
        grid,
        production=False,
    )

    assert len(records) == 9
    assert sum(record.role is DatasetRole.TRAINING for record in records) == 5
    assert sum(record.role is DatasetRole.HELD_OUT for record in records) == 4
    assert {record.n_states for record in records} == {4}
    assert max(record.tvd for record in records) < 1e-12
    with pytest.raises(ValueError, match="65,536 states"):
        evaluate_exact_tvd(result, training_targets, heldout_targets, grid)


def test_each_separate_checkpoint_is_evaluated_on_every_heldout_target(
    tiny_training_fixture,
):
    grid, _, _, _, _, result = tiny_training_fixture
    separate = replace(
        result,
        design=build_model_design(
            ModelFamily.SEPARATE,
            (grid.training[0],),
            grid,
            n_qubits=2,
        ),
    )
    training_targets, heldout_targets = _exact_target_sets_for_result(grid, result)

    records = evaluate_exact_tvd(
        separate,
        training_targets,
        heldout_targets,
        grid,
        production=False,
    )

    assert (
        len([record for record in records if record.role is DatasetRole.HELD_OUT]) == 4
    )
    assert {record.beta for record in records} == set(grid.training + grid.heldout)


def test_representability_runner_and_gate_use_exact_frozen_boundaries(manifest):
    grid = BetaGrid(
        training=manifest.beta_grid.training,
        heldout=manifest.beta_grid.held_out,
        provenance=manifest.calibration.method,
    )
    screened_betas = (grid.training[0], grid.anchor, grid.training[-1])
    tvds = iter([0.25, 0.25, 0.25, 0.25, 0.25, 0.35])
    calls = []

    def evaluator(coupling_seed, beta):
        calls.append((coupling_seed, beta))
        return next(tvds)

    result = run_representability_screen(manifest, evaluator)

    assert calls == [
        (coupling_seed, beta)
        for coupling_seed in REPRESENTABILITY_COUPLING_SEEDS
        for beta in screened_betas
    ]
    assert result.median_tvd == REPRESENTABILITY_MEDIAN_TVD_MAX
    assert result.maximum_tvd == REPRESENTABILITY_CELL_TVD_MAX
    assert result.passed

    cell_failure = list(result.cells)
    cell_failure[-1] = replace(cell_failure[-1], tvd=0.350001)
    failed = representability_gate(cell_failure, grid)
    assert failed.median_rule_passed
    assert not failed.every_cell_rule_passed
    assert not failed.passed


def test_estimator_streams_are_stable_complete_run_provenance(manifest):
    runs = plan_primary(manifest, _tuning_decision(manifest))
    run = runs[0]
    same = EstimatorStreams.for_run(run)
    other = EstimatorStreams.for_run(runs[1])

    assert same == EstimatorStreams.for_run(run)
    assert len(set(same.to_dict().values())) == 3
    assert same != other


def test_result_bundle_records_checkpoint_datasets_estimator_and_exact_metrics(
    tiny_training_fixture,
    manifest,
    tmp_path,
):
    grid, data, _, _, _, result = tiny_training_fixture
    run = next(
        item
        for item in plan_primary(manifest, _tuning_decision(manifest))
        if item.coupling_seed == 666
        and item.optimizer_seed == 0
        and item.model_family is ModelFamily.FIXED_INPUT
    )
    targets = _exact_target_sets_for_result(grid, result)
    evaluation = evaluate_exact_tvd(
        result,
        *targets,
        grid,
        production=False,
    )
    checkpoint = tmp_path / "checkpoint.npz"

    bundle = result_bundle_from_training(
        run,
        result,
        data,
        ZERO_DIGEST,
        evaluation,
        checkpoint,
        checkpoint_artifact_path="checkpoints/checkpoint.npz",
        code_revision="abc123",
        environment={"python": "test"},
        invocation=("run_ising_campaign.py", "--run-id", run.run_id),
        started_at="2026-08-10T10:00:00+00:00",
        completed_at="2026-08-10T10:00:01+00:00",
    )

    assert checkpoint.exists()
    assert bundle.checkpoint.selected_step == result.best_step
    assert bundle.checkpoint.validation_metric == "conditional_mmd"
    assert dict(bundle.estimator_randomness) == result.streams.to_dict()
    assert bundle.raw_metrics["exact_tvd"]
    assert bundle.raw_metrics["max_updates"] == 2
    assert "exact_targets" in dict(bundle.dataset_hashes)

    tuning_run = next(
        item
        for item in plan_tuning(manifest)
        if item.coupling_seed == 666
        and item.optimizer_seed == 0
        and item.model_family is ModelFamily.FIXED_INPUT
    )
    tuning_bundle = result_bundle_from_training(
        tuning_run,
        result,
        data,
        None,
        (),
        tmp_path / "tuning-checkpoint.npz",
        checkpoint_artifact_path="checkpoints/tuning-checkpoint.npz",
        code_revision="abc123",
        environment={"python": "test"},
        invocation=("run_ising_campaign.py", "--stage", "tuning"),
        started_at="2026-08-10T10:00:00+00:00",
        completed_at="2026-08-10T10:00:01+00:00",
    )
    assert tuning_bundle.raw_metrics["exact_tvd"] == []
    assert "exact_targets" not in dict(tuning_bundle.dataset_hashes)
    with pytest.raises(ValueError, match="cannot access exact Targets"):
        result_bundle_from_training(
            tuning_run,
            result,
            data,
            ZERO_DIGEST,
            evaluation,
            tmp_path / "leaked-checkpoint.npz",
            checkpoint_artifact_path="checkpoints/leaked-checkpoint.npz",
            code_revision="abc123",
            environment={"python": "test"},
            invocation=("run_ising_campaign.py", "--stage", "tuning"),
            started_at="2026-08-10T10:00:00+00:00",
            completed_at="2026-08-10T10:00:01+00:00",
        )


def test_model_design_rejects_phase_feature_mismatch():
    with pytest.raises(ValueError, match="required only for the phase family"):
        ModelDesign(
            family=ModelFamily.NO_CONDITION,
            betas=(0.2, 0.3),
            condition_angles=np.zeros((2, 2)),
            normalized_conditions=np.array([0.0, 1.0]),
            phase_feature=PHASE_FEATURE_MAPS[0],
        )


@pytest.mark.parametrize(
    ("stage", "stage_name"),
    (
        (Stage.RUNTIME_FEASIBILITY, "feasibility"),
        (Stage.REPRESENTABILITY_SCREEN, "representability"),
        (Stage.TUNING, "tuning"),
        (Stage.PRIMARY, "primary"),
        (Stage.ANCILLA, "ancilla"),
        (Stage.REPRESENTABILITY_DIAGNOSTIC, "diagnostic"),
    ),
)
def test_failed_novelty_gate_terminally_refuses_store_and_cli_execution(
    tmp_path,
    manifest,
    stage,
    stage_name,
):
    campaign_root = tmp_path / stage_name
    store = CampaignStore(campaign_root, manifest)
    store.release_gate(
        Gate.THEOREM_NOVELTY,
        GateDecision.FAILED,
        evidence_digest=ZERO_DIGEST,
        reason="binding novelty review failed",
    )
    try:
        entries = store.dry_run(stage)
    except GateBlockedError as error:
        assert "theorem_novelty" in str(error)
    else:
        assert entries
        assert all(entry.status is PlanStatus.BLOCKED for entry in entries)
        assert all("theorem_novelty" in entry.reason for entry in entries)
    assert not list((campaign_root / "ledger" / "reservations").glob("*.json"))
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--manifest",
            str(MANIFEST_PATH),
            "--campaign-root",
            str(campaign_root),
            "--repository",
            str(Path(__file__).parents[1]),
            "--stage",
            stage_name,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "theorem_novelty" in completed.stderr
    assert not list((campaign_root / "ledger" / "reservations").glob("*.json"))


def test_cli_resume_refuses_reserved_run_without_optimizer_state(tmp_path, manifest):
    campaign_root = tmp_path / "campaign"
    store = CampaignStore(campaign_root, manifest)
    for gate in (Gate.THEOREM_NOVELTY, Gate.TARGET_CALIBRATION, Gate.SAMPLING_QUALITY):
        release = (
            store.release_gate
            if gate is Gate.THEOREM_NOVELTY
            else store._release_finalized_gate
        )
        release(
            gate,
            GateDecision.PASSED,
            evidence_digest=ZERO_DIGEST,
            reason="fixture predecessor passed",
        )
    run = plan_runtime_feasibility(manifest)[0]
    store.reserve_run(run)

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--manifest",
            str(MANIFEST_PATH),
            "--campaign-root",
            str(campaign_root),
            "--repository",
            str(Path(__file__).parents[1]),
            "--run-id",
            run.run_id,
            "--resume",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "no complete optimizer state" in completed.stderr
    assert store.result(run.run_id) is None
    assert len(list((campaign_root / "ledger" / "reservations").glob("*.json"))) == 1


def test_exact_diagnostic_identity_is_exposed_by_cli_dry_run(tmp_path, manifest):
    campaign_root = tmp_path / "campaign"
    store = CampaignStore(campaign_root, manifest)
    for gate in (
        Gate.THEOREM_NOVELTY,
        Gate.TARGET_CALIBRATION,
        Gate.SAMPLING_QUALITY,
        Gate.RUNTIME_FEASIBILITY,
    ):
        release = (
            store.release_gate
            if gate is Gate.THEOREM_NOVELTY
            else store._release_finalized_gate
        )
        release(
            gate,
            GateDecision.PASSED,
            evidence_digest=ZERO_DIGEST,
            reason="fixture predecessor passed",
        )
    store.release_gate(
        Gate.REPRESENTABILITY,
        GateDecision.FAILED,
        evidence_digest=ZERO_DIGEST,
        reason="fixture sampled objective failed",
    )
    expected = plan_representability_diagnostic(manifest)[0]

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--manifest",
            str(MANIFEST_PATH),
            "--campaign-root",
            str(campaign_root),
            "--stage",
            "diagnostic",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    rows = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert len(rows) == 1
    assert rows[0]["run_id"] == expected.run_id
    assert rows[0]["status"] == "eligible"

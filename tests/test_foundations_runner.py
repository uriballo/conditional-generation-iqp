"""Deployment semantics and a tiny real foundations-campaign execution."""

from pathlib import Path

import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.foundations_campaign import (
    FoundationsCampaignStore,
    FoundationsGate,
    GateDecision,
)
from iqp_cond.foundations_data import prepare_foundations_datasets
from iqp_cond.foundations_estimator import EstimatorSelection
from iqp_cond.foundations_protocol import (
    FoundationsManifest,
    plan_foundations_runtime,
)
from iqp_cond.foundations_runner import (
    execute_foundations_run,
    foundations_grid,
    physical_diagnostics,
)
from iqp_cond.ising_training import build_model_design
from iqp_cond.phase_conditioning import parameter_signature
from iqp_cond.protocol import ModelFamily

REPOSITORY = Path(__file__).parents[1]
MANIFEST_PATH = REPOSITORY / "configs" / "arxiv_foundations_v1.json"
TARGETS = REPOSITORY / "results" / "quantum-campaign-v1" / "targets"


def test_fixed_product_state_uses_one_parameter_signature_for_all_conditions():
    manifest = FoundationsManifest.from_json(MANIFEST_PATH)
    design = build_model_design(
        ModelFamily.FIXED_INPUT,
        manifest.beta_grid.training,
        foundations_grid(manifest),
        n_qubits=16,
    )
    ansatz = complete_ansatz(16)
    signature = parameter_signature(
        np.zeros(ansatz.n_parameters), np.asarray(ansatz.generators)
    )

    assert design.condition_angles.shape == (5, 16)
    assert signature.coefficient_count == 136
    assert signature.generator_count == 136
    assert signature.qubit_count == 16


def test_physical_diagnostics_are_zero_when_model_equals_exact_target():
    from iqp_cond.foundations_data import load_foundations_exact_archive

    manifest = FoundationsManifest.from_json(MANIFEST_PATH)
    exact = load_foundations_exact_archive(TARGETS / "coupling_606.npz", manifest, 606)
    probabilities = np.concatenate(
        (exact.training.probabilities, exact.held_out.probabilities), axis=0
    )
    diagnostics = physical_diagnostics(probabilities, exact)

    assert len(diagnostics) == 9
    assert max(row["energy_absolute_error"] for row in diagnostics) < 1e-12
    assert max(row["correlation_rmse"] for row in diagnostics) < 1e-12


def test_tiny_end_to_end_runtime_run_records_complete_bundle(tmp_path):
    manifest = FoundationsManifest.from_json(MANIFEST_PATH)
    datasets = tmp_path / "datasets"
    prepare_foundations_datasets(manifest, TARGETS, datasets, coupling_seeds=(606,))
    store = FoundationsCampaignStore(tmp_path / "store", manifest)
    store.finalize_gate(
        FoundationsGate.DATA_READY, GateDecision.PASSED, evidence_digest="a" * 64
    )
    store.finalize_gate(
        FoundationsGate.ESTIMATOR_CALIBRATION,
        GateDecision.PASSED,
        evidence_digest="b" * 64,
    )
    selection = EstimatorSelection(
        manifest_digest=manifest.digest,
        operator_budget=128,
        reference_budget=128,
        passing_budgets=(128,),
        evidence_digest="c" * 64,
    )
    run = plan_foundations_runtime(manifest)[0]

    result, evaluation = execute_foundations_run(
        store,
        run,
        datasets_directory=datasets,
        targets_directory=TARGETS,
        selection=selection,
        repository=REPOSITORY,
        invocation=("pytest", "tiny-foundations-run"),
        max_updates=1,
    )

    assert result.history.train_steps.tolist() == [1]
    assert evaluation == ()
    assert store.require_complete(run).run.run_id == run.run_id

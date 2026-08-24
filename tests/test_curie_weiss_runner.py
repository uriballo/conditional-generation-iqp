from pathlib import Path

import jax
import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.curie_weiss_campaign import CurieWeissCampaignStore
from iqp_cond.curie_weiss_data import prepare_dataset_archive
from iqp_cond.curie_weiss_protocol import (
    CurieWeissCampaignManifest,
    CurieWeissFamily,
    plan_curie_weiss_runs,
)
from iqp_cond.curie_weiss_runner import (
    evaluate_run,
    evaluation_angles,
    execute_curie_weiss_run,
    training_config,
    training_seed,
)
from iqp_cond.training import _batch_schedule, initialize_parameters

ROOT = Path(__file__).parents[1]
MANIFEST = CurieWeissCampaignManifest.from_json(
    ROOT / "configs" / "curie_weiss_foundations_v1.json"
)


def _run(family: CurieWeissFamily):
    return next(
        run
        for run in plan_curie_weiss_runs(MANIFEST)
        if run.n_qubits == 8 and run.optimizer_seed == 0 and run.family is family
    )


def test_fixed_circuit_signatures_and_no_condition_reference_behavior():
    conditioned = _run(CurieWeissFamily.CONDITIONED)
    no_condition = _run(CurieWeissFamily.NO_CONDITION)
    generators = np.asarray(complete_ansatz(8).generators)
    params = np.zeros(len(generators))
    assert np.array_equal(generators, complete_ansatz(8).generators)
    assert len({tuple(row) for row in evaluation_angles(MANIFEST, no_condition)}) == 1
    _, arrays = evaluate_run(MANIFEST, no_condition, params, generators)
    assert np.allclose(arrays["model_probabilities"], arrays["model_probabilities"][0])
    assert len({tuple(row) for row in evaluation_angles(MANIFEST, conditioned)}) == 11


def test_optimizer_seed_is_literal_and_matched_across_every_family():
    cell = [
        run
        for run in plan_curie_weiss_runs(MANIFEST)
        if run.n_qubits == 8 and run.optimizer_seed == 3
    ]
    assert len(cell) == 9
    assert {training_seed(run) for run in cell} == {3}
    conditioned = next(
        run for run in cell if run.family is CurieWeissFamily.CONDITIONED
    )
    no_condition = next(
        run for run in cell if run.family is CurieWeissFamily.NO_CONDITION
    )
    assert np.array_equal(
        initialize_parameters(2, training_seed(conditioned)),
        initialize_parameters(2, training_seed(no_condition)),
    )
    config = training_config(MANIFEST, conditioned, updates=2, lightweight=True)
    assert np.array_equal(
        _batch_schedule(6, 10_000, config, 3), _batch_schedule(6, 10_000, config, 3)
    )
    assert np.array_equal(
        jax.random.PRNGKey(training_seed(conditioned)),
        jax.random.PRNGKey(training_seed(no_condition)),
    )


def test_tiny_end_to_end_run_is_immutable_and_resumable(tmp_path):
    datasets = tmp_path / "datasets"
    prepare_dataset_archive(MANIFEST, datasets, n_qubits=8, optimizer_seed=0)
    store = CurieWeissCampaignStore(tmp_path / "store", MANIFEST)
    run = _run(CurieWeissFamily.CONDITIONED)
    result = execute_curie_weiss_run(
        ROOT,
        MANIFEST,
        store,
        datasets,
        run,
        invocation=("test",),
        updates=1,
        require_preflight=False,
        lightweight=True,
    )
    assert len(result.metrics["evaluation_rows"]) == 11
    assert result.metrics["learned_coefficient_count"] == 2
    resumed = execute_curie_weiss_run(
        ROOT,
        MANIFEST,
        store,
        datasets,
        run,
        invocation=("different",),
        updates=1,
        require_preflight=False,
        lightweight=True,
    )
    assert resumed == result

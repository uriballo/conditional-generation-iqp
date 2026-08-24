from pathlib import Path

import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    total_variation_distance,
)
from iqp_cond.gaussian_campaign import GaussianCampaignStore
from iqp_cond.gaussian_data import condition_angle_matrix, prepare_dataset_archive
from iqp_cond.gaussian_protocol import (
    GaussianCampaignManifest,
    GaussianFamily,
    plan_gaussian_runs,
)
from iqp_cond.gaussian_report import _metric
from iqp_cond.gaussian_runner import (
    exact_mmd_squared,
    execute_gaussian_run,
    training_config,
)

REPOSITORY = Path(__file__).parents[1]
MANIFEST = GaussianCampaignManifest.from_json(
    REPOSITORY / "configs" / "gaussian_foundations_v1.json"
)


def test_exact_tvd_and_mmd_boundaries():
    p = np.asarray([0.25, 0.25, 0.25, 0.25])
    q = np.asarray([0.5, 0.0, 0.5, 0.0])
    assert total_variation_distance(p, p) == 0.0
    assert exact_mmd_squared(p, p) == 0.0
    assert total_variation_distance(p, q) == 0.5
    assert exact_mmd_squared(p, q) > 0.0


def test_total_minibatch_is_balanced():
    for run in plan_gaussian_runs(MANIFEST):
        config = training_config(MANIFEST, run)
        n_conditions = 1 if run.family is GaussianFamily.SINGLE_PEAK else run.peak_count
        assert config.batch_size * n_conditions == 450


def test_compiled_signature_is_fixed_and_no_condition_output_is_constant():
    generators = complete_ansatz(8).generators
    params = np.linspace(-0.2, 0.2, generators.shape[0])
    conditions = np.asarray([0.0, 0.5, 1.0])
    conditioned_angles = condition_angle_matrix(conditions, 8)
    assert not np.array_equal(conditioned_angles[0], conditioned_angles[-1])
    conditioned = exact_iqp_probabilities_many(params, generators, conditioned_angles)
    assert conditioned.shape == (3, 256)
    neutral = np.zeros((3, 8), dtype=np.float64)
    no_condition = exact_iqp_probabilities_many(params, generators, neutral)
    assert np.array_equal(no_condition[0], no_condition[1])
    assert np.array_equal(no_condition[1], no_condition[2])
    assert generators.shape == (8 * 9 // 2, 8)


def test_report_metric_contains_exact_tvd_moments_and_mmd():
    bitstrings = enumerate_bitstrings(2)
    target = np.asarray([0.5, 0.0, 0.0, 0.5])
    identical = _metric(target, target, bitstrings, (1.0, 2.0, 4.0))
    assert identical == {
        "profile_tvd": 0.0,
        "bitstring_tvd": 0.0,
        "mean_absolute_error": 0.0,
        "width_absolute_error": 0.0,
        "conditional_mmd": 0.0,
    }
    shifted = _metric(
        target,
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        bitstrings,
        (1.0, 2.0, 4.0),
    )
    assert shifted["bitstring_tvd"] == 0.5
    assert shifted["mean_absolute_error"] == 1.0
    assert shifted["width_absolute_error"] == 1.0
    assert shifted["conditional_mmd"] > 0.0


def test_tiny_end_to_end_run_is_resumable(tmp_path):
    repository = REPOSITORY
    datasets = tmp_path / "datasets"
    prepare_dataset_archive(
        MANIFEST, datasets, n_qubits=8, width_factor=0.35, optimizer_seed=0
    )
    run = next(
        run
        for run in plan_gaussian_runs(MANIFEST)
        if run.n_qubits == 8
        and run.width_factor == 0.35
        and run.optimizer_seed == 0
        and run.family is GaussianFamily.SINGLE_PEAK
        and run.bank_peak_index == 0
    )
    store = GaussianCampaignStore(tmp_path / "results" / "store", MANIFEST)
    bundle = execute_gaussian_run(
        repository,
        MANIFEST,
        store,
        datasets,
        run,
        invocation=("pytest", "tiny"),
        updates=1,
        require_preflight=False,
    )
    assert bundle.run == run
    assert bundle.metrics["optimizer_updates"] == 1
    assert len(bundle.metrics["evaluation_rows"]) == 1
    assert (
        execute_gaussian_run(
            repository,
            MANIFEST,
            store,
            datasets,
            run,
            invocation=("pytest", "tiny"),
            updates=1,
            require_preflight=False,
        )
        == bundle
    )

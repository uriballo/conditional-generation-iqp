"""Contracts for the eight-visible Curie--Weiss ancilla campaign."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np
import pytest

from iqp_cond import curie_weiss_ancilla_report as report_module
from iqp_cond.ancilla import ancilla_ansatz, pad_observables_for_ancillas
from iqp_cond.artifacts import sha256_file
from iqp_cond.curie_weiss_ancilla_campaign import (
    CurieWeissAncillaCampaignError,
    CurieWeissAncillaCampaignStore,
    PartialAncillaResultError,
    UndeclaredAncillaRunError,
)
from iqp_cond.curie_weiss_ancilla_protocol import (
    CurieWeissAncillaFamily,
    CurieWeissAncillaManifest,
    ProtocolValidationError,
    plan_curie_weiss_ancilla_runs,
)
from iqp_cond.curie_weiss_ancilla_report import _interval, _raw_oracle, build_report
from iqp_cond.curie_weiss_ancilla_runner import (
    _generator_identity_initialization,
    condition_angles,
    evaluate_run,
    execute_run,
    load_reused_dataset,
    training_config,
)
from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.exact import enumerate_bitstrings
from iqp_cond.mmd import mmd_mask_weights, mmd_operator_probability

ROOT = Path(__file__).parents[1]
MANIFEST = CurieWeissAncillaManifest.from_json(
    ROOT / "configs/curie_weiss_ancilla_foundations_v1.json"
)
M5_MANIFEST = CurieWeissAncillaManifest.from_json(
    ROOT / "configs/curie_weiss_m5_foundations_v1.json"
)


def _run(ancillas, family, seed=0, beta=None):
    return next(
        run
        for run in plan_curie_weiss_ancilla_runs(MANIFEST)
        if run.ancillas == ancillas
        and run.family is family
        and run.optimizer_seed == seed
        and run.reference_beta == beta
    )


def test_plan_has_exact_cardinality_rungs_and_parameter_counts():
    runs = plan_curie_weiss_ancilla_runs(MANIFEST)
    assert len(runs) == len({run.run_id for run in runs}) == 180
    assert {run.ancillas for run in runs} == {0, 1, 3, 5, 7, 9}
    expected = {0: 36, 1: 45, 3: 66, 5: 91, 7: 120, 9: 153}
    assert {
        rung: _run(rung, CurieWeissAncillaFamily.CONDITIONED).parameter_count
        for rung in expected
    } == expected
    references = [
        run
        for run in runs
        if run.family is CurieWeissAncillaFamily.TEMPERATURE_REFERENCE
    ]
    assert len(references) == 60
    assert {run.ancillas for run in references} == {9}


def test_fixed_m5_plan_has_exact_80_fit_cardinality_and_isolated_identity():
    runs = plan_curie_weiss_ancilla_runs(M5_MANIFEST)
    assert len(runs) == len({run.run_id for run in runs}) == 80
    conditioned = [
        run for run in runs if run.family is CurieWeissAncillaFamily.CONDITIONED
    ]
    controls = [
        run for run in runs if run.family is CurieWeissAncillaFamily.NO_CONDITION
    ]
    references = [
        run
        for run in runs
        if run.family is CurieWeissAncillaFamily.TEMPERATURE_REFERENCE
    ]
    assert (len(conditioned), len(controls), len(references)) == (10, 10, 60)
    assert {run.ancillas for run in runs} == {5}
    assert {run.parameter_count for run in runs} == {91}
    assert {run.campaign_version for run in runs} == {"curie_weiss_m5_foundations_v1"}
    assert not (
        {run.run_id for run in runs}
        & {run.run_id for run in plan_curie_weiss_ancilla_runs(MANIFEST)}
    )


def test_fixed_m5_selection_is_prospective_and_pilot_sha_bound():
    selection = M5_MANIFEST.raw["development_selection"]
    assert selection["tested_ancillas"] == [0, 1, 3]
    assert selection["selected_ancillas"] == 5
    assert selection["optimization_claim"] is False
    pilot_manifest = ROOT / selection["campaign_path"] / "manifest.json"
    assert sha256_file(pilot_manifest) == selection["manifest_sha256"]


def test_dataset_reuse_is_sha_bound_and_has_no_heldout_role():
    run = _run(9, CurieWeissAncillaFamily.CONDITIONED)
    dataset = load_reused_dataset(ROOT, MANIFEST, run)
    assert dataset.training.shape == (6, 10_000, 8)
    assert dataset.validation.shape == (6, 2_000, 8)
    reference = load_reused_dataset(
        ROOT,
        MANIFEST,
        _run(9, CurieWeissAncillaFamily.TEMPERATURE_REFERENCE, beta=0.9),
    )
    assert reference.training.shape == (1, 10_000, 8)
    for seed in MANIFEST.optimizer_seeds:
        reused = load_reused_dataset(
            ROOT,
            MANIFEST,
            _run(9, CurieWeissAncillaFamily.CONDITIONED, seed=seed),
        )
        assert reused.sha256 == MANIFEST.source_for_seed(seed).sha256
        assert not np.array_equal(reused.training[:, :2_000], reused.validation)


def test_broadcast_no_condition_and_generator_identity_initialization():
    conditioned = _run(9, CurieWeissAncillaFamily.CONDITIONED, seed=3)
    no_condition = _run(9, CurieWeissAncillaFamily.NO_CONDITION, seed=3)
    conditioned_angles = condition_angles(MANIFEST, conditioned, evaluation=True)
    no_angles = condition_angles(MANIFEST, no_condition, evaluation=True)
    assert conditioned_angles.shape == (11, 17)
    assert len({tuple(row) for row in conditioned_angles}) == 11
    assert len({tuple(row) for row in no_angles}) == 1
    assert np.allclose(no_angles, 0.0)
    small = ancilla_ansatz(8, 1).generators
    large = ancilla_ansatz(8, 9).generators
    small_values = _generator_identity_initialization(3, small)
    large_values = _generator_identity_initialization(3, large)
    lookup = {
        tuple(int(index) for index in np.flatnonzero(np.asarray(row))): float(value)
        for row, value in zip(large, large_values, strict=True)
    }
    np.testing.assert_allclose(
        small_values,
        np.asarray(
            [
                lookup[tuple(int(index) for index in np.flatnonzero(np.asarray(row)))]
                for row in small
            ]
        ),
    )


def test_all_visible_masks_are_exactly_weighted_and_ancilla_padded():
    masks = enumerate_bitstrings(8)
    padded = pad_observables_for_ancillas(masks, 9)
    assert masks.shape == (256, 8)
    assert padded.shape == (256, 17)
    assert np.all(np.asarray(padded)[:, 8:] == 0)
    weights = mmd_mask_weights(8)
    assert int(jnp.sum(weights == 1)) == 8
    assert int(jnp.sum(weights == 2)) == 28
    for sigma in (1.0, 2.0, 4.0):
        probability = mmd_operator_probability(sigma)
        law = probability**weights * (1 - probability) ** (8 - weights)
        np.testing.assert_allclose(float(law.sum()), 1.0)


def test_exact_marginal_and_no_condition_output_are_normalized_and_constant():
    run = _run(1, CurieWeissAncillaFamily.NO_CONDITION)
    ansatz = ancilla_ansatz(8, 1)
    params = np.zeros(ansatz.n_parameters)
    metrics, arrays = evaluate_run(MANIFEST, run, params, ansatz.generators)
    probabilities = arrays["model_probabilities"]
    assert probabilities.shape == (11, 256)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        probabilities, np.repeat(probabilities[[0]], probabilities.shape[0], axis=0)
    )
    assert len(metrics["evaluation_rows"]) == 11
    assert {row["role"] for row in metrics["evaluation_rows"]} == {
        "training",
        "held_out",
    }
    visible_run = _run(0, CurieWeissAncillaFamily.NO_CONDITION)
    visible_ansatz = ancilla_ansatz(8, 0)
    _, visible_arrays = evaluate_run(
        MANIFEST,
        visible_run,
        np.zeros(visible_ansatz.n_parameters),
        visible_ansatz.generators,
    )
    np.testing.assert_allclose(probabilities, visible_arrays["model_probabilities"])


def test_store_rejects_undeclared_and_partial_results(tmp_path):
    store = CurieWeissAncillaCampaignStore(tmp_path / "store", MANIFEST)
    run = _run(0, CurieWeissAncillaFamily.CONDITIONED)
    store.reserve(run, require_preflight=False)
    with pytest.raises(PartialAncillaResultError):
        store.reserve(run, require_preflight=False)
    with pytest.raises(ProtocolValidationError):
        replace(run, ancillas=2)
    with pytest.raises(UndeclaredAncillaRunError):
        store.validate_declared(replace(run, configuration_digest="0" * 64))


def test_failed_preflight_is_archived_but_cannot_release_runs(tmp_path):
    store = CurieWeissAncillaCampaignStore(tmp_path / "store", MANIFEST)
    evidence = {
        "manifest_digest": MANIFEST.digest,
        "passed": False,
        "projected_seconds_per_model": 3_901.0,
        "performance_gate": False,
    }
    store.record_preflight(evidence)
    assert store.preflight_path.exists()
    with pytest.raises(CurieWeissAncillaCampaignError, match="preflight"):
        store.reserve(_run(9, CurieWeissAncillaFamily.CONDITIONED))


def test_raw_oracle_and_paired_bootstrap_boundaries():
    training = np.asarray(MANIFEST.training_betas)
    references = np.stack(
        [exact_curie_weiss_target(8, beta).probabilities for beta in training]
    )
    selected = _raw_oracle(training, references, 1.0)
    left = references[2]
    right = references[3]
    target = exact_curie_weiss_target(8, 1.0).probabilities
    assert np.isclose(
        np.abs(selected - target).sum(),
        min(np.abs(left - target).sum(), np.abs(right - target).sum()),
    )
    supported = _interval({seed: 0.1 for seed in range(10)})
    mixed = _interval({seed: (-1.0) ** seed for seed in range(10)})
    unsupported = _interval({seed: -0.1 for seed in range(10)})
    assert supported["status"] == "supported"
    assert mixed["status"] == "mixed"
    assert unsupported["status"] == "unsupported"


def test_training_budget_and_matching_are_frozen():
    conditioned = _run(9, CurieWeissAncillaFamily.CONDITIONED, seed=4)
    no_condition = _run(9, CurieWeissAncillaFamily.NO_CONDITION, seed=4)
    reference = _run(
        9,
        CurieWeissAncillaFamily.TEMPERATURE_REFERENCE,
        seed=4,
        beta=0.9,
    )
    conditioned_config = training_config(MANIFEST, conditioned)
    no_condition_config = training_config(MANIFEST, no_condition)
    reference_config = training_config(MANIFEST, reference)
    assert conditioned_config == no_condition_config
    assert conditioned_config.steps == 2_000
    assert conditioned_config.batch_size == 75
    assert conditioned_config.n_ops == conditioned_config.validation_n_ops == 256
    assert conditioned_config.n_iqp_samples == 8_192
    assert conditioned_config.validation_n_iqp_samples == 16_384
    assert reference_config.batch_size == 450


def test_tiny_end_to_end_is_resumable_and_refuses_duplicate_completion():
    results = ROOT / "results"
    with TemporaryDirectory(dir=results, prefix=".ancilla-e2e-") as temporary:
        store = CurieWeissAncillaCampaignStore(Path(temporary) / "store", MANIFEST)
        run = _run(0, CurieWeissAncillaFamily.CONDITIONED)
        result = execute_run(
            ROOT,
            MANIFEST,
            store,
            run,
            invocation=("pytest", "tiny"),
            updates=1,
            require_preflight=False,
            lightweight=True,
        )
        assert result.metrics["optimizer_updates"] == 1
        assert len(result.metrics["evaluation_rows"]) == 11
        resumed = execute_run(
            ROOT,
            MANIFEST,
            store,
            run,
            invocation=("pytest", "resume"),
            updates=1,
            require_preflight=False,
            lightweight=True,
        )
        assert resumed == result
        with pytest.raises(CurieWeissAncillaCampaignError, match="duplicate"):
            store.record_result(result)


def test_complete_report_is_deterministic_and_keeps_both_oracles(monkeypatch):
    class FakeStore:
        manifest = MANIFEST

        @staticmethod
        def completed_runs():
            return plan_curie_weiss_ancilla_runs(MANIFEST)

    evaluation = np.stack(
        [
            exact_curie_weiss_target(8, beta).probabilities
            for beta in MANIFEST.evaluation_betas
        ]
    )
    neutral = np.repeat(
        exact_curie_weiss_target(8, 1.0).probabilities[None, :], 11, axis=0
    )

    def probabilities(_store, run):
        if run.family is CurieWeissAncillaFamily.TEMPERATURE_REFERENCE:
            return exact_curie_weiss_target(8, run.reference_beta).probabilities[
                None, :
            ]
        if run.family is CurieWeissAncillaFamily.NO_CONDITION:
            return neutral
        return evaluation

    monkeypatch.setattr(report_module, "_probabilities", probabilities)
    first = build_report(FakeStore())
    second = build_report(FakeStore())
    assert first == second
    assert first["completed_runs"] == 180
    assert len(first["response_rows"]) == 660
    critical = next(
        row
        for row in first["response_rows"]
        if row["ancillas"] == 9 and row["seed"] == 0 and row["beta"] == 1.0
    )
    assert "profile_oracle" in critical and "raw_oracle" in critical
    assert "adjacent_mixture_profile" in critical


def test_fixed_m5_report_has_no_post_hoc_ladder_comparison(monkeypatch):
    class FakeStore:
        manifest = M5_MANIFEST

        @staticmethod
        def completed_runs():
            return plan_curie_weiss_ancilla_runs(M5_MANIFEST)

    evaluation = np.stack(
        [
            exact_curie_weiss_target(8, beta).probabilities
            for beta in M5_MANIFEST.evaluation_betas
        ]
    )
    neutral = np.repeat(
        exact_curie_weiss_target(8, 1.0).probabilities[None, :], 11, axis=0
    )

    def probabilities(_store, run):
        if run.family is CurieWeissAncillaFamily.TEMPERATURE_REFERENCE:
            return exact_curie_weiss_target(8, run.reference_beta).probabilities[
                None, :
            ]
        return (
            neutral
            if run.family is CurieWeissAncillaFamily.NO_CONDITION
            else evaluation
        )

    monkeypatch.setattr(report_module, "_probabilities", probabilities)
    report = build_report(FakeStore())
    assert report["completed_runs"] == 80
    assert report["headline_ancillas"] == 5
    assert len(report["response_rows"]) == 110
    names = {row["comparison"] for row in report["comparisons"]}
    assert "m5_vs_mixture_profile_tvd_held_out" in names
    assert not any("_vs_m0_" in name for name in names)

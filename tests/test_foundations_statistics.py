"""Publication-gate tests for the compact foundations study."""

from pathlib import Path

import pytest

from iqp_cond.campaign_statistics import PairedEvaluationRecord
from iqp_cond.foundations_protocol import FoundationsManifest
from iqp_cond.foundations_statistics import (
    finalize_conditional_control,
    finalize_interpolation,
)

MANIFEST_PATH = Path(__file__).parents[1] / "configs" / "arxiv_foundations_v1.json"


@pytest.fixture
def manifest() -> FoundationsManifest:
    return FoundationsManifest.from_json(MANIFEST_PATH)


def _records(manifest, split: str, *, fixed=0.30, no_condition=0.50, lookup=0.40):
    conditions = (
        manifest.beta_grid.training
        if split == "training"
        else manifest.beta_grid.held_out
    )
    return [
        PairedEvaluationRecord(
            coupling_instance=str(instance),
            optimizer_seed=seed,
            condition=condition,
            split=split,
            method_tvds={
                "fixed_input": fixed,
                "no_condition": no_condition,
                "condition_modulated_phase": 0.25,
                **({"oracle_adjacent": lookup} if split == "held_out" else {}),
            },
        )
        for instance in manifest.instances.primary
        for seed in manifest.optimizer_seeds.primary
        for condition in conditions
    ]


def test_conditional_control_passes_exact_tvd_and_75_percent_boundaries(manifest):
    records = _records(manifest, "training", fixed=0.35)
    # One of four instances ties the no-condition baseline in every condition.
    records = [
        PairedEvaluationRecord(
            record.coupling_instance,
            record.optimizer_seed,
            record.condition,
            {
                **record.method_tvds,
                "fixed_input": (
                    0.50
                    if record.coupling_instance == str(manifest.instances.primary[-1])
                    else 0.30
                ),
            },
            split="training",
        )
        for record in records
    ]
    result = finalize_conditional_control(manifest, records)

    assert result.training_median_fixed_tvd == pytest.approx(0.30)
    assert result.cell_win_fraction == pytest.approx(0.75)
    assert result.improvement_interval.lower > 0.0
    assert result.passed


def test_conditional_control_fails_each_binding_component(manifest):
    assert not finalize_conditional_control(
        manifest, _records(manifest, "training", fixed=0.350001)
    ).passed
    assert not finalize_conditional_control(
        manifest, _records(manifest, "training", fixed=0.50, no_condition=0.50)
    ).passed

    records = _records(manifest, "training")
    losing_instances = {str(seed) for seed in manifest.instances.primary[2:]}
    records = [
        PairedEvaluationRecord(
            record.coupling_instance,
            record.optimizer_seed,
            record.condition,
            {
                **record.method_tvds,
                "fixed_input": (
                    0.60 if record.coupling_instance in losing_instances else 0.30
                ),
            },
            split="training",
        )
        for record in records
    ]
    assert finalize_conditional_control(manifest, records).cell_win_fraction == 0.5
    assert not finalize_conditional_control(manifest, records).passed


def test_interpolation_passes_at_cell_and_instance_boundaries(manifest):
    records = _records(manifest, "held_out")
    last = str(manifest.instances.primary[-1])
    records = [
        PairedEvaluationRecord(
            record.coupling_instance,
            record.optimizer_seed,
            record.condition,
            {
                **record.method_tvds,
                "fixed_input": 0.40 if record.coupling_instance == last else 0.30,
            },
            split="held_out",
        )
        for record in records
    ]
    result = finalize_interpolation(manifest, records)

    assert result.cell_wins == 12
    assert result.cell_total == 16
    assert result.cell_win_fraction == 0.75
    assert result.instances_improved_vs_lookup == 3
    assert result.improvement_vs_lookup.lower > 0.0
    assert result.improvement_vs_no_condition.lower > 0.0
    assert result.passed


def test_interpolation_requires_both_intervals_and_lookup_win_rules(manifest):
    no_advantage = _records(
        manifest, "held_out", fixed=0.40, no_condition=0.40, lookup=0.50
    )
    assert not finalize_interpolation(manifest, no_advantage).passed

    lookup_ties = _records(
        manifest, "held_out", fixed=0.40, no_condition=0.50, lookup=0.40
    )
    result = finalize_interpolation(manifest, lookup_ties)
    assert result.cell_wins == 0
    assert result.instances_improved_vs_lookup == 0
    assert not result.passed

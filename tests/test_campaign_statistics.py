"""Tests for the paired Ising campaign statistics and frozen gates."""

import numpy as np
import pytest

from iqp_cond.campaign_statistics import (
    BOOTSTRAP_REPLICATES,
    CONDITION_MODULATED_PHASE,
    FIXED_INPUT,
    NO_CONDITION,
    ORACLE_ADJACENT,
    AncillaEvaluationRecord,
    PairedEvaluationRecord,
    ancilla_descriptive_success_rule,
    benefit_retention_gate,
    conservative_benefit_retention,
    hierarchical_bootstrap,
    interpolation_cell_win_fraction,
    interpolation_gate,
    interpolation_regressed_instances,
    oracle_adjacent_beta_lookup,
    paired_interpolation_improvement,
)


def _records(
    instances: int = 2,
    seeds: int = 2,
    conditions: int = 2,
    *,
    fixed: float = 0.4,
    phase: float = 0.2,
    no_condition: float = 0.6,
    oracle: float = 0.5,
) -> list[PairedEvaluationRecord]:
    return [
        PairedEvaluationRecord(
            coupling_instance=f"instance-{instance}",
            optimizer_seed=seed,
            condition=0.2 + 0.1 * condition,
            method_tvds={
                FIXED_INPUT: fixed,
                CONDITION_MODULATED_PHASE: phase,
                NO_CONDITION: no_condition,
                ORACLE_ADJACENT: oracle,
            },
        )
        for instance in range(instances)
        for seed in range(seeds)
        for condition in range(conditions)
    ]


def test_hierarchical_bootstrap_is_frozen_and_preserves_paired_blocks():
    records = _records()

    def paired_mean(sample: tuple[PairedEvaluationRecord, ...]) -> float:
        assert all(
            {FIXED_INPUT, CONDITION_MODULATED_PHASE, NO_CONDITION}
            <= set(record.method_tvds)
            for record in sample
        )
        return float(np.mean([record.tvd(NO_CONDITION) for record in sample]))

    first = hierarchical_bootstrap(records, paired_mean)
    second = hierarchical_bootstrap(records, paired_mean)
    assert first.replicates.shape == (BOOTSTRAP_REPLICATES,)
    assert not first.replicates.flags.writeable
    np.testing.assert_array_equal(first.replicates, second.replicates)
    assert first.point_estimate == pytest.approx(0.6)
    assert first.lower == pytest.approx(0.6)
    assert first.upper == pytest.approx(0.6)


def test_hierarchical_bootstrap_rejects_unmatched_seed_blocks_and_duplicates():
    records = _records()
    unmatched = [
        record
        for record in records
        if not (record.coupling_instance == "instance-1" and record.optimizer_seed == 1)
    ]
    with pytest.raises(ValueError, match="optimizer seeds must be matched"):
        hierarchical_bootstrap(unmatched, lambda sample: 0.0)
    with pytest.raises(ValueError, match="identities must be unique"):
        hierarchical_bootstrap([*records, records[0]], lambda sample: 0.0)


def test_benefit_retention_passes_exact_half_boundary_with_positive_interval():
    result = benefit_retention_gate(_records())
    assert result.denominators_valid
    assert result.median_retention == pytest.approx(0.5)
    assert result.bootstrap.replicates.shape == (10_000,)
    assert result.bootstrap.lower == pytest.approx(0.5)
    assert result.median_at_least_half
    assert result.lower_bound_positive
    assert result.passed


def test_benefit_retention_handles_denominator_edges_conservatively():
    zero = conservative_benefit_retention(0.4, 0.2, 0.4)
    negative = conservative_benefit_retention(0.4, 0.2, 0.5)
    assert zero.value == negative.value == 0.0
    assert not zero.denominator_valid
    assert not negative.denominator_valid

    records = _records()
    bad = records[0]
    records[0] = PairedEvaluationRecord(
        bad.coupling_instance,
        bad.optimizer_seed,
        bad.condition,
        {
            FIXED_INPUT: 0.2,
            CONDITION_MODULATED_PHASE: 0.6,
            NO_CONDITION: 0.6,
            ORACLE_ADJACENT: 0.5,
        },
    )
    result = benefit_retention_gate(records)
    assert not result.denominators_valid
    assert result.invalid_denominator_cells == (bad.identity,)
    assert not result.passed

    missing_phase = _records()
    first = missing_phase[0]
    missing_phase[0] = PairedEvaluationRecord(
        first.coupling_instance,
        first.optimizer_seed,
        first.condition,
        {FIXED_INPUT: 0.4, NO_CONDITION: 0.6, ORACLE_ADJACENT: 0.5},
    )
    with pytest.raises(ValueError, match="missing method"):
        benefit_retention_gate(missing_phase)


def test_oracle_adjacent_beta_lookup_uses_better_neighbor_and_lower_tie():
    target = np.array([0.5, 0.5])
    outputs = np.array([[1.0, 0.0], [0.6, 0.4], [0.2, 0.8]])
    result = oracle_adjacent_beta_lookup(0.75, [0.5, 1.0, 1.5], outputs, target)
    assert result.lower_condition == 0.5
    assert result.upper_condition == 1.0
    assert result.selected_condition == 1.0
    assert result.tvd == pytest.approx(0.1)

    tie_outputs = np.array([[0.7, 0.3], [0.3, 0.7]])
    tied = oracle_adjacent_beta_lookup(0.5, [0.0, 1.0], tie_outputs, target)
    assert tied.selected_condition == 0.0
    with pytest.raises(ValueError, match="must not be a Training"):
        oracle_adjacent_beta_lookup(1.0, [0.0, 1.0], tie_outputs, target)


def _interpolation_records(
    improvements: list[float],
) -> list[PairedEvaluationRecord]:
    oracle = 0.5
    return [
        PairedEvaluationRecord(
            f"instance-{instance}",
            0,
            0.75,
            {FIXED_INPUT: oracle - improvement, ORACLE_ADJACENT: oracle},
        )
        for instance, improvement in enumerate(improvements)
    ]


def test_interpolation_gate_passes_all_three_components_at_75_percent_boundary():
    records = _interpolation_records([0.2, 0.2, 0.2, 0.0])
    improvement = paired_interpolation_improvement(records)
    wins, total, fraction = interpolation_cell_win_fraction(records)
    regressions = interpolation_regressed_instances(records)
    result = interpolation_gate(records)

    assert improvement.lower > 0.0
    assert (wins, total, fraction) == (3, 4, 0.75)
    assert regressions == ()
    assert result.interval_strictly_positive
    assert result.cell_win_rule_passed
    assert result.no_instance_regression
    assert result.passed


def test_interpolation_gate_exercises_each_failure_component():
    interval_failure = interpolation_gate(_interpolation_records([0.0] * 4))
    assert not interval_failure.interval_strictly_positive
    assert interval_failure.no_instance_regression
    assert not interval_failure.passed

    cell_failure = interpolation_gate(
        _interpolation_records([0.2] * 5 + [0.0] * 3),
    )
    assert cell_failure.interval_strictly_positive
    assert cell_failure.cell_win_fraction == 0.625
    assert not cell_failure.cell_win_rule_passed
    assert cell_failure.no_instance_regression
    assert not cell_failure.passed

    regression_failure = interpolation_gate(
        _interpolation_records([0.2, 0.2, 0.2, -0.01]),
    )
    assert regression_failure.interval_strictly_positive
    assert regression_failure.cell_win_fraction == 0.75
    assert regression_failure.regressed_instances == ("instance-3",)
    assert not regression_failure.no_instance_regression
    assert not regression_failure.passed


def _ancilla_records() -> list[AncillaEvaluationRecord]:
    records = []
    for rung in (3, 5):
        for instance in range(2):
            for seed in range(2):
                if rung == 3:
                    ancilla = 0.31 if (instance, seed) == (1, 1) else 0.2
                else:
                    ancilla = 0.35
                records.append(
                    AncillaEvaluationRecord(
                        coupling_instance=f"instance-{instance}",
                        optimizer_seed=seed,
                        condition=0.75,
                        rung=rung,
                        ancilla_tvd=ancilla,
                        m0_tvd=0.3,
                        matched_capacity_tvd=0.3,
                    ),
                )
    return records


def test_ancilla_rule_reports_both_rungs_and_passes_one_at_75_percent():
    result = ancilla_descriptive_success_rule(_ancilla_records())
    m3, m5 = result.rungs
    assert (m3.rung, m5.rung) == (3, 5)
    assert m3.win_fraction_vs_m0 == pytest.approx(0.75)
    assert m3.win_fraction_vs_matched_capacity == pytest.approx(0.75)
    assert m3.median_rules_passed
    assert m3.no_instance_regression
    assert m3.passed
    assert not m5.passed
    assert result.successful_rungs == (3,)
    assert result.passed
    assert not result.population_claim_allowed


def test_ancilla_rule_requires_both_rungs_and_instance_guardrail():
    only_m3 = [record for record in _ancilla_records() if record.rung == 3]
    with pytest.raises(ValueError, match="both m=3 and m=5"):
        ancilla_descriptive_success_rule(only_m3)

    records = _ancilla_records()
    records = [
        AncillaEvaluationRecord(
            record.coupling_instance,
            record.optimizer_seed,
            record.condition,
            record.rung,
            0.5
            if record.rung == 3 and record.coupling_instance == "instance-1"
            else record.ancilla_tvd,
            record.m0_tvd,
            record.matched_capacity_tvd,
        )
        for record in records
    ]
    result = ancilla_descriptive_success_rule(records)
    m3 = result.rungs[0]
    assert m3.regressed_instances_vs_m0 == ("instance-1",)
    assert not m3.no_instance_regression
    assert not m3.passed

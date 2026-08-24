"""Binding paired publication decisions for ``arxiv_foundations_v1``."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .campaign_statistics import (
    FIXED_INPUT,
    NO_CONDITION,
    ORACLE_ADJACENT,
    BootstrapResult,
    PairedEvaluationRecord,
    hierarchical_bootstrap,
)
from .foundations_protocol import FoundationsManifest


@dataclass(frozen=True, slots=True)
class ConditionalControlDecision:
    training_median_fixed_tvd: float
    improvement_interval: BootstrapResult
    cell_wins: int
    cell_total: int
    cell_win_fraction: float
    median_rule_passed: bool
    interval_rule_passed: bool
    cell_rule_passed: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class InterpolationDecision:
    improvement_vs_no_condition: BootstrapResult
    improvement_vs_lookup: BootstrapResult
    cell_wins: int
    cell_total: int
    cell_win_fraction: float
    instances_improved_vs_lookup: int
    improved_instances: tuple[str, ...]
    no_condition_interval_passed: bool
    lookup_interval_passed: bool
    cell_rule_passed: bool
    instance_rule_passed: bool
    passed: bool


def _validate_matrix(
    manifest: FoundationsManifest,
    records: Sequence[PairedEvaluationRecord],
    *,
    split: str,
    required_methods: set[str],
) -> tuple[PairedEvaluationRecord, ...]:
    values = tuple(records)
    expected_conditions = set(
        manifest.beta_grid.training
        if split == "training"
        else manifest.beta_grid.held_out
    )
    expected_instances = {str(value) for value in manifest.instances.primary}
    expected_seeds = set(manifest.optimizer_seeds.primary)
    expected_count = (
        len(expected_instances) * len(expected_seeds) * len(expected_conditions)
    )
    identities = {record.identity for record in values}
    if len(values) != expected_count or len(identities) != expected_count:
        raise ValueError("evaluation matrix has missing or duplicate paired cells")
    if {record.split for record in values} != {split}:
        raise ValueError(f"evaluation matrix must contain only {split} cells")
    if {record.coupling_instance for record in values} != expected_instances:
        raise ValueError("evaluation matrix instances differ from the primary set")
    if {record.optimizer_seed for record in values} != expected_seeds:
        raise ValueError("evaluation matrix optimizer seeds differ")
    if {record.condition for record in values} != expected_conditions:
        raise ValueError("evaluation matrix conditions differ from the frozen grid")
    if any(not required_methods.issubset(record.method_tvds) for record in values):
        raise ValueError("evaluation matrix is missing a required paired method")
    return values


def _mean_improvement(
    records: tuple[PairedEvaluationRecord, ...], comparator: str
) -> float:
    return float(
        np.mean(
            [record.tvd(comparator) - record.tvd(FIXED_INPUT) for record in records]
        )
    )


def _averaged_cells(
    records: tuple[PairedEvaluationRecord, ...], comparator: str
) -> dict[tuple[str, float], float]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record.coupling_instance, record.condition)].append(
            record.tvd(comparator) - record.tvd(FIXED_INPUT)
        )
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def finalize_conditional_control(
    manifest: FoundationsManifest,
    records: Sequence[PairedEvaluationRecord],
) -> ConditionalControlDecision:
    """Apply all three preregistered conditional-control requirements."""
    values = _validate_matrix(
        manifest,
        records,
        split="training",
        required_methods={FIXED_INPUT, NO_CONDITION},
    )
    median = float(np.median([record.tvd(FIXED_INPUT) for record in values]))
    interval = hierarchical_bootstrap(
        values,
        lambda sample: _mean_improvement(sample, NO_CONDITION),
        n_replicates=manifest.publication.bootstrap_replicates,
        seed=manifest.publication.bootstrap_seed,
        confidence_level=manifest.publication.confidence_level,
    )
    cells = _averaged_cells(values, NO_CONDITION)
    wins = sum(improvement > 0.0 for improvement in cells.values())
    fraction = wins / len(cells)
    median_passed = median <= manifest.publication.training_median_tvd_max
    interval_passed = interval.lower > 0.0
    cell_passed = fraction >= manifest.publication.cell_win_fraction_min
    return ConditionalControlDecision(
        training_median_fixed_tvd=median,
        improvement_interval=interval,
        cell_wins=wins,
        cell_total=len(cells),
        cell_win_fraction=fraction,
        median_rule_passed=median_passed,
        interval_rule_passed=interval_passed,
        cell_rule_passed=cell_passed,
        passed=median_passed and interval_passed and cell_passed,
    )


def finalize_interpolation(
    manifest: FoundationsManifest,
    records: Sequence[PairedEvaluationRecord],
) -> InterpolationDecision:
    """Apply the within-range held-out interpolation publication gate."""
    values = _validate_matrix(
        manifest,
        records,
        split="held_out",
        required_methods={FIXED_INPUT, NO_CONDITION, ORACLE_ADJACENT},
    )
    no_condition_interval = hierarchical_bootstrap(
        values,
        lambda sample: _mean_improvement(sample, NO_CONDITION),
        n_replicates=manifest.publication.bootstrap_replicates,
        seed=manifest.publication.bootstrap_seed,
        confidence_level=manifest.publication.confidence_level,
    )
    lookup_interval = hierarchical_bootstrap(
        values,
        lambda sample: _mean_improvement(sample, ORACLE_ADJACENT),
        n_replicates=manifest.publication.bootstrap_replicates,
        seed=manifest.publication.bootstrap_seed,
        confidence_level=manifest.publication.confidence_level,
    )
    cells = _averaged_cells(values, ORACLE_ADJACENT)
    wins = sum(improvement > 0.0 for improvement in cells.values())
    fraction = wins / len(cells)
    instance_improvements = {
        instance: float(
            np.mean(
                [
                    improvement
                    for (cell_instance, _), improvement in cells.items()
                    if cell_instance == instance
                ]
            )
        )
        for instance in sorted({key[0] for key in cells})
    }
    improved = tuple(
        instance
        for instance, improvement in instance_improvements.items()
        if improvement > 0.0
    )
    no_condition_passed = no_condition_interval.lower > 0.0
    lookup_passed = lookup_interval.lower > 0.0
    cell_passed = fraction >= manifest.publication.cell_win_fraction_min
    instance_passed = len(improved) >= manifest.publication.min_instances_improved
    return InterpolationDecision(
        improvement_vs_no_condition=no_condition_interval,
        improvement_vs_lookup=lookup_interval,
        cell_wins=wins,
        cell_total=len(cells),
        cell_win_fraction=fraction,
        instances_improved_vs_lookup=len(improved),
        improved_instances=improved,
        no_condition_interval_passed=no_condition_passed,
        lookup_interval_passed=lookup_passed,
        cell_rule_passed=cell_passed,
        instance_rule_passed=instance_passed,
        passed=(
            no_condition_passed and lookup_passed and cell_passed and instance_passed
        ),
    )

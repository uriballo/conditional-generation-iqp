"""Paired statistical gates for the frozen Ising campaign.

Coupling instances are the primary scientific replicates.  Optimizer seeds
are nested within instances, while Conditions and method values remain paired
inside :class:`PairedEvaluationRecord` throughout resampling.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .exact import total_variation_distance

BOOTSTRAP_REPLICATES = 10_000
HIERARCHICAL_BOOTSTRAP_SEED = 20_260_810
CONFIDENCE_LEVEL = 0.95

FIXED_INPUT = "fixed_input"
CONDITION_MODULATED_PHASE = "condition_modulated_phase"
NO_CONDITION = "no_condition"
ORACLE_ADJACENT = "oracle_adjacent"


@dataclass(frozen=True, slots=True)
class PairedEvaluationRecord:
    """One paired Bitstring-TVD cell for all compared methods."""

    coupling_instance: str
    optimizer_seed: int
    condition: float
    method_tvds: Mapping[str, float]
    split: str = "held_out"

    def __post_init__(self) -> None:
        instance = str(self.coupling_instance)
        if not instance:
            raise ValueError("coupling_instance must not be empty")
        if self.optimizer_seed < 0:
            raise ValueError("optimizer_seed must be non-negative")
        if not np.isfinite(self.condition):
            raise ValueError("condition must be finite")
        if self.split not in {"training", "held_out"}:
            raise ValueError("split must be 'training' or 'held_out'")
        if not self.method_tvds:
            raise ValueError("method_tvds must not be empty")
        normalized = {}
        for method, value in sorted(self.method_tvds.items()):
            if not method:
                raise ValueError("method names must not be empty")
            metric = float(value)
            if not np.isfinite(metric) or metric < 0.0 or metric > 1.0:
                raise ValueError("Bitstring TVD values must be finite and in [0, 1]")
            normalized[str(method)] = metric
        object.__setattr__(self, "coupling_instance", instance)
        object.__setattr__(self, "method_tvds", MappingProxyType(normalized))

    @property
    def identity(self) -> tuple[str, int, str, float]:
        """Return the pairing-preserving evaluation-cell identity."""
        return (
            self.coupling_instance,
            self.optimizer_seed,
            self.split,
            self.condition,
        )

    def tvd(self, method: str) -> float:
        """Return one method's TVD, rejecting an incomplete pairing."""
        try:
            return self.method_tvds[method]
        except KeyError as exc:
            raise ValueError(
                f"record {self.identity} is missing method {method!r}",
            ) from exc


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Point estimate and frozen percentile-bootstrap distribution."""

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    seed: int
    replicates: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.replicates, dtype=np.float64).copy()
        values.setflags(write=False)
        object.__setattr__(self, "replicates", values)


@dataclass(frozen=True, slots=True)
class RetentionValue:
    """One conservative fixed-circuit benefit-retention calculation."""

    value: float
    numerator: float
    denominator: float
    denominator_valid: bool


@dataclass(frozen=True, slots=True)
class BenefitRetentionGateResult:
    """Binding median-retention and uncertainty decision."""

    median_retention: float
    bootstrap: BootstrapResult
    denominators_valid: bool
    invalid_denominator_cells: tuple[tuple[str, int, str, float], ...]
    median_at_least_half: bool
    lower_bound_positive: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class OracleLookupResult:
    """Adjacent-Training-Condition Oracle result for one Held-out Target."""

    heldout_condition: float
    lower_condition: float
    upper_condition: float
    selected_condition: float
    lower_tvd: float
    upper_tvd: float
    tvd: float


@dataclass(frozen=True, slots=True)
class InterpolationGateResult:
    """All three binding Ising interpolation components."""

    paired_improvement: BootstrapResult
    interval_strictly_positive: bool
    cell_wins: int
    cell_total: int
    cell_win_fraction: float
    cell_win_rule_passed: bool
    regressed_instances: tuple[str, ...]
    no_instance_regression: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class AncillaEvaluationRecord:
    """One paired ancilla, primary, and Matched-capacity evaluation cell."""

    coupling_instance: str
    optimizer_seed: int
    condition: float
    rung: int
    ancilla_tvd: float
    m0_tvd: float
    matched_capacity_tvd: float

    def __post_init__(self) -> None:
        if self.rung not in {3, 5}:
            raise ValueError("ancilla rung must be 3 or 5")
        if self.optimizer_seed < 0:
            raise ValueError("optimizer_seed must be non-negative")
        if not str(self.coupling_instance):
            raise ValueError("coupling_instance must not be empty")
        if not np.isfinite(self.condition):
            raise ValueError("condition must be finite")
        metrics = (self.ancilla_tvd, self.m0_tvd, self.matched_capacity_tvd)
        if any(
            not np.isfinite(value) or value < 0.0 or value > 1.0 for value in metrics
        ):
            raise ValueError("all Bitstring TVDs must be finite and in [0, 1]")
        object.__setattr__(self, "coupling_instance", str(self.coupling_instance))

    @property
    def cell_identity(self) -> tuple[str, int, float]:
        """Return the instance-seed-Condition pairing key."""
        return (self.coupling_instance, self.optimizer_seed, self.condition)


@dataclass(frozen=True, slots=True)
class AncillaRungResult:
    """Every component of one descriptive ancilla-rung rule."""

    rung: int
    median_improvement_vs_m0: float
    median_improvement_vs_matched_capacity: float
    median_rules_passed: bool
    wins_vs_m0: int
    wins_vs_matched_capacity: int
    cell_total: int
    win_fraction_vs_m0: float
    win_fraction_vs_matched_capacity: float
    cell_win_rules_passed: bool
    regressed_instances_vs_m0: tuple[str, ...]
    no_instance_regression: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class AncillaSuccessResult:
    """Two-rung descriptive ancilla decision; both rungs are mandatory."""

    rungs: tuple[AncillaRungResult, AncillaRungResult]
    successful_rungs: tuple[int, ...]
    passed: bool
    population_claim_allowed: bool = False


def _validated_paired_records(
    records: Sequence[PairedEvaluationRecord],
) -> tuple[PairedEvaluationRecord, ...]:
    values = tuple(records)
    if not values:
        raise ValueError("records must not be empty")
    identities = [record.identity for record in values]
    if len(set(identities)) != len(identities):
        raise ValueError("paired evaluation identities must be unique")

    records_by_instance_seed: dict[
        tuple[str, int],
        list[PairedEvaluationRecord],
    ] = defaultdict(list)
    seeds_by_instance: dict[str, set[int]] = defaultdict(set)
    for record in values:
        records_by_instance_seed[
            (record.coupling_instance, record.optimizer_seed)
        ].append(record)
        seeds_by_instance[record.coupling_instance].add(record.optimizer_seed)

    seed_sets = {tuple(sorted(seeds)) for seeds in seeds_by_instance.values()}
    if len(seed_sets) != 1:
        raise ValueError("optimizer seeds must be matched across coupling instances")

    reference_cells: tuple[tuple[str, float], ...] | None = None
    reference_methods: tuple[str, ...] | None = None
    for cell_records in records_by_instance_seed.values():
        cells = tuple(
            sorted((record.split, record.condition) for record in cell_records),
        )
        methods = {tuple(record.method_tvds) for record in cell_records}
        if len(methods) != 1:
            raise ValueError("all Conditions must preserve the same method pairing")
        current_methods = next(iter(methods))
        if reference_cells is None:
            reference_cells = cells
            reference_methods = current_methods
        elif cells != reference_cells or current_methods != reference_methods:
            raise ValueError(
                "all instance-seed blocks must preserve Conditions and methods",
            )
    return tuple(sorted(values, key=lambda record: record.identity))


def hierarchical_bootstrap(
    records: Sequence[PairedEvaluationRecord],
    statistic: Callable[[tuple[PairedEvaluationRecord, ...]], float],
    *,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = HIERARCHICAL_BOOTSTRAP_SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> BootstrapResult:
    """Resample instances, then seeds within each selected instance.

    Each selected seed contributes its complete collection of paired method and
    Condition records.  Therefore no resampling step can break a comparison's
    pairing or independently resample Held-out Conditions.
    """
    ordered = _validated_paired_records(records)
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    grouped: dict[tuple[str, int], tuple[PairedEvaluationRecord, ...]] = {}
    instances = sorted({record.coupling_instance for record in ordered})
    seeds_by_instance: dict[str, list[int]] = {}
    for instance in instances:
        seeds = sorted(
            {
                record.optimizer_seed
                for record in ordered
                if record.coupling_instance == instance
            },
        )
        seeds_by_instance[instance] = seeds
        for optimizer_seed in seeds:
            grouped[(instance, optimizer_seed)] = tuple(
                record
                for record in ordered
                if record.coupling_instance == instance
                and record.optimizer_seed == optimizer_seed
            )

    rng = np.random.default_rng(seed)
    replicate_values = np.empty(n_replicates, dtype=np.float64)
    for replicate in range(n_replicates):
        selected_records: list[PairedEvaluationRecord] = []
        instance_indices = rng.integers(0, len(instances), size=len(instances))
        for instance_index in instance_indices:
            instance = instances[int(instance_index)]
            seeds = seeds_by_instance[instance]
            seed_indices = rng.integers(0, len(seeds), size=len(seeds))
            for seed_index in seed_indices:
                selected_records.extend(grouped[(instance, seeds[int(seed_index)])])
        replicate_values[replicate] = float(statistic(tuple(selected_records)))

    point_estimate = float(statistic(ordered))
    if not np.isfinite(point_estimate) or not np.all(np.isfinite(replicate_values)):
        raise ValueError("bootstrap statistic must be finite for every sample")
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(replicate_values, (tail, 1.0 - tail))
    return BootstrapResult(
        point_estimate=point_estimate,
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        seed=seed,
        replicates=replicate_values,
    )


def conservative_benefit_retention(
    no_condition_tvd: float,
    fixed_input_tvd: float,
    phase_conditioned_tvd: float,
    *,
    denominator_tolerance: float = 1e-12,
) -> RetentionValue:
    """Return retention, mapping a nonpositive denominator to a failed zero.

    A zero or negative No-condition-to-phase improvement cannot provide a
    favorable infinite or sign-reversed retention score.  Such a cell is
    marked invalid, assigned zero for resampling, and causes the binding gate
    to fail even if the remaining cells are favorable.
    """
    values = np.asarray(
        [no_condition_tvd, fixed_input_tvd, phase_conditioned_tvd],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("TVD inputs must be finite and in [0, 1]")
    if denominator_tolerance < 0.0:
        raise ValueError("denominator_tolerance must be non-negative")
    numerator = float(no_condition_tvd - fixed_input_tvd)
    denominator = float(no_condition_tvd - phase_conditioned_tvd)
    valid = denominator > denominator_tolerance
    return RetentionValue(
        value=numerator / denominator if valid else 0.0,
        numerator=numerator,
        denominator=denominator,
        denominator_valid=valid,
    )


def _require_held_out(records: Sequence[PairedEvaluationRecord]) -> None:
    if any(record.split != "held_out" for record in records):
        raise ValueError("binding empirical gates accept Held-out records only")


def _retention(record: PairedEvaluationRecord) -> RetentionValue:
    return conservative_benefit_retention(
        record.tvd(NO_CONDITION),
        record.tvd(FIXED_INPUT),
        record.tvd(CONDITION_MODULATED_PHASE),
    )


def benefit_retention_gate(
    records: Sequence[PairedEvaluationRecord],
) -> BenefitRetentionGateResult:
    """Apply the median-half and strictly-positive-lower-bound rules."""
    for record in records:
        record.tvd(NO_CONDITION)
        record.tvd(FIXED_INPUT)
        record.tvd(CONDITION_MODULATED_PHASE)
    ordered = _validated_paired_records(records)
    _require_held_out(ordered)
    retention_by_identity = {record.identity: _retention(record) for record in ordered}

    def statistic(sample: tuple[PairedEvaluationRecord, ...]) -> float:
        return float(
            np.median(
                [retention_by_identity[record.identity].value for record in sample],
            ),
        )

    bootstrap = hierarchical_bootstrap(ordered, statistic)
    invalid = tuple(
        identity
        for identity, retention in retention_by_identity.items()
        if not retention.denominator_valid
    )
    median_passed = bootstrap.point_estimate >= 0.5 or np.isclose(
        bootstrap.point_estimate,
        0.5,
        rtol=0.0,
        atol=1e-12,
    )
    lower_passed = bootstrap.lower > 0.0
    denominators_valid = not invalid
    return BenefitRetentionGateResult(
        median_retention=bootstrap.point_estimate,
        bootstrap=bootstrap,
        denominators_valid=denominators_valid,
        invalid_denominator_cells=invalid,
        median_at_least_half=median_passed,
        lower_bound_positive=lower_passed,
        passed=denominators_valid and median_passed and lower_passed,
    )


def _probability_vector(values: np.ndarray, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise ValueError(f"{name} must be a one-dimensional probability vector")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} must be normalized")
    return probabilities


def oracle_adjacent_beta_lookup(
    heldout_beta: float,
    training_betas: Sequence[float],
    training_probabilities: np.ndarray,
    heldout_target_probabilities: np.ndarray,
) -> OracleLookupResult:
    """Choose the better of the two adjacent Training outputs by exact TVD."""
    betas = np.asarray(training_betas, dtype=np.float64)
    if betas.ndim != 1 or betas.size < 2 or not np.all(np.isfinite(betas)):
        raise ValueError("training_betas must contain at least two finite values")
    if np.any(np.diff(betas) <= 0.0):
        raise ValueError("training_betas must be strictly increasing")
    if not np.isfinite(heldout_beta):
        raise ValueError("heldout_beta must be finite")
    if np.any(np.isclose(betas, heldout_beta, rtol=0.0, atol=1e-14)):
        raise ValueError("heldout_beta must not be a Training Condition")
    upper_index = int(np.searchsorted(betas, heldout_beta, side="right"))
    if upper_index == 0 or upper_index == betas.size:
        raise ValueError("heldout_beta must lie inside the Training interval")

    outputs = np.asarray(training_probabilities, dtype=np.float64)
    if outputs.ndim != 2 or outputs.shape[0] != betas.size:
        raise ValueError(
            "training_probabilities must have shape (n_training_betas, n_states)",
        )
    target = _probability_vector(
        heldout_target_probabilities,
        "heldout_target_probabilities",
    )
    if outputs.shape[1] != target.size:
        raise ValueError("Training outputs and Held-out Target must align")
    for index, output in enumerate(outputs):
        _probability_vector(output, f"training_probabilities[{index}]")

    lower_index = upper_index - 1
    lower_tvd = total_variation_distance(target, outputs[lower_index])
    upper_tvd = total_variation_distance(target, outputs[upper_index])
    selected_index = lower_index if lower_tvd <= upper_tvd else upper_index
    return OracleLookupResult(
        heldout_condition=float(heldout_beta),
        lower_condition=float(betas[lower_index]),
        upper_condition=float(betas[upper_index]),
        selected_condition=float(betas[selected_index]),
        lower_tvd=lower_tvd,
        upper_tvd=upper_tvd,
        tvd=min(lower_tvd, upper_tvd),
    )


def paired_interpolation_improvement(
    records: Sequence[PairedEvaluationRecord],
) -> BootstrapResult:
    """Bootstrap the aggregate paired Oracle-minus-fixed TVD improvement."""
    ordered = _validated_paired_records(records)
    _require_held_out(ordered)

    def statistic(sample: tuple[PairedEvaluationRecord, ...]) -> float:
        return float(
            np.mean(
                [
                    record.tvd(ORACLE_ADJACENT) - record.tvd(FIXED_INPUT)
                    for record in sample
                ],
            ),
        )

    return hierarchical_bootstrap(ordered, statistic)


def interpolation_cell_win_fraction(
    records: Sequence[PairedEvaluationRecord],
) -> tuple[int, int, float]:
    """Compare seed-averaged instance-by-Condition cells."""
    ordered = _validated_paired_records(records)
    _require_held_out(ordered)
    cells: dict[tuple[str, float], list[PairedEvaluationRecord]] = defaultdict(list)
    for record in ordered:
        cells[(record.coupling_instance, record.condition)].append(record)
    wins = 0
    for cell_records in cells.values():
        fixed = np.mean([record.tvd(FIXED_INPUT) for record in cell_records])
        oracle = np.mean([record.tvd(ORACLE_ADJACENT) for record in cell_records])
        wins += int(fixed < oracle)
    total = len(cells)
    return wins, total, wins / total


def interpolation_regressed_instances(
    records: Sequence[PairedEvaluationRecord],
) -> tuple[str, ...]:
    """Return instances whose mean fixed TVD is worse than Oracle."""
    ordered = _validated_paired_records(records)
    _require_held_out(ordered)
    grouped: dict[str, list[PairedEvaluationRecord]] = defaultdict(list)
    for record in ordered:
        grouped[record.coupling_instance].append(record)
    return tuple(
        instance
        for instance, instance_records in sorted(grouped.items())
        if np.mean([record.tvd(FIXED_INPUT) for record in instance_records])
        > np.mean([record.tvd(ORACLE_ADJACENT) for record in instance_records])
    )


def interpolation_gate(
    records: Sequence[PairedEvaluationRecord],
) -> InterpolationGateResult:
    """Apply interval, 75% cell-win, and no-instance-regression rules."""
    improvement = paired_interpolation_improvement(records)
    wins, total, fraction = interpolation_cell_win_fraction(records)
    regressions = interpolation_regressed_instances(records)
    interval_passed = improvement.lower > 0.0
    cell_passed = fraction >= 0.75
    no_regression = not regressions
    return InterpolationGateResult(
        paired_improvement=improvement,
        interval_strictly_positive=interval_passed,
        cell_wins=wins,
        cell_total=total,
        cell_win_fraction=fraction,
        cell_win_rule_passed=cell_passed,
        regressed_instances=regressions,
        no_instance_regression=no_regression,
        passed=interval_passed and cell_passed and no_regression,
    )


def _ancilla_rung_result(
    rung: int,
    records: tuple[AncillaEvaluationRecord, ...],
) -> AncillaRungResult:
    improvements_m0 = np.asarray(
        [record.m0_tvd - record.ancilla_tvd for record in records],
    )
    improvements_matched = np.asarray(
        [record.matched_capacity_tvd - record.ancilla_tvd for record in records],
    )
    median_m0 = float(np.median(improvements_m0))
    median_matched = float(np.median(improvements_matched))
    median_passed = median_m0 > 0.0 and median_matched > 0.0
    wins_m0 = int(np.sum(improvements_m0 > 0.0))
    wins_matched = int(np.sum(improvements_matched > 0.0))
    total = len(records)
    fraction_m0 = wins_m0 / total
    fraction_matched = wins_matched / total
    cells_passed = fraction_m0 >= 0.75 and fraction_matched >= 0.75

    by_instance: dict[str, list[AncillaEvaluationRecord]] = defaultdict(list)
    for record in records:
        by_instance[record.coupling_instance].append(record)
    regressions = tuple(
        instance
        for instance, instance_records in sorted(by_instance.items())
        if np.mean([record.ancilla_tvd for record in instance_records])
        > np.mean([record.m0_tvd for record in instance_records])
    )
    no_regression = not regressions
    return AncillaRungResult(
        rung=rung,
        median_improvement_vs_m0=median_m0,
        median_improvement_vs_matched_capacity=median_matched,
        median_rules_passed=median_passed,
        wins_vs_m0=wins_m0,
        wins_vs_matched_capacity=wins_matched,
        cell_total=total,
        win_fraction_vs_m0=fraction_m0,
        win_fraction_vs_matched_capacity=fraction_matched,
        cell_win_rules_passed=cells_passed,
        regressed_instances_vs_m0=regressions,
        no_instance_regression=no_regression,
        passed=median_passed and cells_passed and no_regression,
    )


def ancilla_descriptive_success_rule(
    records: Sequence[AncillaEvaluationRecord],
) -> AncillaSuccessResult:
    """Evaluate both preregistered rungs and allow no population claim."""
    values = tuple(records)
    if not values:
        raise ValueError("records must not be empty")
    identities = [(record.rung, *record.cell_identity) for record in values]
    if len(identities) != len(set(identities)):
        raise ValueError("ancilla evaluation identities must be unique")
    grouped = {
        rung: tuple(record for record in values if record.rung == rung)
        for rung in (3, 5)
    }
    if any(not rung_records for rung_records in grouped.values()):
        raise ValueError("both m=3 and m=5 rungs must be reported")
    cell_sets = {
        frozenset(record.cell_identity for record in rung_records)
        for rung_records in grouped.values()
    }
    if len(cell_sets) != 1:
        raise ValueError("both ancilla rungs must preserve the same paired cells")
    rung_results = tuple(_ancilla_rung_result(rung, grouped[rung]) for rung in (3, 5))
    successful = tuple(result.rung for result in rung_results if result.passed)
    return AncillaSuccessResult(
        rungs=(rung_results[0], rung_results[1]),
        successful_rungs=successful,
        passed=bool(successful),
    )

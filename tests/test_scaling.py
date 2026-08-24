from itertools import pairwise

import numpy as np
import pytest

from iqp_cond.scaling import (
    SCALING_CONDITION_COUNTS,
    aggregate_condition_runs,
    condition_angle_fractions,
    nested_condition_grids,
    parameter_accounting,
    sample_scaling_datasets,
    scaling_budget,
    strict_comparison_counts,
)


def test_condition_grids_are_nested_and_include_the_frozen_endpoints():
    grids = nested_condition_grids()
    for smaller, larger in pairwise(SCALING_CONDITION_COUNTS):
        smaller_rows = {tuple(row) for row in grids[smaller]}
        larger_rows = {tuple(row) for row in grids[larger]}
        assert smaller_rows <= larger_rows
    assert condition_angle_fractions(2) == pytest.approx([0.0, 0.25])
    assert condition_angle_fractions(9) == pytest.approx(np.arange(9) / 32)


@pytest.mark.parametrize(
    ("track", "n_conditions", "n_training", "n_validation"),
    [
        ("fixed_total", 2, 9_000, 1_800),
        ("fixed_total", 3, 6_000, 1_200),
        ("fixed_total", 5, 3_600, 720),
        ("fixed_total", 9, 2_000, 400),
        ("fixed_per_condition", 9, 10_000, 2_000),
    ],
)
def test_budgets_keep_total_minibatch_balanced(
    track,
    n_conditions,
    n_training,
    n_validation,
):
    budget = scaling_budget(n_conditions, track)

    assert budget.n_training_per_condition == n_training
    assert budget.n_validation_per_condition == n_validation
    assert budget.batch_size_per_condition * n_conditions == 450
    assert budget.total_batch_size == 450
    assert budget.optimizer_updates == 2_000


def test_sampled_datasets_are_condition_balanced_and_binary():
    budget = scaling_budget(2, "fixed_total")
    training, validation = sample_scaling_datasets(seed=4, budget=budget)

    assert training.shape == (2, 9_000, 19)
    assert validation.shape == (2, 1_800, 19)
    assert np.array_equal(training, training.astype(bool))
    assert np.array_equal(validation, validation.astype(bool))


def test_parameter_accounting_counts_the_separate_bank():
    accounting = parameter_accounting(5)

    assert accounting.shared_parameters == 190
    assert accounting.no_condition_parameters == 190
    assert accounting.separate_parameters_per_model == 190
    assert accounting.separate_bank_parameters == 950
    assert accounting.shared_fraction_of_separate == pytest.approx(0.2)


def _fake_run(seed, profile, bitstring):
    return {
        "seed": seed,
        "n_conditions": 3,
        "conditions": [
            {
                "condition": index,
                "angle_fraction_pi": index / 8,
                "profile_tvd": profile[index],
                "bitstring_tvd": bitstring[index],
            }
            for index in range(3)
        ],
    }


def test_aggregation_retains_mean_median_and_worst_condition_views():
    runs = [
        _fake_run(0, [0.1, 0.2, 0.4], [0.3, 0.4, 0.6]),
        _fake_run(1, [0.2, 0.3, 0.5], [0.2, 0.5, 0.7]),
    ]

    summary = aggregate_condition_runs(runs)

    assert summary["aggregate"]["mean"]["profile_tvd"]["median"] == pytest.approx(
        0.2833333333,
    )
    assert summary["aggregate"]["median"]["profile_tvd"]["median"] == pytest.approx(
        0.25,
    )
    assert summary["aggregate"]["worst"]["profile_tvd"]["median"] == pytest.approx(
        0.45,
    )
    assert summary["per_condition"][2]["profile_tvd"]["median"] == pytest.approx(
        0.45,
    )


def test_strict_comparison_counts_each_condition_failure():
    shared = [_fake_run(0, [0.1, 0.4, 0.2], [0.2, 0.6, 0.3])]
    baseline = [_fake_run(0, [0.2, 0.3, 0.2], [0.3, 0.5, 0.3])]

    counts = strict_comparison_counts(shared, baseline, "profile_tvd")

    assert counts == {"failures": 2, "total": 3}

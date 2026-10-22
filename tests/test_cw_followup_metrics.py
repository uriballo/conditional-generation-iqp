import numpy as np

from paper_experiments.cw_followup_metrics import (
    choose_confirmed,
    curve_scores,
    robust_score,
    select_checkpoint,
    shortlist,
)


def test_trapezoid_weights_and_single_point():
    betas = np.array([0.0, 1.0, 2.0])
    assert robust_score([1.0], [1.0]) == 1.0
    assert np.isclose(robust_score([0.0, 1.0, 0.0], betas), 5 / 6)
    result = curve_scores([0.0, 1.0, 0.0], [0, 0, 0], [0, 0, 0], betas)
    assert np.isclose(result["J"], 5 / 6)
    assert np.isclose(result["mean_tvd"], 0.5)
    marker = curve_scores([0.0, 1.0, 0.0], [0, 0, 0], [0, 0, 0], [0.4, 1.0, 2.0])
    assert np.isclose(marker["mean_tvd"], 0.5)


def test_invalid_objective_inputs_raise():
    import pytest

    with pytest.raises(ValueError):
        robust_score([], [])
    with pytest.raises(ValueError):
        robust_score([1, -1], [0, 1])


def test_global_pool_tolerance_and_deterministic_ties():
    records = [
        {"J": 0.10, "P": 0.9, "step": 100, "start": 1},
        {"J": 0.105, "P": 0.2, "step": 200, "start": 0},
        {"J": 0.20, "P": 0.1, "step": 0, "start": 0},
    ]
    assert select_checkpoint(records, tolerance=0.01)["step"] == 200
    tied = [
        {"J": 0.1, "P": 0.1, "step": 2, "start": 1},
        {"J": 0.1, "P": 0.1, "step": 1, "start": 2},
    ]
    assert select_checkpoint(tied)["step"] == 1
    assert choose_confirmed(records, tolerance=0.01)["step"] == 200


def test_shortlist_is_distinct_and_bounded():
    records = [
        {"trial_id": i, "J": 0.1 + i / 100, "P": 0.5 - i / 100, "config": {"x": i // 2}}
        for i in range(8)
    ]
    result = shortlist(records)
    assert 3 <= len(result) <= 4
    assert len({repr(r["config"]) for r in result}) == len(result)


def test_shortlist_order_and_guardrail():
    records = [
        {"trial_id": 0, "J": 0.1, "P": 0.8, "config": {"x": 0}},
        {"trial_id": 1, "J": 0.105, "P": 0.4, "config": {"x": 1}},
        {"trial_id": 2, "J": 0.11, "P": 0.6, "config": {"x": 2}},
        {"trial_id": 3, "J": 0.3, "P": 0.01, "config": {"x": 3}},
    ]
    assert [r["trial_id"] for r in shortlist(records)] == [0, 1, 2, 3]
    records[3]["J"] = 0.105
    assert [r["trial_id"] for r in shortlist(records)] == [0, 3, 1]


def test_canonical_duplicate_uses_sorted_representative():
    records = [
        {"trial_id": 4, "J": 0.2, "P": 0.2, "config": {"a": 1, "b": 2}},
        {"trial_id": 1, "J": 0.1, "P": 0.3, "config": {"b": 2, "a": 1}},
        {"trial_id": 2, "J": 0.15, "P": 0.2, "config": {"z": 1}},
        {"trial_id": 3, "J": 0.16, "P": 0.2, "config": {"z": 2}},
    ]
    assert shortlist(records)[0]["trial_id"] == 1

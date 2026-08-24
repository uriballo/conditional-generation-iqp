import numpy as np
import pytest

from iqp_cond.gaussian_statistics import (
    GaussianMetricRecord,
    adjacent_indices,
    barycentric_mixture,
    oracle_adjacent_lookup,
    oracle_adjacent_profile_lookup,
    paired_bootstrap_interval,
)


def test_typed_gaussian_metric_record_roundtrip_and_boundaries():
    record = GaussianMetricRecord(0.1, 0.2, 0.3, 0.4, 0.05)
    assert GaussianMetricRecord.from_dict(record.to_dict()) == record
    with pytest.raises(ValueError, match="non-negative"):
        GaussianMetricRecord(-0.1, 0.2, 0.3, 0.4, 0.05)
    with pytest.raises(ValueError, match="cannot exceed"):
        GaussianMetricRecord(1.01, 0.2, 0.3, 0.4, 0.05)


def test_target_blind_barycentric_mixture():
    conditions = np.asarray([0.0, 0.5, 1.0])
    stored = np.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    mixture, weights = barycentric_mixture(conditions, stored, 0.25)
    assert np.allclose(mixture, [0.75, 0.25])
    assert weights == {
        "left_index": 0,
        "right_index": 1,
        "left_weight": 0.5,
        "right_weight": 0.5,
    }


def test_oracle_uses_only_adjacent_bank_outputs():
    conditions = np.asarray([0.0, 0.5, 1.0])
    stored = np.asarray([[1.0, 0.0], [0.4, 0.6], [0.0, 1.0]])
    selected, details = oracle_adjacent_lookup(
        conditions, stored, np.asarray([0.45, 0.55]), 0.25
    )
    assert np.array_equal(selected, stored[1])
    assert details["selected_index"] == 1
    with pytest.raises(ValueError, match="unseen"):
        adjacent_indices(conditions, 0.5)


def test_profile_oracle_uses_the_primary_task_metric():
    conditions = np.asarray([0.0, 0.5, 1.0])
    bitstrings = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int8)
    target = np.asarray([0.5, 0.0, 0.0, 0.5])
    stored = np.asarray(
        [
            [0.5, 0.0, 0.0, 0.5],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    selected, details = oracle_adjacent_profile_lookup(
        conditions, stored, target, bitstrings, 0.25
    )
    assert np.array_equal(selected, stored[0])
    assert details["selected_index"] == 0
    assert details["selection_metric"] == "profile_tvd"


def test_paired_bootstrap_is_deterministic_and_boundary_exact():
    values = {seed: float(seed + 1) for seed in range(10)}
    first = paired_bootstrap_interval(values)
    second = paired_bootstrap_interval(values)
    assert first == second
    assert first.estimate == 5.5
    assert first.lower < first.estimate < first.upper
    with pytest.raises(ValueError):
        paired_bootstrap_interval({0: 1.0})

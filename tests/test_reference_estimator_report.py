"""Check the estimator report's symmetry and replicate-bootstrap semantics."""

import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "estimator_report",
    Path(__file__).parents[1] / "scripts/report_reference_estimator_study.py",
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_cross_layer_odd_mask_is_forbidden_only_for_symmetric_inputs():
    ops = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0]])
    np.testing.assert_array_equal(
        report._active_masks(ops, "product")["active"], [False, True, True, True]
    )
    for encoder in ["cat", "bond"]:
        masks = report._active_masks(ops, encoder)
        np.testing.assert_array_equal(masks["active"], [False, False, False, True])
        np.testing.assert_array_equal(masks["forbidden"], [False, True, True, False])


def test_fixed_case_bootstrap_does_not_resample_cases():
    records = []
    for value in [1.0, 3.0]:
        records.append(
            {
                "arrays": {
                    "sample_values": np.full((32, 1), value),
                    "exact_values": np.zeros(1),
                },
                "masks": {"active": np.array([True])},
            }
        )
    point, lo, hi = report.bootstrap_case_mse(records, "moment", seed=42)
    np.testing.assert_allclose([point, lo, hi], np.sqrt(5))

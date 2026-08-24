"""Boundary tests for the preregistered estimator selector."""

from pathlib import Path

import pytest

from iqp_cond.foundations_estimator import (
    EstimatorAuditRow,
    EstimatorSelectionError,
    select_estimator_budget,
)
from iqp_cond.foundations_protocol import FoundationsManifest

MANIFEST_PATH = Path(__file__).parents[1] / "configs" / "arxiv_foundations_v1.json"


def _row(
    budget: int,
    *,
    loss_bias: float = 0.009,
    exact_cosine: float = 0.96,
    nrmse: float = 0.19,
    loss_cv: float = 0.09,
    large_cosine: float = 0.91,
) -> EstimatorAuditRow:
    return EstimatorAuditRow(
        operator_budget=budget,
        reference_budget=budget,
        n8_absolute_loss_bias=loss_bias,
        n8_mean_gradient_cosine=exact_cosine,
        n8_normalized_gradient_rmse=nrmse,
        n16_repeated_loss_cv=loss_cv,
        n16_median_pairwise_gradient_cosine=large_cosine,
    )


def test_selector_chooses_smallest_candidate_that_meets_every_boundary():
    manifest = FoundationsManifest.from_json(MANIFEST_PATH)
    rows = (
        _row(128, loss_bias=0.0100001),
        _row(
            256,
            loss_bias=0.01,
            exact_cosine=0.95,
            nrmse=0.20,
            loss_cv=0.10,
            large_cosine=0.90,
        ),
        _row(512),
        _row(1024),
    )

    selection = select_estimator_budget(manifest, rows)

    assert selection.operator_budget == 256
    assert selection.reference_budget == 256
    assert selection.passing_budgets == (256, 512, 1024)
    assert selection.manifest_digest == manifest.digest


def test_selector_stops_if_no_candidate_passes_and_rejects_missing_candidates():
    manifest = FoundationsManifest.from_json(MANIFEST_PATH)
    with pytest.raises(EstimatorSelectionError, match="no estimator candidate"):
        select_estimator_budget(
            manifest,
            tuple(_row(budget, large_cosine=0.899) for budget in (128, 256, 512, 1024)),
        )

    with pytest.raises(ValueError, match="exactly the preregistered candidates"):
        select_estimator_budget(
            manifest,
            tuple(_row(budget) for budget in (128, 256, 512)),
        )


def test_selector_rejects_unequal_or_duplicate_operator_reference_budgets():
    manifest = FoundationsManifest.from_json(MANIFEST_PATH)
    rows = list(_row(budget) for budget in (128, 256, 512, 1024))
    rows[0] = EstimatorAuditRow(
        operator_budget=128,
        reference_budget=256,
        n8_absolute_loss_bias=0.0,
        n8_mean_gradient_cosine=1.0,
        n8_normalized_gradient_rmse=0.0,
        n16_repeated_loss_cv=0.0,
        n16_median_pairwise_gradient_cosine=1.0,
    )
    with pytest.raises(ValueError, match="equal operator/reference"):
        select_estimator_budget(manifest, tuple(rows))

    rows[0] = _row(256)
    with pytest.raises(ValueError, match="exactly the preregistered candidates"):
        select_estimator_budget(manifest, tuple(rows))

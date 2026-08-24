"""Public contracts for the compact arXiv foundations campaign."""

from pathlib import Path

import pytest

from iqp_cond.foundations_protocol import (
    FoundationsManifest,
    FoundationsTuningDecision,
    plan_foundations_primary,
    plan_foundations_representability,
    plan_foundations_runtime,
    plan_foundations_tuning,
)
from iqp_cond.protocol import ModelFamily, Stage

MANIFEST_PATH = Path(__file__).parents[1] / "configs" / "arxiv_foundations_v1.json"


@pytest.fixture
def manifest() -> FoundationsManifest:
    return FoundationsManifest.from_json(MANIFEST_PATH)


def test_foundations_manifest_freezes_compact_study_without_mutating_v1(manifest):
    assert manifest.campaign_version == "arxiv_foundations_v1"
    assert manifest.source_campaign.version == "quantum_campaign_v1"
    assert manifest.source_campaign.manifest_digest == (
        "8cf5380b7a658381c8b61b3069f921c938f92be814c091418e46d4d905f6f0f3"
    )
    assert manifest.instances.development == (606, 707)
    assert manifest.instances.primary == (666, 101, 202, 303)
    assert manifest.instances.reserved == (404, 505)
    assert manifest.data.source == "exact_gibbs_draws"
    assert manifest.data.training_samples == 5_000
    assert manifest.data.validation_samples == 2_000
    assert manifest.estimator.candidate_budgets == (128, 256, 512, 1024)
    assert manifest.training.max_updates == 1_000
    assert manifest.training.validation_interval == 50
    assert manifest.run_allocation.total == 60
    assert len(manifest.digest) == 64


def test_foundations_planners_return_only_the_frozen_unique_run_matrix(manifest):
    runtime = plan_foundations_runtime(manifest)
    representability = plan_foundations_representability(manifest)
    tuning = plan_foundations_tuning(manifest)
    decision = FoundationsTuningDecision.reference(manifest)
    primary = plan_foundations_primary(manifest, decision)

    assert len(runtime) == 1
    assert len(representability) == 12
    assert len(tuning) == 12
    assert len(primary) == 36
    assert (
        len({run.run_id for run in (*runtime, *representability, *tuning, *primary)})
        == 61
    )
    assert {run.stage for run in representability} == {
        Stage.REPRESENTABILITY_SCREEN,
    }
    assert {run.condition_scope for run in representability} == {
        "train_beta_0",
        "train_beta_2",
        "train_beta_4",
    }
    assert {run.model_family for run in tuning} == {
        ModelFamily.FIXED_INPUT,
        ModelFamily.CONDITION_MODULATED_PHASE,
        ModelFamily.NO_CONDITION,
    }
    assert {run.coupling_seed for run in primary} == set(
        manifest.instances.primary,
    )
    assert {run.optimizer_seed for run in primary} == {0, 1, 2}
    assert all(run.configuration_digest == manifest.digest for run in primary)


def test_primary_planning_rejects_incomplete_or_foreign_tuning_decisions(manifest):
    decision = FoundationsTuningDecision.reference(manifest)
    with pytest.raises(ValueError, match="exactly one selection per shared family"):
        FoundationsTuningDecision(
            manifest_digest=manifest.digest,
            evidence_digest="0" * 64,
            selections=decision.selections[:-1],
        )

    foreign = FoundationsTuningDecision(
        manifest_digest="f" * 64,
        evidence_digest="0" * 64,
        selections=decision.selections,
    )
    with pytest.raises(ValueError, match="different foundations manifest"):
        plan_foundations_primary(manifest, foreign)


def test_manifest_rejects_overlapping_development_primary_or_reserved_seeds(
    manifest,
):
    payload = manifest.to_dict()
    payload["instances"]["reserved"][0] = payload["instances"]["primary"][0]
    with pytest.raises(ValueError, match="instance seed groups must be disjoint"):
        FoundationsManifest.from_dict(payload)

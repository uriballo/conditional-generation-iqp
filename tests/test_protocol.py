import copy
from dataclasses import replace
from pathlib import Path

import pytest

from iqp_cond.protocol import (
    CampaignManifest,
    ModelFamily,
    ProtocolValidationError,
    content_digest,
)

MANIFEST_PATH = Path(__file__).parents[1] / "configs" / "quantum_campaign_v1.json"


def _payload() -> dict[str, object]:
    return CampaignManifest.from_json(MANIFEST_PATH).to_dict()


def test_production_manifest_is_typed_frozen_and_content_addressed():
    manifest = CampaignManifest.from_json(MANIFEST_PATH)

    assert len(manifest.coupling_seeds) == 8
    assert 666 in manifest.coupling_seeds
    assert len(manifest.optimizer_seeds) == 5
    assert len(manifest.data_seeds) == 8
    assert len(manifest.beta_grid.training) == 5
    assert len(manifest.beta_grid.held_out) == 4
    assert manifest.model_protocol.n_qubits == 16
    assert manifest.model_protocol.n_parameters == 136
    assert manifest.estimator_budget.operators_per_bandwidth == 2_000
    assert manifest.estimator_budget.reference_samples_per_operator_condition == 2_000
    assert manifest.estimator_budget.bandwidths == (1.0, 2.0, 4.0)
    assert manifest.sampler_protocol.to_dict()["n_chains"] == 8
    assert manifest.sampler_protocol.to_dict()["burn_in_sweeps"] == 4_000
    assert manifest.sampler_protocol.to_dict()["thinning_sweeps"] == 32
    assert manifest.training_protocol.batch_size == 256
    assert manifest.training_protocol.max_updates == 2_000
    assert len(manifest.training_protocol.tuning_candidates) == 9
    assert manifest.run_allocation.to_dict() == {
        "feasibility_representability": 20,
        "tuning": 36,
        "primary": 320,
        "ancilla": 24,
    }
    assert len(manifest.digest) == 64
    assert CampaignManifest.from_dict(manifest.to_dict()).digest == manifest.digest


@pytest.mark.parametrize(
    "case",
    [
        "seven_couplings",
        "missing_anchor_seed",
        "duplicate_coupling",
        "four_optimizers",
        "missing_beta_anchor",
        "non_midpoint_holdout",
        "wrong_model_vocabulary",
        "wrong_phase_vocabulary",
        "wrong_qubits",
        "wrong_parameters",
        "wrong_operator_budget",
        "wrong_reference_budget",
        "wrong_bandwidths",
        "wrong_sampler_schedule",
        "wrong_sampler_threshold",
        "wrong_batch_size",
        "wrong_max_updates",
        "wrong_checkpoint_rule",
        "wrong_tuning_candidate",
        "transferable_allocation",
    ],
)
def test_manifest_rejects_every_frozen_protocol_drift(case):
    payload = copy.deepcopy(_payload())
    if case == "seven_couplings":
        payload["coupling_seeds"] = payload["coupling_seeds"][:-1]
        payload["data_seeds"] = payload["data_seeds"][:-1]
    elif case == "missing_anchor_seed":
        payload["coupling_seeds"][0] = 999
    elif case == "duplicate_coupling":
        payload["coupling_seeds"][-1] = payload["coupling_seeds"][0]
    elif case == "four_optimizers":
        payload["optimizer_seeds"] = payload["optimizer_seeds"][:-1]
    elif case == "missing_beta_anchor":
        payload["beta_grid"]["training"][2] = 0.34
    elif case == "non_midpoint_holdout":
        payload["beta_grid"]["held_out"][0] += 0.001
    elif case == "wrong_model_vocabulary":
        payload["model_protocol"]["families"].remove("separate")
    elif case == "wrong_phase_vocabulary":
        payload["model_protocol"]["phase_features"][0] = "learned_features"
    elif case == "wrong_qubits":
        payload["model_protocol"]["n_qubits"] = 15
    elif case == "wrong_parameters":
        payload["model_protocol"]["n_parameters"] = 135
    elif case == "wrong_operator_budget":
        payload["estimator_budget"]["operators_per_bandwidth"] = 1_999
    elif case == "wrong_reference_budget":
        payload["estimator_budget"]["reference_samples_per_operator_condition"] = 1_999
    elif case == "wrong_bandwidths":
        payload["estimator_budget"]["bandwidths"] = [1.0, 2.0]
    elif case == "wrong_sampler_schedule":
        payload["sampler_protocol"]["thinning_sweeps"] = 16
    elif case == "wrong_sampler_threshold":
        payload["sampler_protocol"]["min_ess_per_chain"] = 99.0
    elif case == "wrong_batch_size":
        payload["training_protocol"]["batch_size"] = 255
    elif case == "wrong_max_updates":
        payload["training_protocol"]["max_updates"] = 1_999
    elif case == "wrong_checkpoint_rule":
        payload["training_protocol"]["checkpoint_rule"] = "last_checkpoint"
    elif case == "wrong_tuning_candidate":
        payload["training_protocol"]["tuning_candidates"][3][
            "phase_feature_assignment"
        ] = "signed_affine_multipliers"
    else:
        payload["run_allocation"]["primary"] -= 1
        payload["run_allocation"]["tuning"] += 1

    with pytest.raises(ProtocolValidationError):
        CampaignManifest.from_dict(payload)


def test_manifest_rejects_unknown_fields_instead_of_ignoring_schema_drift():
    payload = _payload()
    payload["mutable_threshold_override"] = True

    with pytest.raises(ProtocolValidationError, match="unknown"):
        CampaignManifest.from_dict(payload)


def test_run_identity_digest_covers_every_scientific_coordinate():
    from iqp_cond.campaign import TuningDecision, plan_primary

    manifest = CampaignManifest.from_json(MANIFEST_PATH)
    candidates = {
        family: manifest.training_protocol.candidates_for(family)[1].candidate_id
        for family in (
            ModelFamily.FIXED_INPUT,
            ModelFamily.CONDITION_MODULATED_PHASE,
            ModelFamily.NO_CONDITION,
        )
    }
    decision = TuningDecision.create(
        manifest,
        candidates,
        evidence_digest=content_digest({"evidence": "tuning"}),
    )
    run = plan_primary(manifest, decision)[0]
    variants = (
        replace(run, campaign_version="quantum_campaign_v2"),
        replace(run, coupling_seed=101),
        replace(run, condition_scope="different_scope"),
        replace(run, model_family=ModelFamily.NO_CONDITION),
        replace(run, optimizer_seed=4),
        replace(run, data_seed=9999),
        replace(run, tuning_candidate_id="fixed_input_lr_high"),
        replace(run, configuration_digest=content_digest({"manifest": "other"})),
    )

    assert len({run.run_id, *(variant.run_id for variant in variants)}) == 1 + len(
        variants,
    )

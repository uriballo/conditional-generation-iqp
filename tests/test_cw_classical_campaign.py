import json

import numpy as np
import pytest

from paper_experiments import cw_classical_campaign as campaign
from paper_experiments import cw_classical_config as config
from paper_experiments import cw_classical_fit as fit


def _record(repeat, j=0.2, p=0.1, status="ok"):
    return {"job": {"repeat": repeat}, "status": status, "scores": {"J": j, "P": p}}


def test_protocol_hash_and_parent_and_stage_separation():
    value = config.protocol(smoke=True)
    assert value["smoke"] is True
    assert config.betas("screen", True) != config.betas("confirmation", True)
    assert config.betas("confirmation", True) != config.betas("final", True)
    assert config.target_stream("final", 0, 0.7) == config.target_stream(
        "final", 0, 0.7
    )


def test_architecture_counts_and_limits():
    assert config.limits(True)["n"] == 4
    assert config.limits(False)["n"] == 16
    for family in config.FAMILIES:
        module = config.model(family)
        import jax

        assert (
            module.parameter_count(module.init(jax.random.key(0), capacity="small")) > 0
        )
        assert (
            module.parameter_count(module.init(jax.random.key(0), capacity="large")) > 0
        )


def test_aggregate_means_and_rejects_bad_repeats():
    result = campaign._aggregate([_record(0, 0.2, 0.4), _record(1, 0.6, 0.8)], 2)
    assert np.isclose(result["J"], 0.4) and np.isclose(result["P"], 0.6)
    assert result["valid"] is True
    with pytest.raises(ValueError):
        campaign._aggregate([_record(0), _record(0)], 2)
    assert campaign._aggregate([_record(0, status="failed")], 1)["valid"] is False


def test_tell_rejects_tampered_completed_result():
    import optuna

    study = optuna.create_study(direction="minimize")
    study.ask()
    trial = study.trials[0]
    result = {"J": 0.2, "valid": True}
    campaign._tell(study, trial, result)
    completed = study.trials[0]
    with pytest.raises(ValueError):
        campaign._tell(study, completed, {"J": 0.3, "valid": True})


def test_random_phase_rejects_tpe_before_random(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(campaign, "source_identity", lambda: {"test": "same"})
    with pytest.raises(ValueError, match="random-review-ready"):
        campaign.run_all(tmp_path / "early-tpe", phase="tpe", smoke=True)


def test_target_stream_is_reproducible_and_seeded():
    from paper_experiments.cw_followup_config import stream_record

    stream = config.target_stream("final", 0, 0.7)
    assert stream == stream_record("final", 0, "0.7", "target_pool")
    assert stream["seed"] == int(stream["sha256"][:8], 16)
    assert (
        config.classical_stream("sampler", "rbm")["seed"]
        == config.classical_stream("sampler", "rbm")["seed"]
    )


def test_numerical_failure_is_recorded_and_resumed_without_replacement(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fit, "source_identity", lambda: {"test": "same"})
    calls = []

    def nonconvergence(*args):
        calls.append(True)
        raise FloatingPointError("quadrature did not converge")

    monkeypatch.setattr(fit, "evaluate", nonconvergence)
    job = dict(
        family="rbm",
        capacity="small",
        smoke=True,
        stage="screen",
        repeat=0,
        trial_id=0,
        identity={"test": "same"},
        candidate=dict(
            learning_rate=0.01, init_scale=1.0, b1=0.9, b2=0.99, schedule="constant"
        ),
    )
    receipt = fit.run_fit(job, tmp_path)
    assert receipt["status"] == "numerical_failure"
    assert receipt["search_penalty_not_measured_tvd"] is True
    assert fit.run_fit(job, tmp_path) == receipt
    assert len(calls) == 1
    pool = tmp_path / receipt["target_pools"][0]["path"]
    pool.write_bytes(pool.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="paired target pool"):
        fit.run_fit(job, tmp_path)


def test_protocol_checksum_detects_an_unrecorded_design_change(tmp_path, monkeypatch):
    path = tmp_path / "protocol.json"
    value = json.loads(config.PROTOCOL_PATH.read_text())
    value["training"]["updates"] = 1999
    path.write_text(json.dumps(value))
    monkeypatch.setattr(config, "PROTOCOL_PATH", path)
    with pytest.raises(ValueError, match="hash mismatch"):
        config.protocol()


def test_proposals_bind_transformed_parameters():
    params = dict(
        learning_rate=0.01,
        init_scale=1.0,
        b1=0.9,
        one_minus_beta2=0.01,
        schedule="constant",
    )
    row = dict(
        optuna_params=params,
        config=dict(
            learning_rate=0.01, init_scale=1.0, b1=0.9, b2=0.99, schedule="constant"
        ),
    )
    campaign._validate_proposal(row)
    row["config"]["b2"] = 0.98
    with pytest.raises(ValueError, match="Optuna parameters"):
        campaign._validate_proposal(row)


def test_random_freeze_and_resume_preserve_proposals(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(campaign, "source_identity", lambda: {"test": "same"})
    calls = []

    def fake_fit(root, manifest, family, proposal, stage, executor):
        calls.append((family, proposal["trial_id"]))
        result = dict(
            **proposal, J=0.2 + 0.01 * proposal["trial_id"], P=0.1, valid=True, fits=[]
        )
        campaign.write_json(
            root / stage / family / f"{proposal['trial_id']:03d}.json", result
        )
        return result

    monkeypatch.setattr(campaign, "_fit_proposal", fake_fit)
    monkeypatch.setattr(campaign, "_audit_result", lambda *args: [])
    root = tmp_path / "random"
    result = campaign.run_all(root, smoke=True)
    frozen = (root / "random-proposals.json").read_bytes()
    assert result["status"] == "paused_for_random_review"
    assert result["counts"] == dict.fromkeys(config.FAMILIES, 3)
    assert len(calls) == 9
    assert campaign.run_all(root, smoke=True) == result
    assert len(calls) == 9
    assert (root / "random-proposals.json").read_bytes() == frozen
    monkeypatch.setattr(campaign, "source_identity", lambda: {"test": "changed"})
    with pytest.raises(ValueError, match="identity"):
        campaign.run_all(root, smoke=True)

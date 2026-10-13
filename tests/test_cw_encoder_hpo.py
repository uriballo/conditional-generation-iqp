import json

import optuna
import pytest

from paper_experiments import cw_encoder_hpo as hpo
from paper_experiments.cw_followup_config import candidate as audited_candidate


def test_reupload_search_space_matches_shared_parity_plus_fixed_turn():
    hpo.configure("reupload")
    spec = hpo.protocol()
    assert spec["family"]["name"] == hpo.FAMILY == "shared_reuploaded_parity"
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
    trial = study.ask()
    config = hpo.candidate(trial)
    reference = audited_candidate(
        optuna.trial.FixedTrial(trial.params), "shared_parity"
    )
    assert {k: v for k, v in config.items() if k != "turn"} == reference
    assert config["turn"] == spec["family"]["turn"] == 0.1


def test_squeezed_search_space_replaces_only_the_scale_range():
    hpo.configure("squeezed")
    spec = hpo.protocol()
    assert hpo.FAMILY == "shared_squeezed_parity"
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
    for _ in range(20):
        trial = study.ask()
        config = hpo.candidate(trial)
        reference = audited_candidate(
            optuna.trial.FixedTrial(trial.params), "independent_unconditional"
        )
        assert {k: config[k] for k in reference} == reference
        assert 0.3 <= config["beta_scale"] <= 0.7
        assert config["kappa"] == spec["family"]["kappa"] == 0.4
        assert "turn" not in config


@pytest.mark.parametrize(
    "campaign,option",
    [("reupload", "turn"), ("squeezed", "kappa"), ("squeezed_independent", "kappa")],
)
def test_sharded_smoke_campaign_runs_every_phase(tmp_path, campaign, option):
    auth = "test authorization"
    root = tmp_path / "campaign"
    run = dict(smoke=True, authorization=auth, campaign=campaign)
    for phase in ("random", "confirmation", "final"):
        if phase == "confirmation":
            hpo.run(root, "tpe", **run)
        hpo.run(root, phase, prepare=True, **run)
        for index in range(2):
            status = hpo.run(root, phase, shard=(index, 2), **run)
            assert status["status"] == "shard_complete"
        status = hpo.run(root, phase, **run)
    assert status["status"] == "final_complete"
    data = json.loads((root / "final" / "figure_data.json").read_text())
    assert data["kind"] == f"cw_{campaign}_final"
    assert data["family"] == hpo.FAMILY
    assert option in data["config"]
    assert len(data["repeat_scores"]) == 1
    if campaign == "squeezed_independent":
        assert data["beta"] == [0.4, 1.0, 2.0]
        assert len(data["curves"]["tvd"][0]) == 3
    study = hpo._study(root, None)
    assert len(study.trials) == 6
    phases = [t.user_attrs["phase"] for t in study.trials]
    assert phases == ["random"] * 3 + ["tpe"] * 3


def test_shards_require_frozen_primary(tmp_path):
    with pytest.raises(ValueError):
        hpo.run(tmp_path, "random", smoke=True, authorization="x", shard=(0, 2))
    with pytest.raises(ValueError):
        hpo.run(tmp_path, "tpe", smoke=True, authorization="x", shard=(0, 2))

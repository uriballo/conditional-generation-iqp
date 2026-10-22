import hashlib
import json
from pathlib import Path

import optuna
import pytest

from paper_experiments import cw_encoder_hpo as hpo
from paper_experiments.cw_followup_config import candidate as audited_candidate


@pytest.mark.parametrize(
    "campaign,family",
    [
        ("twisted", "shared_twisted_parity"),
        ("twisted_independent", "independent_twisted_parity"),
    ],
)
def test_twisted_search_space_replaces_only_the_scale_range(campaign, family):
    hpo.configure(campaign)
    spec = hpo.protocol()
    assert hpo.FAMILY == family
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
    [
        ("twisted", "kappa"),
        ("twisted_independent", "kappa"),
    ],
)
def test_sharded_smoke_campaign_runs_every_phase(tmp_path, campaign, option):
    root = tmp_path / "campaign"
    run = dict(smoke=True, campaign=campaign)
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
    if campaign.endswith("_independent"):
        assert data["beta"] == [0.4, 1.0, 2.0]
        assert len(data["curves"]["tvd"][0]) == 3
    study = hpo._study(root, None)
    assert len(study.trials) == 6
    phases = [t.user_attrs["phase"] for t in study.trials]
    assert phases == ["random"] * 3 + ["tpe"] * 3


def test_shards_require_frozen_primary(tmp_path):
    with pytest.raises(ValueError):
        hpo.run(tmp_path, "random", smoke=True, shard=(0, 2))
    with pytest.raises(ValueError):
        hpo.run(tmp_path, "tpe", smoke=True, shard=(0, 2))


def test_twisted_protocols_change_only_the_encoder_identity():
    """The frozen squeezed protocols are retained as the twisted parents."""
    allowed = {"name", "encoder", "kappa_policy", "role"}
    root = Path(hpo.__file__).parents[1]
    for twisted in ("twisted", "twisted_independent"):
        hpo.configure(twisted)
        spec = hpo.protocol()
        source = root / spec["derived_from"]["path"]
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest()
            == spec["derived_from"]["file_sha256"]
        )
        parent = json.loads(source.read_text())
        assert spec["derived_from"]["protocol_sha256"] == parent["protocol_sha256"]
        assert (
            spec["parent_protocol"]["protocol_sha256"]
            == (parent["parent_protocol"]["protocol_sha256"])
        )
        for key in set(parent["family"]) | set(spec["family"]):
            if key not in allowed:
                assert spec["family"][key] == parent["family"][key], key
        assert spec["stages"] == parent["stages"]

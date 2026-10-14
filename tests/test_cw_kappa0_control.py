import json

import pytest

from paper_experiments import cw_kappa0_control as control

WINNER = dict(
    trial_id=99,
    config=dict(
        learning_rate=0.0035,
        init_scale=0.034,
        b1=0.83,
        b2=0.998,
        schedule="constant",
        expected_weights=[2, 4, 7.5],
        beta_scale=0.49,
        kappa=0.4,
    ),
)


def test_jobs_remove_only_the_squeezing():
    jobs = control.jobs(WINNER, scales=("winner", 0.3), smoke=True)
    assert {j["family"] for j in jobs} == {"shared_squeezed_parity"}
    assert {j["candidate"]["kappa"] for j in jobs} == {0.0}
    assert sorted({j["candidate"]["beta_scale"] for j in jobs}) == [0.3, 0.49]
    for job in jobs:
        rest = {
            k: v
            for k, v in job["candidate"].items()
            if k not in ("kappa", "beta_scale")
        }
        assert rest == {
            k: v
            for k, v in WINNER["config"].items()
            if k not in ("kappa", "beta_scale")
        }
        assert job["stage"] == "final"


def test_smoke_control_runs_and_bundles(tmp_path):
    root = tmp_path / "control"
    assert control.run(root, WINNER, scales=("winner",), smoke=True) == ["ok"]
    payload = control.bundle(root, tmp_path / "kappa0_control.json")
    assert payload["kind"] == "cw_kappa0_control" and payload["smoke"]
    (entry,) = payload["scales"].values()
    assert entry["config"]["kappa"] == 0.0
    assert len(entry["repeat_scores"]) == len(entry["repeats"]) == 1
    assert len(entry["curves"]["tvd"][0]) == len(payload["beta"])
    saved = json.loads((tmp_path / "kappa0_control.json").read_text())
    assert saved["fit_receipt_sha256"] == payload["fit_receipt_sha256"]
    other = dict(WINNER, config=dict(WINNER["config"], beta_scale=0.5))
    with pytest.raises(ValueError):
        control.run(root, other, scales=("winner",), smoke=True)

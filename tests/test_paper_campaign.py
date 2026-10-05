"""Inventories and full staged HPO with inexpensive deterministic fit doubles."""

import json
from collections import Counter
from pathlib import Path

import pytest

from paper_experiments import hpo
from paper_experiments.campaign import inventory
from run_experiments import main


def test_reported_inventory():
    jobs = inventory()
    assert len(jobs) == 64
    assert len({j["job_id"] for j in jobs}) == 64
    assert Counter(j["experiment"] for j in jobs) == dict(
        gaussian=10, shared=5, conditional=15, unconditional=15, scaling=18, symmetry=1
    )
    assert len(inventory(smoke=True)) == 10
    assert len(inventory("curie-weiss")) == 35


def test_dry_run_writes_nothing(tmp_path, capsys):
    assert main(["--dry-run", "--output", str(tmp_path)]) == 0
    assert not list(tmp_path.iterdir())
    assert len(json.loads(capsys.readouterr().out)) == 64


@pytest.mark.parametrize("smoke", [True, False])
def test_all_hpo_stages_and_resume(tmp_path, smoke):
    calls = []

    def execute(job, output_root, *, smoke=False):
        calls.append(job)
        p = job["optimizer"]
        score = p["init_scale"] if job["stage"] == "screen" else 1 - p["init_scale"]
        return dict(
            directory=str(Path(output_root) / job["job_id"]),
            development_profile_tvd_mean=score,
            train_profile_tvd_mean=score,
            held_out_profile_tvd_per_condition=[score]
            * (3 if job["experiment"] == "shared" else 1),
            held_out_tvd_per_condition=[score]
            * (3 if job["experiment"] == "shared" else 1),
        )

    result = hpo.run_all(tmp_path, smoke=smoke, executor=execute)
    screen = [j for j in calls if j["stage"] == "screen"]
    assert len(screen) == (
        16 if smoke else 768
    )  # two shared studies x two fits; two baseline studies x six fits
    assert {j["seed"] for j in screen} == ({5200} if smoke else {5200, 5201})
    for kind in ("conditional", "unconditional"):
        assert {j["beta"] for j in screen if j["experiment"] == kind} == {0.4, 1.0, 1.6}
    assert len(result["winners"]) == 3 and len(result["final"]) == 4
    confirms = json.loads((tmp_path / "confirmation.json").read_text())
    assert Counter(r["family"] for r in confirms) == dict(
        shared_adam=2,
        shared_schedule_free=2,
        independent_conditional=2,
        independent_unconditional=2,
    )
    for winner in result["winners"]:
        eligible = [
            r
            for r in confirms
            if (
                r["family"].startswith("shared")
                if winner["family"].startswith("shared")
                else r["family"] == winner["family"]
            )
        ]
        assert winner["score"] == min(r["score"] for r in eligible)
    final = [j for j in calls if j["stage"] == "final"]
    assert {j["seed"] for j in final} == ({5400} if smoke else set(range(5400, 5405)))
    before = len(calls)
    hpo.run_all(tmp_path, smoke=smoke, executor=execute)
    assert len(calls) == before


def test_full_hpo_protocol():
    p = hpo.protocol()
    assert p["trials"] == 48 and p["batch_size"] == 2 and not p["pruning"]
    assert p["screen_seeds"] == [5200, 5201]
    assert p["confirmation_seeds"] == [5300, 5301, 5302]
    assert p["final_seeds"] == [5400, 5401, 5402, 5403, 5404]
    for family in hpo.STUDIES:
        jobs = hpo._jobs(family, hpo._reference(), "screen", p["screen_seeds"])
        assert len(jobs) == (2 if family.startswith("shared") else 6)

"""Search space, pairing and stage rules of the Hamming-weight campaigns."""

import json
import math

import numpy as np
import optuna

from paper_experiments import gaussian2d_diagonal_report as diagonal_report
from paper_experiments import gaussian2d_hpo as hpo
from paper_experiments import gaussian2d_lattice_report as lattice_report


def test_search_space_matches_protocol():
    space = hpo.protocol()["search_space"]
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
    grid = space["kernel"]["tunable_grid"]
    for _ in range(50):
        config = hpo.candidate(study.ask())
        assert 3e-4 <= config["learning_rate"] <= 0.1
        assert 0.01 <= config["init_scale"] <= 1
        assert 0.8 <= config["b1"] <= 0.95 and 0.95 <= config["b2"] <= 0.999
        assert config["schedule"] in space["schedule"]["choices"]
        assert 0 <= config["one_body_mean"] <= math.pi / 2
        first, second, third = config["expected_weights"]
        assert first == 2 and second in grid and third in grid and second <= third


def test_conditions_separate_training_selection_and_reporting():
    sets = hpo.conditions()
    assert [len(sets[k]) for k in ("fitting", "development", "reporting")] == [9, 8, 16]
    for first, second in [
        ("fitting", "development"),
        ("fitting", "reporting"),
        ("development", "reporting"),
    ]:
        assert not any(np.allclose(a, b) for a in sets[first] for b in sets[second])


def test_shortlist_ranks_valid_distinct_configurations():
    rows = [
        dict(trial_id=0, J=0.3, valid=True, config={"a": 1}),
        dict(trial_id=1, J=0.1, valid=True, config={"a": 2}),
        dict(trial_id=2, J=0.1, valid=True, config={"a": 2}),
        dict(trial_id=3, J=0.05, valid=False, config={"a": 3}),
        dict(trial_id=4, J=0.2, valid=True, config={"a": 4}),
    ]
    assert [r["trial_id"] for r in hpo.shortlist(rows, 3)] == [1, 4, 0]


def test_smoke_campaigns_pair_streams_and_bundle(tmp_path):
    receipts = {}
    for campaign in ("input", "diagonal"):
        root = tmp_path / campaign
        for phase in ("random", "confirmation", "final"):
            hpo.run(root, campaign, phase, smoke=True)
        search, fits = hpo.load_campaign(root)
        assert search["screening_trials"] == 3 and len(fits) == 2
        receipts[campaign] = [r for r, _ in fits]
    for first, second in zip(*receipts.values(), strict=True):
        assert first["target_pool"] == second["target_pool"]
        assert first["streams"] == second["streams"]
    assert receipts["input"][0]["target_pool"] != receipts["input"][1]["target_pool"]
    data = tmp_path / "data"
    lattice = lattice_report.extract(tmp_path / "input", data / "gaussian_lattice")
    assert lattice["n_fits"] == 2
    arrays = np.load(data / "gaussian_lattice/DIAGNOSTICS.npz")
    np.testing.assert_allclose(arrays["grid_tvd"][:, 20, 20], 0, atol=1e-12)
    assert np.all(arrays["grid_tvd"] >= arrays["grid_profile_tvd"] - 1e-12)
    summary = diagonal_report.extract(
        tmp_path / "diagonal",
        data / "gaussian2d-diagonal",
        ciqp=data / "gaussian_lattice",
        grid_points=3,
        axis_points=3,
    )
    assert summary["diagonal"]["n_parameters"] == 120
    assert set(diagonal_report.AXIS_KEYS) <= set(summary["ciqp"])
    json.dumps(summary, allow_nan=False)

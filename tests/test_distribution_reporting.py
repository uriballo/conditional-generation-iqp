"""Regression checks for full-distribution TVD reporting."""

import json
from pathlib import Path

import numpy as np

from paper_experiments.figures import (
    _curie_encoder_tvd,
    _curie_tex,
    gaussian_tables,
)

DATA = Path(__file__).resolve().parents[1] / "paper/data"


def test_gaussian_current_aggregates_are_full_and_profile_distinct():
    payload = json.loads((DATA / "gaussian/results.json").read_text())
    rows = gaussian_tables(payload)
    trained = next(row for row in rows if "Conditional" in row["model"])
    encoder = next(row for row in rows if "Encoder" in row["model"])
    np.testing.assert_allclose(trained["full_tvd"], 0.09949910125966158)
    np.testing.assert_allclose(trained["profile_tvd"], 0.0463837406099964)
    np.testing.assert_allclose(encoder["full_tvd"], 0.29216310790044)
    assert trained["seeds"] == 5
    assert trained["full_tvd"] != trained["profile_tvd"]


def test_gaussian_aggregation_is_median_of_run_means():
    payload = {
        "conditions": [0, 1],
        "rows": [
            {
                "control": "conditional",
                "seed": i,
                "bitstring_tvd": full,
                "profile_tvd": profile,
            }
            for i, (full, profile) in enumerate(
                [
                    ([0.0, 0.8], [0.0, 0.1]),
                    ([0.8, 0.0], [0.1, 0.0]),
                    ([0.5, 0.5], [0.3, 0.3]),
                ]
            )
        ]
        + [
            {
                "control": "encoder_only",
                "seed": i,
                "bitstring_tvd": [0.9, 0.9],
                "profile_tvd": [0.4, 0.4],
            }
            for i in range(3)
        ],
    }
    trained = next(
        row for row in gaussian_tables(payload) if "Conditional" in row["model"]
    )
    assert trained["full_tvd"] == 0.4
    conditional = [
        r["bitstring_tvd"] for r in payload["rows"] if r["control"] == "conditional"
    ]
    assert np.mean(np.median(np.asarray(conditional), axis=0)) == 0.5


def test_curie_full_tex_has_all_betas_and_encoder_label():
    payload = json.loads((DATA / "curie-weiss/figure_data.json").read_text())
    text = _curie_tex(payload, "full")
    assert "tab:curie_weiss_full_tvd" in text
    assert "\\begin{table" in text and "\\toprule" in text
    for value in ("0.2175", "0.1709", "0.1180", "0.3842", "0.1240"):
        assert value in text
    assert "full-distribution TVD" in text
    assert "reused for reporting" in text
    assert "Selection remains profile TVD" not in text
    profile = _curie_tex(payload, "profile")
    assert "tab:curie_weiss_profile_tvd" in profile and "0.1301" in profile


def test_curie_encoder_tvd_is_exact_profile_distance():
    payload = json.loads((DATA / "curie-weiss/figure_data.json").read_text())
    values = np.asarray(_curie_encoder_tvd(payload))
    np.testing.assert_allclose(
        values,
        [0.11798753353181994, 0.44063747253259833, 0.8698503185685845],
        atol=1e-12,
        rtol=0,
    )


def test_fresh_gaussian_export_retains_full_distribution_errors(tmp_path):
    from math import comb

    from paper_experiments.figures import export_completed, render_all

    run = tmp_path / "run"
    run.mkdir()
    conditions = (np.arange(80) + 1 / 3) / 80
    target = np.full(65536, 1 / 65536)
    model = np.zeros(65536)
    model[255] = 1  # A single weight-eight string, rather than a uniform shell.
    weights = np.array([i.bit_count() for i in range(65536)])
    chance = np.sin(1.5 / 2) ** 2
    reference = chance**weights * (1 - chance) ** (16 - weights)
    probabilities = np.broadcast_to(model, (80, 65536)).copy()
    # The midpoint exporter independently cross-checks the first saved condition.
    probabilities[0] = reference
    np.savez_compressed(
        run / "checkpoint.npz",
        params=np.zeros(16),
        generators=np.eye(16, dtype=np.int32),
        training_conditions=np.linspace(0, 1, 5),
        evaluation_conditions=conditions,
        held_out_target_probabilities=np.broadcast_to(target, (80, 65536)),
        held_out_probabilities=probabilities,
    )
    (tmp_path / "reported-results.json").write_text(
        json.dumps([dict(experiment="gaussian", seed=17100, directory=str(run))])
    )
    compact = tmp_path / "compact"
    assert export_completed(tmp_path, compact) == ["gaussian"]
    payload = json.loads((compact / "gaussian/results.json").read_text())
    trained = gaussian_tables(payload)[0]
    first_full = 0.5 * np.abs(reference - target).sum()
    first_profile = 0.5 * sum(
        comb(16, k) * abs(chance**k * (1 - chance) ** (16 - k) - 1 / 65536)
        for k in range(17)
    )
    np.testing.assert_allclose(
        trained["full_tvd"], (79 * (1 - 1 / 65536) + first_full) / 80
    )
    np.testing.assert_allclose(
        trained["profile_tvd"],
        (79 * (1 - comb(16, 8) / 65536) + first_profile) / 80,
    )
    assert trained["full_tvd"] > trained["profile_tvd"]
    paths = render_all(compact, tmp_path / "figures", experiment="gaussian")
    assert {p.name for p in paths} >= {
        "gaussian_translation_profiles.pdf",
        "gaussian_tvd.tex",
        "gaussian_translation_profiles.provenance.json",
    }

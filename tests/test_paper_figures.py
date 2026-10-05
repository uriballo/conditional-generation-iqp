"""Compact data must reproduce the current manuscript metrics."""

import json
from pathlib import Path

import numpy as np

from paper_experiments.figures import curie_tables, render_all

DATA = Path(__file__).resolve().parents[1] / "paper/data"


def test_curie_data_are_current_five_seed_results():
    p = json.loads((DATA / "curie-weiss/figure_data.json").read_text())
    assert p["seeds"] == [5400, 5401, 5402, 5403, 5404]
    np.testing.assert_allclose(
        p["metrics"]["shared"]["profile_mean"],
        [0.071222, 0.068337, 0.153045],
        atol=6e-7,
    )
    np.testing.assert_allclose(
        p["metrics"]["independent_conditional"]["profile_mean"],
        [0.052984, 0.033390, 0.295743],
        atol=6e-7,
    )
    np.testing.assert_allclose(
        p["metrics"]["independent_unconditional"]["profile_mean"],
        [0.016995, 0.029623, 0.350017],
        atol=6e-7,
    )
    for _family, arm in p["arms"].items():
        np.testing.assert_allclose(np.asarray(arm["mean"]).sum(1), 1, atol=1e-12)
    assert len(curie_tables(p)) == 18


def test_original_scaling_endpoint_metrics():
    p = json.loads((DATA / "scaling/figure_data.json").read_text())
    for n, initial, final in [(64, 0.0504, 0.0350), (2048, 0.0490, 0.0417)]:
        values = np.asarray(p["curves"][str(n)]["held_out"]["overall_per_seed"])
        np.testing.assert_allclose(
            np.median(values, axis=0)[[0, -1]], [initial, final], atol=5e-5
        )


def test_all_numerical_figures_render(tmp_path):
    paths = render_all(DATA, tmp_path)
    assert len([p for p in paths if p.suffix == ".pdf"]) == 4
    assert len([p for p in paths if p.suffix == ".png"]) == 4
    assert all(p.is_file() and p.stat().st_size > 100 for p in paths)

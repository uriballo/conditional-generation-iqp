"""Compact data must reproduce the current manuscript metrics."""

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.figures import render_all

DATA = Path(__file__).resolve().parents[1] / "data"


def test_scaling_endpoint_metrics():
    p = json.loads((DATA / "scaling/figure_data.json").read_text())
    for n, initial, final in [
        (64, 0.04213059582112235, 0.0014936687410896461),
        (1024, 0.041702844245070696, 0.004333284793505562),
    ]:
        values = np.asarray(p["curves"][str(n)]["held_out"]["overall_per_seed"])
        np.testing.assert_allclose(
            values.mean(axis=0)[[0, -1]], [initial, final], atol=5e-5
        )


def test_all_numerical_figures_render(tmp_path):
    paths = render_all(DATA, tmp_path)
    assert {p.name for p in paths if p.suffix == ".pdf"} == {
        "gaussian_translation_profiles.pdf",
        "curie_weiss_lattice_scaling_mmd.pdf",
        "gaussian2d_lattice_heatmaps.pdf",
        "gaussian2d_lattice_axes_response.pdf",
        "gaussian2d_lattice_grid_response.pdf",
        "curie_weiss_model_comparison.pdf",
        "curie_weiss_physical_response.pdf",
        "curie_weiss_encoder_comparison.pdf",
        "curie_weiss_tvd_fluctuations.pdf",
        "curie_weiss_classical_tvd.pdf",
        "curie_weiss_lattice_scaling_observables.pdf",
    }
    assert len([p for p in paths if p.suffix == ".png"]) == 11
    assert all(p.is_file() and p.stat().st_size > 100 for p in paths)


def test_missing_bundle_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no completed data"):
        render_all(tmp_path, tmp_path / "out", experiment="curie-weiss")

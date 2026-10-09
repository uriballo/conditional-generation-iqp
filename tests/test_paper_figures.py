"""Compact data must reproduce the current manuscript metrics."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from paper_experiments.figures import curie_tables, gaussian_tables, render_all

DATA = Path(__file__).resolve().parents[1] / "paper/data"


def test_curie_data_are_current_five_seed_results():
    p = json.loads((DATA / "curie-weiss/figure_data.json").read_text())
    assert p["seeds"] == [2026100610, 2026100611, 2026100612, 2026100613, 2026100614]
    np.testing.assert_allclose(
        p["metrics"]["shared"]["profile_mean"],
        [0.097619, 0.130125, 0.063118],
        atol=6e-7,
    )
    np.testing.assert_allclose(
        p["metrics"]["independent_conditional"]["profile_mean"],
        [0.093942, 0.088479, 0.394773],
        atol=6e-7,
    )
    np.testing.assert_allclose(
        p["metrics"]["independent_unconditional"]["profile_mean"],
        [0.118591, 0.078020, 0.329344],
        atol=6e-7,
    )
    for _family, arm in p["arms"].items():
        np.testing.assert_allclose(np.asarray(arm["mean"]).sum(1), 1, atol=1e-12)
    assert len(curie_tables(p)) == 18
    assert p["selection_metric"] == "full_tvd"
    assert p["confirmation_reused"]
    assert np.asarray(p["records"]["shared"][0]["target_profile"]).shape == (3, 17)
    for family in ("shared", "independent_conditional", "independent_unconditional"):
        for record in p["records"][family]:
            assert isinstance(record["best_step"], int)
            assert len(record["checkpoint_sha256"]) == 64


def test_original_scaling_endpoint_metrics():
    p = json.loads((DATA / "scaling/figure_data.json").read_text())
    for n, initial, final in [(64, 0.0504, 0.0350), (2048, 0.0490, 0.0417)]:
        values = np.asarray(p["curves"][str(n)]["held_out"]["overall_per_seed"])
        np.testing.assert_allclose(
            np.median(values, axis=0)[[0, -1]], [initial, final], atol=5e-5
        )


def test_all_numerical_figures_render(tmp_path):
    paths = render_all(DATA, tmp_path)
    assert {p.name for p in paths if p.suffix == ".pdf"} == {
        "gaussian_translation_profiles.pdf",
        "curie_weiss_lattice_scaling_mmd.pdf",
        "gaussian2d_phase_heatmaps.pdf",
        "gaussian2d_phase_mean_response.pdf",
        "gaussian2d_phase_width_response.pdf",
        "conditional_symmetry_current.pdf",
        "conditional_symmetry_fluctuation.pdf",
    }
    assert len([p for p in paths if p.suffix == ".png"]) == 7
    assert all(p.is_file() and p.stat().st_size > 100 for p in paths)


def test_historical_curie_data_remains_explicitly_renderable(tmp_path):
    paths = render_all(DATA, tmp_path, experiment="curie-weiss-historical")
    assert any(path.name == "curie_weiss_model_comparison.pdf" for path in paths)

    payload = json.loads((DATA / "curie-weiss/figure_data.json").read_text())
    assert payload["seeds"] == [
        2026100610,
        2026100611,
        2026100612,
        2026100613,
        2026100614,
    ]


def test_gaussian_render_exports_profiles_and_response(tmp_path):
    render_all(DATA, tmp_path, experiment="gaussian")
    for name in (
        "gaussian_translation_profiles.png",
        "gaussian_translation_profiles.pdf",
        "midpoint_profiles.csv",
        "response.csv",
        "training_response.csv",
    ):
        assert (tmp_path / name).is_file()
    mid = json.loads((DATA / "gaussian/midpoint_profiles.json").read_text())
    records = np.asarray([r["profiles"] for r in mid["records"]])
    with (tmp_path / "midpoint_profiles.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    model = np.asarray([[float(r[f"model_{j}"]) for j in range(4)] for r in rows])
    np.testing.assert_allclose(model, records.mean(axis=0).T, atol=1e-12)
    np.testing.assert_allclose(model.sum(axis=0), 1, atol=1e-12)
    with (tmp_path / "response.csv").open(newline="") as f:
        response = list(csv.DictReader(f))
    payload = json.loads((DATA / "gaussian/results.json").read_text())
    runs = np.asarray(
        [r["profile"] for r in payload["rows"] if r["control"] == "conditional"]
    )
    means = runs.reshape(5, -1, runs.shape[-1]) @ np.arange(runs.shape[-1])
    np.testing.assert_allclose(float(response[0]["model"]), np.median(means[:, 0]))
    np.testing.assert_allclose(
        float(response[0]["lower"]), np.quantile(means[:, 0], 0.25)
    )
    np.testing.assert_allclose(
        float(response[0]["upper"]), np.quantile(means[:, 0], 0.75)
    )
    np.testing.assert_allclose([float(r["c"]) for r in response], payload["conditions"])
    with (tmp_path / "training_response.csv").open(newline="") as f:
        assert [float(r["c"]) for r in csv.DictReader(f)] == [0, 0.25, 0.5, 0.75, 1]
    np.testing.assert_allclose(
        [r["full_tvd"] for r in gaussian_tables(payload)],
        [0.0994991013, 0.2921631079],
        atol=5e-11,
    )


def test_gaussian_midpoint_coverage_failures(tmp_path):
    root = tmp_path / "gaussian"
    root.mkdir()
    shutil.copytree(DATA / "gaussian", root, dirs_exist_ok=True)
    import pytest

    for mutate in (
        lambda p: p["records"][0].__setitem__("seed", 99999),
        lambda p: p["records"][0].__setitem__("checkpoint_sha256", "bad"),
        lambda p: p.__setitem__("conditions", [0.1] * 4),
    ):
        p = json.loads((root / "midpoint_profiles.json").read_text())
        mutate(p)
        (root / "midpoint_profiles.json").write_text(json.dumps(p))
        with pytest.raises(ValueError):
            render_all(tmp_path, tmp_path / "out", experiment="gaussian")
        shutil.copy(
            DATA / "gaussian/midpoint_profiles.json", root / "midpoint_profiles.json"
        )

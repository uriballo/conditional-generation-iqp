"""Check compact evidence against the circuit model and architectural baseline."""

import hashlib
import json
from pathlib import Path

import numpy as np

from paper_experiments.gaussian2d_lattice import metrics
from paper_experiments.gaussian2d_lattice_report import evaluator

DATA = Path(__file__).resolve().parents[1] / "paper/data/gaussian_lattice"


def test_compact_parameters_reproduce_reported_full_distributions():
    summary = json.loads((DATA / "SUMMARY.json").read_text())
    for name, key in [
        ("PARAMETERS.npz", "parameters_sha256"),
        ("DIAGNOSTICS.npz", "diagnostic_npz_sha256"),
        ("protocol.json", "protocol_sha256"),
    ]:
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == summary[key]
    params = np.load(DATA / "PARAMETERS.npz")
    for design in ("grid", "axes", "translation"):
        row = summary["results"][design]
        pairs = np.asarray(row["reporting_conditions"])[:3]
        for theta, fit in zip(params[design], row["fits"], strict=True):
            value = metrics(theta, pairs)
            fast = evaluator(theta)(pairs)
            for key in ("full_tvd", "profile_tvd", "generated_mean", "generated_sd"):
                np.testing.assert_allclose(
                    value[key], fit["selected"][key][:3], atol=1e-10, rtol=0
                )
                np.testing.assert_allclose(value[key], fast[key], atol=1e-10, rtol=0)


def test_diagnostics_distinguish_fixed_center_from_learned_width():
    a = np.load(DATA / "DIAGNOSTICS.npz")
    np.testing.assert_allclose(a["grid_tvd"][:, :, 20, 20], 0, atol=1e-12)
    assert np.all(a["grid_tvd"] >= a["grid_profile_tvd"] - 1e-12)
    assert np.all(np.diff(a["model_sd"], axis=-1) > 0)
    assert a["translation_profiles"].shape == (5, 4, 17)
    assert a["translation_response"].shape == (5, 80)
    np.testing.assert_array_equal(
        a["translation_profile_c"], a["translation_reporting_c"][[0, 26, 53, 79]]
    )
    np.testing.assert_allclose(
        a["translation_profiles"].sum(axis=-1), 1, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        a["translation_profiles"] @ np.arange(17),
        a["translation_response"][:, [0, 26, 53, 79]],
        atol=1e-10,
        rtol=0,
    )
    assert not np.intersect1d(
        a["translation_profile_c"], a["translation_fitting_c"]
    ).size
    assert not np.intersect1d(
        a["translation_profile_c"], a["translation_development_c"]
    ).size
    np.testing.assert_allclose(
        a["translation_target_profiles"] @ np.arange(17),
        np.asarray(
            [5.15495467017703, 6.980572218744717, 8.99482401417277, 10.823182933575673]
        ),
        atol=1e-10,
        rtol=0,
    )
    assert not np.any(a["translation_reporting_c"] == 0.5)

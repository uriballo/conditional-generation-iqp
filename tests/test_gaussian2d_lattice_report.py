"""Check compact evidence against the circuit model and architectural baseline."""

import hashlib
import json
from pathlib import Path

import numpy as np

from paper_experiments.gaussian2d_lattice import metrics
from paper_experiments.gaussian2d_lattice_report import evaluator

DATA = Path(__file__).resolve().parents[1] / "data/gaussian_lattice"


def test_compact_parameters_reproduce_reported_full_distributions():
    summary = json.loads((DATA / "SUMMARY.json").read_text())
    for name, key in [
        ("PARAMETERS.npz", "parameters_sha256"),
        ("DIAGNOSTICS.npz", "diagnostic_npz_sha256"),
        ("protocol.json", "protocol_sha256"),
    ]:
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == summary[key]
    row = summary["results"]["axes"]
    pairs = np.asarray(row["reporting_conditions"])[:3]
    thetas = np.load(DATA / "PARAMETERS.npz")["axes"]
    assert len(thetas) == len(row["fits"]) == summary["n_fits"] == 10
    for theta, fit in zip(thetas, row["fits"], strict=True):
        value = metrics(theta, pairs)
        fast = evaluator(theta)(pairs)
        for key in ("full_tvd", "profile_tvd", "generated_mean", "generated_sd"):
            np.testing.assert_allclose(
                value[key], fit["selected"][key][:3], atol=1e-10, rtol=0
            )
            np.testing.assert_allclose(value[key], fast[key], atol=1e-10, rtol=0)


def test_diagnostics_pin_the_uniform_center():
    a = np.load(DATA / "DIAGNOSTICS.npz")
    np.testing.assert_allclose(a["grid_tvd"][:, 20, 20], 0, atol=1e-12)
    assert np.all(a["grid_tvd"] >= a["grid_profile_tvd"] - 1e-12)
    np.testing.assert_allclose(a["model_mean"][:, 1, 10], 8, atol=1e-10)

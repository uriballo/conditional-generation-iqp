"""Independent checks of pairing and the data-by-ancilla interaction."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from report_ashkin_teller_data_efficiency import (
    ARMS,
    METRICS,
    SIZES,
    recompute,
    row_index,
    summarize,
)


def rows(seeds):
    out = []
    for si, s in enumerate(seeds):
        for size in SIZES:
            for e, r in ARMS:
                value = (
                    0.4
                    + si * 0.001
                    - size * 0.0001
                    - (r or 0) * (0.004 + 0.00001 * size)
                )
                if e.endswith("_encoder_only"):
                    value = 0.45
                out.append(
                    {
                        "seed": s,
                        "train_size": size,
                        "model": e,
                        "n_ancillas": r,
                        "train": {m: [value] * 16 for m in METRICS},
                        "test": {
                            m: (value + np.arange(9) * 0.001).tolist() for m in METRICS
                        },
                    }
                )
    return out


def test_interaction_uses_within_seed_difference_not_independent_arms():
    seeds = (1, 2, 3, 4)
    report = summarize(rows(seeds), seeds)
    for e in ("cat", "bond"):
        v = next(
            x
            for x in report["contrasts"]
            if x["kind"] == "interaction_1000_minus_100"
            and x["model"] == e
            and x["n_ancillas"] == 4
            and x["metric"] == "tvd"
        )
        np.testing.assert_allclose(v["seed_values"], 0.036, atol=1e-14)
        np.testing.assert_allclose(v["ci95"], [0.036, 0.036], atol=1e-14)
    assert len(report["curves"]) == 30  # Ancilla counts remain separate.


def test_bootstrap_resamples_four_seed_means_not_36_conditions():
    seeds = (1, 2, 3, 4)
    records = rows(seeds)
    report = summarize(records, seeds)
    v = next(
        x
        for x in report["curves"]
        if x["model"] == "cat" and x["n_ancillas"] == 0 and x["train_size"] == 100
    )
    means = np.array([0.394, 0.395, 0.396, 0.397])
    draws = np.random.default_rng(950702).integers(0, 4, (10000, 4))
    np.testing.assert_allclose(v["metrics"]["tvd"]["seed_values"], means)
    np.testing.assert_allclose(
        v["metrics"]["tvd"]["ci95"], np.quantile(means[draws].mean(1), [0.025, 0.975])
    )


def test_incomplete_duplicate_and_mutating_controls_are_rejected():
    seeds = (1, 2)
    records = rows(seeds)
    with pytest.raises(ValueError, match="missing"):
        row_index(records[:-1], seeds)
    with pytest.raises(ValueError, match="duplicate"):
        row_index([*records, records[0]], seeds)
    next(x for x in records if x["model"] == "cat_encoder_only")["test"]["tvd"][0] += (
        0.1
    )
    with pytest.raises(AssertionError):
        row_index(records, seeds)


def test_independent_metric_recomputation_matches_identity_and_manual_tvd():
    p = np.full((2, 4096), 1 / 4096)
    q = p.copy()
    q[:, 0] += 1 / 4096
    q[:, 1] -= 1 / 4096
    values = recompute(q, p)
    np.testing.assert_allclose(values["tvd"], 1 / 4096)
    for v in recompute(p, p).values():
        np.testing.assert_array_equal(v, 0)

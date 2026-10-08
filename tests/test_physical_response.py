import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.physical_response import _observables, physical_arrays, render

DATA = Path(__file__).resolve().parents[1] / "paper/data/curie-weiss"


def _data():
    sweep = json.loads((DATA / "response_sweep.json").read_text())
    reported = json.loads((DATA / "figure_data.json").read_text())
    return sweep, reported


def test_uniform_and_reference_values():
    p = np.array([math.comb(16, k) / 2**16 for k in range(17)])
    absolute, heat = _observables(p, 2.0)
    assert absolute == pytest.approx(0.196380615234375)
    assert heat == pytest.approx(4 * 15 / 512)
    sweep, reported = _data()
    arrays = physical_arrays(sweep, reported)
    assert arrays["product"][2][-1] == pytest.approx(1.2989738720797575)
    assert arrays["target"][1][-1] == pytest.approx(0.5042506535620443)
    assert arrays["product"][0][-1] == pytest.approx(0.855060112503538)


def test_nonlinear_observable_is_per_seed_before_aggregation():
    p0 = np.zeros(17)
    p0[8] = 1
    p1 = np.zeros(17)
    p1[0] = 1
    profiles = np.array([p0, p1])
    _, individual = _observables(profiles, 1.0)
    _, pooled = _observables(profiles.mean(0), 1.0)
    assert np.allclose(individual, 0)
    assert pooled > 0


@pytest.mark.parametrize(
    "mutate",
    ["seed", "normalization", "marker_missing", "identity", "duplicate", "target"],
)
def test_malformed_inputs_rejected(mutate):
    sweep, reported = _data()
    bad = deepcopy(sweep)
    if mutate == "seed":
        bad["seeds"] = bad["seeds"][:-1]
    if mutate == "normalization":
        bad["arms"]["product"]["profiles_per_seed"][0][0][0] += 0.1
    if mutate == "marker_missing":
        reported["records"]["independent_conditional"].pop()
    if mutate == "identity":
        bad["provenance"]["checkpoint_identities"][0]["checkpoint_sha256"] = "bad"
    if mutate == "duplicate":
        reported["records"]["independent_conditional"].append(
            deepcopy(reported["records"]["independent_conditional"][0])
        )
    if mutate == "target":
        profile = bad["target"]["profiles"][0]
        profile[0] += 0.001
        profile[8] -= 0.001
    with pytest.raises(ValueError):
        physical_arrays(bad, reported)


def test_independent_record_order_is_irrelevant():
    sweep, reported = _data()
    reversed_report = deepcopy(reported)
    reversed_report["records"]["independent_conditional"].reverse()
    reversed_report["records"]["independent_unconditional"].reverse()
    a = physical_arrays(sweep, reported)
    b = physical_arrays(sweep, reversed_report)
    assert np.allclose(
        a["independent"]["conditional"]["absolute_mean"],
        b["independent"]["conditional"]["absolute_mean"],
    )


def test_render_writes_three_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sweep, reported = _data()
    (data_dir / "response_sweep.json").write_text(json.dumps(sweep))
    (data_dir / "figure_data.json").write_text(json.dumps(reported))
    paths = render(data_dir, tmp_path / "out")
    assert len(paths) == 3 and all(path.exists() for path in paths)

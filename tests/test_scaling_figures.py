"""Scaling rendering audits for sampled-MMD trajectories."""

import json

import pytest

from paper_experiments.figures import _scaling


def test_scaling_renderer_rejects_nonfinite_curve_and_keeps_negative_values(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    payload = {
        "sizes": [64],
        "seeds": [100],
        "steps": [0, 2, 4],
        "curves": {
            "64": {
                "held_out": {"overall_per_seed": [[-0.3, -0.2, -0.1]]},
                "training": {"overall_per_seed": [[-0.3, -0.2, -0.1]]},
            }
        },
    }
    (data / "figure_data.json").write_text(json.dumps(payload))
    output = tmp_path / "rendered"
    _scaling(data, output)
    table = (output / "scaling_mmd.csv").read_text()
    assert "-0.3" in table
    payload["curves"]["64"]["held_out"]["overall_per_seed"][0][1] = float("nan")
    (data / "figure_data.json").write_text(json.dumps(payload, allow_nan=True))
    with pytest.raises(ValueError, match="non-finite"):
        _scaling(data, tmp_path / "bad-render")

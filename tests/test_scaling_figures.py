"""Scaling export and rendering audits for sampled-MMD trajectories."""

import json

import pytest

from paper_experiments.figures import _scaling, export_completed


def _record(seed=100, size=64, encoder="parity", value=-0.25, scale=1.1842981420325214):
    rows = [
        {
            "step": step,
            "grid": grid,
            "mean": value + step / 10000,
            "mean_per_condition": [value + step / 10000],
        }
        for grid in ("training", "held_out")
        for step in (0, 2, 4)
    ]
    return {
        "experiment": "scaling",
        "seed": seed,
        "job": {
            "size": size,
            "job_id": f"job-{seed}",
            "config": {"encoder": encoder, "encoder_scale": scale},
        },
        "held_out_evaluation": {"rows": rows},
        "ansatz": {"name": f"lattice_{size}", "n_qubits": size},
        "smoke": True,
    }


def _write_records(root, records):
    root.mkdir()
    (root / "reported-results.json").write_text(json.dumps(records))


def test_scaling_export_records_model_identity_and_sampled_mmd_provenance(tmp_path):
    source = tmp_path / "runs"
    destination = tmp_path / "export"
    _write_records(source, [_record()])
    export_completed(source, destination)
    payload = json.loads((destination / "scaling/figure_data.json").read_text())
    assert payload["model_metadata"]["encoder"] == "parity"
    assert payload["model_metadata"]["encoder_scale"] == 1.1842981420325214
    assert payload["model_metadata"]["ansatz_by_size"]["64"][0]["name"] == "lattice_64"
    assert payload["model_metadata"]["job_ids"] == ["job-100"]
    _scaling(destination / "scaling", tmp_path / "rendered")
    provenance = json.loads(
        (
            tmp_path / "rendered/curie_weiss_lattice_scaling_mmd.provenance.json"
        ).read_text()
    )
    assert provenance["primary_metric"] == "held_out_sampled_mmd_squared"
    assert provenance["aggregation"] == "median and interquartile range across seeds"
    assert len(provenance["input_sha256"]) == 64


def test_scaling_export_rejects_mixed_encoders(tmp_path):
    source = tmp_path / "runs"
    _write_records(source, [_record(), _record(seed=101, encoder="parity_075")])
    with pytest.raises(ValueError, match="mixed encoders"):
        export_completed(source, tmp_path / "export")


def test_scaling_export_rejects_mixed_encoder_scales(tmp_path):
    source = tmp_path / "runs"
    _write_records(source, [_record(), _record(seed=101, scale=0.75)])
    with pytest.raises(ValueError, match="mixed encoder scales"):
        export_completed(source, tmp_path / "export")


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

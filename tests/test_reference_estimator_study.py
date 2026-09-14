import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "reference_estimator_study",
    Path(__file__).parents[1] / "scripts" / "run_reference_estimator_study.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


def _manifest(path: Path) -> Path:
    payload = {
        "study_id": "test",
        "sample_budgets": [8],
        "repetitions": 2,
        "root_seed": 17,
        "cases": [
            {
                "case_id": "n4-product",
                "encoder": "angle",
                "n_qubits": 4,
                "encoding": "angle",
                "angles": [0.1, -0.2, 0.3, 0.4],
                "generators": [[1, 0, 0, 0], [0, 1, 0, 0]],
                "params": [0.2, -0.1],
                "ops": [[1, 0, 0, 0], [0, 1, 1, 0]],
                "phase_regime": "small",
                "condition": "test",
            }
        ],
    }
    path.write_text(json.dumps(payload))
    return path


def test_pilot_shapes_reproducibility_and_resume(tmp_path: Path, monkeypatch):
    manifest = _manifest(tmp_path / "manifest.json")
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv", ["runner", "--manifest", str(manifest), "--output", str(output)]
    )
    main()
    arrays_path = output / "n4-product__b8.npz"
    with np.load(arrays_path) as arrays:
        first = {name: arrays[name].copy() for name in arrays.files}
    monkeypatch.setattr(
        "sys.argv", ["runner", "--manifest", str(manifest), "--output", str(output)]
    )
    main()
    with np.load(arrays_path) as arrays:
        for name, value in first.items():
            np.testing.assert_array_equal(arrays[name], value)


def test_mismatched_resume_is_rejected(tmp_path: Path, monkeypatch):
    manifest = _manifest(tmp_path / "manifest.json")
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv", ["runner", "--manifest", str(manifest), "--output", str(output)]
    )
    main()
    payload = json.loads(manifest.read_text())
    payload["root_seed"] = 99
    manifest.write_text(json.dumps(payload))
    monkeypatch.setattr(
        "sys.argv", ["runner", "--manifest", str(manifest), "--output", str(output)]
    )
    with pytest.raises(RuntimeError, match="stale or mismatched"):
        main()

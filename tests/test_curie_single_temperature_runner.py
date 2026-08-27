from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from iqp_cond.artifacts import sha256_file, write_immutable_json
from iqp_cond.curie_single_temperature import (
    SingleTemperatureManifest,
    plan_single_temperature_runs,
)
from iqp_cond.curie_single_temperature_runner import (
    execute_run,
    target_samples,
    training_config,
)

REPOSITORY = Path(__file__).parents[1]
MANIFEST_PATH = REPOSITORY / "configs" / "curie_weiss_single_temperature_v1.json"
sys.path.insert(0, str(REPOSITORY / "scripts"))

from summarize_curie_single_temperature import (  # noqa: E402
    FinalizationError,
    adjacent_training_mixture,
    load_single_results,
)


def manifest() -> SingleTemperatureManifest:
    return SingleTemperatureManifest.from_json(MANIFEST_PATH)


def first_run():
    return plan_single_temperature_runs(manifest())[0]


def test_target_samples_are_deterministic_and_role_separated() -> None:
    run = first_run()
    training = target_samples(run, role="training", count=64)
    repeated = target_samples(run, role="training", count=64)
    validation = target_samples(run, role="validation", count=64)

    assert np.array_equal(training, repeated)
    assert training.shape == (64, run.n_qubits)
    assert not np.array_equal(training, validation)
    assert training_config(manifest()).batch_size == 75


def test_lightweight_execution_is_restartable_and_detects_corrupt_arrays(
    tmp_path: Path,
) -> None:
    run = first_run()
    campaign = tmp_path / "campaign"
    first = execute_run(
        REPOSITORY,
        manifest(),
        run,
        campaign,
        invocation=("pytest",),
        updates=1,
        lightweight=True,
    )
    repeated = execute_run(
        REPOSITORY,
        manifest(),
        run,
        campaign,
        invocation=("different invocation is ignored on restart",),
        updates=1,
        lightweight=True,
    )

    assert repeated == first
    assert first["run_identity"] == run.to_dict()
    assert first["metrics"]["parameter_count"] == run.parameter_count
    assert first["metrics"]["evaluation"]["role"] == "training"
    arrays_path = campaign / "arrays" / f"{run.run_id}.npz"
    arrays_path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="missing or conflicting arrays"):
        execute_run(
            REPOSITORY,
            manifest(),
            run,
            campaign,
            invocation=("pytest",),
            updates=1,
            lightweight=True,
        )


def test_execution_rejects_orphaned_arrays(tmp_path: Path) -> None:
    run = first_run()
    campaign = tmp_path / "campaign"
    arrays_path = campaign / "arrays" / f"{run.run_id}.npz"
    arrays_path.parent.mkdir(parents=True)
    arrays_path.write_bytes(b"orphan")

    with pytest.raises(RuntimeError, match="orphaned arrays"):
        execute_run(
            REPOSITORY,
            manifest(),
            run,
            campaign,
            invocation=("pytest",),
            updates=1,
            lightweight=True,
        )


def test_adjacent_mixture_uses_only_bracketing_training_outputs() -> None:
    stored = {
        0.5: np.asarray([1.0, 0.0]),
        0.7: np.asarray([0.0, 1.0]),
    }

    mixture = adjacent_training_mixture(0.6, (0.5, 0.7), stored)

    assert np.allclose(mixture, [0.5, 0.5])


def test_finalizer_rejects_incomplete_and_duplicate_campaigns(tmp_path: Path) -> None:
    current_manifest = manifest()
    with pytest.raises(FinalizationError, match="incomplete"):
        load_single_results(current_manifest, [tmp_path])

    run = first_run()
    roots = [tmp_path / "host1", tmp_path / "host2"]
    for root in roots:
        arrays_path = root / "arrays" / f"{run.run_id}.npz"
        arrays_path.parent.mkdir(parents=True)
        arrays_path.write_bytes(b"same")
        payload = {
            "schema_version": 1,
            "manifest_digest": current_manifest.digest,
            "run_identity": run.to_dict(),
            "run_id": run.run_id,
            "source_tree_digest": "a" * 64,
            "code_revision": "b" * 40,
            "optimizer_updates": 2000,
            "lightweight": False,
            "arrays_sha256": sha256_file(arrays_path),
            "metrics": {},
        }
        write_immutable_json(root / "runs" / f"{run.run_id}.json", payload)
        envelope = json.loads((root / "runs" / f"{run.run_id}.json").read_text())
        assert envelope["payload"]["run_id"] == run.run_id

    with pytest.raises(FinalizationError, match="duplicate"):
        load_single_results(current_manifest, roots)

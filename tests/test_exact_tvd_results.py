from pathlib import Path

import numpy as np

from iqp_cond.exact_tvd_results import (
    RunRecord,
    _resolve_probability_path,
    bootstrap_ci,
    comparison_rows,
    interpolate_probabilities,
)


def test_remote_absolute_probability_path_falls_back_after_sync(tmp_path):
    envelope_dir = tmp_path / "envelopes"
    probability_dir = tmp_path / "probabilities"
    envelope_dir.mkdir()
    probability_dir.mkdir()
    artifact = probability_dir / "run.npz"
    artifact.write_bytes(b"immutable")
    resolved = _resolve_probability_path(
        envelope_dir, "/remote/host/results/probabilities/run.npz"
    )
    assert resolved == artifact


def _record(identity, conditions, probabilities, targets):
    metrics = tuple(
        {"condition": float(condition), "profile_tvd": 0.0} for condition in conditions
    )
    return RunRecord(
        identity,
        np.asarray(conditions),
        np.asarray(probabilities),
        np.asarray(targets),
        metrics,
        Path("unused.json"),
    )


def test_interpolation_and_bootstrap_are_deterministic():
    mixed = interpolate_probabilities(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.25)
    assert np.allclose(mixed, [0.75, 0.25])
    assert bootstrap_ci([1.0, 2.0, 3.0], seed=4) == bootstrap_ci(
        [1.0, 2.0, 3.0], seed=4
    )


def test_generalization_uses_adjacent_bank_and_oracle():
    spec = {"training_conditions": [0.0, 1.0], "held_out_conditions": [0.5]}
    config = {"datasets": {"gaussian": spec}}
    identity = {
        "dataset": "gaussian",
        "visible_qubits": 8,
        "ancillas": 0,
        "seed": 100,
        "family": "conditioned",
        "condition_role": "shared",
    }
    target = np.array([[0.5, 0.5]] * 3)
    records = [
        _record(identity, [0.0, 0.5, 1.0], target, target),
        _record(
            {**identity, "family": "independent", "condition_role": "train_0"},
            [0.0],
            [[0.8, 0.2]],
            [[0.5, 0.5]],
        ),
        _record(
            {**identity, "family": "independent", "condition_role": "train_1"},
            [1.0],
            [[0.2, 0.8]],
            [[0.5, 0.5]],
        ),
        _record(
            {**identity, "family": "independent", "condition_role": "held_out_0"},
            [0.5],
            [[0.6, 0.4]],
            [[0.5, 0.5]],
        ),
    ]
    rows = comparison_rows(records, config)
    held = next(row for row in rows if row["split"] == "held_out")
    assert np.isclose(held["conditioned_tvd"], 0.0)
    assert np.isclose(held["interpolation_tvd"], 0.0)
    assert np.isclose(held["oracle_tvd"], 0.1)

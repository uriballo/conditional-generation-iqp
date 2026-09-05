import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "curie_weiss_interval_prototype",
    Path(__file__).parents[1] / "scripts/run_curie_weiss_interval_prototype.py",
)
prototype = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(prototype)


def _args(ancillas):
    return Namespace(
        n_qubits=2,
        ancillas=ancillas,
        seed=3,
        steps=1,
        samples=1,
        batch_size=1,
        validation_samples=1,
        n_ops=1,
        n_iqp_samples=1,
        validation_iqp_samples=1,
        validation_members=1,
        ensemble_members=1,
        half_width_pi=[0.1],
    )


@pytest.mark.parametrize("ancillas", [0, 1])
def test_run_uses_prefix_and_role_specific_widths(monkeypatch, ancillas):
    seen = {}

    def train(*args, **kwargs):
        seen["centers"], seen["widths"] = args[2], args[3]
        return SimpleNamespace(
            best_params=0, generators=[], best_step=0, best_validation_loss=0.0
        )

    def exact(*args):
        seen["inference_angles"] = args[2]
        size = 2 ** (2 + ancillas)
        return np.full((len(args[2]), size), 1 / size)

    monkeypatch.setattr(prototype, "train_with_resampled_angles", train)
    monkeypatch.setattr(prototype, "exact_iqp_probabilities_many", exact)
    payload = prototype.run(_args(ancillas))
    all_betas = np.concatenate((prototype.TRAINING_BETAS, prototype.HELDOUT_BETAS))
    if ancillas:
        ancilla_centers = np.repeat(
            prototype.beta_to_angles(all_betas)[:, None], ancillas, axis=1
        )
        expected = np.asarray(
            prototype.ancilla_only_condition_angles(ancilla_centers, 2)
        )
        np.testing.assert_array_equal(seen["widths"][:2], 0)
        assert np.all(seen["widths"][2:] > 0)
        np.testing.assert_array_equal(seen["inference_angles"][:, :2], expected[:, :2])
    else:
        expected = np.repeat(prototype.beta_to_angles(all_betas)[:, None], 2, axis=1)
        assert np.all(seen["widths"] > 0)
        assert np.any(seen["inference_angles"] != expected)
    np.testing.assert_array_equal(seen["centers"], expected[:6])
    assert len(payload["rows"]) == 11
    assert {row["role"] for row in payload["rows"]} == {"training", "heldout"}

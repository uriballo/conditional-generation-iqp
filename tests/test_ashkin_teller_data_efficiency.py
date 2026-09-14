"""Guard tests for the data-size/ancilla data-efficiency campaign."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import ashkin_teller_data_efficiency as campaign


def test_draw_data_is_paired_and_uses_independent_validation_streams():
    train, validation = campaign.draw_data(4711)
    assert train.shape == validation.shape == (16, 1000, 12)
    assert train.dtype == validation.dtype == np.int8
    assert not np.array_equal(train, validation)
    np.testing.assert_array_equal(train, campaign.draw_data(4711)[0])


def test_freeze_shapes_and_initialization_uses_only_each_prefix(tmp_path):
    manifest = campaign.freeze(tmp_path)
    assert manifest["ancillas"] == [0, 2, 4]
    with np.load(tmp_path / "data.npz") as data:
        assert data["samples"].shape == (10, 2, 16, 1000, 12)
        for r in campaign.ANCILLAS:
            assert data[f"initial_r{r}"].shape == (10, 3, 78 + 12 * r)
            g = data[f"generators_r{r}"]
            expected = campaign.arm_initial(data["samples"][0, 0, :, :100], g, 4711)
            np.testing.assert_allclose(data[f"initial_r{r}"][0, 0], expected)
            changed = data["samples"][0, 0].copy()
            changed[:, 100:] = 1 - changed[:, 100:]
            np.testing.assert_allclose(
                campaign.arm_initial(changed[:, :100], g, 4711), expected
            )
        assert "test_targets" not in data.files


def test_verify_manifest_detects_data_tampering(tmp_path):
    campaign.freeze(tmp_path)
    path = tmp_path / "data.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="Frozen data changed"):
        campaign.verify_manifest(tmp_path)


def test_evaluate_requires_prediction_seal_before_opening_test_laws(
    tmp_path, monkeypatch
):
    campaign.freeze(tmp_path)
    touched = []

    def forbidden(conditions):
        touched.append(np.asarray(conditions).copy())
        raise AssertionError("held-out law accessed before prediction seal")

    monkeypatch.setattr(campaign, "_laws", forbidden)
    with pytest.raises(RuntimeError, match="sealed"):
        campaign.evaluate(tmp_path)
    assert touched == []


@pytest.mark.parametrize("call", ["train", "classical"])
def test_fitting_is_closed_after_evaluation_marker(tmp_path, monkeypatch, call):
    campaign.freeze(tmp_path)
    (tmp_path / "evaluation-started.json").write_text("{}")
    monkeypatch.setattr(campaign, "verify_manifest", lambda out: {})
    with pytest.raises(RuntimeError, match="evaluation"):
        if call == "train":
            campaign.train(tmp_path, "cat", 4711, 100, 0)
        else:
            campaign.classical(tmp_path, 4711, 100)

"""Contract tests for the frozen scalar Gaussian translation protocol."""

from __future__ import annotations

import json
import re
from pathlib import Path

import iqp_cond as ciqp
import numpy as np
import pytest

from experiments.common.config import RunConfig
from experiments.common.datasets import target_mean
from experiments.scalar_gaussian_translation.protocol import (
    ANCHOR_COUNTS,
    ENCODERS,
    INTERVALS,
    anchor_conditions,
    classify_condition,
    encoder_angles,
    evaluation_grid,
    make_encoder,
    matched_exposure_steps,
    target_diagnostics,
    target_moments,
    target_probabilities,
    training_set_label,
)
from experiments.scalar_gaussian_translation.study import build_study


def test_encoder_angle_formulas_and_validation() -> None:
    c = np.array([0.0, 0.5, 1.0])
    assert np.allclose(
        encoder_angles(c, "broadcast_affine"), np.pi * (0.32 - 0.68 * c[:, None])
    )
    t = np.arange(8) / 7
    expected = np.pi * (
        (0.28 + 0.06 * t[None, :]) - (0.62 + 0.08 * t[None, :]) * c[:, None]
    )
    assert np.allclose(encoder_angles(c, "qubit_affine"), expected)
    expected_anti = np.full((3, 8), np.pi / 2)
    expected_anti[:, ::2] = np.pi * c[:, None]
    expected_anti[:, 1::2] = np.pi * (1 - c[:, None])
    assert np.allclose(encoder_angles(c, "anti_block"), expected_anti)
    assert np.array_equal(encoder_angles(c, "condition_blind"), np.zeros((3, 8)))
    for name in ("broadcast_affine", "qubit_affine"):
        assert np.all(np.diff(encoder_angles(c, name), axis=0) < 0)
        theta = encoder_angles(np.linspace(0, 1, 101), name)
        post_h = np.stack(
            (
                np.cos(theta / 2) + np.sin(theta / 2),
                np.cos(theta / 2) - np.sin(theta / 2),
            ),
            axis=-1,
        ) / np.sqrt(2)
        assert np.all(np.abs(post_h) > 1e-8)
        assert not np.allclose(theta[0] + theta[-1], 0.0)
    with pytest.raises(ValueError):
        encoder_angles(np.array([-0.01]), "broadcast_affine")


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("s", [np.pi / 54, np.pi / 27])
def test_exponential_fallback_angles_and_preparation(k: int, s: float) -> None:
    c = np.array([0.0, 0.25, 1.0])
    phase = encoder_angles(c, "exponential", fallback_k=k, fallback_s=s)
    expected = np.zeros((3, 8))
    expected[:, :k] = s * (3.0 ** np.arange(k))[None, :] * c[:, None]
    assert np.allclose(phase, expected)
    encoder = make_encoder(c, "exponential", fallback_k=k, fallback_s=s)
    assert tuple(tuple(axis) for axis in encoder.axes) == (
        tuple("Y" * 8),
        tuple("Z" * 8),
    )
    assert np.allclose(np.asarray(encoder.angles)[:, 0], np.pi / 2)
    assert np.allclose(np.asarray(encoder.angles)[:, 1], expected)


def test_encoder_public_contract_and_register_ansatz() -> None:
    c = np.array([0.17, 0.51, 0.83])
    for name in ENCODERS:
        report = ciqp.check_encoder(
            make_encoder(c, name), n_samples=2048, probes_per_condition=128, n_masks=3
        )
        assert report.ok, report.failures
    config = RunConfig(
        study="scalar_gaussian_translation",
        encoder="broadcast_affine",
        n_visible=8,
        n_ancillas=0,
    )
    study = build_study()
    model = ciqp.ConditionalModel(
        study.make_ansatz(config), study.make_encoder(c, config)
    )
    assert model.ansatz.n_parameters == 36
    assert model.encoder.n_qubits == 8
    assert config.n_ancillas == 0
    for invalid in (
        RunConfig(study="scalar_gaussian_translation", n_visible=7),
        RunConfig(study="scalar_gaussian_translation", n_visible=8, n_ancillas=1),
    ):
        with pytest.raises(ValueError):
            study.make_ansatz(invalid)
    assert build_study().held_out_conditions.tolist() == evaluation_grid("dev").tolist()
    assert (
        build_study(evaluation_split="audit").held_out_conditions.tolist()
        == evaluation_grid("audit").tolist()
    )


def test_protocols_counts_endpoints_and_random_layouts() -> None:
    assert training_set_label("near", 4, "even", None) == "near-k4-even"
    assert training_set_label("near", 4, "random", 1102) == "near-k4-random-1102"
    with pytest.raises(ValueError):
        training_set_label("near", 4, "random", None)
    for name, (low, high) in INTERVALS.items():
        assert anchor_conditions(name, 2).tolist() == [low, high]
        for count in ANCHOR_COUNTS:
            values = anchor_conditions(name, count)
            assert len(values) == count
            assert values[0] == low and values[-1] == high
            assert np.all(np.diff(values) > 0)
            random_a = anchor_conditions(name, count, "random", 1102)
            random_b = anchor_conditions(name, count, "random", 1102)
            assert np.array_equal(random_a, random_b)
            assert random_a[0] == low and random_a[-1] == high
    near = anchor_conditions("near", 5)
    mirror_near = anchor_conditions("mirror_near", 5)
    far = anchor_conditions("far", 5)
    mirror_far = anchor_conditions("mirror_far", 5)
    assert np.allclose(mirror_near, 1 - near[::-1])
    assert np.allclose(mirror_far, 1 - far[::-1])
    random_near = anchor_conditions("near", 5, "random", 1101)
    random_mirror_near = anchor_conditions("mirror_near", 5, "random", 1101)
    assert not np.allclose(random_mirror_near, 1 - random_near[::-1])
    assert classify_condition(0.4, "near", near) == "training"
    assert classify_condition(0.25, "interpolation", near) == "held_out_interpolation"
    assert classify_condition(0.9, "near", near) == "near_extrapolation"
    assert classify_condition(0.6, "far", far) == "far_extrapolation"
    assert classify_condition(0.9, "far", far) == "far_extrapolation"
    assert classify_condition(0.1, "mirror_near", mirror_near) == "mirrored_diagnostic"
    assert matched_exposure_steps(600, 2) == 400
    with pytest.raises(ValueError):
        matched_exposure_steps(301, 2)


def test_eval_grids_disjoint_immutable_and_seeds_separate() -> None:
    grids = [evaluation_grid(name) for name in ("dev", "validation", "audit")]
    assert all(grid.shape == (20,) and not grid.flags.writeable for grid in grids)
    assert len(set(np.concatenate(grids))) == 60
    for grid, offset in zip(grids, (1, 2, 3), strict=True):
        assert np.allclose(grid, 1 - grid[::-1])
        base = np.asarray([(6 * k + offset) / 128 for k in range(10)])
        assert np.allclose(grid, np.sort(np.concatenate((base, 1 - base))))


def test_target_moments_truncation_and_protocol_manifest() -> None:
    for c in (0.0, 0.2, 0.5, 1.0):
        mean, std = target_moments(c)
        assert abs(mean - target_mean(8, c)) < 1e-9
        assert 0 < std < 4
    diagnostic = target_diagnostics(0.0)
    assert diagnostic["nominal_mean"] == target_mean(8, 0.0)
    assert diagnostic["exponent_width"] == pytest.approx(np.sqrt(8) * 0.5)
    assert diagnostic["actual_mean"] == pytest.approx(diagnostic["nominal_mean"])
    assert diagnostic["actual_std"] != pytest.approx(diagnostic["exponent_width"])
    assert diagnostic["boundary_shell_mass"] > 0
    with pytest.raises(ValueError):
        target_moments(0.5, n_visible=12)
    with pytest.raises(ValueError):
        target_diagnostics(0.5, n_visible=12)
    probabilities = target_probabilities(np.array([0.25, 0.75]))
    assert probabilities.shape == (2, 256)
    assert np.allclose(probabilities.sum(axis=1), 1)
    manifest = json.loads(
        (
            Path(__file__).parents[1] / "scalar_gaussian_translation" / "protocol.json"
        ).read_text()
    )
    assert manifest["production_manifest_enabled"] is False
    assert manifest["n_visible"] == 8 and manifest["n_ancillas"] == 0
    assert manifest["forbidden_visible_sizes"] == [12]
    assert len(manifest["stage2"]["candidates"]) == 8
    assert manifest["main"]["encoders"] == [
        "broadcast_affine",
        "qubit_affine",
        "anti_block",
    ]
    assert manifest["main"]["budget_modes"] == ["fixed_total", "matched_exposure"]
    assert manifest["data"] == {
        "train_samples": 512,
        "validation_samples": 512,
        "test_samples": 512,
    }
    assert manifest["training"]["validation_n_ops"] == 64
    assert manifest["training"]["validation_mc_samples"] == 256
    assert manifest["training"]["gemm_dtype"] == "float32"
    assert manifest["training"]["sample_chunk"] == 128
    gate = manifest["stage1_development_gate"]["advance_ry_encoder_if_all"]
    assert gate["median_train_tvd_at_most"] == 0.45
    assert gate["median_held_interpolation_tvd_at_most"] == 0.45
    assert gate["worst_seed_held_mean_tvd_at_most"] == 0.60
    assert (
        manifest["stage1_development_gate"][
            "invoke_exponential_fallback_only_if_both_ry_encoders_fail"
        ]
        is True
    )


def test_public_api_only_imports() -> None:
    source_root = Path(__file__).parents[1] / "scalar_gaussian_translation"
    forbidden = re.compile(r"\biqp_cond\.(?:_|src\b)|from\s+iqp_cond\s+import\s+_")
    for path in source_root.glob("*.py"):
        assert not forbidden.search(path.read_text())

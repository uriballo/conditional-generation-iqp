from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from iqp_cond.curie_weiss_ancilla_protocol import broadcast_angle
from iqp_cond.curie_weiss_capacity_control import (
    COLD_BETAS,
    CRITICAL_BETAS,
    PARAMETER_COUNT,
    conditioned_run,
    evaluate_control,
    load_manifest,
    visible_123local_generators,
)

ROOT = Path(__file__).parents[1]


def test_visible_control_contains_all_one_two_and_three_body_generators() -> None:
    generators = visible_123local_generators()
    assert generators.shape == (PARAMETER_COUNT, 8)
    supports = np.sum(generators, axis=1)
    assert tuple(np.bincount(supports, minlength=4)[1:]) == (8, 28, 56)
    assert len({tuple(row) for row in generators}) == PARAMETER_COUNT


def test_control_reuses_v2_map_and_paired_run_identity() -> None:
    manifest = load_manifest(ROOT / "configs/curie_weiss_m5_foundations_v2.json")
    run = conditioned_run(manifest, 3)
    assert run.optimizer_seed == 3
    assert run.ancillas == 5
    assert np.isclose(broadcast_angle(manifest, 0.5, 8)[0], -0.14 * math.pi)
    assert np.isclose(broadcast_angle(manifest, 1.5, 8)[0], 0.14 * math.pi)
    assert CRITICAL_BETAS == (0.9, 1.0, 1.1)
    assert COLD_BETAS == (1.3, 1.4, 1.5)


def test_exact_control_evaluation_covers_all_temperatures() -> None:
    manifest = load_manifest(ROOT / "configs/curie_weiss_m5_foundations_v2.json")
    rows, probabilities = evaluate_control(manifest, np.zeros(PARAMETER_COUNT))
    assert len(rows) == len(manifest.evaluation_betas) == 11
    assert probabilities.shape == (11, 256)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

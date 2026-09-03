from pathlib import Path

import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.curie_encoder_ablation import (
    ARMS,
    AblationManifest,
    arm_dispatch,
    condition_angles,
    encoder_only_visible_probabilities,
    generic_angle,
    plan_encoder_only,
    plan_runs,
)
from iqp_cond.encoding import ANGLE_ENCODING, SPIN_FLIP_BRANCH_ENCODING
from iqp_cond.exact import exact_iqp_visible_probabilities
from iqp_cond.exact_tvd_runner import _curie_angle


def test_manifest_has_exact_factorial_plan():
    manifest = AblationManifest.from_json(
        Path("configs/curie_encoder_ablation_v1.json")
    )
    runs = plan_runs(manifest)
    assert len(runs) == 240
    assert len({r.run_id for r in runs}) == 240
    assert len(plan_encoder_only(manifest)) == 24


def test_generic_map_is_affine_without_critical_landmark():
    assert np.isclose(generic_angle(0.5), np.pi / 8)
    assert np.isclose(generic_angle(1.5), 3 * np.pi / 8)
    assert np.isclose(generic_angle(1.0), np.pi / 4)
    assert generic_angle(1.0) != _curie_angle(1.0)


def test_dispatch_and_informed_map_match_production():
    for arm in ARMS:
        encoding, angle = arm_dispatch(arm, 1.2)
        assert ("spin_flip" in arm) == (encoding == "spin_flip_branch")
        assert np.isclose(angle, _curie_angle(1.2)) == arm.endswith(
            "informed_piecewise"
        )


def test_encoder_only_is_normalized_and_has_no_leakage_argument():
    for arm in ARMS:
        p = encoder_only_visible_probabilities(8, 10, arm, 1.0)
        assert p.shape == (256,)
        assert np.isclose(p.sum(), 1.0)
        assert np.all(p >= 0)


def test_analytic_encoder_only_matches_zero_phase_exact_evaluator():
    betas = (0.6, 1.0, 1.4)
    for arm in ARMS:
        encoding = (
            SPIN_FLIP_BRANCH_ENCODING if arm.startswith("spin_flip") else ANGLE_ENCODING
        )
        n_visible, ancillas = 8, 2
        generators = np.asarray(complete_ansatz(n_visible + ancillas).generators)
        exact = exact_iqp_visible_probabilities(
            np.zeros(generators.shape[0]),
            generators,
            condition_angles(arm, betas, n_visible + ancillas),
            n_visible,
            ancillas,
            encoding,
        )
        analytic = np.stack(
            [
                encoder_only_visible_probabilities(n_visible, ancillas, arm, beta)
                for beta in betas
            ]
        )
        np.testing.assert_allclose(analytic, exact, atol=1e-12)

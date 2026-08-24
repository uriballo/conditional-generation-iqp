"""Unit tests for the three new core research experiments."""

from __future__ import annotations

import numpy as np

from iqp_cond.ancilla_ladder import (
    compute_reduced_density_matrix_properties,
)
from iqp_cond.ansatz import complete_ansatz
from iqp_cond.matched_capacity import (
    compute_physical_observables,
    generate_k_local_generators,
    matched_visible_ansatz,
)
from iqp_cond.multidimensional_2d import (
    EVAL_CONDITIONS_2D,
    TRAIN_CONDITIONS_2D,
    bilinear_mixture_2d,
    build_2d_gaussian_target,
    encoding_2d,
    gaussian_2d_target_params,
)


def test_matched_capacity_generator_counts():
    """Verify 3-local generator bank yields exactly 91 parameters on 8 qubits."""
    gens = generate_k_local_generators(n_qubits=8, target_count=91)
    assert gens.shape == (91, 8)

    weights = gens.sum(axis=1)
    assert np.count_nonzero(weights == 1) == 8  # 1-local
    assert np.count_nonzero(weights == 2) == 28  # 2-local
    assert np.count_nonzero(weights == 3) == 55  # 3-local (55 of 56)

    ansatz = matched_visible_ansatz(8, 91)
    assert ansatz.n_qubits == 8
    assert ansatz.n_parameters == 91

    # Check 13-qubit 2-local parameter count is also exactly 91
    ansatz_13 = complete_ansatz(13)
    assert ansatz_13.n_parameters == 91


def test_physical_observables_math():
    """Check magnetization and Binder cumulant computation."""
    # Deterministic test distribution: 50% all zeros, 50% all ones
    probs = np.zeros(256)
    probs[0] = 0.5  # all zeros (spins +1) -> M = +8
    probs[255] = 0.5  # all ones (spins -1) -> M = -8

    obs = compute_physical_observables(probs, n_qubits=8)
    assert np.isclose(obs["abs_magnetization"], 8.0)
    assert np.isclose(obs["m2"], 64.0)
    assert np.isclose(obs["m4"], 4096.0)
    # Binder cumulant U4 = 1 - M4 / (3 * M2^2) = 1 - 4096 / (3 * 4096) = 1 - 1/3 = 2/3
    assert np.isclose(obs["binder_cumulant"], 2.0 / 3.0)


def test_multidimensional_2d_encoding_and_targets():
    """Test 2D condition grid, target parameters, and encoding angles."""
    assert TRAIN_CONDITIONS_2D.shape == (9, 2)
    assert EVAL_CONDITIONS_2D.shape == (25, 2)

    angles = encoding_2d(EVAL_CONDITIONS_2D, n_qubits=8)
    assert angles.shape == (25, 8)
    assert np.all(np.isfinite(angles))

    # Verify target parameters
    for c in EVAL_CONDITIONS_2D:
        mean, std = gaussian_2d_target_params(c, n_qubits=8)
        assert 1.0 <= mean <= 7.0
        assert 0.5 <= std <= 3.0
        tgt = build_2d_gaussian_target(c, n_qubits=8)
        assert len(tgt.shell_probabilities) == 9
        assert np.isclose(tgt.shell_probabilities.sum(), 1.0)


def test_bilinear_mixture_2d_interpolation():
    """Verify bilinear mixture recovers corner points exactly."""
    train_conds = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    p00 = np.array([1.0, 0.0, 0.0, 0.0])
    p10 = np.array([0.0, 1.0, 0.0, 0.0])
    p01 = np.array([0.0, 0.0, 1.0, 0.0])
    p11 = np.array([0.0, 0.0, 0.0, 1.0])
    train_probs = np.stack([p00, p10, p01, p11])

    # Corner test
    p_corner = bilinear_mixture_2d(np.array([0.0, 0.0]), train_conds, train_probs)
    assert np.allclose(p_corner, p00)

    # Center test (0.5, 0.5)
    p_center = bilinear_mixture_2d(np.array([0.5, 0.5]), train_conds, train_probs)
    assert np.allclose(p_center, np.full(4, 0.25))


def test_reduced_density_matrix_and_entropy():
    """Verify trace preservation, non-negative eigenvalues, and entropy properties."""
    # Test r=0 (pure visible)
    params_36 = np.zeros(36)
    gens_36 = np.asarray(complete_ansatz(8).generators, dtype=np.int8)
    p_vis_0, ent_0, eff_r_0 = compute_reduced_density_matrix_properties(
        params_36, gens_36, condition_angle=0.0, n_visible=8, n_ancillas=0
    )
    assert len(p_vis_0) == 256
    assert np.isclose(p_vis_0.sum(), 1.0)
    assert np.isclose(ent_0, 0.0)  # Pure state has zero entropy
    assert np.isclose(eff_r_0, 1.0)  # Pure state has rank 1

    # Test r=2 (mixed visible)
    params_55 = np.random.default_rng(42).normal(scale=0.2, size=55)
    gens_55 = np.asarray(complete_ansatz(10).generators, dtype=np.int8)
    p_vis_2, ent_2, eff_r_2 = compute_reduced_density_matrix_properties(
        params_55, gens_55, condition_angle=0.2, n_visible=8, n_ancillas=2
    )
    assert len(p_vis_2) == 256
    assert np.isclose(p_vis_2.sum(), 1.0)
    assert ent_2 >= 0.0  # Entropy must be non-negative
    assert 1.0 <= eff_r_2 <= 4.0  # Rank cannot exceed 2^r = 4

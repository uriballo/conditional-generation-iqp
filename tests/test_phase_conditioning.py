"""Tests for the parameter-matched phase-conditioning baseline."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import exact_iqp_probabilities
from iqp_cond.phase_conditioning import (
    BALANCED_CONSTANT_SLOPE,
    COMMON_AFFINE,
    DETERMINISTIC_SIGNED_AFFINE,
    FIXED_INPUT_FAMILY,
    NO_CONDITION_FAMILY,
    PHASE_CONDITIONED_FAMILY,
    PHASE_FEATURE_MAPS,
    assert_deployment_contract,
    assert_fixed_input_contract,
    assert_no_condition_contract,
    assert_phase_conditioned_contract,
    deployment_signature,
    exact_effective_phases,
    exact_phase_conditioned_probabilities,
    normalize_conditions,
    parameter_signature,
    phase_feature_matrix,
    sampled_effective_phases,
    sampled_phase_conditioned_expectations,
)


def _primary_fixture() -> tuple[np.ndarray, np.ndarray]:
    ansatz = complete_ansatz(16)
    coefficients = np.linspace(-0.2, 0.3, ansatz.n_parameters)
    return coefficients, np.asarray(ansatz.generators)


def test_exactly_three_target_independent_parameter_matched_feature_maps():
    assert PHASE_FEATURE_MAPS == (
        "common_affine_multiplier",
        "signed_affine_multipliers",
        "balanced_constant_slope_bank",
    )
    conditions = np.array([0.0, 0.25, 1.0])

    common = phase_feature_matrix(conditions, COMMON_AFFINE)
    signed = phase_feature_matrix(conditions, DETERMINISTIC_SIGNED_AFFINE)
    balanced = phase_feature_matrix(conditions, BALANCED_CONSTANT_SLOPE)

    assert common.shape == signed.shape == balanced.shape == (3, 136)
    np.testing.assert_allclose(
        common,
        np.repeat(0.5 + conditions[:, None], 136, axis=1),
    )
    centered = 2.0 * conditions[:, None] - 1.0
    np.testing.assert_allclose(
        signed[:, 0::2],
        np.repeat(1.0 + centered, 68, axis=1),
    )
    np.testing.assert_allclose(
        signed[:, 1::2],
        np.repeat(1.0 - centered, 68, axis=1),
    )
    np.testing.assert_allclose(balanced[:, 0::2], 1.0)
    np.testing.assert_allclose(
        balanced[:, 1::2],
        np.repeat(conditions[:, None], 68, axis=1),
    )
    assert balanced.shape[1] // 2 == 68


def test_condition_normalization_is_frozen_and_bounded():
    np.testing.assert_allclose(
        normalize_conditions([0.2, 0.4, 0.6], 0.2, 0.6),
        [0.0, 0.5, 1.0],
    )
    with pytest.raises(ValueError, match="inside the frozen interval"):
        normalize_conditions([0.1, 0.4], 0.2, 0.6)
    with pytest.raises(ValueError, match="strictly greater"):
        normalize_conditions([0.2], 0.6, 0.6)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        phase_feature_matrix(np.array([-0.01]), COMMON_AFFINE)


def test_exact_and_sampled_effective_phases_agree_and_are_differentiable():
    coefficients, _ = _primary_fixture()
    conditions = np.array([0.0, 0.5, 1.0])
    exact = exact_effective_phases(
        coefficients,
        conditions,
        DETERMINISTIC_SIGNED_AFFINE,
    )
    sampled = sampled_effective_phases(
        jnp.asarray(coefficients),
        conditions,
        DETERMINISTIC_SIGNED_AFFINE,
    )
    np.testing.assert_allclose(sampled, exact)

    gradient = jax.grad(
        lambda values: jnp.sum(
            sampled_effective_phases(values, conditions, COMMON_AFFINE),
        ),
    )(jnp.asarray(coefficients))
    np.testing.assert_allclose(gradient, np.full(136, 3.0))


def test_parameter_and_deployment_signatures_capture_exact_identity():
    coefficients, generators = _primary_fixture()
    signature = parameter_signature(coefficients, generators)
    assert signature.coefficient_count == 136
    assert signature.generator_count == 136
    assert signature.qubit_count == 16
    assert len(signature.coefficients_sha256) == 64
    changed = parameter_signature(coefficients + 0.01, generators)
    assert changed.coefficients_sha256 != signature.coefficients_sha256
    assert changed.generators_sha256 == signature.generators_sha256

    phases = np.repeat(coefficients[None, :], 3, axis=0)
    inputs = np.stack(
        [np.zeros(16), np.full(16, 0.1), np.full(16, 0.2)],
    )
    deployment = deployment_signature(
        FIXED_INPUT_FAMILY,
        coefficients,
        generators,
        phases,
        inputs,
    )
    assert deployment.generators_invariant
    assert deployment.phases_invariant
    assert not deployment.inputs_invariant


def test_fixed_input_phase_conditioned_and_no_condition_contracts():
    coefficients, generators = _primary_fixture()
    varying_inputs = np.stack(
        [np.zeros(16), np.full(16, np.pi / 8), np.full(16, np.pi / 4)],
    )
    fixed = assert_fixed_input_contract(coefficients, generators, varying_inputs)
    assert fixed.family == FIXED_INPUT_FAMILY

    repeated_inputs = np.repeat(varying_inputs[[1]], 3, axis=0)
    no_condition = assert_no_condition_contract(
        coefficients,
        generators,
        repeated_inputs,
    )
    assert no_condition.family == NO_CONDITION_FAMILY

    phase = assert_phase_conditioned_contract(
        coefficients,
        generators,
        np.array([0.0, 0.5, 1.0]),
        varying_inputs[1],
        BALANCED_CONSTANT_SLOPE,
    )
    assert phase.family == PHASE_CONDITIONED_FAMILY
    assert not phase.phases_invariant
    assert phase.inputs_invariant

    with pytest.raises(ValueError, match="Condition-varying inputs"):
        assert_fixed_input_contract(coefficients, generators, repeated_inputs)
    with pytest.raises(ValueError, match="invariant input"):
        assert_no_condition_contract(coefficients, generators, varying_inputs)


def test_contract_rejects_recompiled_generators_and_changed_fixed_phases():
    coefficients, generators = _primary_fixture()
    generator_batch = np.repeat(generators[None, :, :], 2, axis=0)
    generator_batch[1, 0, 0] = 0
    inputs = np.stack([np.zeros(16), np.full(16, 0.2)])
    phases = np.repeat(coefficients[None, :], 2, axis=0)
    with pytest.raises(ValueError, match="compiled generators"):
        assert_deployment_contract(
            FIXED_INPUT_FAMILY,
            coefficients,
            generator_batch,
            phases,
            inputs,
        )

    changed_phases = phases.copy()
    changed_phases[1, 0] += 0.1
    with pytest.raises(ValueError, match="learned phase values unchanged"):
        assert_deployment_contract(
            FIXED_INPUT_FAMILY,
            coefficients,
            generators,
            changed_phases,
            inputs,
        )


def test_exact_and_sampled_phase_conditioned_helpers_use_effective_phases():
    coefficients, generators = _primary_fixture()
    conditions = np.array([0.0, 1.0])
    fixed_input = np.full(16, np.pi / 8)
    exact = exact_phase_conditioned_probabilities(
        coefficients,
        generators,
        conditions,
        fixed_input,
        COMMON_AFFINE,
    )
    assert exact.shape == (2, 2**16)
    np.testing.assert_allclose(exact.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        exact[1],
        exact_iqp_probabilities(1.5 * coefficients, generators, fixed_input),
    )

    ops = np.stack([np.eye(16, dtype=np.int8)[0], np.ones(16, dtype=np.int8)])
    sampled = sampled_phase_conditioned_expectations(
        coefficients,
        generators,
        conditions,
        fixed_input,
        COMMON_AFFINE,
        ops,
        jax.random.PRNGKey(7),
        32,
    )
    assert sampled.values.shape == (2, 2)
    assert sampled.standard_errors.shape == (2, 2)
    assert sampled.sample_values.shape == (2, 32, 2)
    assert np.all(np.isfinite(sampled.values))


def test_phase_interface_rejects_non_matched_coefficient_counts():
    _, generators = _primary_fixture()
    with pytest.raises(ValueError, match=r"shape \(136,\)"):
        exact_effective_phases(np.zeros(135), [0.0, 1.0], COMMON_AFFINE)
    with pytest.raises(ValueError, match=r"\(136, n_qubits\)"):
        parameter_signature(np.zeros(136), generators[:-1])

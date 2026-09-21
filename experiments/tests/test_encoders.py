"""Encoder maps: shapes, branch constraint, and library integration."""

from __future__ import annotations

import iqp_cond as ciqp
import numpy as np

from experiments.common import datasets, encoders
from experiments.common.metrics import complement_tvd
from experiments.gaussian.study import target_probabilities


def test_scalar_broadcast_encoder_satisfies_the_protocol() -> None:
    conditions = np.linspace(0.0, 1.0, 3)
    states = encoders.scalar_broadcast_encoder(conditions, 5)
    assert isinstance(states, ciqp.Encoder)
    assert states.n_conditions == 3
    assert states.n_qubits == 5
    assert states.angles.shape == (3, 1, 5)
    assert states.axes == (("Y",) * 5,)


def test_anti_block_encoder_satisfies_the_protocol() -> None:
    conditions = np.linspace(0.0, 1.0, 3)
    states = encoders.anti_block_encoder(conditions, 5)
    assert isinstance(states, ciqp.Encoder)
    assert states.n_conditions == 3
    assert states.n_qubits == 5
    assert states.angles.shape == (3, 1, 5)
    assert states.axes == (("Y",) * 5,)


def test_anti_block_angles_are_anti_phase() -> None:
    angles = encoders.anti_block_angles(np.asarray([0.25]), 4)[0]
    assert np.allclose(angles, [np.pi * 0.25, np.pi * 0.75, np.pi * 0.25, np.pi * 0.75])


def test_anti_block_angles_pair_consecutive_wires_and_neutralize_odd_wire() -> None:
    conditions = np.asarray([0.0, 0.23, 0.5, 1.0])
    for n_qubits in (4, 5, 7):
        angles = encoders.anti_block_angles(conditions, n_qubits)
        excitation = np.sin(angles / 2.0) ** 2
        assert np.allclose(excitation.sum(axis=1), n_qubits / 2)
        assert np.allclose(angles[1, :4], [np.pi * 0.23, np.pi * 0.77] * 2)
        if n_qubits % 2:
            assert np.allclose(angles[:, -1], np.pi / 2)


def test_anti_block_angles_have_complement_angle_relation() -> None:
    conditions = np.linspace(0.0, 1.0, 11)
    angles = encoders.anti_block_angles(conditions, 5)
    reversed_angles = encoders.anti_block_angles(1.0 - conditions, 5)
    assert np.allclose(reversed_angles, np.pi - angles)


def test_anti_block_output_is_complement_covariant_with_ancilla() -> None:
    states = encoders.anti_block_encoder(np.asarray([0.2, 0.8]), 5)
    model = ciqp.ConditionalModel(
        ciqp.Ansatz.complete(5),
        states,
        n_visible=4,
    )
    rng = np.random.default_rng(17)
    params = rng.normal(size=model.ansatz.n_parameters)
    probabilities = model.probabilities(params)
    assert complement_tvd(probabilities[0], probabilities[1]) < 1e-10


def test_scalar_target_is_complement_covariant() -> None:
    probabilities = target_probabilities(np.asarray([0.2, 0.8]), 6)
    assert complement_tvd(probabilities[0], probabilities[1]) < 1e-12


def test_scalar_ablation_encoders_and_skew_properties() -> None:
    conditions = np.linspace(0.0, 1.0, 101)
    balanced = encoders.balanced_angles(conditions, 5)
    ramp = encoders.broadcast_ramp_angles(conditions, 5)
    target_mean = encoders.target_mean_angles(conditions, 5)
    assert np.allclose(balanced, np.pi / 2)
    assert np.allclose(ramp[:, 0], np.pi * conditions)
    target_excitation = np.sin(target_mean[:, 0] / 2.0) ** 2
    assert np.allclose(target_excitation, 0.15 + 0.70 * conditions)

    for delta in (-0.1, 0.1):
        angles = encoders.skew_angles(conditions, 5, delta)
        transformed = angles[:, 0] / np.pi
        assert np.all(np.diff(transformed) > 0.0)
        assert np.allclose(angles[:, 0] + angles[:, 1], np.pi)
        assert not np.allclose(
            encoders.skew_angles(1.0 - conditions, 5, delta),
            np.pi - angles,
        )


def test_skew_encoder_rejects_non_monotone_delta() -> None:
    try:
        encoders.skew_angles(np.asarray([0.0, 1.0]), 4, 1.0 / np.pi)
    except ValueError as error:
        assert "delta" in str(error)
    else:  # pragma: no cover - invalid delta must fail
        raise AssertionError("non-monotone skew delta was accepted")


def test_anti_block_swap_reverses_the_condition() -> None:
    low = encoders.anti_block_angles(np.asarray([0.0]), 8)[0]
    high = encoders.anti_block_angles(np.asarray([1.0]), 8)[0]
    # c = 0 and c = 1 differ by swapping the two block assignments.
    assert np.allclose(low, np.roll(high, 1))


def test_two_block_angles_stay_inside_the_frozen_branch() -> None:
    conditions = datasets.training_conditions()
    angles = encoders.two_block_angles(conditions, 12)
    assert angles.shape == (16, 12)
    assert np.all(np.abs(angles) <= np.pi * encoders.ALPHA_MAX_OVER_PI + 1e-9)
    # Positions alternate between the two blocks.
    blocks = encoders.block_assignment(12)
    assert np.allclose(angles[:, blocks == 0], angles[:, :1])
    assert np.allclose(angles[:, blocks == 1], angles[:, 1:2])


def test_broadcast_ablation_ignores_the_width_coordinate() -> None:
    conditions = np.asarray([[0.5, 0.30], [0.5, 0.45]])
    angles = encoders.broadcast_angles(conditions, 6)
    assert np.allclose(angles[0], angles[1])


def test_encoder_for_dispatches_named_arms() -> None:
    conditions = datasets.training_conditions()
    for name in ("two_block", "broadcast"):
        states = encoders.encoder_for(name, conditions, 8)
        assert isinstance(states, ciqp.Encoder)
        assert states.n_conditions == conditions.shape[0]

    scalar_conditions = np.linspace(0.0, 1.0, 5)
    for name in (
        "anti_block",
        "balanced",
        "broadcast_ramp",
        "target_mean",
        "skew_minus",
        "skew_plus",
        "anti_fixed_phase",
        "anti_reupload",
        "anti_perqubit_sine",
    ):
        states = encoders.encoder_for(name, scalar_conditions, 8)
        assert isinstance(states, ciqp.Encoder)
        assert states.n_conditions == scalar_conditions.size
    try:
        encoders.encoder_for("nope", conditions, 8)
    except ValueError as error:
        assert "unknown encoder" in str(error)
    else:  # pragma: no cover - the lookup must fail
        raise AssertionError("unknown encoder was accepted")


def test_layered_and_per_wire_appendix_controls() -> None:
    conditions = np.asarray([0.2, 0.8])
    fixed = encoders.anti_fixed_phase_encoder(conditions, 6)
    reupload = encoders.anti_reupload_encoder(conditions, 6)
    per_wire = encoders.anti_perqubit_sine_angles(conditions, 6)
    base = encoders.anti_block_angles(conditions, 6)

    assert fixed.angles.shape == (2, 3, 6)
    assert reupload.angles.shape == (2, 3, 6)
    assert fixed.axes == (("Y",) * 6, ("Z",) * 6, ("Y",) * 6)
    assert np.allclose(fixed.angles[:, 1], np.pi / 4.0)
    assert np.allclose(reupload.angles[:, 1, 0], np.pi * conditions)
    assert per_wire.shape == base.shape
    assert not np.allclose(per_wire, base)

"""Encoder maps: shapes, branch constraint, and library integration."""

from __future__ import annotations

import iqp_cond as ciqp
import numpy as np

from experiments.common import datasets, encoders


def test_scalar_broadcast_encoder_is_a_circuit_encoder() -> None:
    conditions = np.linspace(0.0, 1.0, 3)
    states = encoders.scalar_broadcast_encoder(conditions, 5)
    assert isinstance(states, ciqp.CircuitEncoder)
    assert states.n_conditions == 3
    assert states.n_qubits == 5
    ops = states.preparation_ops(0)
    assert len(ops) == 5
    assert all(operation.name == "RY" for operation in ops)


def test_anti_block_encoder_is_a_circuit_encoder() -> None:
    conditions = np.linspace(0.0, 1.0, 3)
    states = encoders.anti_block_encoder(conditions, 5)
    assert isinstance(states, ciqp.CircuitEncoder)
    assert states.n_conditions == 3
    assert states.n_qubits == 5
    ops = states.preparation_ops(0)
    assert len(ops) == 5
    assert all(operation.name == "RY" for operation in ops)


def test_anti_block_angles_are_anti_phase() -> None:
    angles = encoders.anti_block_angles(np.asarray([0.25]), 4)[0]
    assert np.allclose(angles, [np.pi * 0.25, np.pi * 0.75, np.pi * 0.25, np.pi * 0.75])


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
        assert isinstance(states, ciqp.CircuitEncoder)
        assert states.n_conditions == conditions.shape[0]
    try:
        encoders.encoder_for("nope", conditions, 8)
    except ValueError as error:
        assert "unknown encoder" in str(error)
    else:  # pragma: no cover - the lookup must fail
        raise AssertionError("unknown encoder was accepted")

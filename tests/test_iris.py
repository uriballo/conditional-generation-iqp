"""Tests for the Iris ancilla-only benchmark."""

import numpy as np

from iqp_cond.ancilla import ancilla_only_condition_angles
from iqp_cond.iris import load_iris_targets


def test_iris_targets_are_three_class_and_ancilla_only():
    dataset = load_iris_targets(
        seed=7,
        n_training=12,
        n_validation=8,
        n_validation_examples_per_class=5,
    )

    assert dataset.class_names == ("setosa", "versicolor", "virginica")
    assert dataset.training_targets.shape == (3, 12, 4)
    assert dataset.validation_targets.shape == (3, 8, 4)
    assert dataset.ancilla_angles.shape == (3, 2)

    condition_angles = ancilla_only_condition_angles(
        dataset.ancilla_angles,
        dataset.n_data_qubits,
    )
    assert np.all(condition_angles[:, :4] == 0.0)
    np.testing.assert_allclose(condition_angles[:, 4:], dataset.ancilla_angles)


def test_iris_two_bits_per_feature_expands_output_register():
    dataset = load_iris_targets(
        seed=7,
        n_training=12,
        n_validation=8,
        n_validation_examples_per_class=5,
        bits_per_feature=2,
    )

    assert dataset.training_targets.shape == (3, 12, 8)
    assert dataset.validation_targets.shape == (3, 8, 8)
    assert dataset.thresholds.shape == (4, 3)
    assert dataset.n_data_qubits == 8


def test_iris_supports_extra_latent_ancillas():
    dataset = load_iris_targets(
        seed=7,
        n_training=12,
        n_validation=8,
        n_validation_examples_per_class=5,
        n_ancillas=4,
    )

    assert dataset.ancilla_angles.shape == (3, 4)
    assert np.all(dataset.ancilla_angles[:, 2:] == 0.0)

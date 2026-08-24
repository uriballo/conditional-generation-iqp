"""Iris multiclass targets for the ancilla-only conditioning experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris

from .ancilla import (
    ancilla_ansatz,
    ancilla_only_condition_angles,
    marginalize_ancillas,
)
from .ansatz import Ansatz
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    total_variation_distance,
)
from .training import TrainingConfig, TrainingResult, train


@dataclass(frozen=True)
class IrisDataset:
    """A reproducible bitstring version of the three-class Iris dataset."""

    training_targets: np.ndarray
    validation_targets: np.ndarray
    ancilla_angles: np.ndarray
    thresholds: np.ndarray
    bits_per_feature: int
    class_names: tuple[str, ...]

    @property
    def n_classes(self) -> int:
        """Number of class Conditions."""
        return len(self.class_names)

    @property
    def n_data_qubits(self) -> int:
        """Number of output qubits used for discretized Iris features."""
        return int(self.training_targets.shape[-1])

    @property
    def n_ancillas(self) -> int:
        """Number of ancilla selector qubits."""
        return int(self.ancilla_angles.shape[-1])


def _binary_class_codes(
    n_classes: int,
    n_ancillas: int | None = None,
) -> np.ndarray:
    """Return little-endian binary class codes as 0/pi RY angles."""
    if n_classes < 2:
        raise ValueError("n_classes must be at least two")
    minimum_ancillas = int(np.ceil(np.log2(n_classes)))
    if n_ancillas is None:
        n_ancillas = minimum_ancillas
    if n_ancillas < minimum_ancillas:
        raise ValueError(
            f"n_ancillas must be at least {minimum_ancillas} for {n_classes} classes",
        )
    codes = (
        np.arange(n_classes, dtype=np.int64)[:, None]
        >> np.arange(n_ancillas, dtype=np.int64)[None, :]
    ) & 1
    return np.pi * codes.astype(np.float64)


def _discretize_features(
    features: np.ndarray,
    train_indices: np.ndarray,
    bits_per_feature: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-bin features and return little-endian bits plus thresholds."""
    if bits_per_feature not in (1, 2):
        raise ValueError("bits_per_feature must be 1 or 2")
    n_levels = 2**bits_per_feature
    quantiles = np.arange(1, n_levels, dtype=np.float64) / n_levels
    thresholds = np.quantile(
        features[train_indices],
        quantiles,
        axis=0,
    ).T
    bins = np.sum(
        features[:, :, None] >= thresholds[None, :, :],
        axis=2,
        dtype=np.int64,
    )
    bit_positions = np.arange(bits_per_feature, dtype=np.int64)
    encoded = ((bins[:, :, None] >> bit_positions[None, None, :]) & 1).reshape(
        features.shape[0],
        -1,
    )
    return encoded.astype(np.float64), thresholds


def load_iris_targets(
    seed: int = 0,
    n_training: int = 512,
    n_validation: int = 256,
    n_validation_examples_per_class: int = 10,
    bits_per_feature: int = 1,
    n_ancillas: int | None = None,
) -> IrisDataset:
    """Load and discretize Iris into class-conditional bitstring targets.

    Each of the four continuous Iris features becomes one or two output bits
    using quantile thresholds fitted on the training split. Samples are
    resampled with replacement to give the existing Conditional MMD trainer
    fixed-size, class-balanced target arrays. Class labels are encoded as
    binary 0/pi angles on ancillas only; the data-qubit angles are all zero.
    """
    if n_training < 2 or n_validation < 2:
        raise ValueError("n_training and n_validation must be at least two")
    if n_validation_examples_per_class < 1:
        raise ValueError("n_validation_examples_per_class must be positive")

    iris = load_iris()
    features = np.asarray(iris.data, dtype=np.float64)
    labels = np.asarray(iris.target, dtype=np.int64)
    class_names = tuple(str(name) for name in iris.target_names)
    rng = np.random.default_rng(seed)

    train_rows: list[np.ndarray] = []
    validation_rows: list[np.ndarray] = []
    for class_index in range(len(class_names)):
        class_indices = np.flatnonzero(labels == class_index)
        shuffled = rng.permutation(class_indices)
        if n_validation_examples_per_class >= shuffled.size:
            raise ValueError("validation split must leave training examples")
        validation_rows.append(shuffled[:n_validation_examples_per_class])
        train_rows.append(shuffled[n_validation_examples_per_class:])

    train_indices = np.concatenate(train_rows)
    encoded_features, thresholds = _discretize_features(
        features,
        train_indices,
        bits_per_feature,
    )

    training_targets = np.stack(
        [
            encoded_features[rows][
                rng.integers(0, len(rows), size=n_training)
            ]
            for rows in train_rows
        ],
    )
    validation_targets = np.stack(
        [
            encoded_features[rows][
                rng.integers(0, len(rows), size=n_validation)
            ]
            for rows in validation_rows
        ],
    )
    ancilla_angles = _binary_class_codes(len(class_names), n_ancillas)
    return IrisDataset(
        training_targets=training_targets,
        validation_targets=validation_targets,
        ancilla_angles=ancilla_angles,
        thresholds=thresholds,
        bits_per_feature=bits_per_feature,
        class_names=class_names,
    )


def _empirical_probabilities(
    samples: np.ndarray,
    n_data_qubits: int,
) -> np.ndarray:
    """Return empirical probabilities in little-endian integer order."""
    bitstrings = np.asarray(samples, dtype=np.int8)
    weights = 1 << np.arange(n_data_qubits, dtype=np.int64)
    indices = bitstrings.astype(np.int64) @ weights
    counts = np.bincount(indices, minlength=2**n_data_qubits)
    return counts.astype(np.float64) / bitstrings.shape[0]


def run_iris_experiment(
    seed: int,
    config: TrainingConfig,
    n_training: int = 512,
    n_validation: int = 256,
    bits_per_feature: int = 1,
    n_ancillas: int | None = None,
) -> tuple[TrainingResult, dict[str, object]]:
    """Train the complete 2-local ansatz with ancilla-only class encoding."""
    dataset = load_iris_targets(
        seed=seed,
        n_training=n_training,
        n_validation=n_validation,
        bits_per_feature=bits_per_feature,
        n_ancillas=n_ancillas,
    )
    ansatz: Ansatz = ancilla_ansatz(
        dataset.n_data_qubits,
        dataset.n_ancillas,
    )
    condition_angles = ancilla_only_condition_angles(
        dataset.ancilla_angles,
        dataset.n_data_qubits,
    )
    model_name = (
        f"iris_ancilla_only_{bits_per_feature}bit_m{dataset.n_ancillas}_complete"
    )
    result = train(
        model_name,
        ansatz,
        np.asarray(condition_angles),
        dataset.training_targets,
        dataset.validation_targets,
        seed,
        config,
    )

    joint_probabilities = exact_iqp_probabilities_many(
        result.best_params,
        result.generators,
        condition_angles,
    )
    model_probabilities = marginalize_ancillas(
        joint_probabilities,
        dataset.n_data_qubits,
        dataset.n_ancillas,
    )
    target_probabilities = np.stack(
        [
            _empirical_probabilities(targets, dataset.n_data_qubits)
            for targets in dataset.validation_targets
        ],
    )
    bitstrings = enumerate_bitstrings(dataset.n_data_qubits)
    classes = []
    for class_name, target, model in zip(
        dataset.class_names,
        target_probabilities,
        model_probabilities,
        strict=True,
    ):
        classes.append(
            {
                "class": class_name,
                "bitstring_tvd": total_variation_distance(target, model),
                "target_probabilities": target.tolist(),
                "model_probabilities": model.tolist(),
            },
        )

    return result, {
        "dataset": "iris",
        "encoding_scope": "ancilla_only",
        "topology": ansatz.name,
        "n_data_qubits": dataset.n_data_qubits,
        "n_ancillas": dataset.n_ancillas,
        "n_total_qubits": ansatz.n_qubits,
        "n_parameters": ansatz.n_parameters,
        "seed": seed,
        "class_names": list(dataset.class_names),
        "class_ancilla_angles": dataset.ancilla_angles.tolist(),
        "feature_thresholds": dataset.thresholds.tolist(),
        "n_training_per_class": n_training,
        "n_validation_per_class": n_validation,
        "bits_per_feature": dataset.bits_per_feature,
        "bitstrings": bitstrings.tolist(),
        "classes": classes,
    }

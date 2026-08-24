"""Log-K angle path generalization experiment over nested Hamming-ball targets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import log2, pi
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import nnls

from .ansatz import complete_ansatz
from .encoding import ANGLE_ENCODING
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    total_variation_distance,
)
from .hamming_balls import HammingBallsTarget
from .mmd import conditional_mmd
from .training import TrainingConfig, TrainingResult, train

N_QUBITS = 19
TRAIN_CENTER_COUNTS = (1, 2, 4, 8)
MIXTURE_WEIGHT = 0.5
MODE_MASS_FRACTION = 1e-2
NNLS_MIN_WEIGHT_FRACTION = 5e-2
DATA_SEED_OFFSETS = (600_000, 700_000, 800_000)
N_TRAINING = 10_000
N_VALIDATION = 2_000
N_HELDOUT = 2_000


def center_angle_for_count(center_count: int) -> float:
    """Map a planted center count in ``[1, 8]`` to its broadcast encoding angle."""
    count = float(center_count)
    if not 1.0 <= count <= 8.0:
        raise ValueError("center_count must lie between 1 and 8")
    return pi / 4.0 * log2(count) / 3.0


def heldout_angles() -> np.ndarray:
    """Return the midpoint angles between adjacent training classes."""
    train = np.asarray([center_angle_for_count(k) for k in TRAIN_CENTER_COUNTS])
    return 0.5 * (train[:-1] + train[1:])


def class_targets() -> tuple[HammingBallsTarget, ...]:
    """Return the four nested training classes K = 1, 2, 4, 8."""
    return HammingBallsTarget.nested_family(n_qubits=N_QUBITS)


def mixture_targets() -> tuple[BallMixtureTarget, ...]:
    """Return the three midpoint mixture targets between adjacent classes."""
    family = class_targets()
    return tuple(
        BallMixtureTarget(family[i], family[i + 1], MIXTURE_WEIGHT)
        for i in range(len(family) - 1)
    )


@dataclass(frozen=True)
class BallMixtureTarget:
    """A distribution-space mixture of two adjacent Hamming-ball classes."""

    left: HammingBallsTarget
    right: HammingBallsTarget
    weight: float = MIXTURE_WEIGHT

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must lie in [0, 1]")
        if self.left.n_qubits != self.right.n_qubits:
            raise ValueError("mixture components must share n_qubits")

    def bitstring_probabilities(self, bitstrings: np.ndarray) -> np.ndarray:
        """Return the exact mixture probability of each supplied binary string."""
        return (1.0 - self.weight) * self.left.bitstring_probabilities(
            bitstrings
        ) + self.weight * self.right.bitstring_probabilities(bitstrings)

    def sample(self, rng: np.random.Generator, n_samples: int) -> np.ndarray:
        """Draw bitstrings from the two components with the mixture weight."""
        left_count = int(rng.binomial(n_samples, 1.0 - self.weight))
        samples = np.concatenate(
            [
                self.left.sample(rng, left_count),
                self.right.sample(rng, n_samples - left_count),
            ],
        )
        rng.shuffle(samples)
        return samples


def sample_datasets(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned training, validation, and held-out mixture datasets."""
    family = class_targets()
    training_rng = np.random.default_rng(seed + DATA_SEED_OFFSETS[0])
    validation_rng = np.random.default_rng(seed + DATA_SEED_OFFSETS[1])
    heldout_rng = np.random.default_rng(seed + DATA_SEED_OFFSETS[2])
    training = np.stack(
        [target.sample(training_rng, N_TRAINING) for target in family],
    )
    validation = np.stack(
        [target.sample(validation_rng, N_VALIDATION) for target in family],
    )
    heldout = np.stack(
        [target.sample(heldout_rng, N_HELDOUT) for target in mixture_targets()],
    )
    return training, validation, heldout


def ball_columns(
    bitstrings: np.ndarray,
    centers: np.ndarray,
    flip_probability: float,
) -> np.ndarray:
    """Return one normalized single-ball distribution per supplied center."""
    bitstrings = np.asarray(bitstrings, dtype=np.int8)
    distances = np.count_nonzero(
        bitstrings[:, None, :] != centers[None, :, :],
        axis=2,
    )
    return flip_probability**distances * (1.0 - flip_probability) ** (
        bitstrings.shape[1] - distances
    )


def ball_fit_analysis(
    probabilities: np.ndarray,
    columns: np.ndarray,
) -> dict[str, object]:
    """Fit a probability vector as a nonnegative planted-ball mixture.

    The design adds a uniform background column; the effective center count
    is the number of ball weights carrying at least 5% of the fitted mass.
    """
    n_strings = columns.shape[0]
    design = np.column_stack([columns, np.full(n_strings, 1.0 / n_strings)])
    weights, residual = nnls(design, np.asarray(probabilities, dtype=np.float64))
    total = float(weights.sum())
    significant = weights[:-1] >= NNLS_MIN_WEIGHT_FRACTION * total
    return {
        "ball_weights": weights[:-1].tolist(),
        "background_weight": float(weights[-1]),
        "effective_center_count": int(significant.sum()),
        "relative_l2_residual": float(
            residual / np.linalg.norm(probabilities),
        ),
    }


def count_modes(probabilities: np.ndarray, n_qubits: int) -> int:
    """Count hypercube local maxima with mass at least 1% of the peak."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    threshold = MODE_MASS_FRACTION * float(probabilities.max())
    is_mode = np.ones(probabilities.size, dtype=bool)
    for bit in range(n_qubits):
        neighbors = np.arange(probabilities.size) ^ (1 << bit)
        is_mode &= probabilities > probabilities[neighbors]
    is_mode &= probabilities >= threshold
    return int(is_mode.sum())


def _point_metrics(
    angle: float,
    role: str,
    target_probabilities: np.ndarray,
    model_probabilities: np.ndarray,
    columns: np.ndarray,
    n_qubits: int,
) -> dict[str, object]:
    model_fit = ball_fit_analysis(model_probabilities, columns)
    target_fit = ball_fit_analysis(target_probabilities, columns)
    return {
        "role": role,
        "angle": angle,
        "angle_fraction_pi": angle / pi,
        "bitstring_tvd": total_variation_distance(
            target_probabilities,
            model_probabilities,
        ),
        "target_effective_k": target_fit["effective_center_count"],
        "effective_center_count": model_fit["effective_center_count"],
        "ball_weights": model_fit["ball_weights"],
        "background_weight": model_fit["background_weight"],
        "relative_l2_residual": model_fit["relative_l2_residual"],
        "n_modes": count_modes(model_probabilities, n_qubits),
    }


def run_hamming_experiment(
    seed: int,
    config: TrainingConfig,
    encoding: str = ANGLE_ENCODING,
) -> tuple[TrainingResult, dict[str, object]]:
    """Train on four center-count classes and evaluate three midpoint mixtures."""
    family = class_targets()
    mixtures = mixture_targets()
    train_angles = np.asarray(
        [center_angle_for_count(k) for k in TRAIN_CENTER_COUNTS],
    )
    test_angles = heldout_angles()
    training_data, validation_data, heldout_data = sample_datasets(seed)
    ansatz = complete_ansatz()
    model_name = (
        "hamming_complete"
        if encoding == ANGLE_ENCODING
        else f"hamming_{encoding}_complete"
    )
    result = train(
        model_name,
        ansatz,
        np.repeat(train_angles[:, None], N_QUBITS, axis=1),
        training_data,
        validation_data,
        seed,
        config,
        encoding,
    )

    bitstrings = enumerate_bitstrings(N_QUBITS)
    columns = ball_columns(
        bitstrings,
        family[-1].centers,
        family[-1].flip_probability,
    )
    all_angles = np.concatenate([train_angles, test_angles])
    all_roles = ["train"] * len(family) + ["heldout"] * len(mixtures)
    all_targets = family + mixtures
    model_probabilities = exact_iqp_probabilities_many(
        result.best_params,
        result.generators,
        np.repeat(all_angles[:, None], N_QUBITS, axis=1),
        encoding,
    )
    points = []
    for index, (angle, role, target) in enumerate(
        zip(all_angles, all_roles, all_targets, strict=True),
    ):
        target_probabilities = target.bitstring_probabilities(bitstrings)
        point = _point_metrics(
            float(angle),
            role,
            target_probabilities,
            model_probabilities[index],
            columns,
            N_QUBITS,
        )
        if role == "train":
            point["target_k"] = TRAIN_CENTER_COUNTS[index]
        else:
            train_index = index - len(family)
            left_tvd = total_variation_distance(
                target_probabilities,
                model_probabilities[train_index],
            )
            right_tvd = total_variation_distance(
                target_probabilities,
                model_probabilities[train_index + 1],
            )
            point.update(
                {
                    "k_left": TRAIN_CENTER_COUNTS[train_index],
                    "k_right": TRAIN_CENTER_COUNTS[train_index + 1],
                    "mix_weight": MIXTURE_WEIGHT,
                    "left_train_angle": float(train_angles[train_index]),
                    "right_train_angle": float(train_angles[train_index + 1]),
                    "left_bitstring_tvd": left_tvd,
                    "right_bitstring_tvd": right_tvd,
                    "oracle_bitstring_tvd": min(left_tvd, right_tvd),
                },
            )
        points.append(point)

    heldout = conditional_mmd(
        jnp.asarray(result.best_params),
        jnp.asarray(result.generators),
        jnp.asarray(np.repeat(test_angles[:, None], N_QUBITS, axis=1)),
        jnp.asarray(heldout_data),
        config.sigmas,
        config.validation_n_ops,
        config.validation_n_iqp_samples,
        jax.random.PRNGKey(seed + 3_000_000),
        encoding,
    )
    metrics = {
        "model_name": result.model_name,
        "encoding": encoding,
        "topology": result.topology,
        "seed": seed,
        "n_parameters": int(result.generators.shape[0]),
        "best_step": result.best_step,
        "best_validation_loss": result.best_validation_loss,
        "heldout_mmd": float(heldout.loss),
        "heldout_mmd_per_condition": np.asarray(
            heldout.per_condition,
        ).tolist(),
        "target_mapping": "k(alpha) = 2^(12*alpha/pi), p = 0.08, "
        "nested weight-9 centers",
        "n_qubits": N_QUBITS,
        "flip_probability": float(family[-1].flip_probability),
        "pattern_seed": int(family[-1].pattern_seed),
        "train_center_counts": list(TRAIN_CENTER_COUNTS),
        "heldout_angle_fractions_pi": (test_angles / pi).tolist(),
        "centers": family[-1].centers.tolist(),
        "points": points,
    }
    return result, metrics


def save_hamming_result(
    result: TrainingResult,
    metrics: dict[str, object],
    config: TrainingConfig,
    directory: Path,
) -> None:
    """Persist one hamming-ball seed and its complete evaluation."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{result.model_name}_seed{result.seed}"
    np.savez_compressed(
        directory / f"{stem}.npz",
        generators=result.generators,
        condition_angles=result.condition_angles,
        best_params=result.best_params,
        final_params=result.final_params,
        train_steps=result.train_steps,
        train_losses=result.train_losses,
        validation_steps=result.validation_steps,
        validation_losses=result.validation_losses,
        encoding=np.asarray(result.encoding),
    )
    payload = dict(metrics)
    payload["training_config"] = asdict(config)
    (directory / f"{stem}.json").write_text(
        json.dumps(payload, indent=2) + "\n",
    )

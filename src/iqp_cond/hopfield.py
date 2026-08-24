"""Hopfield Boltzmann targets and the beta-conditioned experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import log, pi, sqrt
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from .ancilla import (
    ancilla_ansatz,
    ancilla_condition_angles,
    exact_ancilla_marginal_probabilities,
)
from .ansatz import complete_ansatz
from .encoding import ANGLE_ENCODING
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    shell_probabilities,
    total_variation_distance,
)
from .mmd import (
    exact_conditional_mmd,
    exact_target_fouriers,
)
from .training import (
    TrainingConfig,
    TrainingResult,
    train_exact,
)

N_QUBITS = 19
PATTERN_SEED = 0
BETA_MIN = 0.5
BETA_MAX = 1.25
TRAIN_BETAS = np.asarray([0.5, 0.75, 1.0, 1.25])
HELDOUT_BETAS = np.asarray(
    [np.sqrt(0.5 * 0.75), np.sqrt(0.75 * 1.0), np.sqrt(1.0 * 1.25)],
)

PATTERNS_19 = np.asarray(
    [
        [1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1],
    ],
    dtype=np.float64,
)


def frozen_patterns(
    n_qubits: int = N_QUBITS,
    pattern_seed: int = PATTERN_SEED,
) -> np.ndarray:
    """Return the fixed pattern pair, regenerating it only off-design."""
    if n_qubits == N_QUBITS and pattern_seed == PATTERN_SEED:
        return PATTERNS_19
    rng = np.random.default_rng(pattern_seed)
    return rng.choice([-1.0, 1.0], size=(2, n_qubits))


def hopfield_couplings(patterns: np.ndarray) -> np.ndarray:
    """Return the Hopfield coupling matrix with zero diagonal."""
    patterns = np.asarray(patterns, dtype=np.float64)
    couplings = (
        np.einsum("pn,pm->nm", patterns, patterns) / patterns.shape[1]
    )
    np.fill_diagonal(couplings, 0.0)
    return couplings


def angle_for_beta(beta: float) -> float:
    """Map a beta in ``[BETA_MIN, BETA_MAX]`` to its broadcast angle."""
    beta = float(beta)
    if not BETA_MIN <= beta <= BETA_MAX:
        raise ValueError(f"beta must lie between {BETA_MIN} and {BETA_MAX}")
    return pi / 4.0 * log(beta / BETA_MIN) / log(BETA_MAX / BETA_MIN)


def broadcast_angles(
    betas: np.ndarray,
    n_qubits: int = N_QUBITS,
) -> np.ndarray:
    """Broadcast one angle per beta over all qubits."""
    betas = np.asarray(betas, dtype=np.float64)
    if betas.ndim != 1:
        raise ValueError("betas must be one-dimensional")
    angles = np.asarray([angle_for_beta(float(beta)) for beta in betas])
    return np.repeat(angles[:, None], n_qubits, axis=1)


@dataclass(frozen=True)
class HopfieldTarget:
    """Exact Boltzmann distribution of a two-pattern Hopfield model."""

    beta: float
    patterns: np.ndarray
    couplings: np.ndarray
    probabilities: np.ndarray
    shell_probabilities: np.ndarray
    pattern_indices: np.ndarray

    @classmethod
    def create(
        cls,
        beta: float,
        n_qubits: int = N_QUBITS,
        pattern_seed: int = PATTERN_SEED,
    ) -> HopfieldTarget:
        """Construct the exact target over all ``2**n`` little-endian states."""
        beta = float(beta)
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        patterns = frozen_patterns(n_qubits, pattern_seed)
        couplings = hopfield_couplings(patterns)
        bitstrings = enumerate_bitstrings(n_qubits)
        spins = 1.0 - 2.0 * bitstrings
        energies = -0.5 * np.einsum(
            "bi,bi->b",
            spins,
            spins @ couplings,
        )
        log_partition = logsumexp(-beta * energies)
        probabilities = np.exp(-beta * energies - log_partition)
        shells = shell_probabilities(probabilities, bitstrings)
        pattern_indices = []
        for pattern in (patterns[0], -patterns[0], patterns[1], -patterns[1]):
            bits = (pattern < 0).astype(int)
            pattern_indices.append(
                sum(int(bit) << qubit for qubit, bit in enumerate(bits)),
            )
        return cls(
            beta=beta,
            patterns=patterns,
            couplings=couplings,
            probabilities=probabilities,
            shell_probabilities=shells,
            pattern_indices=np.asarray(pattern_indices, dtype=np.uint32),
        )

    @property
    def pattern_mass(self) -> float:
        """Probability mass on ``{p1, -p1, p2, -p2}``."""
        return float(self.probabilities[self.pattern_indices].sum())

    def bitstring_probabilities(self, bitstrings: np.ndarray) -> np.ndarray:
        """Return the exact probability of each supplied binary string."""
        bitstrings = np.asarray(bitstrings)
        if bitstrings.ndim != 2 or bitstrings.shape[1] != self.patterns.shape[1]:
            raise ValueError("bitstrings must have shape (n_strings, n_qubits)")
        if not np.all((bitstrings == 0) | (bitstrings == 1)):
            raise ValueError("bitstrings must be binary")
        shifts = np.arange(bitstrings.shape[1], dtype=np.uint32)
        indices = (bitstrings.astype(np.uint32) @ (1 << shifts)).astype(np.int64)
        return self.probabilities[indices]

    def sample(self, rng: np.random.Generator, n_samples: int) -> np.ndarray:
        """Draw little-endian bitstrings exactly from the target."""
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        indices = rng.choice(
            self.probabilities.size,
            size=n_samples,
            p=self.probabilities,
        )
        shifts = np.arange(self.patterns.shape[1], dtype=np.uint32)
        return ((indices[:, None] >> shifts) & 1).astype(np.int8)


def hopfield_targets(
    betas: np.ndarray,
    n_qubits: int = N_QUBITS,
) -> tuple[HopfieldTarget, ...]:
    """Construct the exact target for each supplied beta."""
    return tuple(
        HopfieldTarget.create(float(beta), n_qubits)
        for beta in np.asarray(betas)
    )


def sample_beta_targets(
    betas: np.ndarray,
    seed: int,
    n_samples: int,
    n_qubits: int = N_QUBITS,
) -> np.ndarray:
    """Sample an aligned target dataset for each supplied beta."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            target.sample(rng, n_samples)
            for target in hopfield_targets(betas, n_qubits)
        ],
    )


def _point_metrics(
    beta: float,
    role: str,
    target: HopfieldTarget,
    model_probabilities: np.ndarray,
    bitstrings: np.ndarray,
) -> dict[str, object]:
    model_shells = shell_probabilities(model_probabilities, bitstrings)
    model_mass = float(model_probabilities[target.pattern_indices].sum())
    return {
        "role": role,
        "beta": beta,
        "angle": angle_for_beta(beta),
        "angle_fraction_pi": angle_for_beta(beta) / pi,
        "bitstring_tvd": total_variation_distance(
            target.probabilities,
            model_probabilities,
        ),
        "shell_tvd": total_variation_distance(
            target.shell_probabilities,
            model_shells,
        ),
        "target_pattern_mass": target.pattern_mass,
        "model_pattern_mass": model_mass,
        "pattern_mass_error": abs(model_mass - target.pattern_mass),
    }


def evaluate_at_betas(
    params: np.ndarray,
    generators: np.ndarray,
    betas: np.ndarray,
    encoding: str = ANGLE_ENCODING,
    n_qubits: int = N_QUBITS,
    n_ancillas: int = 0,
    ancilla_scheme: str = "broadcast_all",
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Return exact point metrics and model probabilities at the betas."""
    betas = np.asarray(betas, dtype=np.float64)
    targets = hopfield_targets(betas, n_qubits)
    bitstrings = enumerate_bitstrings(n_qubits)
    data_angles = broadcast_angles(betas, n_qubits)
    if n_ancillas > 0:
        model_probabilities = exact_ancilla_marginal_probabilities(
            params,
            generators,
            data_angles,
            n_qubits,
            n_ancillas,
            ancilla_scheme,
            encoding,
        )
    else:
        model_probabilities = exact_iqp_probabilities_many(
            params,
            generators,
            data_angles,
            encoding,
        )
    points = [
        _point_metrics(float(beta), "train", target, probs, bitstrings)
        for beta, target, probs in zip(
            betas,
            targets,
            model_probabilities,
            strict=True,
        )
    ]
    return points, model_probabilities


def run_hopfield(
    seed: int,
    config: TrainingConfig,
    encoding: str = ANGLE_ENCODING,
    n_ancillas: int = 0,
    ancilla_scheme: str = "broadcast_all",
) -> tuple[TrainingResult, dict[str, object]]:
    """Train one shared model at the four training betas and evaluate seven."""
    ansatz = (
        ancilla_ansatz(N_QUBITS, n_ancillas)
        if n_ancillas > 0
        else complete_ansatz(N_QUBITS)
    )
    if n_ancillas > 0:
        train_angles = ancilla_condition_angles(
            broadcast_angles(TRAIN_BETAS),
            n_ancillas,
            ancilla_scheme,
        )
        heldout_angles = ancilla_condition_angles(
            broadcast_angles(HELDOUT_BETAS),
            n_ancillas,
            ancilla_scheme,
        )
    else:
        train_angles = broadcast_angles(TRAIN_BETAS)
        heldout_angles = broadcast_angles(HELDOUT_BETAS)
    if n_ancillas > 0:
        model_name = (
            f"hopfield_ancilla{n_ancillas}_complete"
            if encoding == ANGLE_ENCODING
            else f"hopfield_ancilla{n_ancillas}_{encoding}_complete"
        )
    else:
        model_name = (
            "hopfield_complete"
            if encoding == ANGLE_ENCODING
            else f"hopfield_{encoding}_complete"
        )
    train_targets = hopfield_targets(TRAIN_BETAS)
    result = train_exact(
        model_name,
        ansatz,
        train_angles,
        exact_target_fouriers(
            np.stack([target.probabilities for target in train_targets]),
        ),
        seed,
        config,
        encoding,
    )

    all_betas = np.concatenate([TRAIN_BETAS, HELDOUT_BETAS])
    all_roles = ["train"] * len(TRAIN_BETAS) + ["heldout"] * len(HELDOUT_BETAS)
    targets = hopfield_targets(all_betas)
    bitstrings = enumerate_bitstrings(N_QUBITS)
    if n_ancillas > 0:
        model_probabilities = exact_ancilla_marginal_probabilities(
            result.best_params,
            result.generators,
            broadcast_angles(all_betas),
            N_QUBITS,
            n_ancillas,
            ancilla_scheme,
            encoding,
        )
    else:
        model_probabilities = exact_iqp_probabilities_many(
            result.best_params,
            result.generators,
            broadcast_angles(all_betas),
            encoding,
        )
    points = []
    for beta, role, target, model_probs in zip(
        all_betas,
        all_roles,
        targets,
        model_probabilities,
        strict=True,
    ):
        point = _point_metrics(float(beta), role, target, model_probs, bitstrings)
        if role == "heldout":
            right = int(np.searchsorted(TRAIN_BETAS, float(beta)))
            adjacent_indices = (right - 1, right)
            adjacent_shell_tvds = []
            adjacent_bitstring_tvds = []
            for adjacent_index in adjacent_indices:
                adjacent_probs = model_probabilities[adjacent_index]
                adjacent_shells = shell_probabilities(
                    adjacent_probs,
                    bitstrings,
                )
                adjacent_shell_tvds.append(
                    total_variation_distance(
                        target.shell_probabilities,
                        adjacent_shells,
                    ),
                )
                adjacent_bitstring_tvds.append(
                    total_variation_distance(target.probabilities, adjacent_probs),
                )
            point.update(
                {
                    "nearest_train_betas": [
                        float(TRAIN_BETAS[index]) for index in adjacent_indices
                    ],
                    "nearest_bitstring_tvd": min(adjacent_bitstring_tvds),
                    "nearest_shell_tvd": min(adjacent_shell_tvds),
                },
            )
        points.append(point)

    heldout_targets = hopfield_targets(HELDOUT_BETAS)
    heldout = exact_conditional_mmd(
        jnp.asarray(result.best_params),
        jnp.asarray(result.generators),
        jnp.asarray(heldout_angles),
        exact_target_fouriers(
            np.stack([target.probabilities for target in heldout_targets]),
        ),
        config.sigmas,
        encoding,
    )
    metrics = {
        "model_name": result.model_name,
        "encoding": encoding,
        "topology": result.topology,
        "seed": seed,
        "n_ancillas": n_ancillas,
        "ancilla_scheme": ancilla_scheme,
        "n_parameters": int(result.generators.shape[0]),
        "best_step": result.best_step,
        "best_validation_loss": result.best_validation_loss,
        "heldout_mmd": float(heldout.loss),
        "heldout_mmd_per_beta": np.asarray(heldout.per_condition).tolist(),
        "target_mapping": "alpha(beta) = pi/4 * log(beta/0.5)/log(8)",
        "train_betas": TRAIN_BETAS.tolist(),
        "heldout_betas": HELDOUT_BETAS.tolist(),
        "points": points,
    }
    return result, metrics


def run_no_condition(
    seed: int,
    config: TrainingConfig,
    encoding: str = ANGLE_ENCODING,
) -> tuple[TrainingResult, dict[str, object]]:
    """Train one shared model with all conditions at the center angle."""
    ansatz = complete_ansatz(N_QUBITS)
    center_angle = angle_for_beta(sqrt(BETA_MIN * BETA_MAX))
    model_name = (
        "hopfield_no_condition_complete"
        if encoding == ANGLE_ENCODING
        else f"hopfield_no_condition_{encoding}_complete"
    )
    targets = hopfield_targets(TRAIN_BETAS)
    result = train_exact(
        model_name,
        ansatz,
        np.full((len(TRAIN_BETAS), N_QUBITS), center_angle),
        exact_target_fouriers(
            np.stack([target.probabilities for target in targets]),
        ),
        seed,
        config,
        encoding,
    )
    points, _ = evaluate_at_betas(
        result.best_params,
        result.generators,
        TRAIN_BETAS,
        encoding,
    )
    for point in points:
        point["angle"] = center_angle
        point["angle_fraction_pi"] = center_angle / pi
    metrics = {
        "model_name": result.model_name,
        "encoding": encoding,
        "topology": result.topology,
        "seed": seed,
        "n_parameters": int(result.generators.shape[0]),
        "best_step": result.best_step,
        "best_validation_loss": result.best_validation_loss,
        "condition_angle": center_angle,
        "train_betas": TRAIN_BETAS.tolist(),
        "points": points,
    }
    return result, metrics


def run_separate_models(
    seed: int,
    config: TrainingConfig,
    encoding: str = ANGLE_ENCODING,
) -> list[tuple[TrainingResult, dict[str, object]]]:
    """Train one independent single-condition model per training beta."""
    ansatz = complete_ansatz(N_QUBITS)
    runs = []
    for _local_index, beta in enumerate(TRAIN_BETAS):
        model_name = (
            f"hopfield_separate_{beta!s}_complete"
            if encoding == ANGLE_ENCODING
            else f"hopfield_separate_{beta!s}_{encoding}_complete"
        )
        target = HopfieldTarget.create(float(beta))
        result = train_exact(
            model_name,
            ansatz,
            broadcast_angles(np.asarray([beta])),
            exact_target_fouriers(target.probabilities[None, :]),
            seed,
            config,
            encoding,
        )
        points, _ = evaluate_at_betas(
            result.best_params,
            result.generators,
            np.asarray([beta]),
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
            "train_betas": [float(beta)],
            "points": points,
        }
        runs.append((result, metrics))
    return runs


def save_hopfield_result(
    result: TrainingResult,
    metrics: dict[str, object],
    config: TrainingConfig,
    directory: Path,
) -> None:
    """Persist one seed and its complete evaluation."""
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

"""Two-dimensional multi-conditioning for continuous multi-attribute generation."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ansatz import complete_ansatz
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    shell_probabilities,
    total_variation_distance,
)
from .mmd import conditional_mmd
from .targets import GaussianHammingTarget
from .training import TrainingConfig, _batch_schedule, initialize_parameters

N_2D_QUBITS = 8
TRAIN_GRID_1D = (0.1, 0.5, 0.9)
EVAL_GRID_1D = (0.1, 0.3, 0.5, 0.7, 0.9)


def make_2d_grid(grid_1d: tuple[float, ...]) -> np.ndarray:
    """Create a 2D coordinate grid of shape (N*N, 2)."""
    xx, yy = np.meshgrid(grid_1d, grid_1d, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


TRAIN_CONDITIONS_2D = make_2d_grid(TRAIN_GRID_1D)  # (9, 2)
EVAL_CONDITIONS_2D = make_2d_grid(EVAL_GRID_1D)  # (25, 2)
HELD_OUT_CONDITIONS_2D = np.asarray(
    [
        pt
        for pt in EVAL_CONDITIONS_2D
        if not any(np.allclose(pt, tr, atol=1e-5) for tr in TRAIN_CONDITIONS_2D)
    ]
)  # (16, 2)


def gaussian_2d_target_params(
    c: np.ndarray,
    n_qubits: int = N_2D_QUBITS,
) -> tuple[float, float]:
    """Map 2D condition (c_mu, c_sigma) to continuous mean and std width."""
    c_mu, c_sigma = float(c[0]), float(c[1])
    mean = n_qubits * (0.25 + 0.50 * c_mu)
    std = np.sqrt(n_qubits) * (0.35 + 0.40 * c_sigma)
    return mean, std


def build_2d_gaussian_target(
    c: np.ndarray,
    n_qubits: int = N_2D_QUBITS,
) -> GaussianHammingTarget:
    """Create exact target for one 2D condition."""
    mean, std = gaussian_2d_target_params(c, n_qubits)
    return GaussianHammingTarget.create(n_qubits, mean, std)


def encoding_2d(
    conditions: np.ndarray,
    n_qubits: int = N_2D_QUBITS,
) -> np.ndarray:
    """Map (N_cond, 2) coordinates to (N_cond, n_qubits) product-state angles.

    Uses block-contrasted multi-variable linear encoding.
    """
    conditions = np.asarray(conditions, dtype=np.float64)
    half = n_qubits // 2
    # Coordinate 0 (mean) is weighted across all qubits
    # Coordinate 1 (width) adds a symmetric contrast between the two halves
    c0 = conditions[:, 0:1]
    c1 = conditions[:, 1:2]

    # Angle formula:
    # First half:  pi * (0.35 - 0.70*c0 + 0.25*(c1 - 0.5))
    # Second half: pi * (0.35 - 0.70*c0 - 0.25*(c1 - 0.5))
    angles_left = pi * (0.35 - 0.70 * c0 + 0.25 * (c1 - 0.5))
    angles_right = pi * (0.35 - 0.70 * c0 - 0.25 * (c1 - 0.5))

    left_block = np.repeat(angles_left, half, axis=1)
    right_block = np.repeat(angles_right, n_qubits - half, axis=1)
    return np.concatenate([left_block, right_block], axis=1)


@dataclass(frozen=True, slots=True)
class Multi2DResult:
    seed: int
    train_profile_tvd_mean: float
    heldout_profile_tvd_mean: float
    train_raw_tvd_mean: float
    heldout_raw_tvd_mean: float
    bilinear_heldout_profile_tvd_mean: float
    bilinear_heldout_raw_tvd_mean: float
    grid_profile_tvds: np.ndarray  # shape (5, 5)
    grid_raw_tvds: np.ndarray  # shape (5, 5)


def bilinear_mixture_2d(
    pt: np.ndarray,
    train_conditions: np.ndarray,
    train_probabilities: np.ndarray,
) -> np.ndarray:
    """Barycentric 2D bilinear interpolation from the surrounding grid box."""
    c0, c1 = float(pt[0]), float(pt[1])
    grid_vals = sorted(list(set(train_conditions[:, 0])))  # [0.1, 0.5, 0.9]

    # Find bounding interval for c0 and c1
    def find_interval(val: float, arr: list[float]) -> tuple[float, float]:
        for i in range(len(arr) - 1):
            if arr[i] <= val <= arr[i + 1]:
                return arr[i], arr[i + 1]
        if val <= arr[0]:
            return arr[0], arr[1]
        return arr[-2], arr[-1]

    x0, x1 = find_interval(c0, grid_vals)
    y0, y1 = find_interval(c1, grid_vals)

    def get_prob(x: float, y: float) -> np.ndarray:
        for idx, cond in enumerate(train_conditions):
            if np.isclose(cond[0], x, atol=1e-5) and np.isclose(
                cond[1], y, atol=1e-5
            ):
                return train_probabilities[idx]
        return train_probabilities[0]

    p00 = get_prob(x0, y0)
    p10 = get_prob(x1, y0)
    p01 = get_prob(x0, y1)
    p11 = get_prob(x1, y1)

    wx = 0.5 if np.isclose(x1, x0) else (c0 - x0) / (x1 - x0)
    wy = 0.5 if np.isclose(y1, y0) else (c1 - y0) / (y1 - y0)

    p_interp = (
        (1 - wx) * (1 - wy) * p00
        + wx * (1 - wy) * p10
        + (1 - wx) * wy * p01
        + wx * wy * p11
    )
    return p_interp / p_interp.sum()


def run_2d_gaussian_trial(
    seed: int,
    n_qubits: int = N_2D_QUBITS,
    config: TrainingConfig | None = None,
) -> Multi2DResult:
    """Train and evaluate a fixed-circuit 2D conditional Born machine."""
    if config is None:
        config = TrainingConfig(
            steps=2000,
            learning_rate=0.01,
            final_learning_rate=0.001,
            batch_size=50,
            n_ops=256,
            n_iqp_samples=8192,
            validation_interval=25,
            validation_n_ops=256,
            validation_n_iqp_samples=16384,
            sigmas=(1.0, 2.0, 4.0),
        )

    # 1. Targets
    train_conds = TRAIN_CONDITIONS_2D  # (9, 2)
    eval_conds = EVAL_CONDITIONS_2D  # (25, 2)
    bitstrings = enumerate_bitstrings(n_qubits)

    np_rng = np.random.default_rng(seed + 200)
    train_targets_list = []
    val_targets_list = []
    train_target_probs = []

    for c in train_conds:
        target = build_2d_gaussian_target(c, n_qubits)
        p_raw = target.bitstring_probabilities(bitstrings)
        train_target_probs.append(p_raw)
        train_draws = np_rng.choice(
            len(bitstrings),
            size=config.batch_size * 20,
            p=p_raw,
        )
        val_draws = np_rng.choice(
            len(bitstrings),
            size=config.validation_n_iqp_samples,
            p=p_raw,
        )
        train_targets_list.append(bitstrings[train_draws])
        val_targets_list.append(bitstrings[val_draws])

    train_targets = np.stack(train_targets_list)  # (9, N_samp, n_qubits)
    val_targets = np.stack(val_targets_list)

    # 2. Ansatz & Angles
    ansatz = complete_ansatz(n_qubits)
    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    train_angles = jnp.asarray(encoding_2d(train_conds, n_qubits))
    eval_angles = jnp.asarray(encoding_2d(eval_conds, n_qubits))

    # 3. Training Loop
    init_params = initialize_parameters(ansatz.n_parameters, seed)
    schedule = optax.cosine_decay_schedule(
        config.learning_rate,
        config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(init_params)
    batch_indices = _batch_schedule(
        len(train_conds),
        train_targets.shape[1],
        config,
        seed,
    )
    rng_key = jax.random.PRNGKey(seed)

    def compute_loss(params: jnp.ndarray, targets: jnp.ndarray, key: jax.Array):
        return conditional_mmd(
            params,
            generators,
            train_angles,
            targets,
            config.sigmas,
            config.n_ops,
            config.n_iqp_samples,
            key,
        ).loss

    def compute_val_loss(params: jnp.ndarray):
        return conditional_mmd(
            params,
            generators,
            train_angles,
            jnp.asarray(val_targets),
            config.sigmas,
            config.validation_n_ops,
            config.validation_n_iqp_samples,
            jax.random.PRNGKey(seed + 999),
        ).loss

    val_fn = jax.jit(compute_val_loss)
    grad_fn = jax.jit(jax.value_and_grad(compute_loss))

    best_params = init_params
    best_val = float(val_fn(init_params))
    current_params = init_params

    for step in range(config.steps):
        indices = batch_indices[step]
        batch = np.stack(
            [train_targets[i, indices[i]] for i in range(len(train_conds))]
        )
        _, grad = grad_fn(
            current_params,
            jnp.asarray(batch),
            jax.random.fold_in(rng_key, step),
        )
        updates, opt_state = optimizer.update(grad, opt_state, current_params)
        current_params = optax.apply_updates(current_params, updates)

        if (step + 1) % config.validation_interval == 0:
            val_score = float(val_fn(current_params))
            if val_score < best_val:
                best_val = val_score
                best_params = current_params

    # 4. Evaluate across 5x5 dense 2D grid
    model_eval_probs = exact_iqp_probabilities_many(
        np.asarray(best_params),
        np.asarray(generators, dtype=np.int8),
        np.asarray(eval_angles),
    )

    # Compute trained probabilities on training grid for bilinear baseline
    model_train_probs = exact_iqp_probabilities_many(
        np.asarray(best_params),
        np.asarray(generators, dtype=np.int8),
        np.asarray(train_angles),
    )

    grid_profile_tvds = np.zeros((len(EVAL_GRID_1D), len(EVAL_GRID_1D)))
    grid_raw_tvds = np.zeros((len(EVAL_GRID_1D), len(EVAL_GRID_1D)))

    train_prof_list, heldout_prof_list = [], []
    train_raw_list, heldout_raw_list = [], []
    bilinear_heldout_prof_list, bilinear_heldout_raw_list = [], []

    eval_idx = 0
    for i, _g0 in enumerate(EVAL_GRID_1D):
        for j, _g1 in enumerate(EVAL_GRID_1D):
            pt = eval_conds[eval_idx]
            target = build_2d_gaussian_target(pt, n_qubits)
            tgt_raw = target.bitstring_probabilities(bitstrings)
            tgt_shells = target.shell_probabilities

            mod_raw = model_eval_probs[eval_idx]
            mod_shells = shell_probabilities(mod_raw, bitstrings)

            raw_tvd = total_variation_distance(tgt_raw, mod_raw)
            prof_tvd = total_variation_distance(tgt_shells, mod_shells)

            grid_profile_tvds[i, j] = prof_tvd
            grid_raw_tvds[i, j] = raw_tvd

            is_train = any(
                np.allclose(pt, tr, atol=1e-5) for tr in train_conds
            )
            if is_train:
                train_prof_list.append(prof_tvd)
                train_raw_list.append(raw_tvd)
            else:
                heldout_prof_list.append(prof_tvd)
                heldout_raw_list.append(raw_tvd)

                # Bilinear mixture baseline
                p_bilinear = bilinear_mixture_2d(
                    pt, train_conds, model_train_probs
                )
                p_bilinear_shells = shell_probabilities(p_bilinear, bitstrings)
                bilinear_heldout_raw_list.append(
                    total_variation_distance(tgt_raw, p_bilinear)
                )
                bilinear_heldout_prof_list.append(
                    total_variation_distance(tgt_shells, p_bilinear_shells)
                )

            eval_idx += 1

    return Multi2DResult(
        seed=seed,
        train_profile_tvd_mean=float(np.mean(train_prof_list)),
        heldout_profile_tvd_mean=float(np.mean(heldout_prof_list)),
        train_raw_tvd_mean=float(np.mean(train_raw_list)),
        heldout_raw_tvd_mean=float(np.mean(heldout_raw_list)),
        bilinear_heldout_profile_tvd_mean=float(
            np.mean(bilinear_heldout_prof_list)
        ),
        bilinear_heldout_raw_tvd_mean=float(np.mean(bilinear_heldout_raw_list)),
        grid_profile_tvds=grid_profile_tvds,
        grid_raw_tvds=grid_raw_tvds,
    )

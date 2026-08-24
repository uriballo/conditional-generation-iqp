"""Matched-capacity control experiments for isolating latent mixed-state rank."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import pi

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ancilla import (
    ancilla_condition_angles,
    exact_ancilla_marginal_probabilities,
)
from .ansatz import Ansatz, complete_ansatz
from .curie_weiss_data import exact_curie_weiss_target
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    shell_probabilities,
    total_variation_distance,
)
from .mmd import conditional_mmd
from .training import TrainingConfig, _batch_schedule, initialize_parameters

VISIBLE_QUBITS = 8
TARGET_PARAMETERS = 91
TRAINING_BETAS = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
HELD_OUT_BETAS = (0.6, 0.8, 1.0, 1.2, 1.4)
ALL_BETAS = tuple(sorted(TRAINING_BETAS + HELD_OUT_BETAS))


def generate_k_local_generators(
    n_qubits: int = 8,
    target_count: int = 91,
) -> np.ndarray:
    """Generate canonical 1-, 2-, and 3-local generators up to target_count."""
    rows: list[list[int]] = []
    # 1-local
    for i in range(n_qubits):
        row = [0] * n_qubits
        row[i] = 1
        rows.append(row)

    # 2-local
    for i, j in itertools.combinations(range(n_qubits), 2):
        row = [0] * n_qubits
        row[i] = 1
        row[j] = 1
        rows.append(row)

    # 3-local if needed
    if len(rows) < target_count:
        for i, j, k in itertools.combinations(range(n_qubits), 3):
            row = [0] * n_qubits
            row[i] = 1
            row[j] = 1
            row[k] = 1
            rows.append(row)
            if len(rows) == target_count:
                break

    if len(rows) != target_count:
        raise ValueError(
            f"Unable to produce exactly {target_count} generators on {n_qubits} qubits"
        )
    return np.asarray(rows, dtype=np.int8)


def matched_visible_ansatz(
    n_qubits: int = VISIBLE_QUBITS,
    n_parameters: int = TARGET_PARAMETERS,
) -> Ansatz:
    """Construct an n-qubit pure visible Ansatz with exactly n_parameters."""
    generators = generate_k_local_generators(n_qubits, n_parameters)
    # Dummy connected graph for Ansatz dataclass
    import networkx as nx

    g = nx.complete_graph(n_qubits)
    return Ansatz("visible_matched_91", g, jnp.asarray(generators, dtype=jnp.float64))


def beta_to_angles(betas: tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Linear broadcast: alpha(beta) = pi * (0.35 - 0.70 * (beta - 0.5))."""
    betas_arr = np.asarray(betas, dtype=np.float64)
    return pi * (0.35 - 0.70 * (betas_arr - 0.5))


@dataclass(frozen=True, slots=True)
class ModelEvaluationResult:
    model_name: str
    seed: int
    training_profile_tvd: float
    held_out_profile_tvd: float
    training_raw_tvd: float
    held_out_raw_tvd: float
    per_beta_profile_tvd: dict[float, float]
    per_beta_raw_tvd: dict[float, float]
    per_beta_magnetization: dict[float, float]
    per_beta_binder_cumulant: dict[float, float]


def compute_physical_observables(
    probabilities: np.ndarray,
    n_qubits: int = VISIBLE_QUBITS,
) -> dict[str, float]:
    """Compute exact physical observables on visible spins."""
    bitstrings = enumerate_bitstrings(n_qubits)
    # Spins s_i in {-1, +1} corresponding to bit 0 -> +1, bit 1 -> -1
    spins = 1.0 - 2.0 * bitstrings
    magnetization = spins.sum(axis=1)  # shape (2^n,)

    m1 = float(np.abs(magnetization) @ probabilities)
    m2 = float((magnetization**2) @ probabilities)
    m4 = float((magnetization**4) @ probabilities)

    binder = float(1.0 - (m4 / (3.0 * (m2**2) + 1e-12)))
    return {
        "abs_magnetization": m1,
        "m2": m2,
        "m4": m4,
        "binder_cumulant": binder,
    }


def run_matched_capacity_trial(
    model_type: str,  # 'ancilla_5', 'visible_matched_91', 'visible_2local_36'
    seed: int,
    config: TrainingConfig | None = None,
) -> ModelEvaluationResult:
    """Train and evaluate one model type on Curie-Weiss with matched parameters."""
    if config is None:
        config = TrainingConfig(
            steps=2000,
            learning_rate=0.01,
            final_learning_rate=0.001,
            batch_size=75,
            n_ops=256,
            n_iqp_samples=8192,
            validation_interval=25,
            validation_n_ops=256,
            validation_n_iqp_samples=16384,
            sigmas=(1.0, 2.0, 4.0),
        )

    # 1. Target Data Generation
    train_betas = np.asarray(TRAINING_BETAS)
    all_betas = np.asarray(ALL_BETAS)
    bitstrings_vis = enumerate_bitstrings(VISIBLE_QUBITS)

    # Deterministic sampling from exact Gibbs distributions
    np_rng = np.random.default_rng(seed + 100)
    train_targets_list = []
    val_targets_list = []
    for b in train_betas:
        p_gibbs = exact_curie_weiss_target(VISIBLE_QUBITS, float(b)).probabilities
        train_draws = np_rng.choice(
            len(bitstrings_vis),
            size=config.batch_size * 20,
            p=p_gibbs,
        )
        val_draws = np_rng.choice(
            len(bitstrings_vis),
            size=config.validation_n_iqp_samples,
            p=p_gibbs,
        )
        train_targets_list.append(bitstrings_vis[train_draws])
        val_targets_list.append(bitstrings_vis[val_draws])

    train_targets = np.stack(train_targets_list)  # (n_cond, n_samples, 8)
    val_targets = np.stack(val_targets_list)

    # 2. Model Setup
    train_angles = beta_to_angles(train_betas)[:, None]  # (n_cond, 1)

    if model_type == "ancilla_5":
        n_ancillas = 5
        total_qubits = VISIBLE_QUBITS + n_ancillas
        ansatz = complete_ansatz(total_qubits)
        generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
        input_angles = jnp.asarray(
            ancilla_condition_angles(
                np.repeat(train_angles, VISIBLE_QUBITS, axis=1),
                n_ancillas,
                "broadcast_all",
            )
        )
        n_params = ansatz.n_parameters  # 91
    elif model_type == "visible_matched_91":
        n_ancillas = 0
        total_qubits = VISIBLE_QUBITS
        ansatz = matched_visible_ansatz(VISIBLE_QUBITS, TARGET_PARAMETERS)
        generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
        input_angles = jnp.asarray(np.repeat(train_angles, VISIBLE_QUBITS, axis=1))
        n_params = TARGET_PARAMETERS  # 91
    elif model_type == "visible_2local_36":
        n_ancillas = 0
        total_qubits = VISIBLE_QUBITS
        ansatz = complete_ansatz(VISIBLE_QUBITS)
        generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
        input_angles = jnp.asarray(np.repeat(train_angles, VISIBLE_QUBITS, axis=1))
        n_params = ansatz.n_parameters  # 36
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 3. Training Loop
    init_params = initialize_parameters(n_params, seed)
    schedule = optax.cosine_decay_schedule(
        config.learning_rate,
        config.steps,
        alpha=config.final_learning_rate / config.learning_rate,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(init_params)


    batch_indices = _batch_schedule(
        len(train_betas),
        train_targets.shape[1],
        config,
        seed,
    )
    rng_key = jax.random.PRNGKey(seed)

    def compute_loss(params: jnp.ndarray, targets: jnp.ndarray, key: jax.Array):
        return conditional_mmd(
            params,
            generators,
            input_angles,
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
            input_angles,
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
            [train_targets[i, indices[i]] for i in range(len(train_betas))]
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

    # 4. Exact Full-System Evaluation across all 11 Betas
    all_angles_vis = np.repeat(
        beta_to_angles(all_betas)[:, None], VISIBLE_QUBITS, axis=1
    )
    if n_ancillas > 0:
        model_probs = exact_ancilla_marginal_probabilities(
            np.asarray(best_params),
            np.asarray(generators, dtype=np.int8),
            all_angles_vis,
            n_data_qubits=VISIBLE_QUBITS,
            n_ancillas=n_ancillas,
        )
    else:
        model_probs = exact_iqp_probabilities_many(
            np.asarray(best_params),
            np.asarray(generators, dtype=np.int8),
            all_angles_vis,
        )

    per_beta_profile_tvd = {}
    per_beta_raw_tvd = {}
    per_beta_mag = {}
    per_beta_binder = {}

    for idx, b in enumerate(all_betas):
        target_raw = exact_curie_weiss_target(VISIBLE_QUBITS, float(b)).probabilities
        target_shells = shell_probabilities(target_raw, bitstrings_vis)

        mod_raw = model_probs[idx]
        mod_shells = shell_probabilities(mod_raw, bitstrings_vis)

        raw_tvd = total_variation_distance(target_raw, mod_raw)
        prof_tvd = total_variation_distance(target_shells, mod_shells)

        per_beta_profile_tvd[float(b)] = prof_tvd
        per_beta_raw_tvd[float(b)] = raw_tvd

        obs = compute_physical_observables(mod_raw, VISIBLE_QUBITS)
        per_beta_mag[float(b)] = obs["abs_magnetization"]
        per_beta_binder[float(b)] = obs["binder_cumulant"]

    train_prof_mean = float(
        np.mean([per_beta_profile_tvd[b] for b in TRAINING_BETAS])
    )
    heldout_prof_mean = float(
        np.mean([per_beta_profile_tvd[b] for b in HELD_OUT_BETAS])
    )
    train_raw_mean = float(np.mean([per_beta_raw_tvd[b] for b in TRAINING_BETAS]))
    heldout_raw_mean = float(
        np.mean([per_beta_raw_tvd[b] for b in HELD_OUT_BETAS])
    )

    return ModelEvaluationResult(
        model_name=model_type,
        seed=seed,
        training_profile_tvd=train_prof_mean,
        held_out_profile_tvd=heldout_prof_mean,
        training_raw_tvd=train_raw_mean,
        held_out_raw_tvd=heldout_raw_mean,
        per_beta_profile_tvd=per_beta_profile_tvd,
        per_beta_raw_tvd=per_beta_raw_tvd,
        per_beta_magnetization=per_beta_mag,
        per_beta_binder_cumulant=per_beta_binder,
    )

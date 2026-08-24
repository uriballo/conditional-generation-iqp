"""Ancilla ladder capacity sweep: tracking visible fidelity, entropy, and rank."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ancilla import (
    ancilla_condition_angles,
)
from .ansatz import complete_ansatz
from .curie_weiss_data import exact_curie_weiss_target
from .encoding import ANGLE_ENCODING, post_hadamard_amplitudes
from .exact import (
    _fwht,
    enumerate_bitstrings,
    shell_probabilities,
    total_variation_distance,
)
from .mmd import conditional_mmd
from .training import TrainingConfig, _batch_schedule, initialize_parameters

VISIBLE_QUBITS = 8
ANCILLA_RUNGS = (0, 1, 2, 3, 4, 5, 6)
TRAINING_BETAS = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
HELD_OUT_BETAS = (0.6, 0.8, 1.0, 1.2, 1.4)
ALL_BETAS = tuple(sorted(TRAINING_BETAS + HELD_OUT_BETAS))


def beta_to_angles(betas: tuple[float, ...] | np.ndarray) -> np.ndarray:
    """Linear broadcast angle mapping."""
    betas_arr = np.asarray(betas, dtype=np.float64)
    return pi * (0.35 - 0.70 * (betas_arr - 0.5))


@dataclass(frozen=True, slots=True)
class AncillaLadderResult:
    r: int
    n_total_qubits: int
    n_parameters: int
    seed: int
    train_profile_tvd: float
    heldout_profile_tvd: float
    train_raw_tvd: float
    heldout_raw_tvd: float
    cold_profile_tvd: float  # at beta = 1.5
    cold_raw_tvd: float
    critical_profile_tvd: float  # at beta = 1.0
    mean_entropy: float
    mean_effective_rank: float
    per_beta_entropy: dict[float, float]
    per_beta_effective_rank: dict[float, float]


def compute_reduced_density_matrix_properties(
    params: np.ndarray,
    generators: np.ndarray,
    condition_angle: float,
    n_visible: int = VISIBLE_QUBITS,
    n_ancillas: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Compute exact visible density matrix rho_vis, entropy S, and effective rank."""
    total_qubits = n_visible + n_ancillas
    indices = np.arange(2**total_qubits, dtype=np.uint32)
    phase = np.zeros(2**total_qubits, dtype=np.float64)

    for theta, generator in zip(params, generators, strict=True):
        wires = np.flatnonzero(generator)
        if wires.size == 1:
            bit = (indices >> int(wires[0])) & 1
            character = 1.0 - 2.0 * bit
        elif wires.size == 2:
            parity = (
                (indices >> int(wires[0])) ^ (indices >> int(wires[1]))
            ) & 1
            character = 1.0 - 2.0 * parity
        else:
            mask = 0
            for w in wires:
                mask |= 1 << int(w)
            parity = np.bitwise_count(indices & mask) & 1
            character = 1.0 - 2.0 * parity
        phase += theta * character

    phase_factors = np.exp(1j * phase)

    # Input state amplitudes
    angles = np.full(total_qubits, condition_angle, dtype=np.float64)
    amp_zero, amp_one = post_hadamard_amplitudes(
        jnp.asarray(angles), ANGLE_ENCODING
    )
    amp_zero = np.asarray(amp_zero, dtype=np.complex128)
    amp_one = np.asarray(amp_one, dtype=np.complex128)

    amp_zero_rev = amp_zero[::-1]
    amp_one_rev = amp_one[::-1]
    reference = np.array([amp_zero_rev[0], amp_one_rev[0]], dtype=np.complex128)
    for i in range(1, total_qubits):
        reference = np.outer(reference, [amp_zero_rev[i], amp_one_rev[i]]).ravel()

    diagonal_state = reference * phase_factors
    full_statevector = _fwht(diagonal_state) / np.sqrt(2**total_qubits)

    if n_ancillas == 0:
        visible_probs = np.abs(full_statevector) ** 2
        return visible_probs, 0.0, 1.0

    # Reshape statevector to (2^n_ancillas, 2^n_visible) with little-endian order
    psi_mat = full_statevector.reshape(2**n_ancillas, 2**n_visible)
    # Visible density matrix rho_vis = psi_mat^dagger @ psi_mat
    rho_vis = psi_mat.conj().T @ psi_mat

    # Diagonal elements are the visible marginal probabilities
    visible_probs = np.real(np.diag(rho_vis))
    visible_probs = visible_probs / visible_probs.sum()

    # Eigenvalues of rho_vis
    eigenvalues = np.linalg.eigvalsh(rho_vis)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    eigenvalues = eigenvalues / eigenvalues.sum()

    # Von Neumann entropy
    pos_eig = eigenvalues[eigenvalues > 1e-15]
    entropy = -float(np.sum(pos_eig * np.log(pos_eig)))

    # Effective rank (Participation ratio)
    purity = float(np.sum(eigenvalues**2))
    eff_rank = float(1.0 / (purity + 1e-15))

    return visible_probs, entropy, eff_rank


def run_ancilla_ladder_trial(
    r: int,
    seed: int,
    config: TrainingConfig | None = None,
) -> AncillaLadderResult:
    """Train and evaluate one rung r of the ancilla ladder on Curie-Weiss."""
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

    n_visible = VISIBLE_QUBITS
    n_total = n_visible + r
    train_betas = np.asarray(TRAINING_BETAS)
    all_betas = np.asarray(ALL_BETAS)
    bitstrings_vis = enumerate_bitstrings(n_visible)

    # 1. Target Data
    np_rng = np.random.default_rng(seed + 300)
    train_targets_list, val_targets_list = [], []
    for b in train_betas:
        p_gibbs = exact_curie_weiss_target(n_visible, float(b)).probabilities
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

    train_targets = np.stack(train_targets_list)
    val_targets = np.stack(val_targets_list)

    # 2. Ansatz & Angles
    ansatz = complete_ansatz(n_total)
    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    train_angles_vis = np.repeat(
        beta_to_angles(train_betas)[:, None], n_visible, axis=1
    )

    if r > 0:
        input_angles = jnp.asarray(
            ancilla_condition_angles(train_angles_vis, r, "broadcast_all")
        )
    else:
        input_angles = jnp.asarray(train_angles_vis)

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

    # 4. Exact Visible Reduced Density Matrix and Fidelity Evaluation
    per_beta_entropy = {}
    per_beta_eff_rank = {}
    per_beta_prof_tvd = {}
    per_beta_raw_tvd = {}

    for b in all_betas:
        target_raw = exact_curie_weiss_target(n_visible, float(b)).probabilities
        target_shells = shell_probabilities(target_raw, bitstrings_vis)

        angle = float(beta_to_angles(b))
        mod_raw, ent, eff_r = compute_reduced_density_matrix_properties(
            np.asarray(best_params),
            np.asarray(generators, dtype=np.int8),
            angle,
            n_visible=n_visible,
            n_ancillas=r,
        )
        mod_shells = shell_probabilities(mod_raw, bitstrings_vis)

        per_beta_entropy[float(b)] = ent
        per_beta_eff_rank[float(b)] = eff_r
        per_beta_prof_tvd[float(b)] = total_variation_distance(
            target_shells, mod_shells
        )
        per_beta_raw_tvd[float(b)] = total_variation_distance(
            target_raw, mod_raw
        )

    train_prof_mean = float(
        np.mean([per_beta_prof_tvd[b] for b in TRAINING_BETAS])
    )
    heldout_prof_mean = float(
        np.mean([per_beta_prof_tvd[b] for b in HELD_OUT_BETAS])
    )
    train_raw_mean = float(
        np.mean([per_beta_raw_tvd[b] for b in TRAINING_BETAS])
    )
    heldout_raw_mean = float(
        np.mean([per_beta_raw_tvd[b] for b in HELD_OUT_BETAS])
    )

    return AncillaLadderResult(
        r=r,
        n_total_qubits=n_total,
        n_parameters=ansatz.n_parameters,
        seed=seed,
        train_profile_tvd=train_prof_mean,
        heldout_profile_tvd=heldout_prof_mean,
        train_raw_tvd=train_raw_mean,
        heldout_raw_tvd=heldout_raw_mean,
        cold_profile_tvd=per_beta_prof_tvd[1.5],
        cold_raw_tvd=per_beta_raw_tvd[1.5],
        critical_profile_tvd=per_beta_prof_tvd[1.0],
        mean_entropy=float(np.mean(list(per_beta_entropy.values()))),
        mean_effective_rank=float(np.mean(list(per_beta_eff_rank.values()))),
        per_beta_entropy=per_beta_entropy,
        per_beta_effective_rank=per_beta_eff_rank,
    )

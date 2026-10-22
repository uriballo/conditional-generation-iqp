"""Diagonal-conditioning baseline for the two-coordinate Gaussian benchmark.

The input is the fixed state |0>^16 for every condition, so the condition
enters only through the diagonal phases,

    theta(u, v) = w_0 + u w_u + v w_v,      w_0, w_u, w_v in R^40,

on the same open 4x4 lattice used by the conditional IQP model. The weights
are stored as one flattened (3 * 40,) vector with rows w_0, w_u, w_v. The
quadratic basis appends rows w_uu, w_vv for u^2 and v^2, a (5 * 40,) vector;
the basis is inferred from the weight length. Training
mirrors :func:`paper_experiments.training.make_trainer` step for step (batch,
band, condition, mask and reference streams), but differentiates through the
condition-dependent phase map.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from math import comb

import iqp_cond as ciqp
import jax
import jax.numpy as jnp
import numpy as np
import optax
from iqp_cond.operators import OperatorBank, sample_nonidentity_masks_jax
from iqp_cond.training import TrainingResult

from .ansatz import square_lattice_ansatz
from .gaussian2d_lattice import _conditions, basis, hadamard, target_profiles

N_QUBITS = 16
N_PHASES = 40
N_TERMS = 3
N_WEIGHTS = N_TERMS * N_PHASES
BASES = {"affine": ("1", "u", "v"), "quadratic": ("1", "u", "v", "uu", "vv")}
# Diagonal layers: the open 4x4 lattice (default) or all one- and two-body terms.
ANSATZ_PHASES = {"lattice": N_PHASES, "complete": N_QUBITS * (N_QUBITS + 1) // 2}
# Every (ansatz, basis) pair has a distinct flattened weight length.
LAYOUTS = {
    len(terms) * phases: (ansatz_name, len(terms), phases)
    for ansatz_name, phases in ANSATZ_PHASES.items()
    for terms in BASES.values()
}
EVALUATION_CHUNK = 16


def n_terms(basis_name):
    return len(BASES[basis_name])


def n_phases(ansatz_name="lattice"):
    return ANSATZ_PHASES[ansatz_name]


def ansatz_of(ansatz_name="lattice"):
    if ansatz_name == "lattice":
        return square_lattice_ansatz(N_QUBITS)
    if ansatz_name == "complete":
        return ciqp.Ansatz.complete(N_QUBITS)
    raise ValueError(f"unknown ansatz {ansatz_name!r}")


def _rows(weights, module):
    size = int(np.prod(np.shape(weights)))
    if size not in LAYOUTS:
        raise ValueError(
            "weights must hold 3 (affine) or 5 (quadratic) rows of 40 lattice "
            "or 136 complete phases"
        )
    return module.reshape(module.asarray(weights), (LAYOUTS[size][1], -1))


def theta_of(weights, u, v):
    """Phases at (u, v) from flattened weights; JAX-differentiable.

    Affine rows w_0, w_u, w_v, optionally followed by w_uu, w_vv.
    """
    w = _rows(weights, jnp)
    theta = w[0] + u * w[1] + v * w[2]
    if w.shape[0] == 5:
        theta = theta + u**2 * w[3] + v**2 * w[4]
    return theta


def thetas(weights, conditions):
    """Numpy phases (C, 40) for an array of (u, v) conditions."""
    w = _rows(np.asarray(weights, dtype=np.float64), np)
    pairs = np.asarray(conditions, dtype=np.float64).reshape(-1, 2)
    u, v = pairs[:, :1], pairs[:, 1:]
    theta = w[0][None] + u * w[1][None] + v * w[2][None]
    if w.shape[0] == 5:
        theta = theta + u**2 * w[3][None] + v**2 * w[4][None]
    return theta


def initial_weights(seed, candidate, basis_name="affine", ansatz_name="lattice"):
    """w_0 from the lattice initialization stream; all other rows zero.

    The one-body generators come first in both ansatzes, so local_mean always
    lands on the 16 one-body entries.
    """
    phases = n_phases(ansatz_name)
    w0 = np.random.default_rng(seed).normal(0, candidate["init_scale"], phases)
    w0[:N_QUBITS] += candidate["local_mean"]
    return np.concatenate([w0, np.zeros((n_terms(basis_name) - 1) * phases)])


def zero_encoder(n_conditions):
    """The fixed |0>^16 input as a condition batch of zero RY angles."""
    return ciqp.AngleEncoder(
        np.zeros((n_conditions, 1, N_QUBITS)), axes=("Y" * N_QUBITS,)
    )


@lru_cache
def characters(ansatz_name="lattice"):
    generators = np.asarray(ansatz_of(ansatz_name).generators)
    return 1 - 2 * ((basis(N_QUBITS)[0] @ generators.T) % 2)


def _ansatz_for_phases(width):
    for ansatz_name, phases in ANSATZ_PHASES.items():
        if phases == width:
            return ansatz_name
    raise ValueError("phase vectors must have 40 (lattice) or 136 (complete) entries")


def probabilities_from_thetas(phases):
    """Exact (C, 2^16) little-endian output law of H D(theta_c) H |0>."""
    phases = np.atleast_2d(np.asarray(phases, dtype=np.float64))
    reference = 2.0 ** (-N_QUBITS / 2)
    chars = characters(_ansatz_for_phases(phases.shape[1]))
    state = reference * np.exp(1j * (phases @ chars.T))
    return abs(hadamard(state)) ** 2


def probabilities(weights, conditions):
    return probabilities_from_thetas(thetas(weights, _conditions(conditions)))


def metrics(weights, conditions, *, profiles=False):
    """Exact metrics in the format of :func:`gaussian2d_lattice.metrics`."""
    pairs = _conditions(conditions)
    weights_k = basis(N_QUBITS)[1]
    counts = np.asarray([comb(N_QUBITS, k) for k in range(N_QUBITS + 1)])
    k = np.arange(N_QUBITS + 1)
    keys = (
        "full_tvd",
        "profile_tvd",
        "within_shell_residual",
        "generated_mean",
        "generated_sd",
        "target_mean",
        "target_sd",
    )
    fields = {key: [] for key in keys}
    saved, worst = [], 0.0
    for start in range(0, len(pairs), EVALUATION_CHUNK):
        chunk = pairs[start : start + EVALUATION_CHUNK]
        q = probabilities(weights, chunk)
        worst = max(worst, float(abs(q.sum(1) - 1).max()))
        target = target_profiles(chunk)
        p = target[:, weights_k] / counts[weights_k]
        rho = np.stack(
            [np.bincount(weights_k, weights=row, minlength=N_QUBITS + 1) for row in q]
        )
        mean = rho @ k
        sd = np.sqrt(np.maximum(rho @ k**2 - mean**2, 0))
        tmean = target @ k
        tsd = np.sqrt(np.maximum(target @ k**2 - tmean**2, 0))
        values = (
            abs(q - p).sum(1) / 2,
            abs(rho - target).sum(1) / 2,
            abs(q - rho[:, weights_k] / counts[weights_k]).sum(1) / 2,
            mean,
            sd,
            tmean,
            tsd,
        )
        for key, value in zip(keys, values, strict=True):
            fields[key].extend(value.tolist())
        if profiles:
            saved.extend(rho.tolist())
    fields["normalization_max_error"] = worst
    if profiles:
        fields["profiles"] = saved
    return fields


def make_trainer(optimizer_settings):
    """Sampled trainer over the (120,) weights with the manuscript schedules.

    Mirrors :func:`paper_experiments.training.make_trainer` with explicit
    ``rng_seeds``, step mask refresh and validation MMD disabled; checkpoint
    selection always uses ``selection_metric``.
    """
    settings = dict(optimizer_settings)
    if settings.get("name") != "adam":
        raise ValueError("unsupported optimizer name")
    learning_rate = float(settings["learning_rate"])
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    b1 = float(settings.get("b1", 0.9))
    b2 = float(settings.get("b2", 0.999))
    eps = float(settings.get("eps", 1e-8))
    ratio = float(settings.get("final_learning_rate_ratio", 0.1))
    if not (0 < b1 < 1 and 0 < b2 < 1 and eps > 0 and 0 < ratio <= 1):
        raise ValueError("invalid Adam settings")
    rng_seeds = settings["rng_seeds"]
    if set(rng_seeds) != {"batches", "bands", "conditions", "masks", "reference"}:
        raise ValueError("rng_seeds must have exactly the required keys")

    def train(
        ansatz,
        encoder,
        fitting_conditions,
        training_targets,
        *,
        seed,
        config,
        initial_params,
        selection_metric,
        checkpoint_callback=None,
    ):
        if config.mask_refresh != "step":
            raise ValueError("diagonal trainer requires step mask refresh")
        pairs = jnp.asarray(_conditions(fitting_conditions))
        training_targets = np.asarray(training_targets, dtype=np.float64)
        initial = np.asarray(initial_params, dtype=np.float64)
        n_conditions = encoder.n_conditions
        if initial.ndim != 1 or initial.size not in {
            len(terms) * ansatz.n_parameters for terms in BASES.values()
        }:
            raise ValueError("invalid initial weights")
        if not np.isfinite(initial).all():
            raise ValueError("invalid initial weights")
        if training_targets.ndim != 3 or training_targets.shape[0] != n_conditions:
            raise ValueError("training condition count differs")
        if len(pairs) != n_conditions or training_targets.shape[2] != N_QUBITS:
            raise ValueError("invalid target dimensions")
        if not np.isin(training_targets, [0, 1]).all():
            raise ValueError("targets must be binary strings")
        sigmas = tuple(config.resolved_sigmas(N_QUBITS))
        batches = np.random.default_rng(int(rng_seeds["batches"])).integers(
            0,
            training_targets.shape[1],
            size=(config.steps, n_conditions, config.batch_size),
        )
        bands = np.random.default_rng(int(rng_seeds["bands"])).integers(
            0, len(sigmas), size=config.steps
        )
        conditions = np.random.default_rng(int(rng_seeds["conditions"])).integers(
            0, n_conditions, size=config.steps
        )
        schedule = optax.cosine_decay_schedule(learning_rate, config.steps, alpha=ratio)
        optimizer = optax.adam(schedule, b1=b1, b2=b2, eps=eps)
        params = jnp.asarray(initial)
        state = optimizer.init(params)
        mask_key = jax.random.PRNGKey(int(rng_seeds["masks"]))
        reference_key = jax.random.PRNGKey(int(rng_seeds["reference"]))

        def compile_step(sigma):
            def loss(weights, uv, step_key, target_batch, selected, bank):
                theta = theta_of(weights, uv[0], uv[1])
                return (
                    ciqp.ConditionalModel(ansatz, selected)
                    .mmd(
                        theta,
                        target_batch,
                        step_key,
                        bank=bank,
                        n_samples=config.mc_samples,
                        shared_samples=True,
                        gemm_dtype=config.gemm_dtype,
                        estimator_dtype=config.estimator_dtype,
                        sample_chunk=config.sample_chunk,
                    )
                    .loss
                )

            @jax.jit
            def step(weights, opt_state, uv, step_key, batch, selected, mask_step_key):
                masks, mask_weights = sample_nonidentity_masks_jax(
                    mask_step_key, config.n_ops, N_QUBITS, sigma
                )
                bank = OperatorBank(
                    sigmas=(sigma,),
                    masks=masks[None, None],
                    weights=mask_weights[None, None],
                    n_ops=config.n_ops,
                    n_visible=N_QUBITS,
                    n_masks=config.n_ops,
                )
                value, grad = jax.value_and_grad(loss)(
                    weights, uv, step_key, batch, selected, bank
                )
                updates, opt_state = optimizer.update(grad, opt_state, weights)
                return value, optax.apply_updates(weights, updates), opt_state

            return step

        steps = tuple(compile_step(sigma) for sigma in sigmas)
        train_losses, checkpoint_steps, scores = [], [], []
        best = dict(params=np.asarray(params).copy(), step=0, score=float("inf"))
        started = time.perf_counter()

        def checkpoint(step, weights):
            weights = np.asarray(weights).copy()
            score = float(selection_metric(weights))
            if not np.isfinite(score) or not np.isfinite(weights).all():
                raise FloatingPointError("nonfinite training checkpoint")
            checkpoint_steps.append(step)
            scores.append(score)
            if score < best["score"]:
                best.update(params=weights, step=step, score=score)
            if checkpoint_callback is not None:
                checkpoint_callback(step, weights)
            if step % 500 == 0:
                print(
                    f"diagonal adam step {step}: "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

        checkpoint(0, params)
        for index in range(config.steps):
            condition = int(conditions[index])
            batch = jnp.asarray(
                training_targets[condition : condition + 1, batches[index, condition]]
            )
            value, params, state = steps[int(bands[index])](
                params,
                state,
                pairs[condition],
                jax.random.fold_in(reference_key, index),
                batch,
                encoder.select(condition),
                jax.random.fold_in(mask_key, index),
            )
            train_losses.append(value)
            if (
                index + 1
            ) % config.validation_interval == 0 or index + 1 == config.steps:
                checkpoint(index + 1, params)
        final = np.asarray(params)
        losses = np.asarray(jnp.stack(train_losses))
        if not np.isfinite(final).all() or not np.isfinite(losses).all():
            raise FloatingPointError("nonfinite final optimization")
        return TrainingResult(
            seed=seed,
            best_params=best["params"],
            final_params=final,
            best_step=best["step"],
            best_validation_loss=float("nan"),
            best_selection_score=best["score"],
            train_steps=np.arange(1, config.steps + 1),
            train_losses=losses,
            validation_steps=np.asarray(checkpoint_steps),
            validation_losses=np.full(len(checkpoint_steps), np.nan),
            selection_scores=np.asarray(scores),
        )

    return train

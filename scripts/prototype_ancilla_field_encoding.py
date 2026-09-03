#!/usr/bin/env python3
"""PROTOTYPE: phase-crossing with ancilla-expanded input-state encodings.

The strict latent-input arms forbid selector-weight phase blocks.  The visible
register has one complete two-local IQP phase vector, and the diagonal acts on
the visible register only.  Condition-bearing ancillas instead control RY
rotations in the input preparation.  Conditional on an ancilla string z, the
visible product-state angles shift as

    alpha_i(t, z) = alpha_affine(t) + sum_k L[i, k] (-1)**z_k.

Every branch therefore passes through exactly the same learned IQP phase
vector.  The 2**r product inputs are generated additively by r columns of
controlled-RY couplings rather than by independently parameterized IQP
experts.  In the linear arm, every ancilla has computational-basis Bernoulli
probability t, where t is the normalized declared condition.  The Fourier arm
is a generic nonlinear control.

The file is exploratory and intentionally separate from production runners.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many_jax,
    generator_characters,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.latent_input import exact_latent_input_probabilities
from iqp_cond.matched_capacity import matched_visible_ansatz

jax.config.update("jax_enable_x64", True)

N_VISIBLE = 8
SIGMAS = (0.5, 1.0, 2.0, 4.0)
PROTOCOLS = {
    "critical_gap": {
        "training": (0.5, 0.7, 0.8, 1.2, 1.3, 1.5),
        "test": (0.9, 1.0, 1.1),
    },
    "hot_to_cold": {
        "training": (0.5, 0.6, 0.7, 0.8, 0.9),
        "test": (1.0, 1.1, 1.2, 1.3, 1.4, 1.5),
    },
}
ENCODINGS = (
    "affine_broadcast",
    "ancilla_field",
    "ancilla_field_fourier",
    "broadcast_ancilla_field",
    "latent_input_fourier",
    "latent_input_linear",
    "latent_input_overlap",
    "visible_capacity",
    "visible_plus_ancilla_field",
)


def _parse_integers(specification: str) -> tuple[int, ...]:
    if "," in specification:
        return tuple(int(value) for value in specification.split(","))
    if "-" in specification:
        start, stop = specification.split("-", 1)
        return tuple(range(int(start), int(stop) + 1))
    return (int(specification),)


def _coordinate(beta: np.ndarray | jnp.ndarray) -> np.ndarray | jnp.ndarray:
    return beta - 0.5


def _affine_angles(betas: jnp.ndarray) -> jnp.ndarray:
    alpha = jnp.pi * (0.35 - 0.70 * _coordinate(betas))
    return jnp.repeat(alpha[:, None], N_VISIBLE, axis=1)


def _latent_strings(ancillas: int) -> jnp.ndarray:
    return jnp.asarray(enumerate_bitstrings(ancillas), dtype=jnp.float64)


def _ancilla_probabilities(
    t: jnp.ndarray, ancillas: int, encoding: str
) -> jnp.ndarray:
    if encoding in {"ancilla_field_fourier", "latent_input_fourier"}:
        frequencies = jnp.arange(1, ancillas + 1, dtype=jnp.float64)
        return jnp.sin(jnp.pi * t[:, None] * frequencies[None, :] / 2.0) ** 2
    if encoding == "latent_input_overlap":
        probability = 0.25 + 0.50 * t
        return jnp.repeat(probability[:, None], ancillas, axis=1)
    if encoding == "broadcast_ancilla_field":
        angle = jnp.pi * (0.35 - 0.70 * t)
        probability = (1.0 - jnp.sin(angle)) / 2.0
        return jnp.repeat(probability[:, None], ancillas, axis=1)
    return jnp.repeat(t[:, None], ancillas, axis=1)


def _product_weights(
    probabilities: jnp.ndarray, latent_bits: jnp.ndarray
) -> jnp.ndarray:
    probabilities = probabilities[:, None, :]
    factors = jnp.where(
        latent_bits[None, :, :] > 0.5, probabilities, 1 - probabilities
    )
    return jnp.prod(factors, axis=-1)


def _target_matrix(betas: tuple[float, ...]) -> jnp.ndarray:
    return jnp.asarray(
        np.stack(
            [exact_curie_weiss_target(N_VISIBLE, beta).probabilities for beta in betas]
        ),
        dtype=jnp.float64,
    )


def _mmd_constants() -> tuple[jnp.ndarray, jnp.ndarray]:
    masks = enumerate_bitstrings(N_VISIBLE)
    characters = 1.0 - 2.0 * ((masks @ masks.T) % 2)
    weights = masks.sum(axis=1)
    laws = []
    for sigma in SIGMAS:
        probability = (1.0 - math.exp(-1.0 / (2.0 * sigma**2))) / 2.0
        laws.append(probability**weights * (1.0 - probability) ** (N_VISIBLE - weights))
    return jnp.asarray(characters), jnp.asarray(np.stack(laws))


def _profile(probabilities: np.ndarray) -> np.ndarray:
    return shell_probabilities(probabilities, enumerate_bitstrings(N_VISIBLE))


def _model_builder(encoding: str, ancillas: int):
    ansatz = (
        matched_visible_ansatz(N_VISIBLE, 36 + N_VISIBLE * ancillas)
        if encoding == "visible_capacity"
        else complete_ansatz(N_VISIBLE)
    )
    generators = jnp.asarray(ansatz.generators, dtype=jnp.int8)
    phase_characters = jnp.asarray(generator_characters(np.asarray(ansatz.generators)))
    if encoding in {"affine_broadcast", "visible_capacity"}:
        shape = (ansatz.n_parameters,)

        def probabilities(params: jnp.ndarray, betas: jnp.ndarray) -> jnp.ndarray:
            return exact_iqp_probabilities_many_jax(
                params,
                generators,
                _affine_angles(betas),
                characters=phase_characters,
            )

    elif encoding in {"latent_input_linear", "latent_input_overlap"}:
        if ancillas < 1:
            raise ValueError(f"{encoding} requires at least one ancilla")
        shape = (ansatz.n_parameters + N_VISIBLE * ancillas,)

        def probabilities(params: jnp.ndarray, betas: jnp.ndarray) -> jnp.ndarray:
            return exact_latent_input_probabilities(
                params[: ansatz.n_parameters],
                generators,
                _affine_angles(betas),
                _coordinate(betas),
                params[ansatz.n_parameters :].reshape(N_VISIBLE, ancillas),
                margin=0.25 if encoding == "latent_input_overlap" else 0.0,
                characters=phase_characters,
            )

    elif encoding == "latent_input_fourier":
        if ancillas < 1:
            raise ValueError(f"{encoding} requires at least one ancilla")
        shape = (ansatz.n_parameters + N_VISIBLE * ancillas,)
        latent_bits = _latent_strings(ancillas)
        latent_signs = 1.0 - 2.0 * latent_bits

        def probabilities(params: jnp.ndarray, betas: jnp.ndarray) -> jnp.ndarray:
            phases = params[: ansatz.n_parameters]
            input_couplings = params[ansatz.n_parameters :].reshape(
                N_VISIBLE, ancillas
            )
            angle_shifts = latent_signs @ input_couplings.T
            branch_angles = (
                _affine_angles(betas)[None, :, :] + angle_shifts[:, None, :]
            )

            def one_branch(angles: jnp.ndarray) -> jnp.ndarray:
                return exact_iqp_probabilities_many_jax(
                    phases,
                    generators,
                    angles,
                    characters=phase_characters,
                )

            branches = jax.vmap(one_branch)(branch_angles)
            probabilities = _ancilla_probabilities(
                _coordinate(betas), ancillas, encoding
            )
            weights = _product_weights(probabilities, latent_bits)
            return jnp.einsum("cb,bcp->cp", weights, branches)

    elif encoding in {
        "ancilla_field",
        "ancilla_field_fourier",
        "broadcast_ancilla_field",
        "visible_plus_ancilla_field",
    }:
        if ancillas < 1:
            raise ValueError(f"{encoding} requires at least one ancilla")
        shape = (ansatz.n_parameters + N_VISIBLE * ancillas,)
        latent_bits = _latent_strings(ancillas)
        latent_signs = 1.0 - 2.0 * latent_bits

        def probabilities(params: jnp.ndarray, betas: jnp.ndarray) -> jnp.ndarray:
            base = params[: ansatz.n_parameters]
            couplings = params[ansatz.n_parameters :].reshape(N_VISIBLE, ancillas)
            local_shifts = latent_signs @ couplings.T
            branch_phases = jnp.repeat(base[None, :], 2**ancillas, axis=0)
            branch_phases = branch_phases.at[:, :N_VISIBLE].add(local_shifts)
            visible_angles = (
                _affine_angles(betas)
                if encoding
                in {"broadcast_ancilla_field", "visible_plus_ancilla_field"}
                else jnp.zeros((len(betas), N_VISIBLE), dtype=jnp.float64)
            )

            def one_branch(phases: jnp.ndarray) -> jnp.ndarray:
                return exact_iqp_probabilities_many_jax(
                    phases,
                    generators,
                    visible_angles,
                    characters=phase_characters,
                )

            branches = jax.vmap(one_branch)(branch_phases)
            probabilities = _ancilla_probabilities(
                _coordinate(betas), ancillas, encoding
            )
            weights = _product_weights(probabilities, latent_bits)
            return jnp.einsum("cb,bcp->cp", weights, branches)

    else:
        raise ValueError(f"unknown encoding: {encoding}")
    return shape, probabilities


def _fit(
    encoding: str,
    ancillas: int,
    betas: tuple[float, ...],
    targets: jnp.ndarray,
    seed: int,
    updates: int,
    initialization: str,
    phase_scale: float,
    initial_params: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    shape, model_probabilities = _model_builder(encoding, ancillas)
    condition_values = jnp.asarray(betas, dtype=jnp.float64)
    observable_characters, mask_laws = _mmd_constants()
    target_moments = targets @ observable_characters.T

    def loss(params: jnp.ndarray) -> jnp.ndarray:
        model = model_probabilities(params, condition_values)
        model_moments = model @ observable_characters.T
        squared = (model_moments - target_moments) ** 2
        return jnp.mean(squared @ mask_laws.T)

    key = jax.random.PRNGKey(seed)
    if initial_params is not None:
        params = jnp.asarray(initial_params, dtype=jnp.float64)
        if params.shape != shape:
            raise ValueError(f"initial_params must have shape {shape}")
    elif initialization == "iid":
        params = jax.random.normal(key, shape) * phase_scale
    elif initialization == "permutation_neutral":
        values = jax.random.normal(key, (ancillas + 2,)) * phase_scale
        one_body = jnp.repeat(values[0], N_VISIBLE)
        two_body = jnp.repeat(values[1], 28)
        if encoding == "affine_broadcast":
            params = jnp.concatenate((one_body, two_body))
        else:
            couplings = jnp.tile(values[2:][None, :], (N_VISIBLE, 1)).ravel()
            params = jnp.concatenate((one_body, two_body, couplings))
    else:
        raise ValueError(f"unknown initialization: {initialization}")
    schedule = optax.cosine_decay_schedule(0.01, updates, alpha=0.1)
    optimizer = optax.adam(schedule)
    state = optimizer.init(params)
    value_and_gradient = jax.jit(jax.value_and_grad(loss))
    evaluate_loss = jax.jit(loss)
    best_params = params
    best_loss = float(evaluate_loss(params))
    best_step = 0
    for step in range(updates):
        _, gradient = value_and_gradient(params)
        updates_value, state = optimizer.update(gradient, state, params)
        params = optax.apply_updates(params, updates_value)
        completed = step + 1
        if completed % 25 == 0 or completed == updates:
            current = float(evaluate_loss(params))
            if current < best_loss:
                best_params, best_loss, best_step = params, current, completed
    return np.asarray(best_params), best_loss, best_step


def _evaluate(
    encoding: str,
    ancillas: int,
    params: np.ndarray,
    training_betas: tuple[float, ...],
    test_betas: tuple[float, ...],
    seed: int,
) -> list[dict[str, object]]:
    """Access exact test targets only after the training checkpoint freezes."""
    _, model_probabilities = _model_builder(encoding, ancillas)
    betas = (*training_betas, *test_betas)
    model = np.asarray(
        model_probabilities(jnp.asarray(params), jnp.asarray(betas, dtype=jnp.float64))
    )
    rows = []
    for index, beta in enumerate(betas):
        target = exact_curie_weiss_target(N_VISIBLE, beta)
        visible_magnetization = np.abs(target.magnetization)
        probability = model[index]
        rows.append(
            {
                "encoding": encoding,
                "ancillas": ancillas,
                "seed": seed,
                "beta": beta,
                "role": "training" if beta in training_betas else "test",
                "profile_tvd": total_variation_distance(
                    target.profile, _profile(probability)
                ),
                "raw_tvd": total_variation_distance(target.probabilities, probability),
                "absolute_magnetization_error": abs(
                    float(visible_magnetization @ target.probabilities)
                    - float(visible_magnetization @ probability)
                ),
            }
        )
    return rows


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    models = sorted({(str(row["encoding"]), int(row["ancillas"])) for row in rows})
    for encoding, ancillas in models:
        key = f"{encoding}_r{ancillas}"
        summary[key] = {}
        for role in ("training", "test"):
            selected = [
                row
                for row in rows
                if row["encoding"] == encoding
                and row["ancillas"] == ancillas
                and row["role"] == role
            ]
            summary[key][role] = {
                metric: float(np.mean([float(row[metric]) for row in selected]))
                for metric in (
                    "profile_tvd",
                    "raw_tvd",
                    "absolute_magnetization_error",
                )
            }
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol = PROTOCOLS[args.protocol]
    training_betas = protocol["training"]
    test_betas = protocol["test"]
    training_targets = _target_matrix(training_betas)
    rows: list[dict[str, object]] = []
    fits = []
    for encoding in args.encoding:
        rungs = (
            (0,)
            if encoding == "affine_broadcast"
            else _parse_integers(args.ancillas)
        )
        for ancillas in rungs:
            for seed in _parse_integers(args.seeds):
                print(
                    f"fit protocol={args.protocol} encoding={encoding} "
                    f"r={ancillas} seed={seed}",
                    flush=True,
                )
                warm_start = None
                warm_loss = None
                if args.initialization == "affine_warm_start":
                    if encoding not in {
                        "broadcast_ancilla_field",
                        "visible_plus_ancilla_field",
                    }:
                        raise ValueError(
                            "affine_warm_start requires an affine-visible encoding"
                        )
                    base, warm_loss, _ = _fit(
                        "affine_broadcast",
                        0,
                        training_betas,
                        training_targets,
                        seed,
                        args.updates,
                        "iid",
                        args.phase_scale,
                    )
                    warm_start = np.concatenate(
                        (base, np.zeros(N_VISIBLE * ancillas, dtype=np.float64))
                    )
                params, best_loss, best_step = _fit(
                    encoding,
                    ancillas,
                    training_betas,
                    training_targets,
                    seed,
                    args.updates,
                    "iid" if warm_start is not None else args.initialization,
                    args.phase_scale,
                    warm_start,
                )
                fits.append(
                    {
                        "encoding": encoding,
                        "ancillas": ancillas,
                        "seed": seed,
                        "best_training_mmd_squared": best_loss,
                        "best_step": best_step,
                        "parameter_count": int(params.size),
                        "warm_start_training_mmd_squared": warm_loss,
                    }
                )
                rows.extend(
                    _evaluate(
                        encoding,
                        ancillas,
                        params,
                        training_betas,
                        test_betas,
                        seed,
                    )
                )
    return {
        "status": "prototype",
        "question": (
            "Can an ancilla-expanded product-input encoding cross the "
            "Curie--Weiss gap while every latent branch uses one shared "
            "visible-only IQP phase vector?"
        ),
        "protocol": args.protocol,
        "training_betas": training_betas,
        "test_betas": test_betas,
        "visible_qubits": N_VISIBLE,
        "ancilla_rungs": _parse_integers(args.ancillas),
        "initialization": args.initialization,
        "phase_scale": args.phase_scale,
        "encoding": {
            "coordinate": "t=(beta-0.5)/(1.5-0.5)",
            "visible_input": {
                "ancilla_field": "fixed zero angle, hence uniform post-H amplitudes",
                "ancilla_field_fourier": (
                    "fixed zero angle with linearly spaced ancilla frequencies"
                ),
                "broadcast_ancilla_field": (
                    "the same target-independent affine angle on visible and ancillas"
                ),
                "latent_input_linear": (
                    "affine visible angles plus learned low-rank latent angle shifts"
                ),
                "latent_input_overlap": (
                    "the same latent input map with p(t)=1/4+t/2, keeping all "
                    "branches supported across the declared condition domain"
                ),
                "latent_input_fourier": (
                    "the same latent input map with fixed integer-frequency weights"
                ),
                "visible_plus_ancilla_field": "target-independent affine broadcast",
            },
            "ancilla_input": (
                "independent computational-basis controls with repeated "
                "Bernoulli(t) or fixed integer-frequency sinusoidal probabilities"
            ),
        },
        "no_multiplexing_audit": {
            "independent_visible_phase_blocks": False,
            "latent_input_models_use_one_visible_phase_vector": True,
            "latent_input_diagonal_acts_on_ancillas": False,
            "latent_input_modulation": (
                "alpha_i(t,z)=alpha_affine(t)+sum_k L_ik*(-1)^z_k"
            ),
            "latent_input_learned_diagonal_rotations": 36,
            "latent_input_learned_preparation_couplings": "8r",
            "visible_two_body_phases_shared_across_all_latent_strings": True,
            "latent_input_preparation": (
                "ancilla RY amplitudes followed by commuting ancilla-controlled "
                "visible RY rotations"
            ),
            "learned_parameters": "36 visible IQP phases + 8r input couplings",
            "visible_iqp_compiled_pauli_locality": 2,
            "component_count": "2^r structured branches induced by tracing ancillas",
        },
        "leakage_audit": {
            "encoding_uses_target_samples": False,
            "encoding_uses_curie_weiss_energy": False,
            "encoding_uses_spin_flip_or_permutation_symmetry": False,
            "encoding_uses_known_critical_point": False,
            "test_targets_accessed_before_checkpoint_freeze": False,
            "training_uses_exact_targets": True,
        },
        "fits": fits,
        "rows": rows,
        "summary": _summarize(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument(
        "--encoding", nargs="+", choices=ENCODINGS, default=list(ENCODINGS)
    )
    parser.add_argument("--ancillas", default="1,2,3,5")
    parser.add_argument("--seeds", default="0-2")
    parser.add_argument("--updates", type=int, default=3_000)
    parser.add_argument(
        "--initialization",
        choices=("iid", "permutation_neutral", "affine_warm_start"),
        default="iid",
    )
    parser.add_argument("--phase-scale", type=float, default=0.1 / math.sqrt(36.0))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.updates < 1
        or args.phase_scale <= 0
        or any(rung < 1 for rung in _parse_integers(args.ancillas))
    ):
        parser.error("updates and ancilla rungs must be positive")
    payload = run(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

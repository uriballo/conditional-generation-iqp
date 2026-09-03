#!/usr/bin/env python3
"""PROTOTYPE: target-independent encodings for Curie--Weiss phase OOD.

This exact-enumeration screen asks whether a generic Bernstein selector
register can replace the target-shaped Curie--Weiss encoder.  It is not a
production campaign and writes nothing unless ``--output`` is supplied.

The Bernstein arm is the visible marginal of one fixed IQP circuit with
selector ancillas.  After the first Hadamards, each selector is prepared as
``sqrt(1-t)|0> + sqrt(t)|1>``.  A selector-permutation-tied diagonal block
assigns one visible IQP expert to each selector Hamming weight.  Tracing the
selectors gives the target-independent mixture

    q(x|t) = sum_j C(r,j) t**j (1-t)**(r-j) q_j(x).

No Curie--Weiss energy, symmetry, critical point, or evaluation outcome enters
this encoding.  Exact targets at test temperatures are first constructed only
after the best training checkpoint has been frozen.
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

jax.config.update("jax_enable_x64", True)

N_VISIBLE = 8
SIGMAS = (0.5, 1.0, 2.0, 4.0)
PROTOCOLS = {
    "critical_gap": {
        "training": (0.5, 0.7, 0.8, 1.2, 1.3, 1.5),
        "test": (0.9, 1.0, 1.1),
        "identifiability": (
            "Both regimes occur in training; the withheld interval tests whether "
            "the learned response crosses the transition without a critical knot."
        ),
    },
    "hot_to_cold": {
        "training": (0.5, 0.6, 0.7, 0.8, 0.9),
        "test": (1.0, 1.1, 1.2, 1.3, 1.4, 1.5),
        "identifiability": (
            "Training is one-sided. No distribution-free method can identify an "
            "arbitrary unseen phase; this is a stress test of the model prior."
        ),
    },
}
ENCODINGS = (
    "affine_broadcast",
    "bernstein_linear",
    "bernstein_quadratic",
    "bernstein_selector",
    "hybrid_bernstein",
)


def _parse_seeds(specification: str) -> tuple[int, ...]:
    if "," in specification:
        return tuple(int(value) for value in specification.split(","))
    if "-" in specification:
        start, stop = specification.split("-", 1)
        return tuple(range(int(start), int(stop) + 1))
    return (int(specification),)


def _coordinate(beta: np.ndarray | jnp.ndarray) -> np.ndarray | jnp.ndarray:
    return (beta - 0.5) / (1.5 - 0.5)


def _affine_angles(betas: jnp.ndarray) -> jnp.ndarray:
    alpha = jnp.pi * (0.35 - 0.70 * _coordinate(betas))
    return jnp.repeat(alpha[:, None], N_VISIBLE, axis=1)


def _bernstein_weights(t: jnp.ndarray, selectors: int) -> jnp.ndarray:
    degrees = jnp.arange(selectors + 1, dtype=jnp.float64)
    coefficients = jnp.asarray(
        [math.comb(selectors, degree) for degree in range(selectors + 1)],
        dtype=jnp.float64,
    )
    return (
        coefficients
        * t[:, None] ** degrees
        * (1.0 - t[:, None]) ** (selectors - degrees)
    )


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


def _model_builder(encoding: str, selectors: int):
    ansatz = complete_ansatz(N_VISIBLE)
    generators = jnp.asarray(ansatz.generators, dtype=jnp.int8)
    phase_characters = jnp.asarray(generator_characters(np.asarray(ansatz.generators)))
    zero_angles = jnp.zeros((1, N_VISIBLE), dtype=jnp.float64)

    if encoding == "affine_broadcast":
        shape = (ansatz.n_parameters,)

        def probabilities(params: jnp.ndarray, betas: jnp.ndarray) -> jnp.ndarray:
            return exact_iqp_probabilities_many_jax(
                params,
                generators,
                _affine_angles(betas),
                characters=phase_characters,
            )

    elif encoding.startswith("bernstein_") or encoding == "hybrid_bernstein":
        if encoding == "bernstein_selector":
            shape = (selectors + 1, ansatz.n_parameters)

            def expert_parameters(params: jnp.ndarray) -> jnp.ndarray:
                return params

        elif encoding in {"bernstein_linear", "bernstein_quadratic"}:
            degree = 1 if encoding == "bernstein_linear" else 2
            shape = (degree + 1, ansatz.n_parameters)
            selector_coordinate = jnp.linspace(-1.0, 1.0, selectors + 1)
            basis_columns = [jnp.ones_like(selector_coordinate), selector_coordinate]
            if degree == 2:
                basis_columns.append(2.0 * selector_coordinate**2 - 1.0)
            selector_basis = jnp.stack(basis_columns, axis=1)

            def expert_parameters(params: jnp.ndarray) -> jnp.ndarray:
                return selector_basis @ params

        else:
            shape = (selectors + 1, ansatz.n_parameters)

            def expert_parameters(params: jnp.ndarray) -> jnp.ndarray:
                return params

        def one_expert_zero(params: jnp.ndarray) -> jnp.ndarray:
            return exact_iqp_probabilities_many_jax(
                params,
                generators,
                zero_angles,
                characters=phase_characters,
            )[0]

        def probabilities(params: jnp.ndarray, betas: jnp.ndarray) -> jnp.ndarray:
            expanded = expert_parameters(params)
            weights = _bernstein_weights(_coordinate(betas), selectors)
            if encoding == "hybrid_bernstein":
                angles = _affine_angles(betas)

                def one_expert_affine(expert: jnp.ndarray) -> jnp.ndarray:
                    return exact_iqp_probabilities_many_jax(
                        expert,
                        generators,
                        angles,
                        characters=phase_characters,
                    )

                experts = jax.vmap(one_expert_affine)(expanded)
                return jnp.einsum("ce,ecp->cp", weights, experts)
            experts = jax.vmap(one_expert_zero)(expanded)
            return weights @ experts

    else:
        raise ValueError(f"unknown encoding: {encoding}")
    return shape, probabilities


def _fit(
    encoding: str,
    selectors: int,
    betas: tuple[float, ...],
    targets: jnp.ndarray,
    seed: int,
    updates: int,
) -> tuple[np.ndarray, float, int]:
    shape, model_probabilities = _model_builder(encoding, selectors)
    condition_values = jnp.asarray(betas, dtype=jnp.float64)
    observable_characters, mask_laws = _mmd_constants()
    target_moments = targets @ observable_characters.T

    def loss(params: jnp.ndarray) -> jnp.ndarray:
        model = model_probabilities(params, condition_values)
        model_moments = model @ observable_characters.T
        squared = (model_moments - target_moments) ** 2
        per_bandwidth = squared @ mask_laws.T
        return jnp.mean(per_bandwidth)

    scale = 0.1 / math.sqrt(36.0)
    params = jax.random.normal(jax.random.PRNGKey(seed), shape) * scale
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
    selectors: int,
    params: np.ndarray,
    training_betas: tuple[float, ...],
    test_betas: tuple[float, ...],
    seed: int,
) -> list[dict[str, object]]:
    """The first access to exact test targets occurs inside this function."""
    _, model_probabilities = _model_builder(encoding, selectors)
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
    encodings = sorted({str(row["encoding"]) for row in rows})
    for encoding in encodings:
        summary[encoding] = {}
        for role in ("training", "test"):
            selected = [
                row
                for row in rows
                if row["encoding"] == encoding and row["role"] == role
            ]
            per_seed = {}
            for seed in sorted({int(row["seed"]) for row in selected}):
                seed_rows = [row for row in selected if row["seed"] == seed]
                per_seed[str(seed)] = {
                    metric: float(np.mean([float(row[metric]) for row in seed_rows]))
                    for metric in (
                        "profile_tvd",
                        "raw_tvd",
                        "absolute_magnetization_error",
                    )
                }
            summary[encoding][role] = {
                metric: float(np.mean([float(row[metric]) for row in selected]))
                for metric in (
                    "profile_tvd",
                    "raw_tvd",
                    "absolute_magnetization_error",
                )
            }
            summary[encoding][role]["per_seed"] = per_seed
    if "affine_broadcast" in summary:
        for encoding in encodings:
            if encoding.startswith("bernstein_"):
                summary[f"test_profile_effect_{encoding}_minus_affine"] = (
                    summary[encoding]["test"]["profile_tvd"]
                    - summary["affine_broadcast"]["test"]["profile_tvd"]
                )
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    protocol = PROTOCOLS[args.protocol]
    training_betas = protocol["training"]
    test_betas = protocol["test"]
    training_targets = _target_matrix(training_betas)
    rows: list[dict[str, object]] = []
    fits = []
    for encoding in args.encoding:
        for seed in _parse_seeds(args.seeds):
            params, best_loss, best_step = _fit(
                encoding,
                args.selectors,
                training_betas,
                training_targets,
                seed,
                args.updates,
            )
            fits.append(
                {
                    "encoding": encoding,
                    "seed": seed,
                    "best_training_mmd_squared": best_loss,
                    "best_step": best_step,
                    "parameter_count": int(params.size),
                }
            )
            rows.extend(
                _evaluate(
                    encoding,
                    args.selectors,
                    params,
                    training_betas,
                    test_betas,
                    seed,
                )
            )
    return {
        "status": "prototype",
        "protocol": args.protocol,
        "training_betas": training_betas,
        "test_betas": test_betas,
        "identifiability": protocol["identifiability"],
        "visible_qubits": N_VISIBLE,
        "selector_ancillas": args.selectors,
        "encodings": {
            "affine_broadcast": (
                "alpha_i(t)=pi(0.35-0.70t), fixed before target access"
            ),
            "bernstein_selector": (
                "r parallel selector ancillas with post-H Bernoulli probability t; "
                "selector-weight blocks produce Bernstein mixture weights"
            ),
            "bernstein_linear": (
                "same selector encoding, with each visible phase affine in normalized "
                "selector Hamming weight"
            ),
            "bernstein_quadratic": (
                "same selector encoding, with each visible phase quadratic in "
                "normalized selector Hamming weight"
            ),
            "hybrid_bernstein": (
                "the affine visible map composed with the same target-independent "
                "Bernstein selector register"
            ),
        },
        "ansatz": {
            "affine_broadcast": "one complete two-local 8-visible IQP block",
            "bernstein_selector": (
                "one selector-weight-multiplexed diagonal with r+1 complete "
                "two-local visible blocks; equivalent direct marginal used here"
            ),
            "bernstein_linear": (
                "selector-permutation-tied diagonal with at most one selector in "
                "each visible-generator control term"
            ),
            "bernstein_quadratic": (
                "selector-permutation-tied diagonal with at most two selectors in "
                "each visible-generator control term"
            ),
            "hybrid_bernstein": (
                "selector-weight-multiplexed diagonal blocks driven in parallel "
                "with the affine visible input path"
            ),
        },
        "leakage_audit": {
            "encoding_uses_target_samples": False,
            "encoding_uses_curie_weiss_energy": False,
            "encoding_uses_spin_flip_or_permutation_symmetry": False,
            "encoding_uses_known_critical_point": False,
            "test_targets_accessed_before_checkpoint_freeze": False,
            "training_uses_exact_targets": True,
            "note": (
                "Exact training is a capacity/optimization screen, not finite-sample "
                "campaign evidence."
            ),
        },
        "fits": fits,
        "rows": rows,
        "summary": _summarize(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument(
        "--encoding",
        nargs="+",
        choices=ENCODINGS,
        default=list(ENCODINGS),
    )
    parser.add_argument("--selectors", type=int, default=3)
    parser.add_argument("--seeds", default="0-2")
    parser.add_argument("--updates", type=int, default=1_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.selectors < 1 or args.updates < 1:
        parser.error("selectors and updates must be positive")
    payload = run(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

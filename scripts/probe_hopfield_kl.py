"""Single-condition exact-KL feasibility probe at the feasible beta grid.

Measures whether exact-gradient training can place pattern mass on the
Hopfield target at the study's feasible betas, under two objectives:

- exact KL, which isolates representability (no kernel, so no MMD
  localization barrier);
- exact zero-variance MMD, as a control that should reproduce the
  ~0.004 pattern mass already measured for the trained single-condition
  models.

Unlike ``analyze_hopfield_boundary.py`` (preregistered beta=4 protocol),
this probe matches the study's per-condition setup: the reference state is
built at the condition's own angle ``alpha(beta)`` over all
``n_data + n_ancillas`` qubits, with ancilla angles broadcast from the
data angle (the study's ``broadcast_all`` scheme).

Writes ``results/hopfield/boundary_feasible.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from iqp_cond.ancilla import ancilla_ansatz, ancilla_condition_angles
from iqp_cond.ansatz import complete_ansatz
from iqp_cond.encoding import post_hadamard_amplitudes
from iqp_cond.exact import (
    enumerate_bitstrings,
    fwht,
    fwht_jax,
    generator_characters,
)
from iqp_cond.hopfield import N_QUBITS, HopfieldTarget, angle_for_beta
from iqp_cond.mmd import mmd_operator_probability


def _reference_amplitudes(alpha: float, n_qubits: int) -> jnp.ndarray:
    zero, one = post_hadamard_amplitudes(jnp.asarray([alpha, alpha]))
    a0 = float(zero[0].real)
    a1 = float(one[0].real)
    weights = enumerate_bitstrings(n_qubits).sum(axis=1).astype(np.float64)
    return jnp.asarray(
        a0 ** (n_qubits - weights) * a1**weights,
        dtype=jnp.complex128,
    )


def _marginal_probs(
    params: jnp.ndarray,
    characters: jnp.ndarray,
    reference: jnp.ndarray,
    n_ancillas: int,
) -> jnp.ndarray:
    n_total = reference.shape[0].bit_length() - 1
    state = fwht_jax(reference * jnp.exp(1j * (characters @ params)))
    joint = jnp.abs(state) ** 2 / 2**n_total
    if n_ancillas == 0:
        return joint
    return joint.reshape(2**n_ancillas, 2**N_QUBITS).sum(axis=0)


def _kl_loss(probs: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return -jnp.sum(target * jnp.log(probs))


def _exact_mmd_loss(
    probs: jnp.ndarray,
    target_fourier: jnp.ndarray,
    mask_weights: jnp.ndarray,
    sigmas: tuple[float, ...],
) -> jnp.ndarray:
    model_fourier = fwht_jax(probs)
    loss = 0.0
    for sigma in sigmas:
        p = mmd_operator_probability(sigma)
        q = (p**mask_weights) * ((1.0 - p) ** (N_QUBITS - mask_weights))
        loss = loss + jnp.sum(
            q * (model_fourier - target_fourier) ** 2,
        )
    return loss / len(sigmas)


def _run(
    name: str,
    beta: float,
    n_ancillas: int,
    objective: str,
    steps: int,
    seed: int,
) -> dict[str, object]:
    ansatz = (
        ancilla_ansatz(N_QUBITS, n_ancillas)
        if n_ancillas > 0
        else complete_ansatz(N_QUBITS)
    )
    n_total = N_QUBITS + n_ancillas
    characters = jnp.asarray(
        generator_characters(np.asarray(ansatz.generators)),
        dtype=jnp.float64,
    )
    alpha = angle_for_beta(beta)
    if n_ancillas > 0:
        angles = ancilla_condition_angles(
            jnp.full((1, N_QUBITS), alpha),
            n_ancillas,
        )[0]
    else:
        angles = jnp.full((n_total,), alpha)
    weights = enumerate_bitstrings(n_total).sum(axis=1).astype(np.float64)
    zero, one = post_hadamard_amplitudes(jnp.asarray(angles))
    reference = jnp.asarray(
        float(zero[0].real) ** (n_total - weights)
        * float(one[0].real) ** weights,
        dtype=jnp.complex128,
    )
    target = HopfieldTarget.create(beta)
    target_probs = jnp.asarray(target.probabilities)
    target_fourier = jnp.asarray(fwht(target.probabilities).real)
    mask_weights = jnp.asarray(
        enumerate_bitstrings(N_QUBITS).sum(axis=1),
        dtype=jnp.float64,
    )

    def loss(params: jnp.ndarray) -> jnp.ndarray:
        probs = _marginal_probs(params, characters, reference, n_ancillas)
        if objective == "kl":
            return _kl_loss(probs, target_probs)
        return _exact_mmd_loss(probs, target_fourier, mask_weights, (1.0, 2.0, 4.0))

    rng = np.random.default_rng(seed)
    params = jnp.asarray(
        rng.normal(0.0, 0.1 / np.sqrt(ansatz.n_parameters), ansatz.n_parameters),
        dtype=jnp.float64,
    )
    optimizer = optax.adam(3e-2)
    state = optimizer.init(params)
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    masses = []
    losses = []
    started = time.monotonic()
    for step in range(steps):
        loss_value, grad = value_and_grad(params)
        updates, state = optimizer.update(grad, state, params)
        params = optax.apply_updates(params, updates)
        if step % 50 == 0:
            probs = np.asarray(
                _marginal_probs(params, characters, reference, n_ancillas),
            )
            masses.append(float(probs[target.pattern_indices].sum()))
            losses.append(float(loss_value))
    return {
        "name": name,
        "beta": beta,
        "objective": objective,
        "n_ancillas": n_ancillas,
        "n_parameters": ansatz.n_parameters,
        "target_pattern_mass": target.pattern_mass,
        "steps": steps,
        "elapsed_seconds": time.monotonic() - started,
        "final_pattern_mass": masses[-1],
        "final_loss": losses[-1],
        "mass_every_50": masses,
        "loss_every_50": losses,
    }


def main() -> None:
    """Run the feasible-grid probe and write ``boundary_feasible.json``."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.5, 0.75, 1.0, 1.25],
    )
    parser.add_argument("--ancillas", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--objectives", nargs="+", default=["kl"])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hopfield/boundary_feasible.json"),
    )
    args = parser.parse_args()

    runs = []
    for beta in args.betas:
        for n_ancillas in args.ancillas:
            for objective in args.objectives:
                name = f"beta{beta}_m{n_ancillas}_{objective}"
                print(f"run {name} steps={args.steps}", flush=True)
                result = _run(name, beta, n_ancillas, objective, args.steps, args.seed)
                runs.append(result)
                ratio = (
                    result["final_pattern_mass"] / result["target_pattern_mass"]
                    if result["target_pattern_mass"] > 0
                    else float("nan")
                )
                print(
                    f"done {name} mass={result['final_pattern_mass']:.5f} "
                    f"target={result['target_pattern_mass']:.5f} "
                    f"ratio={ratio:.3f} seconds={result['elapsed_seconds']:.0f}",
                    flush=True,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(runs, indent=2) + "\n")
    print(f"probe written to {args.output}", flush=True)


if __name__ == "__main__":
    main()

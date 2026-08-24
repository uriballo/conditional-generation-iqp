"""Measure the Hopfield pattern-mass representability boundary.

Runs single-condition exact-gradient optimization at beta=4 (target pattern
mass 0.984) for the complete 2-local ansatz with m in {0, 1, 2} ancillas,
under two objectives:

- exact KL, which isolates representability;
- exact zero-variance MMD, which isolates the kernel objective.

Together with the sampled-estimator feasibility runs, these three curves
locate the stacked barriers documented in ``docs/hopfield-design.md``.  The
default step count reproduces the boundary in about an hour; reduce
``--steps`` for a smoke check.
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

from iqp_cond.ancilla import ancilla_ansatz
from iqp_cond.ansatz import complete_ansatz
from iqp_cond.exact import (
    enumerate_bitstrings,
    fwht,
    fwht_jax,
    generator_characters,
)
from iqp_cond.hopfield import N_QUBITS, HopfieldTarget
from iqp_cond.mmd import mmd_operator_probability


def _reference_amplitudes(n_qubits: int, alpha: float) -> jnp.ndarray:
    from iqp_cond.encoding import post_hadamard_amplitudes

    zero, one = post_hadamard_amplitudes(jnp.full(n_qubits, alpha))
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
    n_ancillas: int,
    objective: str,
    steps: int,
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
    reference = _reference_amplitudes(n_total, np.pi / 8.0)
    target = HopfieldTarget.create(4.0)
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

    rng = np.random.default_rng(10_000)
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
        "objective": objective,
        "n_ancillas": n_ancillas,
        "target_pattern_mass": target.pattern_mass,
        "steps": steps,
        "elapsed_seconds": time.monotonic() - started,
        "final_pattern_mass": masses[-1],
        "final_loss": losses[-1],
        "mass_every_50": masses,
        "loss_every_50": losses,
    }


def main() -> None:
    """Measure the boundary and write ``results/hopfield/boundary.json``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hopfield/boundary.json"),
    )
    args = parser.parse_args()

    runs = []
    for n_ancillas, objective in (
        (0, "kl"),
        (0, "mmd"),
        (1, "kl"),
        (1, "mmd"),
        (2, "kl"),
        (2, "mmd"),
    ):
        name = f"beta4_m{n_ancillas}_{objective}"
        print(f"run {name} steps={args.steps}", flush=True)
        result = _run(name, n_ancillas, objective, args.steps)
        runs.append(result)
        print(
            f"done {name} mass={result['final_pattern_mass']:.4f} "
            f"seconds={result['elapsed_seconds']:.0f}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(runs, indent=2) + "\n")
    print(f"boundary written to {args.output}")


if __name__ == "__main__":
    main()

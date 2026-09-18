"""Performance review of the conditional-IQP estimator.

Measures the per-stage cost of the fixed-bank evaluation and the live variants,
so optimisation choices rest on numbers:

    stages           sample / ratios / phases / one entry, isolated
    banked           fixed-bank objective, per-entry reference draws
    shared           fixed-bank objective, one draw per condition
    fresh            masks drawn from the call key inside the jit
    bank_build       host-side compressed bank construction

Run:

    .venv/bin/python experiments/library_perf/review.py
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from common import problem, save
from iqp_cond.estimator import _phase_differences, estimate_pauli_z_unchecked
from iqp_cond.operators import OperatorBank, operator_bank, sample_nonidentity_masks

import iqp_cond as iqc
from iqp_cond.ancilla import pad_visible_masks


def bench(fn, repeats: int = 20) -> float:
    """Median milliseconds of a fully blocking call."""
    for _ in range(3):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def main() -> None:
    config = iqc.TrainingConfig()
    n_visible, n_conditions = 12, 5
    ansatz, encoders, targets = problem(
        n_visible=n_visible,
        n_conditions=n_conditions,
        n_targets=2048,
        seed=0,
    )
    model = iqc.ConditionalModel(ansatz, encoders)
    sigmas = config.resolved_sigmas(n_visible)
    generators = jnp.asarray(ansatz.generators)
    params = iqc.random_parameters(ansatz.n_parameters, 0)
    batch = jnp.asarray(targets[:, : config.batch_size])
    bank: OperatorBank = operator_bank(
        sigmas,
        config.n_ops,
        n_conditions,
        n_visible,
        seed=0,
    )
    key = jax.random.PRNGKey(0)
    report: dict = {"case": {"n_visible": n_visible, "n_conditions": n_conditions}}

    # ---------------------------------------------------------------- stages
    condition = 0
    encoder = iqc.encoders.select_encoder(encoders, condition)
    masks, _weights = bank.entry(0, condition)
    model_masks = pad_visible_masks(masks, model.n_ancillas)
    samples = encoder.sample(key, config.mc_samples)
    report["stages_ms"] = {
        "sample": bench(lambda: encoder.sample(key, config.mc_samples)),
        "amplitude_ratios": bench(
            lambda: encoder.amplitude_ratios(samples, model_masks),
        ),
        "phase_differences": bench(
            lambda: _phase_differences(params, generators, model_masks, samples),
        ),
        "estimate_one_entry": bench(
            lambda: estimate_pauli_z_unchecked(
                params,
                generators,
                encoder,
                model_masks,
                key=key,
                n_samples=config.mc_samples,
            ),
        ),
    }

    # ---------------------------------------------------------------- variants
    def banked(current_params, target_batch, step_key):
        return model.mmd(
            current_params,
            target_batch,
            step_key,
            bank=bank,
            n_samples=config.mc_samples,
        ).loss

    def shared(current_params, target_batch, step_key):
        return model.mmd(
            current_params,
            target_batch,
            step_key,
            bank=bank,
            n_samples=config.mc_samples,
            shared_samples=True,
        ).loss

    def fresh(current_params, target_batch, step_key):
        return model.mmd(
            current_params,
            target_batch,
            step_key,
            sigmas=sigmas,
            n_ops=config.n_ops,
            n_samples=config.mc_samples,
        ).loss

    variants = {
        "banked": jax.jit(jax.value_and_grad(banked)),
        "shared": jax.jit(jax.value_and_grad(shared)),
        "fresh": jax.jit(jax.value_and_grad(fresh)),
    }
    report["variants_ms"] = {
        name: bench(lambda fn=fn: fn(params, batch, key))
        for name, fn in variants.items()
    }

    # ---------------------------------------------------------------- bank
    report["bank_build_ms"] = {
        "operator_bank": bench(
            lambda: operator_bank(
                sigmas,
                config.n_ops,
                n_conditions,
                n_visible,
                seed=1,
            ),
            repeats=20,
        ),
        "sampler_only": bench(
            lambda: [
                sample_nonidentity_masks(
                    np.random.default_rng(0),
                    config.n_ops,
                    n_visible,
                    sigma,
                )
                for sigma in sigmas
            ],
            repeats=10,
        ),
    }

    for section, values in report.items():
        print(f"\n{section}:")
        for name, value in values.items():
            print(f"  {name}: {value}")
    save("review.json", report)


if __name__ == "__main__":
    main()

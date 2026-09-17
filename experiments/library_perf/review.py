"""jit/vmap performance review of the conditional-IQP estimator.

Measures the per-stage cost of the fixed-bank evaluation and four candidate
variants, so optimisation choices rest on numbers:

    shared-samples   draw one reference sample set per condition and reuse it
                     across bandwidths (saves 2/3 of the sampler cost)
    vmap-bands       evaluate the three bandwidths with one ``jax.vmap`` after
                     padding to a common width (loses per-bandwidth widths)
    vmap-conditions  evaluate all conditions with one ``jax.vmap`` after padding
                     to a common width (grows the live sample x mask buffer)
    sample-in-jit    the pure-JAX ``sampled_mmd`` path

Run:

    PYTHONPATH=packages/iqp-cond/src .venv/bin/python experiments/library_perf/review.py
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from common import problem, save
from iqp_cond.estimator import _phase_differences, estimate_pauli_z_unchecked
from iqp_cond.loss import (
    _condition_mmd,
    banked_mmd,
    operator_bank,
    sample_nonidentity_masks,
)

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
    generators = jnp.asarray(ansatz.generators)
    params = iqc.initialize_parameters(ansatz.n_parameters, 0)
    batch = jnp.asarray(targets[:, : config.batch_size])
    bank = operator_bank(config.sigmas, config.n_ops, n_conditions, n_visible, seed=0)
    n_ancillas = encoders.n_qubits - n_visible
    key = jax.random.PRNGKey(0)
    report: dict = {"case": {"n_visible": n_visible, "n_conditions": n_conditions}}

    # ---------------------------------------------------------------- stages
    condition = 0
    encoder = encoders.select(condition)
    masks, weights = bank.entry(0, condition)
    model_masks = pad_visible_masks(masks, n_ancillas)
    samples = encoder.sample(key, config.n_iqp_samples)

    report["stages_ms"] = {
        "sample": bench(lambda: encoder.sample(key, config.n_iqp_samples)),
        "amplitude_ratios": bench(
            lambda: encoder.amplitude_ratios(samples, model_masks)
        ),
        "phase_differences": bench(
            lambda: _phase_differences(params, generators, model_masks, samples),
        ),
        "estimate_one_entry": bench(
            lambda: estimate_pauli_z_unchecked(
                params,
                generators,
                model_masks,
                encoder,
                key,
                config.n_iqp_samples,
            ),
        ),
        "condition_mmd": bench(
            lambda: _condition_mmd(
                params,
                generators,
                encoder,
                targets[condition : condition + 1],
                masks,
                weights,
                n_ancillas,
                config.n_iqp_samples,
                key,
            ),
        ),
    }

    # ---------------------------------------------------------------- baseline
    def loss(params, target_batch, current_bank, step_key):
        return banked_mmd(
            params,
            generators,
            encoders,
            target_batch,
            current_bank,
            config.n_iqp_samples,
            step_key,
        ).loss

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    report["baseline_ms"] = {
        "banked_forward": bench(lambda: value_and_grad(params, batch, bank, key)),
    }

    # ------------------------------------------------------- shared samples
    def shared_loss(current_params, target_batch, current_bank, step_key):
        return banked_mmd(
            current_params,
            generators,
            encoders,
            target_batch,
            current_bank,
            config.n_iqp_samples,
            step_key,
            shared_samples=True,
        ).loss

    shared = jax.jit(jax.value_and_grad(shared_loss))
    report["shared_samples_ms"] = {
        "forward_grad": bench(lambda: shared(params, batch, bank, key)),
    }

    # ---------------------------------------------------------- vmap bands
    # Like-for-like with the baseline: all conditions, bandwidths vmapped after
    # padding to a common width (the per-bandwidth widths cannot survive vmap).
    def padded_entries(current_condition: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        width = max(bank.width(b) for b in range(bank.n_bandwidths))
        masks = np.zeros((bank.n_bandwidths, width, n_visible))
        weights = np.zeros((bank.n_bandwidths, width))
        for bandwidth in range(bank.n_bandwidths):
            entry_masks, entry_weights = bank.entry(bandwidth, current_condition)
            count = entry_masks.shape[0]
            masks[bandwidth, :count] = np.asarray(entry_masks)
            weights[bandwidth, :count] = np.asarray(entry_weights)
        return jnp.asarray(masks), jnp.asarray(weights)

    padded = [padded_entries(c) for c in range(n_conditions)]
    condition_encoders = [encoders.select(c) for c in range(n_conditions)]

    def vmap_bands_loss(current_params, target_batch, step_key):
        keys = jax.random.split(step_key, n_conditions)
        condition_values = []
        for current_condition in range(n_conditions):
            masks, weights = padded[current_condition]
            condition_encoder = condition_encoders[current_condition]

            target_condition = target_batch[current_condition : current_condition + 1]

            def one(
                entry_masks,
                entry_weights,
                subkey,
                encoder=condition_encoder,
                target=target_condition,
            ):
                return _condition_mmd(
                    current_params,
                    generators,
                    encoder,
                    target,
                    entry_masks,
                    entry_weights,
                    n_ancillas,
                    config.n_iqp_samples,
                    subkey,
                )

            condition_values.append(
                jnp.mean(
                    jax.vmap(one)(
                        masks,
                        weights,
                        jax.random.split(keys[current_condition], bank.n_bandwidths),
                    ),
                ),
            )
        return jnp.mean(jnp.stack(condition_values))

    vmapped = jax.jit(jax.value_and_grad(vmap_bands_loss))
    report["vmap_bands_ms"] = {
        "forward_grad": bench(
            lambda: vmapped(params, batch, key),
        ),
    }

    # ----------------------------------------------------- vmap conditions
    # Products only: the encoder fields are vmapped directly, so this measures
    # the condition axis without touching the library interface.
    condition_masks = []
    condition_weights = []
    for bandwidth in range(bank.n_bandwidths):
        width = bank.width(bandwidth)
        masks = np.zeros((n_conditions, width, n_visible))
        weights = np.zeros((n_conditions, width))
        for current_condition in range(n_conditions):
            entry_masks, entry_weights = bank.entry(bandwidth, current_condition)
            count = entry_masks.shape[0]
            masks[current_condition, :count] = np.asarray(entry_masks)
            weights[current_condition, :count] = np.asarray(entry_weights)
        condition_masks.append(jnp.asarray(masks))
        condition_weights.append(jnp.asarray(weights))

    def vmap_conditions_loss(current_params, target_batch, step_key):
        keys = jax.random.split(step_key, bank.n_bandwidths * n_conditions)
        bandwidth_values = []
        for bandwidth in range(bank.n_bandwidths):
            level_keys = keys[bandwidth * n_conditions : (bandwidth + 1) * n_conditions]

            def one(angle_row, amp_zero, amp_one, masks, weights, target, subkey):
                condition_encoder = iqc.ProductEncoder(
                    name="product",
                    angles=angle_row[None, :],
                    amp_zero=amp_zero[None, :],
                    amp_one=amp_one[None, :],
                )
                return _condition_mmd(
                    current_params,
                    generators,
                    condition_encoder,
                    target[None, :],
                    masks,
                    weights,
                    n_ancillas,
                    config.n_iqp_samples,
                    subkey,
                )

            bandwidth_values.append(
                jnp.mean(
                    jax.vmap(one)(
                        encoders.angles,
                        encoders.amp_zero,
                        encoders.amp_one,
                        condition_masks[bandwidth],
                        condition_weights[bandwidth],
                        target_batch,
                        level_keys,
                    ),
                ),
            )
        return jnp.mean(jnp.stack(bandwidth_values))

    vmap_conditions = jax.jit(jax.value_and_grad(vmap_conditions_loss))
    report["vmap_conditions_ms"] = {
        "forward_grad": bench(
            lambda: vmap_conditions(params, batch, key),
        ),
    }

    # --------------------------------------------------------- bank build
    def build_bank(seed: int) -> float:
        operator_bank(config.sigmas, config.n_ops, n_conditions, n_visible, seed=seed)
        return 0.0

    build_samples = []
    for seed in range(20):
        start = time.perf_counter()
        build_bank(seed)
        build_samples.append((time.perf_counter() - start) * 1e3)
    unique_samples = []
    for seed in range(20):
        start = time.perf_counter()
        for sigma in config.sigmas:
            for _ in range(n_conditions):
                sample_nonidentity_masks(
                    np.random.default_rng(seed),
                    config.n_ops,
                    n_visible,
                    sigma,
                )
        unique_samples.append((time.perf_counter() - start) * 1e3)
    report["bank_build_ms"] = {
        "operator_bank": statistics.median(build_samples),
        "sampler_only": statistics.median(unique_samples),
    }

    def pure_jax_loss(current_params, target_batch, step_key):
        return iqc.sampled_mmd(
            current_params,
            generators,
            encoders,
            target_batch,
            step_key,
            sigmas=config.sigmas,
            n_ops=config.n_ops,
            n_samples=config.n_iqp_samples,
        ).loss

    pure_jax = jax.jit(jax.value_and_grad(pure_jax_loss))
    report["baseline_ms"]["sampled_mmd_forward_grad"] = bench(
        lambda: pure_jax(params, batch, key),
    )

    save("review.json", report)
    for section, values in report.items():
        print(f"\n{section}:")
        for name, value in values.items():
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()

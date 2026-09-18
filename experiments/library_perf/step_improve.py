"""Interleaved A/B measurement of the per-step optimisations.

The machine is usually under background load, so absolute wall clock drifts by
tens of percent between runs.  This harness compiles every variant once and then
interleaves them round-robin within one process, so the comparisons are ratios
measured under identical conditions.

    .venv/bin/python \
        experiments/library_perf/step_improve.py
"""

from __future__ import annotations

import dataclasses
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from common import block, problem, save
from iqp_cond.loss import mmd
from iqp_cond.operators import _assemble_bank, _draw_entrywise, operator_bank

import iqp_cond as ciqp


def interleaved(variants: dict[str, object], rounds: int = 15) -> dict[str, float]:
    """Median milliseconds per variant, measured round-robin."""
    samples: dict[str, list[float]] = {name: [] for name in variants}
    for fn in variants.values():
        block(fn())
    for _ in range(rounds):
        for name, fn in variants.items():
            start = time.perf_counter()
            block(fn())
            samples[name].append((time.perf_counter() - start) * 1e3)
    return {name: statistics.median(values) for name, values in samples.items()}


def old_style_bank(sigmas, n_ops, n_conditions, n_visible, seed, caps):
    """Rebuild the pre-batch bank path: per-entry sampler plus assembly."""
    drawn = _draw_entrywise(
        np.random.default_rng(seed),
        tuple(sigmas),
        n_ops,
        n_conditions,
        n_visible,
        tuple(caps),
        64,
    )
    return _assemble_bank(
        tuple(sigmas),
        drawn,
        n_ops,
        n_visible,
        max(caps),
        tuple(caps),
    )


def main() -> None:
    # Pinned to the regime the recorded numbers were taken in (64 operators,
    # 8192 Monte Carlo samples, fixed sigma triple).  The library defaults have
    # since moved to 512/1000 and the width-aware bodyness schedule, and the
    # tight-cap baseline below is only meaningful at this draw budget.
    config = ciqp.TrainingConfig(
        n_ops=64,
        mc_samples=8_192,
        sigmas=(1.0, 2.0, 4.0),
    )
    ansatz, encoders, targets = problem(
        n_visible=12,
        n_conditions=5,
        n_targets=2048,
        seed=0,
    )
    params = ciqp.random_parameters(ansatz.n_parameters, 0)
    batch = jnp.asarray(targets[:, : config.batch_size])
    key = jax.random.PRNGKey(0)
    n_ops, n_conditions, n_visible = config.n_ops, 5, 12
    sigmas = config.resolved_sigmas(n_visible)
    report: dict = {}

    # ------------------------------------------------------------- bank build
    pilot = operator_bank(sigmas, n_ops, n_conditions, n_visible, seed=0)
    calibrated = tuple(int(width * 1.15) + 2 for width in pilot.widths)
    print(f"caps: {calibrated}")
    report["bank_build_ms"] = interleaved(
        {
            "batched": lambda: operator_bank(
                sigmas,
                n_ops,
                n_conditions,
                n_visible,
                seed=1,
                max_masks=calibrated,
            ),
            "entrywise": lambda: old_style_bank(
                sigmas,
                n_ops,
                n_conditions,
                n_visible,
                1,
                calibrated,
            ),
        },
    )

    # ------------------------------------------------------------- step cost
    def make_step(caps):
        bank = operator_bank(
            sigmas,
            n_ops,
            n_conditions,
            n_visible,
            seed=0,
            max_masks=caps,
        )

        def loss(current_params, target_batch, current_bank, step_key):
            return mmd(
                current_params,
                ansatz.generators,
                encoders,
                target_batch,
                step_key,
                bank=current_bank,
                n_samples=config.mc_samples,
            ).loss

        value_and_grad = jax.jit(jax.value_and_grad(loss))
        return value_and_grad, bank

    natural = operator_bank(sigmas, n_ops, n_conditions, n_visible, seed=0)
    steps: dict[str, object] = {}
    for name, caps in (
        ("caps_calibrated", calibrated),
        ("caps_previous", (66, 39, 23)),
        ("natural", natural.widths),
    ):
        value_and_grad, bank = make_step(caps)
        steps[name] = lambda vg=value_and_grad, b=bank: vg(params, batch, b, key)
    report["step_ms"] = interleaved(steps, rounds=21)

    # ------------------------------------------------------------ ratio dtype
    # Samples and masks are jit arguments, not closures: closed-over arrays are
    # compile-time constants and XLA would fold the whole ratio away.
    masks = natural.entry(0, 0)[0]
    samples = encoders.sample(key, config.mc_samples)
    complex_enc = dataclasses.replace(
        encoders,
        amp_zero=encoders.amp_zero.astype(jnp.complex128),
        amp_one=encoders.amp_one.astype(jnp.complex128),
    )

    def real_ratios(drawn, visible):
        return encoders.amplitude_ratios(drawn, visible)

    def complex_ratios(drawn, visible):
        return complex_enc.amplitude_ratios(drawn, visible)

    jitted_real = jax.jit(real_ratios)
    jitted_complex = jax.jit(complex_ratios)
    report["ratio_ms"] = interleaved(
        {
            "real": lambda: jitted_real(samples, masks),
            "complex": lambda: jitted_complex(samples, masks),
        },
        rounds=21,
    )

    for section, values in report.items():
        print(f"\n{section}:")
        for name, value in values.items():
            print(f"  {name}: {value:.3f} ms")
    save("step_improve.json", report)


if __name__ == "__main__":
    main()

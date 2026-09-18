"""Cost breakdown in the experiment regime: 1000-2000 operators, 500-1000 samples.

Reports, for one (ancillas, n_ops, samples) case:

    bank_ms         operator_bank build (host, NumPy)
    step_ms         jitted loss + gradient with the bank as an argument
    validation_ms   one fixed-bank validation pass
    chunk_*_ms      the same step with sample_chunk set (only when smaller)

    .venv/bin/python \
        experiments/library_perf/regime.py --n-ops 2000 --samples 1000 --ancillas 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
from common import block, problem
from iqp_cond.loss import mmd
from iqp_cond.operators import operator_bank

import iqp_cond as iqc

SIGMAS = (1.0, 2.0, 4.0)
CHUNKS = (256, 512, 1024)


def median_ms(fn, repeats: int = 11) -> float:
    for _ in range(2):
        block(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        block(fn())
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ancillas", type=int, default=0)
    parser.add_argument("--n-ops", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument(
        "--gemm-dtype", choices=["float64", "float32"], default="float64"
    )
    args = parser.parse_args()

    n_visible, n_conditions = 12, 5
    ansatz, encoders, targets = problem(
        n_visible=n_visible,
        n_conditions=n_conditions,
        n_targets=1000,
        seed=0,
        n_ancillas=args.ancillas,
    )
    params = iqc.random_parameters(ansatz.n_parameters, 0)
    generators = jnp.asarray(ansatz.generators)
    bank = operator_bank(
        SIGMAS,
        args.n_ops,
        n_conditions,
        n_visible,
        seed=0,
    )

    gemm_dtype = jnp.float32 if args.gemm_dtype == "float32" else None

    def loss(current_params, target_batch, current_bank, step_key, chunk=None):
        return mmd(
            current_params,
            generators,
            encoders,
            target_batch,
            step_key,
            bank=current_bank,
            n_samples=args.samples,
            shared_samples=True,
            gemm_dtype=gemm_dtype,
            sample_chunk=chunk,
        ).loss

    batch = jnp.asarray(targets[:, :100])
    key = jax.random.PRNGKey(0)
    step = jax.jit(jax.value_and_grad(lambda p, t, b, k: loss(p, t, b, k)))
    report: dict = {
        "ancillas": args.ancillas,
        "n_ops": args.n_ops,
        "samples": args.samples,
        "bank_widths": bank.widths,
        "gemm_dtype": args.gemm_dtype,
    }
    report["bank_ms"] = median_ms(
        lambda: operator_bank(SIGMAS, args.n_ops, n_conditions, n_visible, seed=1),
        repeats=args.repeats,
    )
    report["step_ms"] = median_ms(
        lambda: step(params, batch, bank, key),
        repeats=args.repeats,
    )
    validation = jax.jit(
        lambda p: (
            mmd(
                p,
                generators,
                encoders,
                jnp.asarray(targets),
                jax.random.PRNGKey(1),
                bank=bank,
                n_samples=args.samples,
                shared_samples=True,
            ).loss
        ),
    )
    report["validation_ms"] = median_ms(
        lambda: validation(params), repeats=args.repeats
    )

    for chunk in CHUNKS:
        if chunk >= args.samples:
            continue
        chunked = jax.jit(
            jax.value_and_grad(
                lambda p, t, b, k, c=chunk: loss(p, t, b, k, c),
            ),
        )
        report[f"chunk_{chunk}_ms"] = median_ms(
            lambda fn=chunked: fn(params, batch, bank, key),
            repeats=args.repeats,
        )
    print(json.dumps(report))


if __name__ == "__main__":
    main()

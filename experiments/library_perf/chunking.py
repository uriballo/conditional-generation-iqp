"""Sample-axis chunking for the conditional MMD step, iqpopt-gpu style.

The live tensor of the estimator is the product-encoder ratio
``(samples, masks, model qubits)``.  The GPU repository streams its operator
axis with ``lax.scan`` to cap peak memory; here the sample axis is streamed
instead, because the ancilla axis multiplies that tensor's qubit dimension.

This script measures one configuration per process and prints JSON:

    PYTHONPATH=packages/iqp-cond/src .venv/bin/python \
        experiments/library_perf/chunking.py --ancillas 10 --samples 16384 \
        --chunk 4096

    --chunk 0 means monolithic (no streaming).
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
from iqp_cond.estimator import estimate_pauli_z_from_samples
from iqp_cond.loss import target_moments

import iqp_cond as iqc
from iqp_cond.ancilla import pad_visible_masks

TRAINING_CONDITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def build(ancillas: int, samples: int, n_ops: int) -> dict:
    n_visible, n_conditions = 12, len(TRAINING_CONDITIONS)
    n_model = n_visible + ancillas
    ansatz = iqc.complete_ansatz(n_model)
    conditions = np.asarray(TRAINING_CONDITIONS)
    angles = np.repeat(
        (np.pi * (0.35 - 0.70 * conditions))[:, None],
        n_model,
        axis=1,
    )
    encoders = iqc.encoder(iqc.PRODUCT, angles)
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 2, size=(n_conditions, 100, n_visible)).astype(float)
    params = iqc.initialize_parameters(ansatz.n_parameters, 0)
    bank = iqc.operator_bank((1.0, 2.0, 4.0), n_ops, n_conditions, n_visible, seed=0)
    return {
        "ansatz": ansatz,
        "encoders": encoders,
        "generators": jnp.asarray(ansatz.generators),
        "targets": targets,
        "params": params,
        "bank": bank,
        "samples": samples,
        "ancillas": ancillas,
    }


def monolithic_loss(params, generators, encoders, targets, bank, samples, key):
    return iqc.banked_mmd(
        params,
        generators,
        encoders,
        targets,
        bank,
        samples,
        key,
        shared_samples=True,
    ).loss


def chunked_loss(
    params,
    generators,
    encoders,
    targets,
    bank,
    samples,
    key,
    chunk: int,
):
    """Same objective with the sample axis streamed in ``chunk`` slices.

    The per-mask U-statistic only needs ``sum_s Y`` and ``sum_s Y^2``, so a
    running accumulator over sample chunks is exact up to summation order.
    """
    n_ancillas = encoders.n_qubits - targets.shape[2]
    condition_values = []
    for condition in range(encoders.n_conditions):
        encoder = encoders.select(condition)
        drawn = encoder.sample(jax.random.fold_in(key, condition), samples)
        bandwidth_values = []
        for bandwidth in range(bank.n_bandwidths):
            masks, weights = bank.entry(bandwidth, condition)
            model_masks = pad_visible_masks(masks, n_ancillas)
            total = jnp.zeros(masks.shape[0])
            total_sq = jnp.zeros(masks.shape[0])
            for start in range(0, samples, chunk):
                part = drawn[:, start : start + chunk]
                estimate = estimate_pauli_z_from_samples(
                    params,
                    generators,
                    model_masks,
                    encoder,
                    part,
                )
                values = estimate.sample_values[0]
                total = total + jnp.sum(values, axis=0)
                total_sq = total_sq + jnp.sum(values**2, axis=0)
            target_means, target_squares = target_moments(
                targets[condition : condition + 1],
                masks,
            )
            model_squares = (total**2 - total_sq) / (samples * (samples - 1))
            per_mask = (
                model_squares - 2.0 * (total / samples) * target_means + target_squares
            )
            bandwidth_values.append(jnp.sum(per_mask[0] * weights))
        condition_values.append(jnp.stack(bandwidth_values))
    return jnp.mean(jnp.stack(condition_values, axis=1))


class RssSampler(threading.Thread):
    """Sample this process's resident set while the repeats run.

    ``ru_maxrss`` is monotone and includes compilation, so it hides whether the
    steady-state buffers actually shrink.
    """

    def __init__(self, interval: float = 0.005) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mib = 0.0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval)

    def _sample(self) -> None:
        try:
            with open("/proc/self/statm") as handle:
                pages = int(handle.read().split()[1])
        except OSError:
            return
        self.peak_mib = max(self.peak_mib, pages * 4096.0 / (1024.0 * 1024.0))

    def stop(self) -> float:
        self._stop.set()
        self.join()
        return self.peak_mib


def peak_rss_mib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    return rss / divisor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ancillas", type=int, default=10)
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--n-ops", type=int, default=1000)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    case = build(args.ancillas, args.samples, args.n_ops)
    key = jax.random.PRNGKey(0)

    def loss(current_params, target_batch, step_key):
        if args.chunk:
            return chunked_loss(
                current_params,
                case["generators"],
                case["encoders"],
                target_batch,
                case["bank"],
                args.samples,
                step_key,
                args.chunk,
            )
        return monolithic_loss(
            current_params,
            case["generators"],
            case["encoders"],
            target_batch,
            case["bank"],
            args.samples,
            step_key,
        )

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    start = time.perf_counter()
    output = value_and_grad(case["params"], case["targets"], key)
    jax.block_until_ready(output)
    compile_s = time.perf_counter() - start
    sampler = RssSampler()
    sampler.start()
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        jax.block_until_ready(value_and_grad(case["params"], case["targets"], key))
        times.append((time.perf_counter() - start) * 1e3)
    runtime_peak = sampler.stop()
    print(
        json.dumps(
            {
                "ancillas": args.ancillas,
                "samples": args.samples,
                "n_ops": args.n_ops,
                "chunk": args.chunk,
                "compile_and_first_call_s": compile_s,
                "median_ms": statistics.median(times),
                "process_peak_rss_mib": peak_rss_mib(),
                "runtime_peak_rss_mib": runtime_peak,
            },
        ),
    )


if __name__ == "__main__":
    main()

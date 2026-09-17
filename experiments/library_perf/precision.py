"""Precision levers for the conditional MMD step.

Variants, all sharing one bank and one problem:

    f64              the shipped estimator (float64 throughout)
    f32_gemm         shipped mixed precision: the phase contraction runs in
                     float32 and casts back to float64, so exponentiation and
                     every reduction stay precise
    f32_everything   the estimator cast end to end, including the accumulators
                     (the accuracy cost of what the GPU repo calls plain fp32)

Reports interleaved step times, loss deviation and gradient agreement against
the shipped float64 estimator, and a sample-budget scaling sweep.

    PYTHONPATH=packages/iqp-cond/src .venv/bin/python \
        experiments/library_perf/precision.py
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from common import block, problem, save
from iqp_cond.loss import mmd
from iqp_cond.operators import operator_bank

import iqp_cond as iqc
from iqp_cond.ancilla import pad_visible_masks


def interleaved(variants: dict[str, object], rounds: int = 21) -> dict[str, float]:
    samples: dict[str, list[float]] = {name: [] for name in variants}
    for fn in variants.values():
        block(fn())
    for _ in range(rounds):
        for name, fn in variants.items():
            start = time.perf_counter()
            block(fn())
            samples[name].append((time.perf_counter() - start) * 1e3)
    return {name: statistics.median(values) for name, values in samples.items()}


def cast_encoder(encoders: iqc.ProductEncoder, dtype: jnp.dtype) -> iqc.ProductEncoder:
    return iqc.ProductEncoder(
        name=encoders.name,
        angles=jnp.asarray(encoders.angles, dtype=dtype),
        amp_zero=jnp.asarray(encoders.amp_zero, dtype=dtype),
        amp_one=jnp.asarray(encoders.amp_one, dtype=dtype),
    )


def select(encoders: iqc.ProductEncoder, condition: int) -> iqc.ProductEncoder:
    row = slice(condition, condition + 1)
    return iqc.ProductEncoder(
        name=encoders.name,
        angles=encoders.angles[row],
        amp_zero=encoders.amp_zero[row],
        amp_one=encoders.amp_one[row],
    )


def entry_mmd(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoder: iqc.Encoder,
    target: jnp.ndarray,
    masks: jnp.ndarray,
    weights: jnp.ndarray,
    samples: jnp.ndarray,
    n_ancillas: int,
) -> jnp.ndarray:
    """One bandwidth entry identical to the library, in whatever dtype holds."""
    model_masks = pad_visible_masks(masks, n_ancillas)
    ratios = encoder.amplitude_ratios(samples, model_masks)
    chars = 1.0 - 2.0 * jnp.mod(
        jnp.einsum("bsn,mn->bsm", samples, generators),
        2.0,
    )
    weighted_flips = (
        -2.0
        * jnp.mod(
            jnp.einsum("rn,mn->rm", model_masks, generators),
            2.0,
        )
        * params[None, :]
    )
    phases = jnp.einsum("bsm,rm->bsr", chars, weighted_flips)
    sample_values = jnp.real(ratios * jnp.exp(1j * phases))
    values = jnp.mean(sample_values, axis=1)

    target_characters = 1.0 - 2.0 * jnp.mod(
        jnp.einsum("btn,rn->btr", target, masks),
        2.0,
    )
    n_targets = target.shape[1]
    target_means = jnp.mean(target_characters, axis=1)
    target_sums = jnp.sum(target_characters, axis=1)
    target_squares = (target_sums**2 - n_targets) / (n_targets * (n_targets - 1))

    n_samples = samples.shape[1]
    sums = jnp.sum(sample_values, axis=1)
    square_sums = jnp.sum(sample_values**2, axis=1)
    model_squares = (sums**2 - square_sums) / (n_samples * (n_samples - 1))
    per_mask = model_squares - 2.0 * values * target_means + target_squares
    return jnp.sum(per_mask[0] * weights)


def everything_f32_loss(
    params: jnp.ndarray,
    generators: jnp.ndarray,
    encoders: iqc.ProductEncoder,
    targets: jnp.ndarray,
    bank: iqc.OperatorBank,
    n_samples: int,
    key: jax.Array,
) -> jnp.ndarray:
    """End-to-end float32, including the accumulators."""
    params = jnp.asarray(params, dtype=jnp.float32)
    targets = jnp.asarray(targets, dtype=jnp.float32)
    condition_values = []
    for condition in range(encoders.n_conditions):
        encoder = select(encoders, condition)
        samples = encoder.sample(jax.random.fold_in(key, condition), n_samples)
        entries = []
        for bandwidth in range(bank.n_bandwidths):
            masks, weights = bank.entry(bandwidth, condition)
            entries.append(
                entry_mmd(
                    params,
                    generators,
                    encoder,
                    targets[condition : condition + 1],
                    jnp.asarray(masks, dtype=jnp.float32),
                    jnp.asarray(weights, dtype=jnp.float32),
                    samples,
                    0,
                ),
            )
        condition_values.append(jnp.stack(entries))
    return jnp.mean(jnp.stack(condition_values, axis=1))


def main() -> None:
    config = iqc.TrainingConfig()
    n_visible, n_conditions = 12, 5
    ansatz, encoders, targets = problem(
        n_visible=n_visible,
        n_conditions=n_conditions,
        n_targets=2048,
        seed=0,
    )
    params = iqc.random_parameters(ansatz.n_parameters, 0)
    batch = jnp.asarray(targets[:, : config.batch_size])
    key = jax.random.PRNGKey(0)
    bank = operator_bank(
        config.sigmas,
        config.n_ops,
        n_conditions,
        n_visible,
        seed=0,
    )
    generators = jnp.asarray(ansatz.generators, dtype=jnp.float64)
    report: dict = {"case": {"n_visible": n_visible, "n_conditions": n_conditions}}

    def shipped(current_params, target_batch, step_key, gemm_dtype=None):
        return mmd(
            current_params,
            generators,
            encoders,
            target_batch,
            step_key,
            bank=bank,
            n_samples=config.n_iqp_samples,
            shared_samples=True,
            gemm_dtype=gemm_dtype,
        ).loss

    encoders32 = cast_encoder(encoders, jnp.float32)
    generators32 = jnp.asarray(ansatz.generators, dtype=jnp.float32)

    def everything_f32(current_params, target_batch, step_key):
        return everything_f32_loss(
            current_params,
            generators32,
            encoders32,
            target_batch,
            bank,
            config.n_iqp_samples,
            step_key,
        )

    variants = {
        "f64": jax.jit(jax.value_and_grad(shipped)),
        "f32_gemm": jax.jit(
            jax.value_and_grad(
                lambda p, t, k: shipped(p, t, k, gemm_dtype=jnp.float32),
            ),
        ),
        "f32_everything": jax.jit(jax.value_and_grad(everything_f32)),
    }
    args = {
        "f64": (params, batch, key),
        "f32_gemm": (params, batch, key),
        "f32_everything": (
            jnp.asarray(params, dtype=jnp.float32),
            jnp.asarray(batch, dtype=jnp.float32),
            key,
        ),
    }
    report["step_ms"] = interleaved(
        {name: (lambda fn=fn, a=args[name]: fn(*a)) for name, fn in variants.items()},
    )

    reference_loss, reference_grad = variants["f64"](params, batch, key)
    reference_grad = np.asarray(reference_grad, dtype=np.float64)
    rows = {}
    for name in ("f32_gemm", "f32_everything"):
        loss, grad = variants[name](*args[name])
        grad = np.asarray(grad, dtype=np.float64)
        delta = grad - reference_grad
        rows[name] = {
            "loss": float(loss),
            "loss_deviation": abs(float(loss) - float(reference_loss)),
            "gradient_relative_error": float(
                np.linalg.norm(delta) / np.linalg.norm(reference_grad),
            ),
            "gradient_cosine": float(
                np.dot(grad, reference_grad)
                / (np.linalg.norm(grad) * np.linalg.norm(reference_grad)),
            ),
        }
    report["accuracy_vs_f64"] = rows

    sweep = {}
    for n_samples in (250, 500, 1000, 2000, 4000):

        def sweep_loss(p, t, k, s=n_samples):
            return mmd(
                p,
                generators,
                encoders,
                t,
                k,
                bank=bank,
                n_samples=s,
                shared_samples=True,
            ).loss

        step = jax.jit(jax.value_and_grad(sweep_loss))
        sweep[str(n_samples)] = interleaved(
            {"step": lambda fn=step: fn(params, batch, key)},
            rounds=11,
        )["step"]
    report["sample_sweep_ms"] = sweep

    # ------------------------------------------------------- paper scale
    paper_config = iqc.TrainingConfig(batch_size=2048, n_ops=256, n_iqp_samples=16384)
    paper_ansatz, paper_encoders, paper_targets = problem(
        n_visible=12,
        n_conditions=10,
        n_ancillas=10,
        n_targets=2048,
        seed=1,
    )
    paper_params = iqc.random_parameters(paper_ansatz.n_parameters, 0)
    paper_batch = jnp.asarray(paper_targets)
    paper_bank = operator_bank(
        paper_config.sigmas,
        paper_config.n_ops,
        paper_encoders.n_conditions,
        12,
        seed=0,
    )
    paper_generators = jnp.asarray(paper_ansatz.generators, dtype=jnp.float64)

    def paper_shipped(current_params, target_batch, step_key, gemm_dtype=None):
        return mmd(
            current_params,
            paper_generators,
            paper_encoders,
            target_batch,
            step_key,
            bank=paper_bank,
            n_samples=paper_config.n_iqp_samples,
            gemm_dtype=gemm_dtype,
        ).loss

    paper_variants = {
        "f64": jax.jit(jax.value_and_grad(paper_shipped)),
        "f32_gemm": jax.jit(
            jax.value_and_grad(
                lambda p, t, k: paper_shipped(p, t, k, gemm_dtype=jnp.float32),
            ),
        ),
    }
    report["paper_scale_step_ms"] = interleaved(
        {
            name: (lambda fn=fn: fn(paper_params, paper_batch, key))
            for name, fn in paper_variants.items()
        },
        rounds=11,
    )
    paper_loss, paper_grad = paper_variants["f64"](paper_params, paper_batch, key)
    mixed_loss, mixed_grad = paper_variants["f32_gemm"](
        paper_params,
        paper_batch,
        key,
    )
    paper_grad = np.asarray(paper_grad, dtype=np.float64)
    mixed_grad = np.asarray(mixed_grad, dtype=np.float64)
    report["paper_scale_accuracy"] = {
        "loss_deviation": abs(float(mixed_loss) - float(paper_loss)),
        "gradient_relative_error": float(
            np.linalg.norm(mixed_grad - paper_grad) / np.linalg.norm(paper_grad),
        ),
        "gradient_cosine": float(
            np.dot(mixed_grad, paper_grad)
            / (np.linalg.norm(mixed_grad) * np.linalg.norm(paper_grad)),
        ),
    }

    for section, values in report.items():
        print(f"\n{section}:")
        if isinstance(values, dict):
            for name, value in values.items():
                print(f"  {name}: {value}")
        else:
            print(f"  {values}")
    save("precision.json", report)


if __name__ == "__main__":
    main()

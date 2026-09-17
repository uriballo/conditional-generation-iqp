"""Full training benchmark on the prepared 12-qubit Gaussian dataset.

Uses the archived campaign data
(``results/exact-tvd-ancilla-v1/data/gaussian-n12.npz``): five training
conditions ``[0, 0.25, 0.5, 0.75, 1.0]``, 1000 samples each, and
the eight interleaved held-out conditions for a generalization score.  The
encoder is the broadcast product-state map ``alpha(c) = pi (0.35 - 0.70 c)``.

    PYTHONPATH=packages/iqp-cond/src .venv/bin/python \
        experiments/library_perf/gaussian_fit.py --steps 2000 --n-ops 1000 \
        --samples 500 --label gaussian-n12

Everything the run needs is exposed on the public API: ``TrainingConfig``
carries the update budget, per-condition batch, bandwidths, operator draws, and
Monte Carlo samples; ``train(..., mask_refresh=...)`` selects the mask
protocol; ``estimate_pauli_z`` and ``mmd`` take explicit
``sigmas``, ``n_ops``, and ``n_samples`` the way ``iqpopt.mmd_loss_iqp`` takes
``sigma``, ``n_ops``, ``n_samples``.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from common import REPO, save
from iqp_cond.operators import operator_bank

import iqp_cond as iqc

TRAINING_CONDITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_DATA = (
    "results/exact-tvd-ancilla-v1/data/gaussian-n12.npz",
    "data/gaussian-n12.npz",
)


def peak_rss_mib() -> float:
    """Peak resident set size of this process; ru_maxrss is KiB on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    return rss / divisor


def load_dataset(path: Path) -> dict:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def condition_angles(conditions: np.ndarray, n_qubits: int) -> np.ndarray:
    """The campaign's broadcast product-state angle map, as (C, n) columns."""
    angles = np.pi * (0.35 - 0.70 * conditions)
    return np.repeat(angles[:, None], n_qubits, axis=1)


def find_data(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for candidate in DEFAULT_DATA:
        path = REPO / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"gaussian-n12 dataset not found under {REPO}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--n-ops", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ancillas", type=int, default=0)
    parser.add_argument("--refresh", choices=[iqc.STEP, iqc.CACHE], default=iqc.STEP)
    parser.add_argument(
        "--gemm-dtype", choices=["float64", "float32"], default="float64"
    )
    parser.add_argument("--sample-chunk", type=int, default=0)
    parser.add_argument("--label", default="gaussian-n12")
    args = parser.parse_args()

    data = load_dataset(find_data(args.data))
    conditions = np.asarray(data["conditions"], dtype=np.float64)
    n_visible = int(data["visible_qubits"])
    n_model = n_visible + args.ancillas
    selected = np.asarray(
        [int(np.argmin(np.abs(conditions - value))) for value in TRAINING_CONDITIONS],
    )
    held_out = np.asarray(
        [index for index in range(conditions.size) if index not in set(selected)],
    )
    training = np.asarray(data["train"], dtype=np.float64)[selected]
    validation = np.asarray(data["validation"], dtype=np.float64)[selected]
    held_out_targets = np.asarray(data["test"], dtype=np.float64)[held_out]

    ansatz = iqc.complete_ansatz(n_model)
    generators = np.asarray(ansatz.generators)
    encoders = iqc.encoder(iqc.PRODUCT, condition_angles(conditions[selected], n_model))
    held_out_encoders = iqc.encoder(
        iqc.PRODUCT,
        condition_angles(conditions[held_out], n_model),
    )
    initial_params = iqc.parameters_from_data(
        training[0],
        generators,
        n_visible=n_visible,
        seed=args.seed,
    )
    model = iqc.ConditionalModel(
        ansatz,
        encoders,
        n_visible=n_visible if args.ancillas else None,
    )

    config = iqc.TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        sigmas=(1.0, 2.0, 4.0),
        n_ops=args.n_ops,
        n_iqp_samples=args.samples,
        validation_n_ops=args.n_ops,
        validation_n_iqp_samples=args.samples,
        gemm_dtype=jnp.float32 if args.gemm_dtype == "float32" else None,
        sample_chunk=args.sample_chunk or None,
        mask_refresh=args.refresh,
    )

    start = time.perf_counter()
    result = iqc.train(
        model,
        training,
        validation,
        seed=args.seed,
        config=config,
        initial_params=initial_params,
    )
    wall = time.perf_counter() - start

    # Generalization to the eight held-out conditions, on an independent bank.
    key = jax.random.PRNGKey(args.seed + 4242)
    generalization_bank = operator_bank(
        config.sigmas,
        args.n_ops,
        held_out_encoders.n_conditions,
        n_visible,
        seed=args.seed + 31,
    )
    held_out_score = float(
        iqc.mmd(
            result.best_params,
            ansatz.generators,
            held_out_encoders,
            jnp.asarray(held_out_targets),
            key,
            bank=generalization_bank,
            n_samples=args.samples,
            shared_samples=True,
        ).loss,
    )

    # Checkpoint consistency against the trainer's own validation bank and key.
    expected_bank = operator_bank(
        config.sigmas,
        config.validation_n_ops,
        encoders.n_conditions,
        n_visible,
        seed=args.seed + 1,
    )
    reproduced = float(
        iqc.mmd(
            result.best_params,
            ansatz.generators,
            encoders,
            jnp.asarray(validation),
            jax.random.PRNGKey(args.seed + 1_000_000),
            bank=expected_bank,
            n_samples=config.validation_n_iqp_samples,
        ).loss,
    )

    payload = {
        "label": args.label,
        "data": str(find_data(args.data)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "n_ops": args.n_ops,
        "n_iqp_samples": args.samples,
        "sigmas": list(config.sigmas),
        "mask_refresh": args.refresh,
        "gemm_dtype": args.gemm_dtype,
        "sample_chunk": args.sample_chunk,
        "seed": args.seed,
        "ancillas": args.ancillas,
        "n_parameters": ansatz.n_parameters,
        "training_conditions": [float(value) for value in conditions[selected]],
        "held_out_conditions": [float(value) for value in conditions[held_out]],
        "wall_s": wall,
        "per_step_ms": 1e3 * wall / args.steps,
        "best_step": int(result.best_step),
        "best_validation_loss": float(result.best_validation_loss),
        "final_train_loss": float(np.mean(result.train_losses[-20:])),
        "held_out_mmd": held_out_score,
        "checkpoint_consistent": abs(reproduced - result.best_validation_loss) <= 1e-12,
        "peak_rss_mib": peak_rss_mib(),
    }
    for name, value in payload.items():
        print(f"{name}: {value}")
    save(f"{args.label}.json", payload)


if __name__ == "__main__":
    main()

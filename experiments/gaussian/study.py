"""One-parameter Gaussian study: translated Hamming-profile targets.

The target at a scalar condition ``c`` is the truncated discrete Gaussian over
Hamming weight with mean ``n (0.15 + 0.70 c)`` and width ``0.5 sqrt(n)``; the
encoder broadcasts one angle per condition, ``pi (0.35 - 0.70 c)``.  Runs write
to ``results/gaussian/<run_id>/`` through the shared harness.

    uv run python -m experiments.gaussian.study --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import iqp_cond as ciqp
import numpy as np

from experiments.common import runner, tracking
from experiments.common.config import RunConfig
from experiments.common.datasets import gaussian_target
from experiments.common.encoders import scalar_broadcast_encoder
from experiments.common.runner import Study

TRAINING_CONDITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
HELD_OUT_CONDITIONS = (0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375)
CONFIG_PATH = Path(__file__).with_name("config.json")


def _bitstring_states(n_visible: int) -> np.ndarray:
    indices = np.arange(2**n_visible, dtype=np.int64)
    return ((indices[:, None] >> np.arange(n_visible)) & 1).astype(np.float64)


def make_targets(
    conditions: np.ndarray,
    n_visible: int,
    n_samples: int,
    rng: np.random.Generator,
    width_factor: float = 0.5,
) -> np.ndarray:
    """Draw ``(conditions, samples, visible)`` target bitstrings."""
    return np.stack(
        [
            gaussian_target(n_visible, width_factor, float(condition))
            .sample(
                rng,
                n_samples,
            )
            .astype(np.float64)
            for condition in np.asarray(conditions).ravel()
        ],
    )


def target_probabilities(
    conditions: np.ndarray,
    n_visible: int,
    width_factor: float = 0.5,
) -> np.ndarray:
    """Return the exact ``(conditions, 2**n_visible)`` target distributions."""
    states = _bitstring_states(n_visible)
    return np.stack(
        [
            gaussian_target(
                n_visible, width_factor, float(condition)
            ).bitstring_probabilities(states)
            for condition in np.asarray(conditions).ravel()
        ],
    )


def make_encoder(conditions: np.ndarray, config: RunConfig) -> ciqp.Encoder:
    """Broadcast scalar angle map on the joint register."""
    return scalar_broadcast_encoder(np.asarray(conditions).ravel(), config.n_total)


def build_study(width_factor: float = 0.5) -> Study:
    """Return the study descriptor with the paper's frozen parameters."""
    return Study(
        name="gaussian",
        training_conditions=np.asarray(TRAINING_CONDITIONS),
        held_out_conditions=np.asarray(HELD_OUT_CONDITIONS),
        make_targets=lambda conditions, n, samples, rng: make_targets(
            conditions,
            n,
            samples,
            rng,
            width_factor,
        ),
        target_probabilities=lambda conditions, n: target_probabilities(
            conditions,
            n,
            width_factor,
        ),
        make_encoder=make_encoder,
    )


def _paper_defaults() -> dict:
    with CONFIG_PATH.open() as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> None:
    """Run the study for the configured seeds and index the results."""
    defaults = _paper_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-visible", type=int, default=defaults["visible_qubits"][0])
    parser.add_argument("--ancillas", type=int, default=0)
    parser.add_argument("--seed", type=int, default=defaults["seeds"][0])
    parser.add_argument("--seeds", default=None, help="comma-separated overrides")
    parser.add_argument("--steps", type=int, default=defaults["training"]["updates"])
    parser.add_argument("--width-factor", type=float, default=defaults["width"])
    parser.add_argument(
        "--smoke", action="store_true", help="tiny budgets for a quick run"
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)

    seeds = (
        tuple(int(value) for value in args.seeds.split(","))
        if args.seeds
        else (args.seed,)
    )
    if args.smoke:
        config = RunConfig(
            study="gaussian",
            encoder="broadcast",
            n_visible=args.n_visible,
            n_ancillas=args.ancillas,
            width_factor=args.width_factor,
            n_train=64,
            n_validation=64,
            n_test=64,
            steps=5,
            batch_size=16,
            n_ops=16,
            mc_samples=64,
            validation_interval=5,
            seeds=seeds,
        )
    else:
        training = defaults["training"]
        config = RunConfig(
            study="gaussian",
            encoder="broadcast",
            n_visible=args.n_visible,
            n_ancillas=args.ancillas,
            width_factor=args.width_factor,
            n_train=defaults["data"]["train_samples"],
            n_validation=defaults["data"]["validation_samples"],
            n_test=defaults["data"]["test_samples"],
            steps=args.steps,
            batch_size=training["batch_size"],
            n_ops=training["n_ops"],
            mc_samples=training["n_samples"],
            learning_rate=training["learning_rate"],
            final_learning_rate=training["final_learning_rate"],
            validation_interval=training["validation_interval"],
            seeds=seeds,
        )

    study = build_study(args.width_factor)
    records = runner.run_all_seeds(config, study, output_root=args.output_root)
    for record in records:
        tracking.append_index(record, args.output_root)
        print(
            f"{record['run_id']}: train TVD {record['train_tvd_mean']:.4f}, "
            f"held-out TVD {record['held_out_tvd_mean']:.4f}, "
            f"best step {record['best_step']}",
        )


if __name__ == "__main__":
    main()

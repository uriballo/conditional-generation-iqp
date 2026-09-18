"""Two-parameter Gaussian study: conditions vary mean and width jointly.

The condition is ``(u, v) = (mean / n, sd / sqrt(n))`` on the frozen 5x5 grid;
the 16 perimeter conditions train the shared circuit and the nine interior
conditions are held out.  Two encoder arms share the grid: the two-block map
with location and width channels, and the single-channel broadcast ablation.

    uv run python -m experiments.gaussian2d.study --smoke --encoder two_block
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import iqp_cond as ciqp
import numpy as np

from experiments.common import runner, tracking
from experiments.common.config import RunConfig, study_run_config
from experiments.common.datasets import (
    held_out_conditions,
    prescribed_target,
    training_conditions,
)
from experiments.common.encoders import encoder_for
from experiments.common.runner import Study

CONFIG_PATH = Path(__file__).with_name("config.json")


def _bitstring_states(n_visible: int) -> np.ndarray:
    indices = np.arange(2**n_visible, dtype=np.int64)
    return ((indices[:, None] >> np.arange(n_visible)) & 1).astype(np.float64)


def make_targets(
    conditions: np.ndarray,
    n_visible: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``(conditions, samples, visible)`` target bitstrings."""
    return np.stack(
        [
            prescribed_target(n_visible, condition)
            .sample(rng, n_samples)
            .astype(
                np.float64,
            )
            for condition in np.asarray(conditions)
        ],
    )


def target_probabilities(
    conditions: np.ndarray,
    n_visible: int,
) -> np.ndarray:
    """Return the exact ``(conditions, 2**n_visible)`` target distributions."""
    states = _bitstring_states(n_visible)
    return np.stack(
        [
            prescribed_target(n_visible, condition).bitstring_probabilities(states)
            for condition in np.asarray(conditions)
        ],
    )


def make_encoder(conditions: np.ndarray, config: RunConfig) -> ciqp.Encoder:
    """Build the configured two-parameter encoder arm."""
    return encoder_for(config.encoder, np.asarray(conditions), config.n_total)


def build_study() -> Study:
    """Return the study descriptor with the frozen grid and maps."""
    return Study(
        name="gaussian2d",
        training_conditions=training_conditions(),
        held_out_conditions=held_out_conditions(),
        make_targets=make_targets,
        target_probabilities=target_probabilities,
        make_encoder=make_encoder,
    )


def _paper_defaults() -> dict:
    with CONFIG_PATH.open() as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> None:
    """Run one encoder arm and index the result."""
    defaults = _paper_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder",
        choices=sorted(defaults["encoders"]),
        default="two_block",
    )
    parser.add_argument("--n-visible", type=int, default=defaults["visible_qubits"][0])
    parser.add_argument("--ancillas", type=int, default=0)
    parser.add_argument("--seed", type=int, default=defaults["seeds"][0])
    parser.add_argument("--seeds", default=None, help="comma-separated overrides")
    parser.add_argument("--steps", type=int, default=defaults["training"]["updates"])
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
            study="gaussian2d",
            encoder=args.encoder,
            n_visible=args.n_visible,
            n_ancillas=args.ancillas,
            n_train=32,
            n_validation=32,
            n_test=32,
            steps=5,
            batch_size=16,
            n_ops=16,
            mc_samples=64,
            validation_interval=5,
            seeds=seeds,
        )
    else:
        config = study_run_config(
            "gaussian2d",
            defaults,
            encoder=args.encoder,
            n_visible=args.n_visible,
            n_ancillas=args.ancillas,
            steps=args.steps,
            seeds=seeds,
        )

    study = build_study()
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

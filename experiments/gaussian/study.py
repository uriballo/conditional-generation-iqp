"""One-parameter Gaussian study: translated Hamming-profile targets.

The target at a scalar condition ``c`` is the truncated discrete Gaussian over
Hamming weight with mean ``n (0.15 + 0.70 c)`` and width ``0.5 sqrt(n)``. The
main model uses the paired anti-phase encoder. Named protocol and encoder arms
support the performance matrix, ablations, and no-input independent baselines.

    uv run python -m experiments.gaussian.study --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import iqp_cond as ciqp
import numpy as np

from experiments.common import runner, tracking
from experiments.common.config import RunConfig, study_run_config
from experiments.common.datasets import gaussian_target
from experiments.common.encoders import encoder_for
from experiments.common.runner import Study

CONDITION_PROTOCOLS = {
    "full3": (0.0, 0.5, 1.0),
    "half3": (0.0, 0.25, 0.5),
    "full5": (0.0, 0.25, 0.5, 0.75, 1.0),
}
TRAINING_CONDITIONS = CONDITION_PROTOCOLS["full5"]
HELD_OUT_CONDITIONS = tuple((2 * index + 1) / 32 for index in range(16))
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
    """Build the named scalar encoder on the joint register."""
    return encoder_for(config.encoder, np.asarray(conditions).ravel(), config.n_total)


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
        condition_protocols={
            name: np.asarray(conditions, dtype=np.float64)
            for name, conditions in CONDITION_PROTOCOLS.items()
        },
        condition_dimension=1,
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
    parser.add_argument("--encoder", default="anti_block")
    parser.add_argument(
        "--protocol",
        choices=tuple(CONDITION_PROTOCOLS),
        default="full5",
    )
    parser.add_argument(
        "--model-kind",
        choices=("conditional", "encoder_only", "independent_direct"),
        default="conditional",
    )
    parser.add_argument(
        "--fit-condition",
        type=float,
        default=None,
        help="required scalar condition for an independent_direct fit",
    )
    parser.add_argument("--seed", type=int, default=defaults["seeds"][0])
    parser.add_argument("--seeds", default=None, help="comma-separated overrides")
    parser.add_argument("--steps", type=int, default=defaults["training"]["updates"])
    parser.add_argument(
        "--restarts",
        type=int,
        default=defaults["training"]["restarts"],
    )
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
            encoder=args.encoder,
            condition_protocol=args.protocol,
            model_kind=args.model_kind,
            fit_condition=(
                (args.fit_condition,) if args.fit_condition is not None else None
            ),
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
            restarts=min(args.restarts, 1),
            seeds=seeds,
        )
    else:
        config = study_run_config(
            "gaussian",
            defaults,
            encoder=args.encoder,
            n_visible=args.n_visible,
            n_ancillas=args.ancillas,
            condition_protocol=args.protocol,
            model_kind=args.model_kind,
            fit_condition=(
                (args.fit_condition,) if args.fit_condition is not None else None
            ),
            restarts=args.restarts,
            width_factor=args.width_factor,
            steps=args.steps,
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

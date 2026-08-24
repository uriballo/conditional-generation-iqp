"""Run or resume the frozen condition-count scaling matrix."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from iqp_cond.scaling import (
    SCALING_CONDITION_COUNTS,
    SCALING_TRACKS,
    TOTAL_MINIBATCH_SIZE,
    run_scaling,
    save_scaling_result,
    scaling_budget,
)
from iqp_cond.training import TrainingConfig

METHODS = ("shared_angle", "no_condition", "separate_bank")


def _parse_ints(value: str) -> list[int]:
    if "-" in value:
        start, end = (int(part) for part in value.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def _parse_values(value: str, allowed: tuple[str, ...]) -> list[str]:
    values = value.split(",")
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"unknown values {unknown}; expected one of {allowed}")
    return values


def main() -> None:
    """Run each selected K/track/method/seed as an independent file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default="2,3,5,9")
    parser.add_argument("--tracks", default=",".join(SCALING_TRACKS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/scaling/runs"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--steps", type=int, default=2_000)
    args = parser.parse_args()

    conditions = _parse_ints(args.conditions)
    unknown_conditions = sorted(set(conditions) - set(SCALING_CONDITION_COUNTS))
    if unknown_conditions:
        raise ValueError(
            f"unknown Condition counts {unknown_conditions}; "
            f"expected {SCALING_CONDITION_COUNTS}",
        )
    tracks = _parse_values(args.tracks, SCALING_TRACKS)
    methods = _parse_values(args.methods, METHODS)
    seeds = _parse_ints(args.seeds)
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    for track in tracks:
        for n_conditions in conditions:
            budget = scaling_budget(
                n_conditions,
                track,
                optimizer_updates=args.steps,
            )
            config = TrainingConfig(
                steps=args.steps,
                batch_size=budget.total_batch_size // n_conditions,
            )
            if config.batch_size * n_conditions != TOTAL_MINIBATCH_SIZE:
                raise AssertionError("scaling minibatch is not balanced")
            for method in methods:
                model_name = (
                    f"scaling_{track}_k{n_conditions}_{method}"
                )
                for seed in seeds:
                    destination = args.output / f"{model_name}_seed{seed}.json"
                    if args.skip_existing and destination.exists():
                        print(f"skip {model_name} seed={seed}")
                        continue
                    started = time.monotonic()
                    print(
                        f"run {model_name} seed={seed} steps={args.steps}",
                        flush=True,
                    )
                    results, metrics = run_scaling(
                        seed,
                        n_conditions,
                        track,
                        method,
                        config,
                    )
                    save_scaling_result(results, metrics, args.output)
                    elapsed = time.monotonic() - started
                    print(
                        f"done {model_name} seed={seed} seconds={elapsed:.1f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()

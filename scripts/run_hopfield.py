"""Run the Hopfield-beta conditional experiment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from iqp_cond.encoding import ENCODINGS
from iqp_cond.hopfield import (
    run_hopfield,
    run_no_condition,
    run_separate_models,
    save_hopfield_result,
)
from iqp_cond.training import EXACT_GRADIENT_STRATEGIES, TrainingConfig


def _parse_seeds(value: str) -> list[int]:
    if "-" in value:
        start, end = (int(part) for part in value.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def _run_one(
    function,
    seed: int,
    config: TrainingConfig,
    encoding: str,
    output: Path,
    **kwargs,
) -> float:
    result, metrics = function(seed, config, encoding, **kwargs)
    save_hopfield_result(result, metrics, config, output)
    return float(metrics["best_validation_loss"])


def _dense_model_name(encoding: str, n_ancillas: int) -> str:
    """Return the persisted filename stem for a dense Hopfield run."""
    if n_ancillas > 0:
        return (
            f"hopfield_ancilla{n_ancillas}_complete"
            if encoding == "angle"
            else f"hopfield_ancilla{n_ancillas}_{encoding}_complete"
        )
    return (
        "hopfield_complete"
        if encoding == "angle"
        else f"hopfield_{encoding}_complete"
    )


def main() -> None:
    """Run selected seeds for the dense model and both baselines."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hopfield/runs"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--final-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--gradient-strategy",
        choices=EXACT_GRADIENT_STRATEGIES,
        default="mean",
    )
    parser.add_argument("--encoding", choices=ENCODINGS, default="angle")
    parser.add_argument("--ancillas", type=int, default=0)
    parser.add_argument(
        "--ancilla-scheme",
        choices=("broadcast_all", "fixed_zero"),
        default="broadcast_all",
    )
    parser.add_argument(
        "--models",
        choices=("dense", "no_condition", "separate", "all"),
        default="all",
    )
    args = parser.parse_args()
    config = TrainingConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        exact_gradient_strategy=args.gradient_strategy,
    )

    for seed in _parse_seeds(args.seeds):
        if args.models in ("dense", "all"):
            dense_name = _dense_model_name(args.encoding, args.ancillas)
            destination = args.output / f"{dense_name}_seed{seed}.json"
            if args.skip_existing and destination.exists():
                print(f"skip hopfield dense seed={seed}")
            else:
                started = time.monotonic()
                print(f"run hopfield dense seed={seed}", flush=True)
                best = _run_one(
                    run_hopfield,
                    seed,
                    config,
                    args.encoding,
                    args.output,
                    n_ancillas=args.ancillas,
                    ancilla_scheme=args.ancilla_scheme,
                )
                elapsed = time.monotonic() - started
                print(
                    f"done hopfield dense seed={seed} seconds={elapsed:.1f} "
                    f"best_validation_loss={best:.6f}",
                    flush=True,
                )
        if args.models in ("no_condition", "all"):
            destination = (
                args.output / f"hopfield_no_condition_complete_seed{seed}.json"
            )
            if args.skip_existing and destination.exists():
                print(f"skip hopfield no_condition seed={seed}")
            else:
                started = time.monotonic()
                print(f"run hopfield no_condition seed={seed}", flush=True)
                best = _run_one(
                    run_no_condition,
                    seed,
                    config,
                    args.encoding,
                    args.output,
                )
                elapsed = time.monotonic() - started
                print(
                    f"done hopfield no_condition seed={seed} "
                    f"seconds={elapsed:.1f} best_validation_loss={best:.6f}",
                    flush=True,
                )
        if args.models in ("separate", "all"):
            separate_directory = args.output / "separate"
            separate_directory.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            print(f"run hopfield separate seed={seed}", flush=True)
            for result, metrics in run_separate_models(
                seed,
                config,
                args.encoding,
            ):
                save_hopfield_result(
                    result,
                    metrics,
                    config,
                    separate_directory,
                )
            elapsed = time.monotonic() - started
            print(
                f"done hopfield separate seed={seed} seconds={elapsed:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

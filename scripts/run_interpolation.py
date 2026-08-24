"""Run the held-out angle interpolation experiment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from iqp_cond.encoding import ENCODINGS
from iqp_cond.interpolation import (
    run_interpolation,
    save_interpolation_result,
)
from iqp_cond.training import TrainingConfig


def _parse_seeds(value: str) -> list[int]:
    if "-" in value:
        start, end = (int(part) for part in value.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def main() -> None:
    """Run selected seeds and save each independently."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/interpolation/runs"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--encoding", choices=ENCODINGS, default="angle")
    args = parser.parse_args()
    config = TrainingConfig(steps=args.steps)

    for seed in _parse_seeds(args.seeds):
        model_name = (
            "interpolation_complete"
            if args.encoding == "angle"
            else f"interpolation_{args.encoding}_complete"
        )
        destination = args.output / f"{model_name}_seed{seed}.json"
        if args.skip_existing and destination.exists():
            print(f"skip {model_name} seed={seed}")
            continue
        started = time.monotonic()
        print(f"run {model_name} seed={seed}", flush=True)
        result, metrics = run_interpolation(seed, config, args.encoding)
        save_interpolation_result(result, metrics, config, args.output)
        elapsed = time.monotonic() - started
        print(
            f"done {model_name} seed={seed} seconds={elapsed:.1f} "
            f"heldout_mmd={metrics['heldout_mmd']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

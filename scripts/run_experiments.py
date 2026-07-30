"""Run the agreed conditional IQP experiment matrix."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from iqp_cond.experiment import model_specs, run_one, save_result
from iqp_cond.training import TrainingConfig


def _parse_seeds(value: str) -> list[int]:
    if "-" in value:
        start, end = (int(part) for part in value.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def main() -> None:
    """Run selected configurations and save each result independently."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--output", type=Path, default=Path("results/runs"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--steps", type=int, default=2_000)
    args = parser.parse_args()

    specs = model_specs()
    names = list(specs) if args.models == "all" else args.models.split(",")
    unknown = sorted(set(names) - set(specs))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    config = TrainingConfig(steps=args.steps)

    for seed in _parse_seeds(args.seeds):
        for name in names:
            destination = args.output / f"{name}_seed{seed}.json"
            if args.skip_existing and destination.exists():
                print(f"skip {name} seed={seed}")
                continue
            started = time.monotonic()
            print(f"run {name} seed={seed}", flush=True)
            result, metrics = run_one(specs[name], seed, config)
            save_result(result, metrics, config, args.output)
            elapsed = time.monotonic() - started
            print(
                f"done {name} seed={seed} seconds={elapsed:.1f} "
                f"heldout_mmd={metrics['heldout_mmd']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

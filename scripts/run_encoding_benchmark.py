"""Run matched-seed encoding benchmark on the 19-qubit interpolation task."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from iqp_cond.encoding import (
    ANGLE_ENCODING,
    BLOCK_ZZ_PI4_ENCODING,
    BLOCK_ZZ_PI8_ENCODING,
    CALIBRATED_REUPLOAD_ENCODING,
    ENCODINGS,
    SYMMETRIC_REUPLOAD_ENCODING,
)
from iqp_cond.interpolation import (
    run_interpolation,
    save_interpolation_result,
)
from iqp_cond.training import TrainingConfig

BENCHMARK_ENCODINGS = (
    ANGLE_ENCODING,
    SYMMETRIC_REUPLOAD_ENCODING,
    CALIBRATED_REUPLOAD_ENCODING,
    BLOCK_ZZ_PI8_ENCODING,
    BLOCK_ZZ_PI4_ENCODING,
)


def _parse_seeds(value: str) -> list[int]:
    if "-" in value:
        start, end = (int(part) for part in value.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def main() -> None:
    """Run all benchmark encodings on requested seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/encoding_benchmark/runs"),
    )
    parser.add_argument(
        "--encodings",
        nargs="+",
        choices=ENCODINGS,
        default=list(BENCHMARK_ENCODINGS),
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    config = TrainingConfig(steps=args.steps)
    args.output.mkdir(parents=True, exist_ok=True)

    all_jobs = [
        (encoding, seed)
        for encoding in args.encodings
        for seed in seeds
    ]
    sharded_jobs = [
        job for i, job in enumerate(all_jobs)
        if i % args.shard_count == args.shard_index
    ]

    total_runs = len(sharded_jobs)
    completed_runs = 0
    print(
        f"Starting shard [{args.shard_index}/{args.shard_count}]: "
        f"{total_runs} runs assigned."
    )

    for encoding, seed in sharded_jobs:
        model_name = (
            "interpolation_complete"
            if encoding == "angle"
            else f"interpolation_{encoding}_complete"
        )
        destination = args.output / f"{model_name}_seed{seed}.json"
        if args.skip_existing and destination.exists():
            print(f"skip {model_name} seed={seed}", flush=True)
            completed_runs += 1
            continue

        started = time.monotonic()
        print(
            f"[{completed_runs + 1}/{total_runs}] run {model_name} seed={seed}",
            flush=True,
        )
        result, metrics = run_interpolation(seed, config, encoding)
        save_interpolation_result(result, metrics, config, args.output)
        elapsed = time.monotonic() - started
        completed_runs += 1
        print(
            f"done {model_name} seed={seed} seconds={elapsed:.1f} "
            f"heldout_mmd={metrics['heldout_mmd']:.6f} "
            f"shell_tvd={metrics['points'][4]['shell_tvd']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

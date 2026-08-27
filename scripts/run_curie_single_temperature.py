#!/usr/bin/env python3
"""Inspect or execute the frozen Curie--Weiss single-temperature campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from iqp_cond.curie_single_temperature import (
    SingleTemperatureManifest,
    plan_single_temperature_runs,
)
from iqp_cond.curie_single_temperature_runner import describe_run, execute_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/curie_weiss_single_temperature_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/curie-weiss-single-temperature-v1"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--n-qubits", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def select_runs(
    manifest: SingleTemperatureManifest, args: argparse.Namespace
) -> list[object]:
    """Apply coordinate filters, then a deterministic disjoint shard."""
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index/count are invalid")
    if args.updates is not None and args.updates < 1:
        raise ValueError("updates must be positive")
    runs = list(plan_single_temperature_runs(manifest))
    if args.run_id:
        runs = [run for run in runs if run.run_id == args.run_id]
        if not runs:
            raise ValueError("run-id is not declared by the manifest")
    if args.n_qubits is not None:
        runs = [run for run in runs if run.n_qubits == args.n_qubits]
    if args.beta is not None:
        runs = [run for run in runs if run.beta == args.beta]
    if args.seed is not None:
        runs = [run for run in runs if run.seed == args.seed]
    runs = [
        run
        for index, run in enumerate(runs)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        runs = runs[: args.limit]
    return runs


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = Path.cwd().resolve()
    manifest = SingleTemperatureManifest.from_json(args.manifest)
    runs = select_runs(manifest, args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "count": len(runs),
                    "runs": [json.loads(describe_run(run)) for run in runs],
                },
                sort_keys=True,
            )
        )
        return 0

    invocation = tuple(sys.argv if argv is None else [sys.argv[0], *argv])
    for number, run in enumerate(runs, start=1):
        print(f"[{number}/{len(runs)}] {describe_run(run)}", flush=True)
        execute_run(
            repository,
            manifest,
            run,
            args.output,
            invocation=invocation,
            updates=args.updates,
            lightweight=args.lightweight,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

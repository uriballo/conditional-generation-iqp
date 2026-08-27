"""Inspect or execute the frozen confirmatory Curie--Weiss campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from iqp_cond.confirmatory_curie import (
    CurieConfirmatoryArm,
    CurieConfirmatoryManifest,
    plan_curie_confirmatory_runs,
)
from iqp_cond.confirmatory_curie_runner import describe_run, execute_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/curie_weiss_confirmatory_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/curie-weiss-confirmatory-v1"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--arm", choices=[arm.value for arm in CurieConfirmatoryArm])
    parser.add_argument("--n-qubits", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--lightweight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index/count are invalid")
    if args.updates is not None and args.updates < 1:
        raise ValueError("updates must be positive")
    repository = Path.cwd().resolve()
    manifest = CurieConfirmatoryManifest.from_json(args.manifest)
    runs = list(plan_curie_confirmatory_runs(manifest))
    if args.run_id:
        runs = [run for run in runs if run.run_id == args.run_id]
        if not runs:
            raise ValueError("run-id is not declared by the manifest")
    if args.arm:
        runs = [run for run in runs if run.arm is CurieConfirmatoryArm(args.arm)]
    if args.n_qubits is not None:
        runs = [run for run in runs if run.n_qubits == args.n_qubits]
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

    for number, run in enumerate(runs, start=1):
        print(f"[{number}/{len(runs)}] {describe_run(run)}", flush=True)
        execute_run(
            repository,
            manifest,
            run,
            args.output,
            invocation=tuple(sys.argv if argv is None else [sys.argv[0], *argv]),
            updates=args.updates,
            lightweight=args.lightweight,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the frozen eight-fit grid-Ising capacity pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from iqp_cond.grid_ising_capacity_pilot import (  # noqa: E402
    evaluate_run,
    identities,
    load_manifest,
    plan,
    prepare_data_archive,
    run_by_id,
    status,
    summarize,
    train_run,
)


def _launch_phase(
    action: str, manifest: Path, output: Path, run_ids: list[str], workers: int
) -> None:
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def invoke(run_id: str) -> None:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            action,
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--run-id",
            run_id,
        ]
        with (logs / f"{run_id}-{action}.log").open("a") as stream:
            subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(invoke, run_id): run_id for run_id in run_ids}
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                future.result()
            except Exception as error:
                raise RuntimeError(f"{action} failed for {run_id}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "plan",
            "prepare",
            "status",
            "train",
            "evaluate",
            "run",
            "summarize",
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY / "configs/grid_ising_capacity_pilot_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "results/grid-ising-capacity-pilot-v1",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config = load_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.action == "plan":
        result: object = plan(config)
    elif args.action == "prepare":
        path, digest = prepare_data_archive(config, args.output)
        result = {"path": str(path), "sha256": digest}
    elif args.action == "status":
        result = status(config, args.output)
    elif args.action in {"train", "evaluate"}:
        if not args.run_id:
            parser.error(f"{args.action} requires --run-id")
        run = run_by_id(config, args.run_id)
        result = (
            train_run(config, args.output, run)
            if args.action == "train"
            else evaluate_run(config, args.output, run)
        )
    elif args.action == "run":
        if args.workers < 1:
            parser.error("--workers must be positive")
        prepare_data_archive(config, args.output)
        runs = identities(config)
        pending_train = [
            run.run_id
            for run in runs
            if not (args.output / "training" / f"{run.run_id}.json").exists()
        ]
        _launch_phase("train", args.manifest, args.output, pending_train, args.workers)
        pending_evaluation = [
            run.run_id
            for run in runs
            if not (args.output / "evaluation" / f"{run.run_id}.json").exists()
        ]
        _launch_phase(
            "evaluate",
            args.manifest,
            args.output,
            pending_evaluation,
            args.workers,
        )
        result = summarize(config, args.output)
    else:
        result = summarize(config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

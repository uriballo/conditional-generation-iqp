#!/usr/bin/env python3
"""Run one durable, CPU-pinned campaign wave on a Strix host."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _cpu_ranges(workers: int) -> list[str]:
    cpus = os.cpu_count() or 1
    base, remainder = divmod(cpus, workers)
    result, start = [], 0
    for index in range(workers):
        width = base + (index < remainder)
        result.append(f"{start}-{start + width - 1}")
        start += width
    return result


def _jobs(args: argparse.Namespace) -> list[tuple[int, list[str]]]:
    python = sys.executable
    root = args.repository
    common = [
        "--manifest",
        str(args.manifest),
        "--data-directory",
        str(args.data_directory),
    ]
    jobs = []
    if args.phase in {"calibration", "optimizer"}:
        for global_index in range(90):
            if global_index % args.host_count != args.host_index:
                continue
            cell, candidate = divmod(global_index, 5)
            command = [
                python,
                str(root / "scripts/run_exact_tvd_development.py"),
                *common,
                "--output-directory",
                str(args.output_directory / "development"),
                args.phase,
                "--cell-index",
                str(cell),
                "--candidate-index",
                str(candidate),
            ]
            if args.phase == "optimizer":
                if args.n_samples is None:
                    raise ValueError("optimizer wave requires --n-samples")
                command += ["--n-samples", str(args.n_samples)]
            jobs.append((global_index, command))
        return jobs
    phase = "train" if args.phase == "training" else "evaluate"
    for run_index in range(args.max_production_runs):
        if run_index % args.host_count != args.host_index:
            continue
        command = [
            python,
            str(root / "scripts/run_exact_tvd_campaign.py"),
            *common,
            "--output-directory",
            str(args.output_directory),
            phase,
            "--run-index",
            str(run_index),
            "--resume",
        ]
        if phase == "train":
            command += ["--selection", str(args.selection)]
        jobs.append((run_index, command))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("calibration", "optimizer", "training", "evaluation")
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/exact_tvd_ancilla_v1.json")
    )
    parser.add_argument(
        "--data-directory", type=Path, default=Path("results/exact-tvd-ancilla-v1/data")
    )
    parser.add_argument(
        "--output-directory", type=Path, default=Path("results/exact-tvd-ancilla-v1")
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("results/exact-tvd-ancilla-v1/protocol-selection.json"),
    )
    parser.add_argument("--n-samples", type=int)
    parser.add_argument(
        "--max-production-runs",
        type=int,
        default=4300,
        help="Cap the frozen production plan; use 3900 to skip the MFC tail.",
    )
    parser.add_argument("--host-index", type=int, required=True)
    parser.add_argument("--host-count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    for name in (
        "repository",
        "manifest",
        "data_directory",
        "output_directory",
        "selection",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if not 0 <= args.host_index < args.host_count:
        raise SystemExit("invalid host shard")
    if not 1 <= args.max_production_runs <= 4300:
        raise SystemExit("max production runs must lie between 1 and 4300")
    jobs = _jobs(args)
    logs = args.output_directory / "logs" / args.phase
    logs.mkdir(parents=True, exist_ok=True)
    ranges = _cpu_ranges(args.workers)
    started = time.monotonic()

    def worker(worker_index: int) -> list[dict]:
        outcomes = []
        for job_index, command in jobs[worker_index :: args.workers]:
            log_path = logs / f"host{args.host_index}-job{job_index}.log"
            with log_path.open("ab") as stream:
                completed = subprocess.run(
                    ["taskset", "-c", ranges[worker_index], *command],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            outcomes.append(
                {
                    "job": job_index,
                    "returncode": completed.returncode,
                    "log": str(log_path),
                }
            )
            if completed.returncode:
                break
        return outcomes

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        nested = list(pool.map(worker, range(args.workers)))
    outcomes = [row for group in nested for row in group]
    failures = [row for row in outcomes if row["returncode"]]
    payload = {
        "schema_version": 1,
        "phase": args.phase,
        "host_index": args.host_index,
        "host_count": args.host_count,
        "workers": args.workers,
        "cpu_ranges": ranges,
        "planned_jobs": len(jobs),
        "completed_jobs": len(outcomes) - len(failures),
        "failures": failures,
        "runtime_seconds": time.monotonic() - started,
    }
    summary = (
        args.summary
        or args.output_directory / f"wave-{args.phase}-host{args.host_index}.json"
    )
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

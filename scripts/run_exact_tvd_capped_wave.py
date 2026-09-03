#!/usr/bin/env python3
"""Resume the frozen Exact-TVD training plan with an upper run-index cap."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _parse_cpus(value: str) -> list[int]:
    cpus = []
    for part in value.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    if not cpus or len(cpus) != len(set(cpus)):
        raise ValueError("CPU set must contain unique CPU indices")
    return cpus


def _ranges(cpus: list[int], workers: int) -> list[str]:
    base, remainder = divmod(len(cpus), workers)
    if base == 0:
        raise ValueError("workers cannot exceed assigned CPUs")
    result, cursor = [], 0
    for index in range(workers):
        width = base + (index < remainder)
        result.append(",".join(map(str, cpus[cursor : cursor + width])))
        cursor += width
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--host-index", type=int, required=True)
    parser.add_argument("--host-count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--cpu-set", default="0-26")
    parser.add_argument("--min-runs", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=3900)
    parser.add_argument(
        "--phase", choices=("training", "evaluation", "both"), default="training"
    )
    args = parser.parse_args()
    if not 0 <= args.host_index < args.host_count:
        raise SystemExit("invalid host shard")
    if not 0 <= args.min_runs < args.max_runs <= 3900:
        raise SystemExit("the capped wave may include only Gaussian and Curie-Weiss")
    jobs = [
        index
        for index in range(args.min_runs, args.max_runs)
        if index % args.host_count == args.host_index
    ]
    cpus = _parse_cpus(args.cpu_set)
    ranges = _ranges(cpus, args.workers)
    threads_per_worker = max(1, len(cpus) // args.workers)
    logs = args.output_directory / "logs" / f"{args.phase}-capped-no-mfc"
    logs.mkdir(parents=True, exist_ok=True)
    cache = args.output_directory / ".jax-compilation-cache"
    cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def command(run_index: int, phase: str) -> list[str]:
        values = [
            sys.executable,
            str(args.repository / "scripts/run_exact_tvd_campaign.py"),
            "--manifest",
            str(args.repository / "configs/exact_tvd_ancilla_v1.json"),
            "--data-directory",
            str(args.data_directory),
            "--output-directory",
            str(args.output_directory),
            "train" if phase == "training" else "evaluate",
            "--run-index",
            str(run_index),
            "--resume",
        ]
        if phase == "training":
            values += ["--selection", str(args.selection)]
        return values

    def worker(worker_index: int) -> list[dict]:
        outcomes = []
        environment = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = str(threads_per_worker)
        worker_cache = cache / f"host{args.host_index}-worker{worker_index}"
        worker_cache.mkdir(parents=True, exist_ok=True)
        environment["JAX_COMPILATION_CACHE_DIR"] = str(worker_cache)
        environment["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
        for run_index in jobs[worker_index :: args.workers]:
            log = logs / f"host{args.host_index}-job{run_index}.log"
            returncode = 0
            with log.open("ab") as stream:
                phases = (
                    ("training", "evaluation")
                    if args.phase == "both"
                    else (args.phase,)
                )
                for phase in phases:
                    completed = subprocess.run(
                        [
                            "taskset",
                            "-c",
                            ranges[worker_index],
                            *command(run_index, phase),
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                        env=environment,
                    )
                    returncode = completed.returncode
                    if returncode:
                        break
            outcomes.append(
                {"job": run_index, "returncode": returncode, "log": str(log)}
            )
            if returncode:
                break
        return outcomes

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = [
            row for group in pool.map(worker, range(args.workers)) for row in group
        ]
    failures = [row for row in outcomes if row["returncode"]]
    payload = {
        "schema_version": 1,
        "phase": args.phase,
        "scope": "gaussian_curie_only",
        "min_runs": args.min_runs,
        "max_runs": args.max_runs,
        "host_index": args.host_index,
        "workers": args.workers,
        "threads_per_worker": threads_per_worker,
        "cpu_ranges": ranges,
        "completed_jobs": len(outcomes) - len(failures),
        "failures": failures,
        "runtime_seconds": time.monotonic() - started,
    }
    summary = args.output_directory / (
        f"wave-{args.phase}-{args.min_runs}-{args.max_runs}-host{args.host_index}.json"
    )
    summary.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark 2/4/6 CPU-pinned campaign workers and record thermal headroom."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _temperatures() -> list[float]:
    values = []
    for directory in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            sensor_name = (directory / "name").read_text().strip()
        except OSError:
            continue
        if sensor_name != "k10temp":
            continue
        for path in directory.glob("temp*_input"):
            try:
                value = float(path.read_text().strip()) / 1000.0
            except (OSError, ValueError):
                continue
            if 0.0 < value < 150.0:
                values.append(value)
    return values


def _cpu_ranges(workers: int) -> list[str]:
    cpus = os.cpu_count() or 1
    if workers > cpus:
        raise ValueError("worker count exceeds logical CPUs")
    base, remainder = divmod(cpus, workers)
    result = []
    start = 0
    for index in range(workers):
        width = base + (1 if index < remainder else 0)
        result.append(f"{start}-{start + width - 1}")
        start += width
    return result


def benchmark(args: argparse.Namespace, workers: int) -> dict:
    started = time.monotonic()
    maximum_temperature = max(_temperatures(), default=float("nan"))
    processes = []
    with tempfile.TemporaryDirectory(prefix=f"exact-tvd-bench-{workers}-") as output:
        for index, cpu_range in enumerate(_cpu_ranges(workers)):
            command = [
                "taskset",
                "-c",
                cpu_range,
                sys.executable,
                str(args.repository / "scripts/run_exact_tvd_development.py"),
                "--manifest",
                str(args.manifest),
                "--data-directory",
                str(args.data_directory),
                "--output-directory",
                output,
                "calibration",
                "--cell-index",
                str(index % 18),
                "--candidate-index",
                "0",
                "--repeats",
                "2",
            ]
            processes.append(
                subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            )
        while any(process.poll() is None for process in processes):
            maximum_temperature = max(
                maximum_temperature, max(_temperatures(), default=float("nan"))
            )
            time.sleep(0.5)
        failures = []
        for index, process in enumerate(processes):
            stdout, stderr = process.communicate()
            if process.returncode:
                failures.append(
                    {
                        "worker": index,
                        "returncode": process.returncode,
                        "stdout": stdout.decode(errors="replace")[-2000:],
                        "stderr": stderr.decode(errors="replace")[-2000:],
                    }
                )
    elapsed = time.monotonic() - started
    return {
        "workers": workers,
        "logical_cpus": os.cpu_count(),
        "cpu_ranges": _cpu_ranges(workers),
        "elapsed_seconds": elapsed,
        "cells_per_hour": workers * 3600.0 / elapsed,
        "maximum_temperature_c": maximum_temperature,
        "below_thermal_limit": maximum_temperature < args.maximum_temperature,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/exact_tvd_ancilla_v1.json")
    )
    parser.add_argument(
        "--data-directory", type=Path, default=Path("results/exact-tvd-ancilla-v1/data")
    )
    parser.add_argument("--worker-counts", type=int, nargs="+", default=(2, 4, 6))
    parser.add_argument("--maximum-temperature", type=float, default=90.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.repository = args.repository.resolve()
    args.manifest = args.manifest.resolve()
    args.data_directory = args.data_directory.resolve()
    results = [benchmark(args, workers) for workers in args.worker_counts]
    eligible = [
        row for row in results if row["below_thermal_limit"] and not row["failures"]
    ]
    selected = (
        max(eligible, key=lambda row: row["cells_per_hour"])["workers"]
        if eligible
        else None
    )
    payload = {"schema_version": 1, "results": results, "selected_workers": selected}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if selected is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

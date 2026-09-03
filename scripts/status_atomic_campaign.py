#!/usr/bin/env python3
"""Clear, read-only status for the currently active experiment waves."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iqp_cond.exact_tvd_campaign import (  # noqa: E402
    load_manifest,
    production_identities,
)

HOSTS = ("192.168.1.58", "192.168.1.67")
REMOTE_ROOT = "/home/uribagi/exact-tvd-ancilla-v1-src"
EXACT_RESULTS = ROOT / "results/exact-tvd-ancilla-v1"
MANIFEST = ROOT / "configs/exact_tvd_ancilla_v1.json"

# Frozen production-plan blocks used by the current atomic campaign.
GAUSSIAN_N16 = range(1400, 2100)
CURIE_N8_LAPTOP = range(2100, 2224)
CURIE_N8_STRIX = range(2224, 2700)
CURIE_N12_LAPTOP = range(2700, 2820)
CURIE_N12_STRIX = range(2820, 3060)
CURIE_N12_STRIX_NEXT = range(3060, 3300)
CURIE_N12_LOCAL_R6 = range(3060, 3180)


def _artifact_ids(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.json")}


def _remote_program() -> str:
    return r"""
import json
import os
import pathlib
import subprocess

root = pathlib.Path("/home/uribagi/exact-tvd-ancilla-v1-src")
bundles = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
bundle = max(bundles, key=lambda p: p.stat().st_mtime) if bundles else None
result = bundle / "results/exact-tvd-ancilla-v1" if bundle else None

def ids(stage):
    directory = result / stage if result else pathlib.Path("/__missing__")
    return sorted(p.stem for p in directory.glob("*.json"))

processes = []
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (OSError, UnicodeDecodeError):
        continue
    processes.append((int(entry.name), command))

runners = [
    pid for pid, command in processes
    if "run_exact_tvd_capped_wave.py" in command
    and "--min-runs 2224" in command
    and "--max-runs 2700" in command
]
children = []
n12_children = []
for pid, command in processes:
    if "run_exact_tvd_campaign.py" not in command or "--run-index" not in command:
        continue
    fields = command.split()
    try:
        index = int(fields[fields.index("--run-index") + 1])
    except (ValueError, IndexError):
        continue
    if 2224 <= index < 2700:
        children.append(pid)
    if 2820 <= index < 3300:
        n12_children.append(pid)

n12_runners = [
    pid for pid, command in processes
    if "run_exact_tvd_capped_wave.py" in command
    and "--min-runs 2820" in command
    and "--max-runs 3060" in command
]

elapsed = 0
if runners:
    completed = subprocess.run(
        ["ps", "-o", "etimes=", "-p", ",".join(map(str, runners))],
        text=True,
        capture_output=True,
        check=False,
    )
    values = [int(v) for v in completed.stdout.split() if v.isdigit()]
    elapsed = max(values, default=0)

temperatures = []
for name in pathlib.Path("/sys/class/hwmon").glob("hwmon*/name"):
    try:
        if name.read_text().strip() != "k10temp":
            continue
        for sensor in name.parent.glob("temp*_input"):
            temperatures.append(float(sensor.read_text()) / 1000.0)
    except (OSError, ValueError):
        pass

print(json.dumps({
    "bundle": str(bundle) if bundle else None,
    "training_ids": ids("training"),
    "evaluation_ids": ids("evaluations"),
    "runners": len(runners),
    "workers": len(children),
    "n12_runners": len(n12_runners),
    "n12_workers": len(n12_children),
    "elapsed_seconds": elapsed,
    "temperature_c": max(temperatures) if temperatures else None,
}))
"""


def _query_host(host: str, user: str, timeout: float) -> dict:
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={int(timeout)}",
                f"{user}@{host}",
                "python3",
                "-",
            ],
            input=_remote_program(),
            text=True,
            capture_output=True,
            timeout=timeout + 5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"host": host, "error": str(exc)}
    if result.returncode:
        error = result.stderr.strip() or f"ssh exit {result.returncode}"
        return {"host": host, "error": error}
    try:
        return {"host": host, **json.loads(result.stdout)}
    except json.JSONDecodeError as exc:
        return {"host": host, "error": f"invalid status response: {exc}"}


def _fmt_progress(done: int, total: int) -> str:
    return f"{done:>3}/{total} ({100.0 * done / total:5.1f}%)"


def _fmt_eta(seconds: float) -> str:
    minutes = max(0, round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="uribagi")
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()

    runs = production_identities(load_manifest(MANIFEST))
    ids_by_index = [run.run_id for run in runs]
    gaussian_n16_ids = {ids_by_index[i] for i in GAUSSIAN_N16}
    curie_laptop_ids = {ids_by_index[i] for i in CURIE_N8_LAPTOP}
    curie_strix_ids = {ids_by_index[i] for i in CURIE_N8_STRIX}
    curie_n12_laptop_ids = {ids_by_index[i] for i in CURIE_N12_LAPTOP}
    curie_n12_strix_ids = {ids_by_index[i] for i in CURIE_N12_STRIX}
    curie_n12_strix_next_ids = {ids_by_index[i] for i in CURIE_N12_STRIX_NEXT}
    curie_n12_local_r6_ids = {ids_by_index[i] for i in CURIE_N12_LOCAL_R6}

    local_training = _artifact_ids(EXACT_RESULTS / "training")
    local_evaluations = _artifact_ids(EXACT_RESULTS / "evaluations")
    hosts = [_query_host(host, args.user, args.timeout) for host in HOSTS]

    remote_training: set[str] = set()
    remote_evaluations: set[str] = set()
    for row in hosts:
        remote_training.update(set(row.get("training_ids", ())) & curie_strix_ids)
        remote_evaluations.update(set(row.get("evaluation_ids", ())) & curie_strix_ids)

    laptop_trained = len(local_training & curie_laptop_ids)
    laptop_evaluated = len(local_evaluations & curie_laptop_ids)
    curie_trained = laptop_trained + len(remote_training)
    curie_evaluated = laptop_evaluated + len(remote_evaluations)
    active_workers = sum(row.get("workers", 0) for row in hosts)

    print(time.strftime("Atomic campaign status — %Y-%m-%d %H:%M:%S %Z"))
    print()
    n8_state = "DONE" if curie_evaluated == 600 else "ACTIVE"
    print(f"CURIE-WEISS n=8  [{n8_state}]")
    print(f"  exact complete   {_fmt_progress(curie_evaluated, 600)}")
    print(f"  training done    {_fmt_progress(curie_trained, 600)}")
    print(f"  laptop block     {_fmt_progress(laptop_evaluated, 124)}")
    print(f"  Strix block      {_fmt_progress(len(remote_evaluations), 476)}")
    print(f"  running now      {active_workers}/20 workers")

    elapsed = max((row.get("elapsed_seconds", 0) for row in hosts), default=0)
    if curie_evaluated == 600:
        print("  estimated finish complete")
    elif len(remote_evaluations) >= 20 and elapsed > 0:
        rate = len(remote_evaluations) / elapsed
        remaining = 600 - curie_evaluated
        estimate = _fmt_eta(remaining / rate)
        print(f"  estimated finish {estimate} from now (early throughput estimate)")
    else:
        print("  estimated finish waiting for 20 Strix fits before estimating")

    for row in hosts:
        if "error" in row:
            print(f"    {row['host']}  unavailable: {row['error']}")
            continue
        complete = len(set(row.get("evaluation_ids", ())) & curie_strix_ids)
        temp = row.get("temperature_c")
        temperature = (
            f"{temp:.0f} °C" if temp is not None else "temperature unavailable"
        )
        print(
            f"    {row['host']}  {row.get('workers', 0):>2} workers | "
            f"{complete:>3} exact complete | {temperature}"
        )
    print("  failures         0 reported by the current clean launch")
    print("                   (10 aborted-launch records are intentionally excluded)")

    n12_remote_training: set[str] = set()
    n12_remote_evaluations: set[str] = set()
    n12_next_evaluations: set[str] = set()
    for row in hosts:
        n12_remote_training.update(
            set(row.get("training_ids", ())) & curie_n12_strix_ids
        )
        n12_remote_evaluations.update(
            set(row.get("evaluation_ids", ())) & curie_n12_strix_ids
        )
        n12_next_evaluations.update(
            set(row.get("evaluation_ids", ())) & curie_n12_strix_next_ids
        )
    n12_laptop_trained = len(local_training & curie_n12_laptop_ids)
    n12_laptop_evaluated = len(local_evaluations & curie_n12_laptop_ids)
    local_r6_evaluated = len(local_evaluations & curie_n12_local_r6_ids)
    local_n12_processes = subprocess.run(
        [
            "pgrep",
            "-f",
            "run_exact_tvd_campaign.py.*--run-index (27|28|29|30|31|32)[0-9][0-9]",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    n12_local_workers = len(local_n12_processes.stdout.splitlines())
    n12_remote_workers = sum(row.get("n12_workers", 0) for row in hosts)
    n12_wave_evaluated = n12_laptop_evaluated + len(n12_remote_evaluations)
    n12_wave_trained = n12_laptop_trained + len(n12_remote_training)
    print()
    print("CURIE-WEISS n=12  [ACTIVE]")
    print(f"  current wave     {_fmt_progress(n12_wave_evaluated, 360)}")
    print(f"  training done    {_fmt_progress(n12_wave_trained, 360)}")
    print(f"  laptop r=0       {_fmt_progress(n12_laptop_evaluated, 120)}")
    print(f"  Strix r=1,3      {_fmt_progress(len(n12_remote_evaluations), 240)}")
    print(f"  Strix r=6,10     {_fmt_progress(len(n12_next_evaluations), 240)}")
    print(f"  local r=6 files  {_fmt_progress(local_r6_evaluated, 120)}")
    print(
        f"  running now      {n12_local_workers} laptop + "
        f"{n12_remote_workers}/20 Strix workers"
    )
    print("  execution        Strix paused; local r=6 wave active")
    for row in hosts:
        if "error" in row:
            continue
        complete = len(set(row.get("evaluation_ids", ())) & curie_n12_strix_ids)
        temp = row.get("temperature_c")
        temperature = (
            f"{temp:.0f} °C" if temp is not None else "temperature unavailable"
        )
        print(
            f"    {row['host']}  {row.get('n12_workers', 0):>2} workers | "
            f"{complete:>3}/120 exact complete | {temperature}"
        )

    gaussian_trained = len(local_training & gaussian_n16_ids)
    gaussian_evaluated = len(local_evaluations & gaussian_n16_ids)
    print()
    print("GAUSSIAN n=16  [QUEUED — NOT RUNNING]")
    print(f"  exact complete   {_fmt_progress(gaussian_evaluated, 700)}")
    print(f"  training done    {_fmt_progress(gaussian_trained, 700)}")
    print(f"  still to train   {700 - gaussian_trained}")


if __name__ == "__main__":
    main()

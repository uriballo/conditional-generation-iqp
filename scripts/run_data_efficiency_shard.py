#!/usr/bin/env python3
"""Execution-only four-worker CPU shard; all scientific choices are frozen."""

from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--host-id", required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    tasks = queue.Queue()
    seeds = [int(x) for x in a.seeds.split(",")]
    for seed in seeds:
        for size in (100, 300, 1000):
            tasks.put(("classical", seed, size, 0))
    for seed in seeds:
        for size in (100, 300, 1000):
            for r in (0, 2, 4):
                for e in ("cat", "bond"):
                    tasks.put((e, seed, size, r))
    failed = []
    complete = []
    lock = threading.Lock()
    stop = threading.Event()
    affinities = ("0-3,16-19", "4-7,20-23", "8-11,24-27", "12-15,28-31")
    status = a.output / f"shard-{a.host_id}.json"

    def record(state):
        payload = {
            "status": state,
            "hostname": socket.gethostname(),
            "seeds": seeds,
            "workers": 4,
            "affinities": affinities,
            "complete_jobs": list(complete),
            "failed_jobs": list(failed),
            "remaining_queued": tasks.qsize(),
            "updated_unix": time.time(),
        }
        temp = status.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temp, status)

    def worker(slot):
        env = {
            **os.environ,
            "PYTHONPATH": str(root / "src"),
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "1",
        }
        while not stop.is_set():
            try:
                e, seed, size, r = tasks.get_nowait()
            except queue.Empty:
                return
            name = (
                f"classical-s{size}-{seed}"
                if e == "classical"
                else f"{e}-s{size}-r{r}-{seed}"
            )
            cmd = [
                "taskset",
                "-c",
                affinities[slot],
                sys.executable,
                str(root / "scripts/ashkin_teller_data_efficiency.py"),
                "classical" if e == "classical" else "train",
                "--output",
                str(a.output),
                "--seed",
                str(seed),
                "--size",
                str(size),
            ]
            if e != "classical":
                cmd += ["--encoder", e, "--ancillas", str(r)]
            print("START", name, "slot", slot, flush=True)
            with (a.output / (name + ".log")).open("a") as log:
                result = subprocess.run(
                    cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            with lock:
                if result.returncode:
                    failed.append({"name": name, "exit": result.returncode})
                    stop.set()
                else:
                    complete.append(name)
                record("Failed" if failed else "Running")
            tasks.task_done()
            print("FINISH", name, "exit", result.returncode, flush=True)

    a.output.mkdir(parents=True, exist_ok=True)
    record("Running")
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(worker, range(4)))
    record("Failed" if failed or not tasks.empty() else "Complete")
    if failed or not tasks.empty():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

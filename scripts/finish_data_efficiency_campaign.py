#!/usr/bin/env python3
"""Synchronize both shards, then seal/evaluate/report only complete predictions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from iqp_cond.artifacts import write_immutable_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--remote-root", required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = a.directory.resolve()
    hosts = ("192.168.1.58", "192.168.1.67")
    remote_out = a.remote_root + "/artifacts/ashkin-teller-data-efficiency-2026-09-07"
    status = out / "completion-status.json"

    def record(state, **extra):
        temporary = status.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {"status": state, "updated_unix": time.time(), **extra}, indent=2
            )
            + "\n"
        )
        os.replace(temporary, status)

    failures = 0
    try:
        while True:
            states = []
            for host in hosts:
                result = subprocess.run(
                    [
                        "rsync",
                        "-az",
                        "--exclude=source.tar.gz",
                        "--exclude=data.npz",
                        "--exclude=manifest.json",
                        f"uribagi@{host}:{remote_out}/",
                        str(out) + "/",
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    failures += 1
                    record("Sync retry", host=host, error=result.stderr[-1000:])
                    break
                shard = json.loads((out / f"shard-{host}.json").read_text())
                states.append(shard)
            if len(states) != 2:
                if failures >= 10:
                    raise RuntimeError(
                        "Ten synchronization failures; inspect remote processes"
                    )
                time.sleep(60)
                continue
            failures = 0
            if any(s["status"] == "Failed" for s in states):
                raise RuntimeError("A shard failed; targets remain closed")
            count = sum(len(s["complete_jobs"]) for s in states)
            record("Running", completed_bundles=count, expected_bundles=210)
            print("PROGRESS", count, "/210 bundles", flush=True)
            if all(s["status"] == "Complete" for s in states):
                break
            time.sleep(60)
        cmd = [sys.executable, str(root / "scripts/ashkin_teller_data_efficiency.py")]
        env = {**os.environ, "PYTHONPATH": str(root / "src")}
        record("Verifying prediction seal")
        subprocess.run(
            [*cmd, "seal", "--output", str(out)], cwd=root, env=env, check=True
        )
        from ashkin_teller_n12 import digest

        marker = {"seal_sha256": digest(out / "predictions-seal.json")}
        write_immutable_json(out / "evaluation-started.json", marker)
        # Close both remote fitting entry points before any new interior target call.
        for host in hosts:
            subprocess.run(
                [
                    "scp",
                    str(out / "evaluation-started.json"),
                    f"uribagi@{host}:{remote_out}/evaluation-started.json",
                ],
                check=True,
            )
        record("Evaluating sealed predictions")
        subprocess.run(
            [*cmd, "evaluate", "--output", str(out)], cwd=root, env=env, check=True
        )
        record("Auditing numerical report")
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts/report_ashkin_teller_data_efficiency.py"),
                "--directory",
                str(out),
                "--output",
                str(out / "report"),
                "--no-plots",
            ],
            cwd=root,
            env=env,
            check=True,
        )
        record(
            "Needs review",
            completed_bundles=210,
            reason="Numerical report complete; figures and manuscript need review",
        )
        print(
            "ALL RESULTS VERIFIED; READY FOR FIGURE AND MANUSCRIPT REVIEW", flush=True
        )
    except Exception as exc:
        record("Needs review", error=str(exc))
        raise


if __name__ == "__main__":
    main()

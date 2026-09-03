#!/usr/bin/env python3
# ruff: noqa: E501
"""Read-only progress monitor for the Exact-TVD Strix campaign."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time

DEFAULT_HOSTS = ("192.168.1.58", "192.168.1.67")
REMOTE_ROOT = "/home/uribagi/exact-tvd-ancilla-v1-src"


def _remote_script(root: str) -> str:
    q = shlex.quote(root)
    return f'''root={q}
bundle=$(find "$root" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\\n' 2>/dev/null | sort -nr | awk 'NR==1 {{sub(/^[^ ]+ /, ""); print}}')
if [ -z "$bundle" ]; then echo NO_BUNDLE; exit 0; fi
count() {{ find "$bundle/$1" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' '; }}
active=$(pgrep -af 'run_exact_tvd_(campaign|development|strix_wave|capped_wave)' 2>/dev/null | wc -l | tr -d ' ')
phases=$(pgrep -af 'run_exact_tvd_(strix|capped)_wave.py' 2>/dev/null | awk '{{for (i=1; i<=NF; i++) if ($i ~ /run_exact_tvd_(strix|capped)_wave.py$/) print $(i+1)}}' | grep -E '^(calibration|optimizer|training|evaluation)$' | sort -u | paste -sd, -)
failures=$(python3 - "$bundle/results/exact-tvd-ancilla-v1" <<'PY'
import json
import pathlib
import sys

total = 0
for path in pathlib.Path(sys.argv[1]).glob("wave-*.json"):
    try:
        total += len(json.loads(path.read_text()).get("failures", []))
    except (OSError, ValueError):
        pass
print(total)
PY
)
printf 'BUNDLE\\t%s\\nACTIVE\\t%s\\nPHASES\\t%s\\nFAILURES\\t%s\\nOPTIMIZER\\t%s\\nTRAINING\\t%s\\nEVALUATION\\t%s\\n' "$bundle" "$active" "$phases" "$failures" "$(count results/exact-tvd-ancilla-v1/development/optimizer-rows)" "$(count results/exact-tvd-ancilla-v1/training)" "$(count results/exact-tvd-ancilla-v1/evaluations)"
'''


def query_host(host: str, user: str, timeout: float) -> dict:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={int(timeout)}",
        f"{user}@{host}",
        "bash",
        "-s",
    ]
    try:
        result = subprocess.run(
            cmd,
            input=_remote_script(REMOTE_ROOT),
            text=True,
            capture_output=True,
            timeout=timeout + 5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"host": host, "error": str(exc)}
    if result.returncode:
        return {"host": host, "error": result.stderr.strip() or f"ssh exit {result.returncode}"}
    data = {"host": host}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("\t")
        if key in {"ACTIVE", "FAILURES", "OPTIMIZER", "TRAINING", "EVALUATION"}:
            data[key.lower()] = int(value or 0)
        elif key:
            data[key.lower()] = value
    return data


def snapshot(hosts: tuple[str, ...], user: str, timeout: float) -> dict:
    rows = [query_host(host, user, timeout) for host in hosts]
    return {
        "hosts": rows,
        "aggregate": {
            key: sum(row.get(key, 0) for row in rows)
            for key in ("failures", "optimizer", "training", "evaluation")
        },
    }


def _print_human(report: dict) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    for row in report["hosts"]:
        if "error" in row:
            print(f"{row['host']}: unavailable ({row['error']})")
            continue
        state = (
            "idle"
            if row.get("active", 0) == 0
            else (row.get("phases") or "active")
        )
        print(
            f"{row['host']}: {state}; "
            f"optimizer {row.get('optimizer', 0)}/45; "
            f"training {row.get('training', 0)}/1950; "
            f"evaluation {row.get('evaluation', 0)}/1950; "
            f"failures {row.get('failures', 0)}"
        )
    aggregate = report["aggregate"]
    print(
        f"TOTAL: optimizer {aggregate['optimizer']}/90; "
        f"training {aggregate['training']}/3900; "
        f"evaluation {aggregate['evaluation']}/3900; "
        f"failures {aggregate['failures']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="uribagi")
    parser.add_argument("--host", dest="hosts", action="append")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--watch", type=float, metavar="SECONDS")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    while True:
        report = snapshot(tuple(args.hosts or DEFAULT_HOSTS), args.user, args.timeout)
        if args.as_json:
            print(json.dumps(report, indent=2))
        else:
            _print_human(report)
        if args.watch is None:
            return
        time.sleep(max(0.1, args.watch))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only progress and resource dashboard for the current Strix queues."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from iqp_cond.exact_tvd_campaign import (  # noqa: E402
    load_manifest,
    production_identities,
)

REMOTE_ROOT = (
    "/home/uribagi/exact-tvd-ancilla-v1-src/"
    "614cbc5eb7dab1af29f4fa39fc8e27e745be9579f4cfaa2ee43822b880435562"
)
HOSTS = ("58", "67")

# Sent over stdin, not installed on the hosts. All operations are read-only.
PROBE = r"""
import base64, json, os, time, zlib
from pathlib import Path
root = Path(__ROOT__)
host = __HOST__
out = root / 'results/exact-tvd-ancilla-v1'
control = out / 'controls/remaining-20260903'
plan = json.loads((control / f'queue-{host}.json').read_text())
status_path = control / f'queue-{host}.status.json'
state = json.loads(status_path.read_text()) if status_path.exists() else {}
def cpu():
    v = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    return sum(v), v[3] + v[4]
before = cpu()
workers, schedulers = [], []
uptime = float(Path('/proc/uptime').read_text().split()[0])
ticks, page_size = os.sysconf('SC_CLK_TCK'), os.sysconf('SC_PAGE_SIZE')
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = (proc/'cmdline').read_bytes().decode().split('\0')
        names = {Path(arg).name for arg in args if arg}
        if 'runner.py' in names and f'queue-{host}.json' in names:
            schedulers.append(int(proc.name))
        if 'run_exact_tvd_campaign.py' not in names or '--run-index' not in args:
            continue
        index = int(args[args.index('--run-index')+1])
        if not 0 <= index < 3900:
            continue
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        elapsed = max(0.001, uptime - int(fields[19])/ticks)
        workers.append(dict(pid=int(proc.name), index=index,
            phase='train' if 'train' in args else 'evaluate',
            elapsed=elapsed, rss_gib=int(fields[21])*page_size/2**30,
            cpu_avg=100*(int(fields[11])+int(fields[12]))/ticks/elapsed,
            cpus=sorted(os.sched_getaffinity(int(proc.name)))))
    except (OSError, ValueError, IndexError):
        continue
memory = dict(line.split(':', 1) for line in
              Path('/proc/meminfo').read_text().splitlines())
total = int(memory['MemTotal'].split()[0])/1024**2
available = int(memory['MemAvailable'].split()[0])/1024**2
temps = []
for label in Path('/sys/class/hwmon').glob('hwmon*/temp*_label'):
    try:
        if label.read_text().strip() == 'Tctl':
            sensor = label.with_name(label.name.replace('_label','_input'))
            temps.append(float(sensor.read_text())/1000)
    except (OSError, ValueError):
        pass
time.sleep(0.2)
after = cpu()
delta = after[0]-before[0]
payload = dict(plan=plan, state=state, workers=workers,
    scheduler_pids=schedulers, temperature_c=max(temps) if temps else None,
    cpu_busy=100*(1-(after[1]-before[1])/delta) if delta else None,
    logical_cpus=os.cpu_count(), memory_total=total, memory_available=available,
    memory_used=total-available,
    training_ids=[p.stem for p in (out/'training').glob('*.json')],
    evaluation_ids=[p.stem for p in (out/'evaluations').glob('*.json')])
print(base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode())
"""


def read(host):
    program = PROBE.replace("__ROOT__", repr(REMOTE_ROOT)).replace(
        "__HOST__", repr(host)
    )
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                f"uribagi@192.168.1.{host}",
                "python3",
                "-",
            ],
            input=program,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return host, json.loads(zlib.decompress(base64.b64decode(result.stdout)))
    except (subprocess.SubprocessError, OSError, ValueError, zlib.error) as exc:
        return host, {"error": str(exc)}


def campaign_progress(runs, snapshots, local_training=(), local_evaluations=()):
    trained, evaluated = set(local_training), set(local_evaluations)
    for snapshot in snapshots.values():
        trained.update(snapshot.get("training_ids", []))
        evaluated.update(snapshot.get("evaluation_ids", []))
    rows = []
    for dataset in ("gaussian", "curie_weiss"):
        for n in (8, 12, 16):
            ids = {
                r["run_id"]
                for r in runs
                if r["dataset"] == dataset and r["visible_qubits"] == n
            }
            rows.append(
                dict(
                    dataset=dataset,
                    n=n,
                    total=len(ids),
                    trained=len(ids & trained),
                    evaluated=len(ids & evaluated),
                )
            )
    return rows


def duration(seconds):
    if seconds is None:
        return "n/a"
    minutes = max(0, int(seconds)) // 60
    return f"{minutes // 60}h{minutes % 60:02d}m"


def number(value, suffix="", digits=0):
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def warnings(snapshot):
    state = snapshot.get("state", {})
    result = []
    if state.get("status") in ("running", "waiting", "backfilling"):
        if not snapshot.get("scheduler_pids"):
            result.append("scheduler is absent; saved state is stale")
    live = {w["pid"] for w in snapshot.get("workers", [])}
    missing = [
        v["pid"] for v in state.get("active", {}).values() if v["pid"] not in live
    ]
    if missing:
        result.append(f"saved worker PIDs no longer live: {missing}")
    return result


def format_worker(worker, run, group):
    family = "shared" if run["family"] == "conditioned" else "indep"
    condition = run.get("condition")
    role = run.get("condition_role", "")
    target = "all train" if condition is None else f"{condition:g}/{role}"
    dataset = "CW" if run["dataset"] == "curie_weiss" else "G"
    return (
        f"  #{worker['index']:<4} {group:<5} {worker['phase']:<8} "
        f"{dataset} n{run['visible_qubits']} r{run['ancillas']:<2} {family:<6} "
        f"s{run['seed']} {target:<13} {duration(worker.get('elapsed')):>6} "
        f"{len(worker.get('cpus', [])):>2} thr "
        f"{number(worker.get('cpu_avg'), '%'):>5} "
        f"{number(worker.get('rss_gib'), 'G', 1):>5}"
    )


def render(snapshots, runs, progress, compact=False, now=None):
    now = time.time() if now is None else now
    stamp = dt.datetime.fromtimestamp(now).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = ["Exact-TVD campaign | " + stamp]
    total = sum(row["total"] for row in progress)
    trained = sum(row["trained"] for row in progress)
    evaluated = sum(row["evaluated"] for row in progress)
    offline = any("error" in s for s in snapshots.values())
    suffix = " [lower bound: host unavailable]" if offline else ""
    lines.append(
        f"FULL CAMPAIGN: trained {trained}/{total} | exact {evaluated}/{total}{suffix}"
    )
    lines.append("Dataset          Training      Exact done    To train / to evaluate")
    for row in progress:
        name = "Gaussian" if row["dataset"] == "gaussian" else "Curie-Weiss"
        lines.append(
            f"{name:<11} n{row['n']:<2}  {row['trained']:>4}/{row['total']:<4}"
            f"     {row['evaluated']:>4}/{row['total']:<4}       "
            f"{row['total'] - row['trained']:>4} / {row['total'] - row['evaluated']}"
        )
    for host, snapshot in snapshots.items():
        lines.append("")
        if "error" in snapshot:
            lines.append(f".{host} OFFLINE: {snapshot['error'][:150]}")
            continue
        plan, state = snapshot["plan"], snapshot["state"]
        workers = snapshot.get("workers", [])
        active = {v["pid"] for v in state.get("active", {}).values()}
        queue_workers = sum(w["pid"] in active for w in workers)
        pinned = {cpu for w in workers for cpu in w.get("cpus", [])}
        lines.append(
            f".{host} {state.get('status', 'unknown').upper()} | "
            f"{state.get('block', 'waiting for old wave')} | {state.get('phase', '-')}"
        )
        lines.append(
            f"  CPU {number(snapshot.get('cpu_busy'), '%')} busy | "
            f"pinned {len(pinned)}/{snapshot.get('logical_cpus', '?')} threads | "
            f"temp {number(snapshot.get('temperature_c'), ' C')} | "
            f"RAM {number(snapshot.get('memory_used'), '', 1)}/"
            f"{number(snapshot.get('memory_total'), ' GiB', 1)} used, "
            f"{number(snapshot.get('memory_available'), ' GiB', 1)} available"
        )
        lines.append(
            f"  Workers {len(workers)} total: {queue_workers} queue + "
            f"{len(workers) - queue_workers} old/other | "
            f"failures {len(state.get('failures', []))}"
        )
        jobs = [j for b in plan["blocks"] for j in b["jobs"]]
        train_ids, eval_ids = (
            set(snapshot["training_ids"]),
            set(snapshot["evaluation_ids"]),
        )
        t = sum(j["run_id"] in train_ids for j in jobs)
        e = sum(j["run_id"] in eval_ids for j in jobs)
        deadline = plan.get("deadline")
        cutoff = (
            dt.datetime.fromtimestamp(deadline).astimezone().strftime("%H:%M %Z")
            if deadline
            else "n/a"
        )
        countdown = (
            duration(deadline - now) if deadline and deadline > now else "reached"
        )
        lines.append(
            f"  Host queue: trained {t}/{len(jobs)}, exact {e}/{len(jobs)} | "
            f"cutoff {cutoff}, in {countdown}"
        )
        pending = [
            b
            for b in plan["blocks"]
            if any(j["run_id"] not in eval_ids for j in b["jobs"])
        ]
        upcoming = [b["name"] for b in pending if b["name"] != state.get("block")][:2]
        lines.append(
            "  Pending blocks: "
            + (" -> ".join(upcoming) if upcoming else "none after current")
        )
        for warning in warnings(snapshot):
            lines.append("  WARNING: " + warning)
        if state.get("reason"):
            lines.append("  Reason: " + str(state["reason"]))
        for failure in state.get("failures", [])[:2]:
            lines.append("  Failure: " + str(failure)[:150])
        if not compact and workers:
            lines.append(
                "  RUN   GROUP PHASE    MODEL / SEED / CONDITION       "
                "ELAPSED  THREADS CPUavg RSS"
            )
            for worker in sorted(workers, key=lambda w: w["index"]):
                if 0 <= worker["index"] < len(runs):
                    group = "queue" if worker["pid"] in active else "old"
                    lines.append(format_worker(worker, runs[worker["index"]], group))
    lines += [
        "",
        "Counts deduplicate local + remote artifacts; MFC excluded. "
        "Host queues are subsets.",
        "CPU busy is sampled; worker CPUavg is lifetime average, "
        "100% = one logical CPU. G = GiB.",
        "Cutoff is a stop time, not a completion ETA. "
        "--compact hides individual workers.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config = load_manifest(ROOT / "configs/exact_tvd_ancilla_v1.json")
    runs = [
        dict(r.to_dict(), run_id=r.run_id) for r in production_identities(config)[:3900]
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        snapshots = dict(pool.map(read, HOSTS))
    out = ROOT / "results/exact-tvd-ancilla-v1"
    progress = campaign_progress(
        runs,
        snapshots,
        [p.stem for p in (out / "training").glob("*.json")],
        [p.stem for p in (out / "evaluations").glob("*.json")],
    )
    if args.json:
        print(json.dumps(dict(progress=progress, hosts=snapshots), indent=2))
    else:
        print(render(snapshots, runs, progress, compact=args.compact))


if __name__ == "__main__":
    main()

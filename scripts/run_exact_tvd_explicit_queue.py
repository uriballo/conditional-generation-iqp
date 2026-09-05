#!/usr/bin/env python3
"""Execute an audited explicit queue against frozen campaign source on one host."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def validate(plan):
    if not isinstance(plan.get("enforce_temperature_limit", True), bool):
        raise ValueError("enforce_temperature_limit must be boolean")
    jobs = [job for block in plan["blocks"] for job in block["jobs"]]
    indices = [job["index"] for job in jobs]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate identities")
    if not all(0 <= index < 3900 for index in indices):
        raise ValueError("out-of-scope identity")
    if len({job["run_id"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate run IDs")
    cpus = plan.get("cpu_set", list(range(32)))
    if not cpus or len(cpus) != len(set(cpus)) or any(c not in range(32) for c in cpus):
        raise ValueError("invalid CPU allocation")
    for block in plan["blocks"]:
        for key in ("training_workers", "evaluation_workers"):
            if not 1 <= block[key] <= len(cpus):
                raise ValueError("workers exceed assigned CPUs")
    return jobs


def temperature_limit_reached(plan, temperatures):
    """Only the scheduler guard is optional; firmware controls are untouched."""
    return bool(
        plan.get("enforce_temperature_limit", True)
        and temperatures
        and max(temperatures) >= 90
    )


def worker_cpus(plan, workers, slot):
    return plan.get("cpu_set", list(range(32)))[slot::workers]


def old_cpu_groups():
    return [list(range(0, 4)), list(range(4, 8))] + [
        list(range(start, start + 3)) for start in range(8, 32, 3)
    ]


def slot_complete(slot, canonical, out):
    return all(
        (out / "evaluations" / f"{canonical[index]['run_id']}.json").exists()
        for index in list(range(3060, 3180, 4))[slot::10]
    )


def process_matches(record):
    """Do not mistake a recycled PID or an unreaped zombie for a live fit."""
    try:
        fields = (
            Path(f"/proc/{record['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        )
        return fields[0] != "Z" and int(fields[19]) == record["start_ticks"]
    except FileNotFoundError:
        return False


def validate_backfill(plan, canonical, out):
    backfill = plan["backfill"]
    old_groups = old_cpu_groups()
    slots = backfill["finished_slots"]
    if (
        not slots
        or len(slots) != len(set(slots))
        or any(s not in range(10) for s in slots)
    ):
        raise ValueError("invalid finished slots")
    if backfill["cpu_groups"] != [old_groups[s] for s in slots]:
        raise ValueError("CPU groups do not match old worker slots")
    for slot in slots:
        for index in list(range(3060, 3180, 4))[slot::10]:
            if not (
                out / "evaluations" / f"{canonical[index]['run_id']}.json"
            ).exists():
                raise ValueError("old slot has unfinished work")
    selected = backfill["indices"]
    jobs = {job["index"]: job for job in validate(plan)}
    if len(selected) != len(set(selected)) or not set(selected) <= jobs.keys():
        raise ValueError("backfill must be a unique subset of this host queue")
    if any(
        not 3180 <= index < 3300 or jobs[index]["family"] != "independent"
        for index in selected
    ):
        raise ValueError("backfill is restricted to independent CW n12 r10 fits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    jobs = validate(plan)
    if args.validate_only:
        print(f"Validated {len(jobs)} unique jobs")
        return
    if args.status:
        out = Path(plan["repository"]) / "results/exact-tvd-ancilla-v1"
        status = args.plan.with_suffix(".status.json")
        state = (
            json.loads(status.read_text())
            if status.exists()
            else {"status": "not launched"}
        )
        state["jobs"] = len(jobs)
        previous_pid = plan.get("wait_pid")
        previous_children = set()
        for children in Path(f"/proc/{previous_pid}/task").glob("*/children"):
            try:
                previous_children.update(children.read_text().split())
            except FileNotFoundError:
                pass
        state["previous_wave_workers"] = len(previous_children)
        state["trained"] = sum(
            (out / "training" / f"{j['run_id']}.json").exists() for j in jobs
        )
        state["evaluated"] = sum(
            (out / "evaluations" / f"{j['run_id']}.json").exists() for j in jobs
        )
        print(json.dumps(state))
        return
    lock = args.plan.with_suffix(".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = Path(plan["repository"])
    canonical = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(root / "scripts/run_exact_tvd_campaign.py"),
                "--manifest",
                str(root / "configs/exact_tvd_ancilla_v1.json"),
                "plan",
                "--json",
            ],
            text=True,
            cwd=root,
        )
    )
    for job in jobs:
        if canonical[job["index"]]["run_id"] != job["run_id"]:
            raise ValueError("queue does not match frozen manifest")
    out = root / "results/exact-tvd-ancilla-v1"
    state_path = args.plan.with_suffix(".status.json")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    guard = threading.RLock()
    state = {
        "host": plan["host"],
        "status": "waiting",
        "active": {},
        "failures": [],
        "finished_tasks": 0,
        "jobs": len(jobs),
        "deadline": plan["deadline"],
    }

    def save():
        state["updated"] = time.time()
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(state_path)

    deadline_check = [0.0]

    def expired():
        # Allow explicit cutoff changes without another scheduler handover.
        now = time.monotonic()
        if now - deadline_check[0] >= 1:
            with guard:
                updated_plan = json.loads(args.plan.read_text())
                plan["deadline"] = float(updated_plan["deadline"])
                state["deadline"] = plan["deadline"]
                deadline_check[0] = now
        return stop.is_set() or time.time() >= plan["deadline"]

    save()

    def resource_watch():
        while not stop.wait(10):
            memory = dict(
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text().splitlines()
            )
            available_gib = int(memory["MemAvailable"].split()[0]) / 1024**2
            temperatures = []
            for label in Path("/sys/class/hwmon").glob("hwmon*/temp*_label"):
                if label.read_text().strip() == "Tctl":
                    temperatures.append(
                        float(
                            label.with_name(
                                label.name.replace("_label", "_input")
                            ).read_text()
                        )
                        / 1000
                    )
            reason = None
            if available_gib < 8:
                reason = f"low memory: {available_gib:.1f} GiB available"
            if temperature_limit_reached(plan, temperatures):
                reason = f"temperature limit: {max(temperatures):.1f} C"
            if reason:
                with guard:
                    state["failures"].append({"resource_guard": reason})
                    stop.set()
                    save()
                return

    threading.Thread(target=resource_watch, daemon=True).start()

    def wave(block, phase, workers, backfill=None):
        pending = queue.Queue()
        block_indices = {job["index"] for job in block["jobs"]}
        adopted = (
            backfill.get("adopted", [])
            if backfill
            else [
                item
                for item in plan.get("adopted", [])
                if item["index"] in block_indices
                and item.get("phase", "train") == phase
            ]
        )
        adopted_indices = {item["index"] for item in adopted}
        folder = "training" if phase == "train" else "evaluations"
        for job in block["jobs"]:
            if (
                job["index"] not in adopted_indices
                and not (out / folder / (job["run_id"] + ".json")).exists()
            ):
                pending.put(job)
        state.update(
            status="backfilling" if backfill else "running",
            block=block["name"],
            phase=phase,
        )
        save()

        def worker(slot):
            allocation = dict(
                plan,
                cpu_set=block.get(
                    f"{phase}_cpu_set", plan.get("cpu_set", list(range(32)))
                ),
            )
            cpus = (
                backfill["cpu_groups"][slot]
                if backfill
                else worker_cpus(allocation, workers, slot)
            )
            env = os.environ.copy()
            for key in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]:
                env[key] = str(len(cpus))
            env["JAX_COMPILATION_CACHE_DIR"] = str(
                out / ".jax-compilation-cache" / f"explicit-{slot}"
            )
            env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
            inherited = next((item for item in adopted if item["cpus"] == cpus), None)
            if inherited:
                with guard:
                    state["active"][str(slot)] = inherited
                    save()
                while process_matches(inherited) and not expired():
                    time.sleep(2)
                if process_matches(inherited):
                    os.killpg(inherited["pid"], signal.SIGINT)
                    until = time.monotonic() + 30
                    while process_matches(inherited) and time.monotonic() < until:
                        time.sleep(1)
                    if process_matches(inherited):
                        os.killpg(inherited["pid"], signal.SIGKILL)
                with guard:
                    state["active"].pop(str(slot))
                    if (out / folder / f"{inherited['run_id']}.json").exists():
                        state["finished_tasks"] += 1
                    elif not expired():
                        state["failures"].append(
                            dict(
                                inherited,
                                reason="adopted fit exited without training artifact",
                            )
                        )
                        stop.set()
                    save()
            while not expired():
                # Stop admitting backfill when the old wave exits. Let already
                # running fits finish before handing all CPUs to the main queue.
                if backfill and not Path(f"/proc/{plan['wait_pid']}/cmdline").exists():
                    return
                if backfill and backfill.get("dynamic"):
                    if not slot_complete(slot, canonical, out):
                        time.sleep(5)
                        continue
                    # The last old evaluator can still be exiting just after
                    # publishing its artifact. Wait until its process is gone.
                    occupied = False
                    for task in Path(f"/proc/{plan['wait_pid']}/task").glob(
                        "*/children"
                    ):
                        try:
                            children = task.read_text().split()
                        except FileNotFoundError:
                            continue
                        for child_pid in children:
                            try:
                                occupied |= bool(
                                    set(cpus) & os.sched_getaffinity(int(child_pid))
                                )
                            except ProcessLookupError:
                                pass
                    if occupied:
                        time.sleep(2)
                        continue
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    return
                command = [
                    "taskset",
                    "-c",
                    ",".join(map(str, cpus)),
                    sys.executable,
                    str(root / "scripts/run_exact_tvd_campaign.py"),
                    "--manifest",
                    str(root / "configs/exact_tvd_ancilla_v1.json"),
                    "--data-directory",
                    str(out / "data"),
                    "--output-directory",
                    str(out),
                    phase,
                    "--run-index",
                    str(job["index"]),
                    "--resume",
                ]
                if phase == "train":
                    command += ["--selection", str(out / "protocol-selection.json")]
                log = args.plan.parent / f"{phase}-{job['index']}.log"
                with log.open("ab") as stream:
                    child = subprocess.Popen(
                        command,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        env=env,
                        cwd=root,
                        start_new_session=True,
                    )
                    with guard:
                        state["active"][str(slot)] = dict(job, pid=child.pid, cpus=cpus)
                        save()
                    while child.poll() is None and not expired():
                        time.sleep(2)
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGINT)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                    with guard:
                        state["active"].pop(str(slot))
                        if child.returncode and not expired():
                            state["failures"].append(
                                dict(job, phase=phase, returncode=child.returncode)
                            )
                            stop.set()
                        elif not child.returncode:
                            state["finished_tasks"] += 1
                        save()

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(worker, range(workers)))

    pid = plan.get("wait_pid")
    backfill = plan.get("backfill")
    if (
        backfill
        and pid
        and (Path(f"/proc/{pid}/cmdline").exists() or backfill.get("adopted"))
        and not expired()
    ):
        # Each reclaimed slot must have finished its ENTIRE static shard, not
        # merely have no current child between two jobs.
        validate_backfill(plan, canonical, out)
        adopted = backfill.get("adopted", [])
        if len({item["index"] for item in adopted}) != len(adopted):
            raise ValueError("duplicate adopted jobs")
        for item in adopted:
            if (
                item["index"] not in backfill["indices"]
                or item["cpus"] not in backfill["cpu_groups"]
            ):
                raise ValueError("adopted fit outside reserved queue/CPUs")
            if item["run_id"] != canonical[item["index"]]["run_id"]:
                raise ValueError("adopted fit identity mismatch")
        ignored = {item["pid"] for item in adopted if process_matches(item)}
        claimed = {cpu for group in backfill["cpu_groups"] for cpu in group}
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                args_bytes = (proc / "cmdline").read_bytes().split(b"\0")
                is_runner = any(
                    arg.endswith(b"/run_exact_tvd_campaign.py") for arg in args_bytes
                )
                if is_runner and (b"train" in args_bytes or b"evaluate" in args_bytes):
                    if int(proc.name) not in ignored and claimed & os.sched_getaffinity(
                        int(proc.name)
                    ):
                        raise RuntimeError("backfill CPUs overlap an active worker")
            except (FileNotFoundError, ProcessLookupError):
                continue
        selected = set(backfill["indices"])
        if backfill.get("dynamic"):
            # Startup validation covers only reclaimed slots. Runtime admission
            # checks cover the other slots as their old shards finish.
            backfill = dict(backfill, cpu_groups=old_cpu_groups())
        wave(
            {
                "name": "curie_weiss-n12-r10-backfill",
                "jobs": [job for job in jobs if job["index"] in selected],
            },
            "train",
            len(backfill["cpu_groups"]),
            backfill,
        )
    state.update(status="waiting", active={})
    save()
    while pid and Path(f"/proc/{pid}/cmdline").exists():
        if expired():
            state["status"] = "failed" if state["failures"] else "paused"
            save()
            return
        time.sleep(5)
    if pid:
        summary = out / "wave-both-3060-3180-host0.json"
        if not summary.exists() or json.loads(summary.read_text()).get("failures"):
            state.update(status="failed", reason="previous wave did not finish cleanly")
            save()
            return
    allowed_adopted = {
        item["pid"] for item in plan.get("adopted", []) if process_matches(item)
    }
    processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    if any(
        "run_exact_tvd_campaign.py" in line
        and (" train " in line or " evaluate " in line)
        and int(line.split(None, 1)[0]) not in allowed_adopted
        for line in processes.splitlines()
    ):
        raise RuntimeError("another campaign worker is active")

    for block in plan["blocks"]:
        for phase, workers in [
            ("train", block["training_workers"]),
            ("evaluate", block["evaluation_workers"]),
        ]:
            if expired():
                break
            wave(block, phase, workers)
        if expired():
            break
    state["status"] = (
        "failed" if state["failures"] else ("paused" if expired() else "complete")
    )
    save()
    stop.set()


if __name__ == "__main__":
    main()

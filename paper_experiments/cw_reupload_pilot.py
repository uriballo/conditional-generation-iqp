"""Pilot: re-uploaded parity encoder on the sixteen-qubit Curie--Weiss task.

The pilot reuses the audited shared-parity configuration and adds the turn
angle per unit inverse temperature. A zero turn reproduces the parity encoder.
Fits run at the confirmation stage, so they select on the development grid and
evaluate on the development and confirmation grids only. No test temperature
is evaluated. Random streams depend only on stage, repeat and restart, which
pairs every pilot fit with the published confirmation fit of the same repeat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from .cw_followup_fit import run_fit, write_json

FAMILY = "shared_reuploaded_parity"
ROOT = Path(__file__).resolve().parents[1]
WINNERS = ROOT / "paper" / "data" / "curie-weiss-followup" / "provenance"
SOURCES = ("reuploaded_parity_encoder.py", "cw_followup_fit.py", "cw_reupload_pilot.py")


def shared_parity_config():
    """Return the audited selected configuration of the shared-parity family."""
    final = json.loads((WINNERS / "final-results.json").read_text())
    config = final["shared_parity"][0]["config"]
    return {key: config[key] for key in sorted(config)}


def _revision():
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_hashes():
    folder = Path(__file__).resolve().parent
    return {n: hashlib.sha256((folder / n).read_bytes()).hexdigest() for n in SOURCES}


def jobs(turns, repeats, smoke=False, scales=None):
    base = shared_parity_config()
    scales = [base["beta_scale"]] if scales is None else scales
    identity = dict(
        kind="cw_reupload_pilot",
        base_family="shared_parity",
        base_config=base,
        stage_semantics="confirmation grids only; no test temperatures",
    )
    for scale in scales:
        for turn in turns:
            yield from _repeats(
                base, float(scale), float(turn), repeats, smoke, identity
            )


def _repeats(base, scale, turn, repeats, smoke, identity):
    candidate = dict(base, beta_scale=scale, turn=turn)
    # Keep the original trial names for the audited scale.
    tag = "" if scale == base["beta_scale"] else f"scale-{scale:g}-"
    for repeat in range(repeats):
        yield dict(
            family=FAMILY,
            candidate=candidate,
            trial_id=f"pilot-{tag}turn-{turn:g}",
            stage="confirmation",
            seed_stage="confirmation",
            repeat=repeat,
            beta=None,
            smoke=smoke,
            identity=identity,
        )


def summarize(records):
    groups = {}
    for record in records:
        candidate = record["job"]["candidate"]
        key = (candidate["beta_scale"], candidate["turn"])
        groups.setdefault(key, []).append(record)
    rows = []
    for (scale, turn), group in sorted(groups.items()):
        ok = [r for r in group if r["status"] == "ok"]
        row = dict(beta_scale=scale, turn=turn, fits=len(group), ok=len(ok))
        for key in ("J", "P", "selection_J", "selection_P"):
            values = np.array([r[key] for r in ok], dtype=float)
            if len(values):
                row[key] = dict(
                    mean=float(values.mean()),
                    std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    values=values.tolist(),
                )
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turns", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--scales", type=float, nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    planned = list(jobs(args.turns, args.repeats, args.smoke, args.scales))
    write_json(
        output / "manifest.json",
        dict(
            kind="cw_reupload_pilot",
            authorization="Author approved the re-uploaded encoder path 2026-10-13",
            revision=_revision(),
            sources_sha256=_source_hashes(),
            turns=args.turns,
            scales=args.scales,
            repeats=args.repeats,
            smoke=args.smoke,
            jobs=planned,
        ),
    )
    records = []
    for index, job in enumerate(planned, 1):
        record = run_fit(job, output)
        records.append(record)
        print(
            f"[{index}/{len(planned)}] scale={job['candidate']['beta_scale']:g} "
            f"turn={job['candidate']['turn']:g} "
            f"repeat={job['repeat']} status={record['status']} "
            f"J={record.get('J')} seconds={record.get('wall_seconds', 0):.0f}",
            flush=True,
        )
        write_json(output / "summary.json", dict(rows=summarize(records)))
    print(json.dumps(summarize(records), indent=1), flush=True)


if __name__ == "__main__":
    main()

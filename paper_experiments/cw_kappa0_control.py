"""Squeezing ablation for the Curie--Weiss squeezed parity encoder.

Refits the selected shared squeezed configuration with kappa=0, which keeps the
nonlinear angle schedule sin t = tanh(a beta^2) but removes the squeezing, so
the encoder is a phase encoder. Each schedule constant ``a`` gets the final-stage
jobs of the audited protocol, whose fit streams pair with the published
squeezed and parity final fits. Optimizer settings stay at the values selected
for kappa=0.4; there is no new hyperparameter search.

Run:    python -m paper_experiments.cw_kappa0_control run ROOT --winner CAMPAIGN_JSON
Bundle: python -m paper_experiments.cw_kappa0_control bundle ROOT --output FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import cw_encoder_hpo as hpo
from . import cw_followup_hpo as audited
from .cw_followup_config import protocol as audited_protocol
from .cw_followup_config import stage_betas
from .cw_followup_fit import _verify_artifacts, digest, write_json
from .cw_followup_metrics import curve_scores

SCALES = ("winner", 0.3, 0.7)
CURVES = ("tvd", "magnetization_error", "heat_error")


def jobs(winner, scales=SCALES, smoke=False):
    hpo.configure("squeezed")
    identity = dict(hpo.source_identity(), control="kappa0 schedule-matched, local CPU")
    context = dict(family=hpo.FAMILY, smoke=smoke, identity=identity)
    out = []
    for scale in scales:
        value = winner["config"]["beta_scale"] if scale == "winner" else float(scale)
        proposal = dict(
            trial_id=winner["trial_id"],
            config=dict(winner["config"], kappa=0.0, beta_scale=value),
        )
        out += list(audited.jobs(context, proposal, stage="final"))
    return out


def _fit(arguments):
    job, root = arguments
    return audited._fit_with_retry(job, Path(root))["status"]


def run(root, winner, scales=SCALES, workers=1, smoke=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    planned = jobs(winner, scales, smoke)
    path = root / "jobs.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(
        json.dumps(planned)
    ):
        raise ValueError("planned control jobs differ from the recorded jobs")
    write_json(path, planned)
    arguments = [(job, str(root)) for job in planned]
    if workers == 1:
        return [_fit(a) for a in arguments]
    with ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        return list(pool.map(_fit, arguments))


def bundle(root, destination):
    """Per-scale curves and scores of every recorded control fit; no omissions."""
    root = Path(root)
    planned = json.loads((root / "jobs.json").read_text())
    smoke = planned[0]["smoke"]
    grid = stage_betas(audited_protocol(smoke), "final")
    scales, receipts = {}, {}
    for job in planned:
        fit_dir = root / "fits" / digest(job)
        record = json.loads((fit_dir / "metrics.json").read_text())
        if record["job"] != job or record["status"] != "ok":
            raise ValueError("control fit missing or invalid")
        _verify_artifacts(fit_dir, record["artifact_sha256"])
        receipts[digest(job)] = hashlib.sha256(
            (fit_dir / "metrics.json").read_bytes()
        ).hexdigest()
        curve = {k: np.asarray(record["metrics_curves"][k]) for k in CURVES}
        if not np.array_equal(record["metrics_curves"]["beta"], grid):
            raise ValueError("control temperature grid differs")
        key = repr(job["candidate"]["beta_scale"])
        entry = scales.setdefault(
            key,
            dict(config=job["candidate"], repeats=[], curves={k: [] for k in CURVES}),
        )
        entry["repeats"].append(job["repeat"])
        for k in CURVES:
            entry["curves"][k].append(curve[k].tolist())
        entry.setdefault("repeat_scores", []).append(
            curve_scores(*(curve[k] for k in CURVES), np.asarray(grid))
        )
    payload = dict(
        kind="cw_kappa0_control",
        family="shared_squeezed_parity",
        trial_id=planned[0]["trial_id"],
        smoke=smoke,
        beta=list(map(float, grid)),
        scales=scales,
        fit_receipt_sha256=receipts,
    )
    write_json(Path(destination), payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("root", type=Path)
    r.add_argument("--winner", type=Path, required=True)
    r.add_argument("--scales", nargs="+", default=list(SCALES))
    r.add_argument("--workers", type=int, default=1)
    b = sub.add_parser("bundle")
    b.add_argument("root", type=Path)
    b.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        winner = json.loads(args.winner.read_text())["winner"]
        scales = [s if s == "winner" else float(s) for s in args.scales]
        print(run(args.root, winner, scales, args.workers))
    else:
        payload = bundle(args.root, args.output)
        for key, entry in payload["scales"].items():
            print(key, np.mean([s["J"] for s in entry["repeat_scores"]]))


if __name__ == "__main__":
    main()

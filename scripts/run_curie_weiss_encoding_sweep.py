#!/usr/bin/env python3
"""Benchmark noncommuting reupload and spin-flip two-branch encodings on Curie--Weiss.

Compares conditioned-model held-out TVD against the frozen broadcast-RY
baseline (``curie-weiss-m5-foundations-v2``).  TVD is the objective metric;
MMD is reported only as a training diagnostic, not a claim.

Subcommands
-----------
run     train + evaluate the given encodings over the given seeds, writing one
        immutable row per (encoding, seed).  Skips rows that already exist.
report  aggregate all rows plus the baseline and print a markdown table.

Examples
--------
    # smoke test: one seed, 50 updates, lightweight samples
    python scripts/run_curie_weiss_encoding_sweep.py run \
        --encoding reupload_l2 --seeds 0 --updates 50 --lightweight
    # full: three encodings over 10 seeds (sharded across processes)
    python scripts/run_curie_weiss_encoding_sweep.py run \
        --encoding reupload_l2 --seeds 0-9
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from iqp_cond.curie_weiss_ancilla_campaign import CurieWeissAncillaCampaignStore
from iqp_cond.curie_weiss_ancilla_protocol import (
    CurieWeissAncillaFamily,
    CurieWeissAncillaManifest,
    plan_curie_weiss_ancilla_runs,
)
from iqp_cond.curie_weiss_ancilla_runner import (
    evaluate_run,
    load_reused_dataset,
    train_model,
    training_config,
)
from iqp_cond.encoding import (
    ANGLE_ENCODING,
    REUPLOAD_L2_ENCODING,
    REUPLOAD_L3_ENCODING,
    SPIN_FLIP_BRANCH_ENCODING,
)
from iqp_cond.exact import (
    enumerate_bitstrings,
    shell_probabilities,
    total_variation_distance,
)

ENCODINGS = {
    "angle": ANGLE_ENCODING,
    "reupload_l2": REUPLOAD_L2_ENCODING,
    "reupload_l3": REUPLOAD_L3_ENCODING,
    "spin_flip_branch": SPIN_FLIP_BRANCH_ENCODING,
}
NEW_ENCODINGS = ("reupload_l2", "reupload_l3", "spin_flip_branch")
HELD_OUT_BETAS = (0.6, 0.8, 1.0, 1.2, 1.4)

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "configs/curie_weiss_m5_foundations_v2.json"
BASELINE_STORE = REPO / "results/curie-weiss-m5-foundations-v2/store"
OUT_DIR = REPO / "results/curie-weiss-encoding-sweep-v1"
ROWS_DIR = OUT_DIR / "rows"


def _tvd_vectors(
    target_probabilities: np.ndarray,
    model_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bits = enumerate_bitstrings(8)
    raw = np.asarray(
        [
            total_variation_distance(t, m)
            for t, m in zip(target_probabilities, model_probabilities, strict=True)
        ]
    )
    profile = np.asarray(
        [
            total_variation_distance(
                shell_probabilities(t, bits), shell_probabilities(m, bits)
            )
            for t, m in zip(target_probabilities, model_probabilities, strict=True)
        ]
    )
    return raw, profile


def _load_baseline(
    store: CurieWeissAncillaCampaignStore,
) -> dict[int, dict[str, np.ndarray]]:
    planned = plan_curie_weiss_ancilla_runs(store.manifest)
    conditioned = [
        r
        for r in planned
        if r.family is CurieWeissAncillaFamily.CONDITIONED and r.ancillas == 5
    ]
    out: dict[int, dict[str, np.ndarray]] = {}
    for run in conditioned:
        bundle = store.load_result(run)
        with np.load(bundle.probabilities_path, allow_pickle=False) as archive:
            betas = np.asarray(archive["betas"], dtype=np.float64)
            target = np.asarray(archive["target_probabilities"], dtype=np.float64)
            model = np.asarray(archive["model_probabilities"], dtype=np.float64)
        raw, profile = _tvd_vectors(target, model)
        out[run.optimizer_seed] = {"betas": betas, "raw": raw, "profile": profile}
    return out


def _run_one(
    manifest: CurieWeissAncillaManifest,
    run,
    encoding_name: str,
    *,
    updates: int | None,
    lightweight: bool,
) -> dict:
    started = time.perf_counter()
    dataset = load_reused_dataset(REPO, manifest, run)
    config = training_config(manifest, run, updates=updates, lightweight=lightweight)
    result = train_model(
        manifest, run, dataset, config, encoding=ENCODINGS[encoding_name]
    )
    _metrics, arrays = evaluate_run(
        manifest,
        run,
        result.best_params,
        result.generators,
        encoding=ENCODINGS[encoding_name],
    )
    betas = np.asarray(arrays["betas"], dtype=np.float64)
    target = np.asarray(arrays["target_probabilities"], dtype=np.float64)
    model = np.asarray(arrays["model_probabilities"], dtype=np.float64)
    raw, profile = _tvd_vectors(target, model)
    return {
        "encoding": encoding_name,
        "seed": run.optimizer_seed,
        "betas": betas.tolist(),
        "raw_tvd": raw.tolist(),
        "profile_tvd": profile.tolist(),
        "best_validation_loss": result.best_validation_loss,
        "best_step": result.best_step,
        "runtime_seconds": time.perf_counter() - started,
    }


def _seed_list(spec: str) -> list[int]:
    if "," in spec:
        return [int(s) for s in spec.split(",")]
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def run(args: argparse.Namespace) -> int:
    manifest = CurieWeissAncillaManifest.from_json(MANIFEST_PATH)
    planned = plan_curie_weiss_ancilla_runs(manifest)
    conditioned = {
        r.optimizer_seed: r
        for r in planned
        if r.family is CurieWeissAncillaFamily.CONDITIONED and r.ancillas == 5
    }
    seeds = _seed_list(args.seeds)
    for encoding in args.encoding:
        for seed in seeds:
            row_path = ROWS_DIR / encoding / f"seed{seed}.json"
            if row_path.exists():
                print(f"skip {encoding} seed{seed} (already complete)", flush=True)
                continue
            print(f"start {encoding} seed{seed}", flush=True)
            row = _run_one(
                manifest,
                conditioned[seed],
                encoding,
                updates=args.updates,
                lightweight=args.lightweight,
            )
            row_path.parent.mkdir(parents=True, exist_ok=True)
            row_path.write_text(json.dumps(row, indent=2, sort_keys=True))
            print(
                f"done  {encoding} seed{seed} "
                f"held-out profile TVD={_held_out_mean(row):.4f}",
                flush=True,
            )
    return 0


def _held_out_mean(row: dict) -> float:
    betas = np.asarray(row["betas"])
    profile = np.asarray(row["profile_tvd"])
    mask = np.isin(betas, HELD_OUT_BETAS)
    return float(profile[mask].mean())


def _paired_summary(rows: list[dict], baseline: dict[int, dict]) -> dict:
    seeds = sorted(r["seed"] for r in rows)
    held_out_profile = []
    held_out_raw = []
    delta_profile = []
    delta_raw = []
    for r in rows:
        betas = np.asarray(r["betas"])
        mask = np.isin(betas, HELD_OUT_BETAS)
        held_out_profile.append(np.asarray(r["profile_tvd"])[mask].mean())
        held_out_raw.append(np.asarray(r["raw_tvd"])[mask].mean())
        b = baseline[r["seed"]]
        bmask = np.isin(b["betas"], HELD_OUT_BETAS)
        bprof = b["profile"][bmask].mean()
        braw = b["raw"][bmask].mean()
        delta_profile.append(held_out_profile[-1] - bprof)
        delta_raw.append(held_out_raw[-1] - braw)
    return {
        "seeds": seeds,
        "held_out_profile_mean": float(np.mean(held_out_profile)),
        "held_out_profile_se": float(
            np.std(held_out_profile, ddof=1) / np.sqrt(len(seeds))
        ),
        "held_out_raw_mean": float(np.mean(held_out_raw)),
        "held_out_raw_se": float(np.std(held_out_raw, ddof=1) / np.sqrt(len(seeds))),
        "delta_profile_mean": float(np.mean(delta_profile)),
        "delta_profile_se": float(np.std(delta_profile, ddof=1) / np.sqrt(len(seeds))),
        "delta_raw_mean": float(np.mean(delta_raw)),
        "delta_raw_se": float(np.std(delta_raw, ddof=1) / np.sqrt(len(seeds))),
        "per_seed_delta_profile": dict(zip(seeds, delta_profile, strict=True)),
    }


def report(args: argparse.Namespace) -> int:
    manifest = CurieWeissAncillaManifest.from_json(MANIFEST_PATH)
    store = CurieWeissAncillaCampaignStore(BASELINE_STORE, manifest)
    baseline = _load_baseline(store)
    baseline_profile = [
        np.asarray(v["profile"])[np.isin(v["betas"], HELD_OUT_BETAS)].mean()
        for v in baseline.values()
    ]
    baseline_raw = [
        np.asarray(v["raw"])[np.isin(v["betas"], HELD_OUT_BETAS)].mean()
        for v in baseline.values()
    ]

    rows: list[tuple[str, dict]] = []
    for name in NEW_ENCODINGS:
        enc_rows = []
        for seed in range(10):
            p = ROWS_DIR / name / f"seed{seed}.json"
            if p.exists():
                enc_rows.append(json.loads(p.read_text()))
        if enc_rows:
            rows.append((name, _paired_summary(enc_rows, baseline)))

    print("\n# Curie--Weiss encoding benchmark (held-out TVD, lower is better)\n")
    print(
        "| encoding | held-out profile TVD | held-out raw TVD | "
        "Δ profile vs baseline | Δ raw vs baseline |"
    )
    print("|---|---|---|---|---|")
    base_prof = float(np.mean(baseline_profile))
    base_raw = float(np.mean(baseline_raw))
    base_prof_se = np.std(baseline_profile, ddof=1) / np.sqrt(10)
    base_raw_se = np.std(baseline_raw, ddof=1) / np.sqrt(10)
    print(
        f"| baseline (broadcast RY) | {base_prof:.4f} ± {base_prof_se:.4f} "
        f"| {base_raw:.4f} ± {base_raw_se:.4f} | — | — |"
    )
    for name, s in rows:
        profile_cell = (
            f"{s['held_out_profile_mean']:.4f} ± {s['held_out_profile_se']:.4f}"
        )
        print(
            f"| {name} | {profile_cell} "
            f"| {s['held_out_raw_mean']:.4f} ± {s['held_out_raw_se']:.4f} "
            f"| {s['delta_profile_mean']:+.4f} ± {s['delta_profile_se']:.4f} "
            f"| {s['delta_raw_mean']:+.4f} ± {s['delta_raw_se']:.4f} |"
        )

    payload = {
        "baseline": {
            "held_out_profile_mean": base_prof,
            "held_out_raw_mean": base_raw,
            "per_seed_profile": {
                s: float(
                    np.asarray(v["profile"])[np.isin(v["betas"], HELD_OUT_BETAS)].mean()
                )
                for s, v in baseline.items()
            },
        },
        "encodings": {name: s for name, s in rows},
    }
    out = OUT_DIR / "summary.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--encoding", nargs="+", choices=NEW_ENCODINGS, required=True)
    run_p.add_argument("--seeds", default="0-9")
    run_p.add_argument("--updates", type=int, default=None)
    run_p.add_argument("--lightweight", action="store_true")
    run_p.set_defaults(func=run)
    rep_p = sub.add_parser("report")
    rep_p.set_defaults(func=report)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

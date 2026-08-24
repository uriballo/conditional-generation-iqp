"""Runner for Experiment 2: Two-Dimensional Multi-Conditioning."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from iqp_cond.multidimensional_2d import (
    EVAL_GRID_1D,
    TRAIN_GRID_1D,
    run_2d_gaussian_trial,
)

ROOT = Path(__file__).parents[1]
OUTPUT_DIR = ROOT / "results" / "multidimensional-2d-v1"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_seeds = 10
    n_qubits = 8

    print("=" * 70)
    print("EXPERIMENT 2: Two-Dimensional Multi-Conditioning (c_mu, c_sigma)")
    print(f"System: {n_qubits} qubits | Complete 2-local IQP (36 parameters)")
    print(
        f"Training grid: {len(TRAIN_GRID_1D)}x{len(TRAIN_GRID_1D)} = 9 conditions"
    )
    print(
        f"Evaluation grid: {len(EVAL_GRID_1D)}x{len(EVAL_GRID_1D)} = 25 conditions "
        "(16 unseen test points)"
    )
    print(f"Seeds: {n_seeds}")
    print("=" * 70)

    start_time = time.time()
    results = []
    grid_prof_accum = np.zeros((len(EVAL_GRID_1D), len(EVAL_GRID_1D)))
    grid_raw_accum = np.zeros((len(EVAL_GRID_1D), len(EVAL_GRID_1D)))

    for s in range(n_seeds):
        t0 = time.time()
        res = run_2d_gaussian_trial(seed=s, n_qubits=n_qubits)
        elapsed = time.time() - t0

        grid_prof_accum += res.grid_profile_tvds
        grid_raw_accum += res.grid_raw_tvds

        record = {
            "seed": res.seed,
            "train_profile_tvd_mean": res.train_profile_tvd_mean,
            "heldout_profile_tvd_mean": res.heldout_profile_tvd_mean,
            "train_raw_tvd_mean": res.train_raw_tvd_mean,
            "heldout_raw_tvd_mean": res.heldout_raw_tvd_mean,
            "bilinear_heldout_profile_tvd_mean": res.bilinear_heldout_profile_tvd_mean,
            "bilinear_heldout_raw_tvd_mean": res.bilinear_heldout_raw_tvd_mean,
            "grid_profile_tvds": res.grid_profile_tvds.tolist(),
            "grid_raw_tvds": res.grid_raw_tvds.tolist(),
            "runtime_seconds": elapsed,
        }
        results.append(record)
        print(
            f"  [Seed {s:2d}] Train: {res.train_profile_tvd_mean:.4f} | "
            f"Heldout: {res.heldout_profile_tvd_mean:.4f} "
            f"(Bilinear: {res.bilinear_heldout_profile_tvd_mean:.4f}) | "
            f"Raw: {res.heldout_raw_tvd_mean:.4f} ({elapsed:.1f}s)"
        )

    grid_prof_mean = grid_prof_accum / n_seeds
    grid_raw_mean = grid_raw_accum / n_seeds
    total_elapsed = time.time() - start_time

    tr_prof_all = [r["train_profile_tvd_mean"] for r in results]
    ho_prof_all = [r["heldout_profile_tvd_mean"] for r in results]
    ho_bilin_all = [r["bilinear_heldout_profile_tvd_mean"] for r in results]

    print("\n" + "=" * 70)
    print("2D MULTI-CONDITIONING SUMMARY")
    print("=" * 70)
    tr_m, tr_s = np.mean(tr_prof_all), np.std(tr_prof_all)
    ho_m, ho_s = np.mean(ho_prof_all), np.std(ho_prof_all)
    bi_m, bi_s = np.mean(ho_bilin_all), np.std(ho_bilin_all)
    print(f"Training Profile TVD Mean : {tr_m:.4f} ± {tr_s:.4f}")
    print(f"Held-out Profile TVD Mean : {ho_m:.4f} ± {ho_s:.4f}")
    print(f"Bilinear Baseline TVD Mean: {bi_m:.4f} ± {bi_s:.4f}")

    output_payload = {
        "experiment": "multidimensional_2d_gaussian",
        "n_qubits": n_qubits,
        "n_seeds": n_seeds,
        "train_grid_1d": list(TRAIN_GRID_1D),
        "eval_grid_1d": list(EVAL_GRID_1D),
        "total_runtime_seconds": total_elapsed,
        "summary": {
            "train_profile_tvd_mean": float(np.mean(tr_prof_all)),
            "train_profile_tvd_std": float(np.std(tr_prof_all)),
            "heldout_profile_tvd_mean": float(np.mean(ho_prof_all)),
            "heldout_profile_tvd_std": float(np.std(ho_prof_all)),
            "bilinear_heldout_profile_tvd_mean": float(np.mean(ho_bilin_all)),
            "bilinear_heldout_profile_tvd_std": float(np.std(ho_bilin_all)),
            "grid_profile_tvd_mean": grid_prof_mean.tolist(),
            "grid_raw_tvd_mean": grid_raw_mean.tolist(),
        },
        "runs": results,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(output_payload, indent=2))
    print(f"Artifact saved to: {OUTPUT_DIR / 'summary.json'}")
    print(f"Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()

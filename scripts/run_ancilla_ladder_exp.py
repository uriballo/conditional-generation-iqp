"""Runner for Experiment 3: Ancilla Ladder Capacity Sweep."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from iqp_cond.ancilla_ladder import (
    ANCILLA_RUNGS,
    run_ancilla_ladder_trial,
)

ROOT = Path(__file__).parents[1]
OUTPUT_DIR = ROOT / "results" / "ancilla-ladder-v1"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_seeds = 10
    rungs = ANCILLA_RUNGS  # (0, 1, 2, 3, 4, 5, 6)

    print("=" * 70)
    print("EXPERIMENT 3: Ancilla Ladder Capacity Scaling Sweep")
    print(f"Sweeping ancilla counts r in {rungs} on 8 visible spins")
    print(f"Seeds: {n_seeds} per rung (Total {len(rungs)*n_seeds} fits)")
    print("=" * 70)

    start_time = time.time()
    results_by_rung: dict[int, list[dict[str, object]]] = {r: [] for r in rungs}

    for r in rungs:
        n_tot = 8 + r
        n_params = n_tot * (n_tot + 1) // 2
        print(f"\n---> Rung r={r} (N={n_tot} qubits, {n_params} params)")
        for s in range(n_seeds):
            t0 = time.time()
            res = run_ancilla_ladder_trial(r=r, seed=s)
            elapsed = time.time() - t0

            record = {
                "r": res.r,
                "n_total_qubits": res.n_total_qubits,
                "n_parameters": res.n_parameters,
                "seed": res.seed,
                "train_profile_tvd": res.train_profile_tvd,
                "heldout_profile_tvd": res.heldout_profile_tvd,
                "train_raw_tvd": res.train_raw_tvd,
                "heldout_raw_tvd": res.heldout_raw_tvd,
                "cold_profile_tvd": res.cold_profile_tvd,
                "cold_raw_tvd": res.cold_raw_tvd,
                "critical_profile_tvd": res.critical_profile_tvd,
                "mean_entropy": res.mean_entropy,
                "mean_effective_rank": res.mean_effective_rank,
                "per_beta_entropy": res.per_beta_entropy,
                "per_beta_effective_rank": res.per_beta_effective_rank,
                "runtime_seconds": elapsed,
            }
            results_by_rung[r].append(record)
            print(
                f"  [r={r}, Seed {s:2d}] Heldout: {res.heldout_profile_tvd:.4f} | "
                f"Cold: {res.cold_profile_tvd:.4f} | "
                f"S(rho): {res.mean_entropy:.3f} | "
                f"Rank: {res.mean_effective_rank:.2f} ({elapsed:.1f}s)"
            )

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("ANCILLA LADDER SUMMARY ACROSS RUNGS")
    print("=" * 70)
    summary_data = {}
    for r in rungs:
        recs = results_by_rung[r]
        ho_prof = [rec["heldout_profile_tvd"] for rec in recs]
        ho_raw = [rec["heldout_raw_tvd"] for rec in recs]
        cold_prof = [rec["cold_profile_tvd"] for rec in recs]
        ent = [rec["mean_entropy"] for rec in recs]
        eff_r = [rec["mean_effective_rank"] for rec in recs]

        summary_data[str(r)] = {
            "r": r,
            "n_total_qubits": 8 + r,
            "n_parameters": (8 + r) * (9 + r) // 2,
            "heldout_profile_tvd_mean": float(np.mean(ho_prof)),
            "heldout_profile_tvd_std": float(np.std(ho_prof)),
            "heldout_raw_tvd_mean": float(np.mean(ho_raw)),
            "heldout_raw_tvd_std": float(np.std(ho_raw)),
            "cold_profile_tvd_mean": float(np.mean(cold_prof)),
            "cold_profile_tvd_std": float(np.std(cold_prof)),
            "mean_entropy": float(np.mean(ent)),
            "mean_effective_rank": float(np.mean(eff_r)),
        }
        ho_m, ho_s = np.mean(ho_prof), np.std(ho_prof)
        print(
            f"r={r} (N={8+r:2d}): Heldout TVD = {ho_m:.4f} ± {ho_s:.4f} | "
            f"Cold TVD = {np.mean(cold_prof):.4f} | "
            f"Entropy = {np.mean(ent):.3f} | Rank = {np.mean(eff_r):.2f}"
        )

    output_payload = {
        "experiment": "ancilla_ladder_scaling_curie_weiss",
        "rungs": list(rungs),
        "total_runtime_seconds": total_elapsed,
        "summary": summary_data,
        "runs": {str(k): v for k, v in results_by_rung.items()},
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(output_payload, indent=2))
    print(f"Artifact saved to: {OUTPUT_DIR / 'summary.json'}")
    print(f"Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()

"""Runner for Experiment 1: Matched-capacity mixed-state rank control."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from iqp_cond.matched_capacity import run_matched_capacity_trial

ROOT = Path(__file__).parents[1]
OUTPUT_DIR = ROOT / "results" / "matched-capacity-v1"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models = ["ancilla_5", "visible_matched_91", "visible_2local_36"]
    n_seeds = 10
    results_by_model: dict[str, list[dict[str, object]]] = {m: [] for m in models}

    print("=" * 70)
    print("EXPERIMENT 1: Matched-Capacity Control on 8-Spin Curie--Weiss")
    print(f"Comparing 3 Models over {n_seeds} seeds:")
    print("  1. ancilla_5: N=13 (8 vis + 5 ancilla, 91 params, rank <= 32)")
    print("  2. visible_matched_91: n=8 (91 params [1-, 2-, 3-local], rank = 1)")
    print("  3. visible_2local_36: n=8 (36 params [1-, 2-local], rank = 1)")
    print("=" * 70)

    start_time = time.time()
    for m in models:
        print(f"\n---> Running Model: {m}")
        for s in range(n_seeds):
            t0 = time.time()
            res = run_matched_capacity_trial(m, seed=s)
            elapsed = time.time() - t0
            record = {
                "model_name": res.model_name,
                "seed": res.seed,
                "train_profile_tvd": res.training_profile_tvd,
                "heldout_profile_tvd": res.held_out_profile_tvd,
                "train_raw_tvd": res.training_raw_tvd,
                "heldout_raw_tvd": res.held_out_raw_tvd,
                "per_beta_profile_tvd": res.per_beta_profile_tvd,
                "per_beta_raw_tvd": res.per_beta_raw_tvd,
                "per_beta_magnetization": res.per_beta_magnetization,
                "per_beta_binder_cumulant": res.per_beta_binder_cumulant,
                "runtime_seconds": elapsed,
            }
            results_by_model[m].append(record)
            print(
                f"  [Seed {s:2d}] Train Prof TVD: {res.training_profile_tvd:.4f} | "
                f"Heldout Prof TVD: {res.held_out_profile_tvd:.4f} | "
                f"Heldout Raw TVD: {res.held_out_raw_tvd:.4f} ({elapsed:.1f}s)"
            )

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS (10-Seed Means ± Std)")
    print("=" * 70)
    summary_data = {}
    for m in models:
        recs = results_by_model[m]
        tr_prof = [r["train_profile_tvd"] for r in recs]
        ho_prof = [r["heldout_profile_tvd"] for r in recs]
        tr_raw = [r["train_raw_tvd"] for r in recs]
        ho_raw = [r["heldout_raw_tvd"] for r in recs]

        cold_prof = [r["per_beta_profile_tvd"][1.5] for r in recs]
        cold_raw = [r["per_beta_raw_tvd"][1.5] for r in recs]

        summary_data[m] = {
            "train_profile_tvd_mean": float(np.mean(tr_prof)),
            "train_profile_tvd_std": float(np.std(tr_prof)),
            "heldout_profile_tvd_mean": float(np.mean(ho_prof)),
            "heldout_profile_tvd_std": float(np.std(ho_prof)),
            "train_raw_tvd_mean": float(np.mean(tr_raw)),
            "train_raw_tvd_std": float(np.std(tr_raw)),
            "heldout_raw_tvd_mean": float(np.mean(ho_raw)),
            "heldout_raw_tvd_std": float(np.std(ho_raw)),
            "cold_endpoint_profile_tvd_mean": float(np.mean(cold_prof)),
            "cold_endpoint_raw_tvd_mean": float(np.mean(cold_raw)),
        }
        print(f"Model: {m}")
        print(
            f"  Training Profile TVD : {np.mean(tr_prof):.4f} ± {np.std(tr_prof):.4f}"
        )
        print(
            f"  Held-out Profile TVD : {np.mean(ho_prof):.4f} ± {np.std(ho_prof):.4f}"
        )
        print(
            f"  Held-out Raw TVD     : {np.mean(ho_raw):.4f} ± {np.std(ho_raw):.4f}"
        )
        print(f"  Cold End Profile TVD : {np.mean(cold_prof):.4f}")
        print(f"  Cold End Raw TVD     : {np.mean(cold_raw):.4f}\n")

    output_payload = {
        "experiment": "matched_capacity_control_curie_weiss",
        "total_runtime_seconds": total_elapsed,
        "summary": summary_data,
        "runs": results_by_model,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(output_payload, indent=2))
    print(f"Artifact saved to: {OUTPUT_DIR / 'summary.json'}")
    print(f"Total experiment time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()

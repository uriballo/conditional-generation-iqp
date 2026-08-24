"""Report hamming experiment run status and preview per-seed metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqp_cond.hamming_experiment import TRAIN_CENTER_COUNTS


def main() -> None:
    """Print one row per completed seed and a completion summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/hamming/runs"),
    )
    args = parser.parse_args()

    paths = sorted(args.runs.glob("*.json"))
    if not paths:
        print(f"no runs found under {args.runs}")
        return

    rows = []
    for path in paths:
        run = json.loads(path.read_text())
        points = run["points"]
        effective_by_k = {
            point["target_k"]: point["effective_center_count"]
            for point in points
            if point["role"] == "train"
        }
        heldout_tvds = [
            point["bitstring_tvd"] for point in points if point["role"] == "heldout"
        ]
        mean_heldout_tvd = (
            sum(heldout_tvds) / len(heldout_tvds) if heldout_tvds else float("nan")
        )
        rows.append(
            {
                "seed": int(run["seed"]),
                "best_step": int(run["best_step"]),
                "best_validation_loss": float(run["best_validation_loss"]),
                "heldout_mmd": float(run["heldout_mmd"]),
                "mean_heldout_tvd": float(mean_heldout_tvd),
                "effective_k": effective_by_k,
            },
        )
    rows.sort(key=lambda row: row["seed"])

    header = "Seed | Step | Val loss | Held-out MMD | Held-out TVD | " + " | ".join(
        f"Eff K={k}" for k in TRAIN_CENTER_COUNTS
    )
    print(header)
    print("|".join("---" for _ in header.split("|")))
    for row in rows:
        effective = [
            row["effective_k"].get(k, float("nan")) for k in TRAIN_CENTER_COUNTS
        ]
        print(
            f"{row['seed']} | {row['best_step']} | "
            f"{row['best_validation_loss']:.5f} | {row['heldout_mmd']:.5f} | "
            f"{row['mean_heldout_tvd']:.4f} | "
            + " | ".join(f"{value:.0f}" for value in effective),
        )

    seeds = {row["seed"] for row in rows}
    missing = sorted(set(range(10)) - seeds)
    print(f"\n{len(rows)}/10 seeds complete")
    if missing:
        print(f"missing: {', '.join(str(seed) for seed in missing)}")
    else:
        print("all seeds complete; run scripts/summarize_hamming_experiment.py")


if __name__ == "__main__":
    main()

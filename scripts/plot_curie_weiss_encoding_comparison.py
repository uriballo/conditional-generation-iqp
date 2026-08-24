#!/usr/bin/env python3
"""Plot the frozen role-separated encoding comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from iqp_cond.artifacts import read_immutable_json

COLORS = {
    "broadcast_baseline": "#555555",
    "role_linear": "#0072B2",
    "role_quadratic": "#D55E00",
}
LABELS = {
    "broadcast_baseline": "broadcast baseline",
    "role_linear": "linear ancillas",
    "role_quadratic": "quadratic ancillas",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/curie-weiss-encoding-comparison-v1/report/report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/curie-weiss-encoding-comparison-v1/summary"),
    )
    args = parser.parse_args(argv)
    report = read_immutable_json(args.report)
    if report.get("completed_new_fits") != 20:
        raise RuntimeError("plot requires all twenty new fits")
    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    betas = np.asarray([row["beta"] for row in report["per_temperature"]])
    for arm in COLORS:
        values = np.asarray(
            [row["arms"][arm]["profile_tvd"] for row in report["per_temperature"]]
        )
        axes[0].plot(
            betas,
            values,
            color=COLORS[arm],
            marker="o",
            label=LABELS[arm],
        )
    axes[0].set_xlabel(r"inverse temperature $\beta$")
    axes[0].set_ylabel("profile TVD")
    axes[0].axvline(1.0, color="0.75", linestyle="--", linewidth=0.8)

    baseline = report["held_out_seed_means"]["broadcast_baseline"]["profile_tvd"]
    linear = report["held_out_seed_means"]["role_linear"]["profile_tvd"]
    quadratic = report["held_out_seed_means"]["role_quadratic"]["profile_tvd"]
    seeds = np.arange(10)
    axes[1].scatter(
        seeds - 0.08,
        [baseline[str(seed)] - linear[str(seed)] for seed in seeds],
        color=COLORS["role_linear"],
        label="baseline minus linear",
    )
    axes[1].scatter(
        seeds + 0.08,
        [baseline[str(seed)] - quadratic[str(seed)] for seed in seeds],
        color=COLORS["role_quadratic"],
        label="baseline minus quadratic",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("optimizer seed")
    axes[1].set_ylabel("held-out profile TVD improvement")
    axes[1].set_xticks(seeds)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    for suffix, options in (("pdf", {}), ("png", {"dpi": 220})):
        figure.savefig(
            args.output / f"encoding_comparison.{suffix}",
            bbox_inches="tight",
            **options,
        )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Figure generator for Experiment 2 (Two-Dimensional Multi-Conditioning)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parents[1]
RESULTS_PATH = ROOT / "results" / "multidimensional-2d-v1" / "summary.json"
OUTPUT_FIG_PDF = (
    ROOT / "paper" / "gaussian_figures" / "multidimensional_2d_surface.pdf"
)
OUTPUT_FIG_PNG = (
    ROOT / "paper" / "gaussian_figures" / "multidimensional_2d_surface.png"
)


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"Results file not found: {RESULTS_PATH}. Run the experiment first.")
        return

    OUTPUT_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS_PATH.read_text())
    summary = data["summary"]
    grid_prof = np.asarray(summary["grid_profile_tvd_mean"])
    grid_raw = np.asarray(summary["grid_raw_tvd_mean"])
    eval_grid = np.asarray(data["eval_grid_1d"])
    train_grid = np.asarray(data["train_grid_1d"])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
        }
    )

    _fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel (a): 2D Profile TVD Heatmap
    ax = axes[0]
    im = ax.imshow(
        grid_prof,
        origin="lower",
        extent=[
            eval_grid[0] - 0.1,
            eval_grid[-1] + 0.1,
            eval_grid[0] - 0.1,
            eval_grid[-1] + 0.1,
        ],
        cmap="Blues_r",
        vmin=0.0,
        vmax=0.35,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Profile TVD")

    # Mark training grid points (red circles) vs evaluation grid points
    for tx in train_grid:
        for ty in train_grid:
            ax.scatter(
                tx,
                ty,
                color="#D55E00",
                s=55,
                edgecolors="black",
                zorder=5,
                label="Train Coord" if tx == 0.1 and ty == 0.1 else "",
            )

    ax.set_xlabel(r"Mean Condition $c_\mu$ (Peak Location)")
    ax.set_ylabel(r"Width Condition $c_\sigma$ (Variance)")
    ax.set_title(
        r"\textbf{a} 2D Profile TVD Surface",
        loc="left",
    )
    ax.set_xticks(eval_grid)
    ax.set_yticks(eval_grid)
    ax.legend(frameon=True, loc="upper right")

    # Panel (b): 2D Raw Bitstring TVD Heatmap
    ax = axes[1]
    im2 = ax.imshow(
        grid_raw,
        origin="lower",
        extent=[
            eval_grid[0] - 0.1,
            eval_grid[-1] + 0.1,
            eval_grid[0] - 0.1,
            eval_grid[-1] + 0.1,
        ],
        cmap="Greens_r",
        vmin=0.0,
        vmax=0.35,
    )
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="Raw TVD")

    for tx in train_grid:
        for ty in train_grid:
            ax.scatter(
                tx,
                ty,
                color="#D55E00",
                s=55,
                edgecolors="black",
                zorder=5,
            )

    ax.set_xlabel(r"Mean Condition $c_\mu$ (Peak Location)")
    ax.set_ylabel(r"Width Condition $c_\sigma$ (Variance)")
    ax.set_title(
        r"\textbf{b} 2D Raw Bitstring TVD Surface",
        loc="left",
    )
    ax.set_xticks(eval_grid)
    ax.set_yticks(eval_grid)

    # Panel (c): Comparison vs Bilinear Baseline on 16 Held-out coordinates
    ax = axes[2]
    runs = data["runs"]
    mod_heldout = [r["heldout_profile_tvd_mean"] for r in runs]
    bilin_heldout = [r["bilinear_heldout_profile_tvd_mean"] for r in runs]

    seeds = np.arange(len(runs))
    width = 0.35
    ax.bar(
        seeds - width / 2,
        mod_heldout,
        width,
        label="2D Shared Conditional IQP",
        color="#0072B2",
        alpha=0.9,
    )
    ax.bar(
        seeds + width / 2,
        bilin_heldout,
        width,
        label="2D Bilinear Barycentric Mixture",
        color="#7F7F7F",
        alpha=0.7,
    )

    ax.axhline(
        np.mean(mod_heldout),
        color="#0072B2",
        linestyle="--",
        linewidth=1.5,
        label=f"IQP Mean ({np.mean(mod_heldout):.3f})",
    )
    ax.axhline(
        np.mean(bilin_heldout),
        color="#7F7F7F",
        linestyle=":",
        linewidth=1.5,
        label=f"Bilinear Mean ({np.mean(bilin_heldout):.3f})",
    )

    ax.set_xlabel("Matched Optimization Seed")
    ax.set_ylabel("Held-Out 2D Profile TVD (16 Test Points)")
    ax.set_title(
        r"\textbf{c} 2D Interpolation vs Bilinear Baseline",
        loc="left",
    )
    ax.set_xticks(seeds)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_PDF, bbox_inches="tight")
    plt.savefig(OUTPUT_FIG_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to:\n  - {OUTPUT_FIG_PDF}\n  - {OUTPUT_FIG_PNG}")


if __name__ == "__main__":
    main()

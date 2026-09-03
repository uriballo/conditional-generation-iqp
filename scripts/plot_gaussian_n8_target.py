#!/usr/bin/env python3
"""Plot the analytic n=8 Gaussian target family used in the production campaign."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from journal_style import (  # noqa: E402
    MM,
    SINGLE_COL_WIDTH,
    apply_journal_style,
    save_figure,
)

from iqp_cond.gaussian_data import gaussian_target, target_mean  # noqa: E402

TRAINING = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
HELD_OUT = np.asarray([0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375])


def make_plot(output: Path) -> None:
    """Render exact shell probabilities over the continuous condition coordinate."""
    conditions = np.linspace(0.0, 1.0, 401)
    probabilities = np.asarray(
        [
            gaussian_target(8, 0.5, float(condition)).shell_probabilities
            for condition in conditions
        ]
    )
    weights = np.arange(9)
    edges = np.r_[conditions, 1.0]
    y_edges = np.arange(-0.5, 9.5)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 68 * MM), layout="constrained")
    mesh = ax.pcolormesh(
        edges,
        y_edges,
        probabilities.T,
        cmap="viridis",
        shading="flat",
        rasterized=True,
    )
    mean = np.asarray([target_mean(8, float(condition)) for condition in conditions])
    ax.plot(conditions, mean, color="white", linewidth=1.2, zorder=3)
    ax.plot(conditions, mean, color="black", linewidth=0.55, zorder=4)
    for values, filled in ((TRAINING, True), (HELD_OUT, False)):
        means = np.asarray([target_mean(8, float(condition)) for condition in values])
        ax.scatter(
            values,
            means,
            s=16,
            facecolors="black" if filled else "white",
            edgecolors="white" if filled else "black",
            linewidths=0.7,
            zorder=5,
            clip_on=False,
            label="training" if filled else "held-out",
        )
    ax.set(
        xlim=(0, 1),
        ylim=(-0.5, 8.5),
        xlabel=r"Condition $c$",
        ylabel="Hamming weight $k$",
    )
    ax.set_yticks(weights)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", ncol=2, handletextpad=0.3, columnspacing=0.8)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=28)
    colorbar.set_label("Shell probability")
    colorbar.ax.minorticks_off()
    save_figure(fig, output.with_suffix(""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "Overleaf" / "figures" / "gaussian_n8_target",
    )
    args = parser.parse_args()
    apply_journal_style()
    make_plot(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

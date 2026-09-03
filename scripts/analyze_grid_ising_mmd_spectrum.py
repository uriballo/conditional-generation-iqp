#!/usr/bin/env python3
"""Audit how the production Hamming-RBF bandwidths see grid-Ising targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from journal_style import (  # noqa: E402
    DOUBLE_COL_WIDTH,
    MM,
    apply_journal_style,
    save_figure,
)

from iqp_cond.grid_ising import spectrum_diagnostic  # noqa: E402

SHAPES = ((2, 4), (3, 4), (4, 4))
BETAS = (0.2, 0.35, 0.44, 0.55, 0.8)
FIELDS = (-0.1, 0.0, 0.1)


def run_analysis() -> list[dict[str, object]]:
    return [
        spectrum_diagnostic(shape, beta, field)
        for shape in SHAPES
        for beta in BETAS
        for field in FIELDS
    ]


def make_plot(records: list[dict[str, object]], output: Path) -> None:
    """Plot 4x4 zero-field signal magnitude and its two-body locality split."""
    selected = [
        record
        for record in records
        if record["shape"] == [4, 4] and record["field"] == 0.0
    ]
    selected.sort(key=lambda record: float(record["beta"]))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL_WIDTH, 66 * MM),
        layout="constrained",
    )

    betas = np.asarray([record["beta"] for record in selected], dtype=float)
    for band_index in range(3):
        band_rows = [record["bands"][band_index] for record in selected]
        axes[0].plot(
            betas,
            [row["total_mmd_squared_vs_uniform"] for row in band_rows],
            marker="o",
            color=colors[band_index],
            label=rf"bandwidth {band_index + 1}",
        )
        axes[1].plot(
            betas,
            [row["weight2_edge_fraction"] for row in band_rows],
            marker="o",
            color=colors[band_index],
            label=rf"bandwidth {band_index + 1}",
        )
    axes[0].set(
        xlabel=r"Inverse temperature $\beta$",
        ylabel=r"Exact MMD$^2$ from uniform",
    )
    axes[1].set(
        xlabel=r"Inverse temperature $\beta$",
        ylabel="Edge share of two-body signal",
        ylim=(0, 1.02),
    )
    axes[0].legend(handlelength=1.4)
    for label, ax in zip("ab", axes, strict=True):
        ax.axvline(0.4406868, color="black", linestyle=":", linewidth=1.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.15, 1.05, f"({label})", transform=ax.transAxes)
    save_figure(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "grid_ising_exploration" / "spectrum.json",
    )
    args = parser.parse_args()
    records = run_analysis()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    apply_journal_style()
    make_plot(records, args.output.with_name("spectrum"))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

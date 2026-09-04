#!/usr/bin/env python3
"""Plot the completed Gaussian exact-TVD ancilla ladders for Figure 3."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iqp_cond.exact_tvd_campaign import load_manifest  # noqa: E402
from iqp_cond.exact_tvd_results import aggregate, load_records  # noqa: E402

ANCILLAS = (0, 1, 3, 6, 10)
FIELDS = (
    ("conditioned_tvd", "Conditioned", "o", "-", "#0072B2"),
    (
        "interpolation_tvd",
        "Adjacent independent interpolation",
        "s",
        "--",
        "#D55E00",
    ),
    ("oracle_tvd", "Target-aware oracle", "^", ":", "#009E73"),
)


def _style() -> None:
    latex = shutil.which("latex") is not None
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.prop_cycle": mpl.cycler(
                color=["#0072B2", "#D55E00", "#009E73"]
            ),
            "lines.linewidth": 1.5,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "lines.markersize": 4,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            **(
                {
                    "text.usetex": True,
                    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
                }
                if latex
                else {"text.usetex": False, "mathtext.fontset": "cm"}
            ),
        }
    )


def _bands(
    seed_rows: list[dict], n_visible: int, field: str
) -> tuple[np.ndarray, ...]:
    medians: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for ancillas in ANCILLAS:
        values = np.asarray(
            [
                float(row[field])
                for row in seed_rows
                if row["dataset"] == "gaussian"
                and int(row["visible_qubits"]) == n_visible
                and row["group"] == "held_out"
                and int(row["ancillas"]) == ancillas
            ],
            dtype=np.float64,
        )
        if values.size != 10 or not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"expected ten finite Gaussian n={n_visible}, r={ancillas} values "
                f"for {field}; found {values.size}"
            )
        medians.append(float(np.median(values)))
        lower.append(float(np.quantile(values, 0.25)))
        upper.append(float(np.quantile(values, 0.75)))
    return np.asarray(medians), np.asarray(lower), np.asarray(upper)


def _draw_completed(ax: plt.Axes, seed_rows: list[dict], n_visible: int) -> None:
    for field, label, marker, line_style, color in FIELDS:
        median, lower, upper = _bands(seed_rows, n_visible, field)
        ax.plot(
            ANCILLAS,
            median,
            marker=marker,
            linestyle=line_style,
            color=color,
            label=label,
            zorder=3,
        )
        ax.fill_between(
            ANCILLAS,
            lower,
            upper,
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=2,
        )


def _format_axis(ax: plt.Axes, panel: str, n_visible: int) -> None:
    ax.set_title(rf"\textbf{{{panel}}}\quad $n={n_visible}$", loc="left")
    ax.set_xlabel(r"Ancillas $r$")
    ax.set_xlim(-0.6, 10.6)
    ax.set_xticks(ANCILLAS)
    ax.set_ylim(0.0, 0.34)
    ax.set_yticks((0.0, 0.1, 0.2, 0.3))
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.5, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        type=Path,
        default=ROOT / "results/exact-tvd-ancilla-v1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/exact_tvd_ancilla_v1.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "Overleaf/figures/gaussian_results.pdf",
    )
    args = parser.parse_args()

    config = load_manifest(args.manifest)
    records = load_records(
        args.campaign / "evaluations", config, require_complete=False
    )
    seed_rows = aggregate(records, config)["seed_rows"]

    _style()
    mm = 1 / 25.4
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(170 * mm, 61 * mm),
        sharey=True,
    )
    # Reserve space for the shared legend below both axes.
    fig.subplots_adjust(left=0.085, right=0.985, top=0.88, bottom=0.30, wspace=0.28)

    _draw_completed(axes[0], seed_rows, 8)
    _draw_completed(axes[1], seed_rows, 12)
    _format_axis(axes[0], "a", 8)
    _format_axis(axes[1], "b", 12)
    axes[0].set_ylabel("Held-out exact TVD")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        columnspacing=1.8,
        handlelength=3.0,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Publication figures for the frozen Ashkin--Teller confirmation result."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
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
        "legend.frameon": False,
        "axes.xmargin": 0.02,
        "axes.ymargin": 0.02,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        **(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
            }
            if shutil.which("latex")
            else {"text.usetex": False, "mathtext.fontset": "cm"}
        ),
    }
)

METHODS = [
    "cat_encoder_only",
    "cat",
    "bond_encoder_only",
    "bond",
    "log_composition",
    "log_linear",
]
LABELS = ["Cat/input", "Cat/IQP", "Bond/input", "Bond/IQP", "Log/comp.", "Log/linear"]
COLORS = ["#7F7F7F", OKABE_ITO[0], "#7F7F7F", OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[4]]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _metric(result: dict, method: str, metric: str) -> dict:
    return result["models"][method]["test"][metric]


def _style_axes(ax: mpl.axes.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(bottom=0)


def _save(fig: mpl.figure.Figure, output: Path, stem: str) -> list[str]:
    paths = [output / f"{stem}.pdf", output / f"{stem}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return [str(path) for path in paths]


def summary_figure(result: dict, output: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 85 * MM), layout="constrained")
    x = np.arange(len(METHODS), dtype=float)
    jitter = np.linspace(-0.12, 0.12, 10)
    for ax, metric, ylabel in zip(
        axes,
        ("tvd", "correlation_rmse"),
        ("Joint TVD", "Connected-correlation RMSE"),
        strict=True,
    ):
        for i, method in enumerate(METHODS):
            entry = _metric(result, method, metric)
            seeds = np.asarray(entry["seed_means"], dtype=float)
            if method.endswith("encoder_only"):
                ax.scatter(x[i], float(entry["mean"]), s=25, color=COLORS[i])
                continue
            ax.scatter(
                x[i] + jitter[: len(seeds)], seeds, s=9, alpha=0.28, color=COLORS[i]
            )
            mean = float(entry["mean"])
            lo, hi = entry["ci95"]
            ax.errorbar(
                x[i],
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt="o",
                ms=5,
                color=COLORS[i],
                capsize=3,
                lw=1.2,
                zorder=4,
            )
        ax.set_xticks(x, [label.replace("/", "/\n") for label in LABELS])
        ax.set_ylabel(ylabel)
        _style_axes(ax)
    axes[0].set_title("Joint distribution")
    axes[1].set_title("Interlayer dependence")
    axes[1].text(
        0.99,
        -0.25,
        "IQP/baselines: 10 seeds and 95"
        + (r"\%" if mpl.rcParams["text.usetex"] else "%")
        + " CI; inputs: fixed",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    return _save(fig, output, "confirmation_paired_summary")


def grid_figure(result: dict, output: Path) -> list[str]:
    coords = np.asarray(result["test_conditions"], dtype=float)
    levels = np.unique(coords)
    if len(levels) != 3 or len(coords) != 9:
        raise ValueError(
            "test_conditions must contain the nine 3x3 confirmation points"
        )

    def grid(metric: str, method: str) -> np.ndarray:
        values = np.asarray(
            _metric(result, method, metric)["condition_means"], dtype=float
        )
        return np.array(
            [
                [values[np.where((coords == (a, b)).all(axis=1))[0][0]] for a in levels]
                for b in levels
            ]
        )

    fig, axes = plt.subplots(2, 2, figsize=(170 * MM, 135 * MM), layout="constrained")
    specs = [
        ("tvd", "cat", "Cat/IQP TVD"),
        ("tvd", "bond", "Bond/IQP TVD"),
        ("correlation_rmse", "cat", "Cat/IQP correlation RMSE"),
        ("correlation_rmse", "bond", "Bond/IQP correlation RMSE"),
    ]
    for ax, (metric, method, title) in zip(axes.flat, specs, strict=True):
        values = grid(metric, method)
        row = 0 if metric == "tvd" else 1
        vmax = max(np.max(grid(metric, "cat")), np.max(grid(metric, "bond")))
        image = ax.imshow(
            values,
            origin="lower",
            vmin=0,
            vmax=vmax,
            interpolation="none",
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xticks(range(3), [f"{v:.2f}" for v in levels])
        ax.set_yticks(range(3), [f"{v:.2f}" for v in levels])
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        for iy in range(3):
            for ix in range(3):
                ax.text(
                    ix,
                    iy,
                    f"{values[iy, ix]:.3f}",
                    ha="center",
                    va="center",
                    color="black" if values[iy, ix] > vmax * 0.55 else "white",
                    fontsize=8,
                )
        ax.minorticks_off()
        if ax is axes[row, 1]:
            fig.colorbar(image, ax=axes[row, :], shrink=0.86, label="error")
    return _save(fig, output, "confirmation_test_grid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ashkin-teller-confirmation-2026-09-07/figures"),
    )
    args = parser.parse_args()
    result = _load(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    files = summary_figure(result, args.output) + grid_figure(result, args.output)
    provenance = {
        "input": str(args.input),
        "input_sha256": _hash(args.input),
        "aggregation": (
            "provided test means and paired-seed bootstrap ci95; "
            "condition means reshaped by (a,b)"
        ),
        "uncertainty": (
            "ci95 fields from input; plotted as paired-seed bootstrap "
            "intervals, not condition intervals"
        ),
        "files": {str(Path(p).name): _hash(Path(p)) for p in files},
    }
    (args.output / "figure-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

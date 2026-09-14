#!/usr/bin/env python3
"""Publication summary figure for the two Ashkin--Teller sizes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
            {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"}
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
    try:
        with path.open() as stream:
            result = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read report JSON {path}: {exc}") from exc
    if not isinstance(result, dict) or not isinstance(result.get("models"), dict):
        raise ValueError(f"invalid report schema in {path}: missing models")
    missing = [m for m in METHODS if m not in result["models"]]
    if missing:
        raise ValueError(f"invalid report schema in {path}: missing models {missing}")
    for method in METHODS:
        entry = result["models"][method].get("test", {})
        for metric in ("tvd", "correlation_rmse"):
            if (
                metric not in entry
                or "mean" not in entry[metric]
                or "ci95" not in entry[metric]
            ):
                raise ValueError(
                    f"invalid report schema in {path}: {method}/test/{metric}"
                )
    conditions = np.asarray(result.get("test_conditions"), dtype=float)
    levels = (0.1, 0.22, 0.38, 0.54, 0.7)
    expected = np.asarray([(a, b) for a in levels[1:-1] for b in levels[1:-1]])
    if conditions.ndim != 2 or conditions.shape != (9, 2):
        raise ValueError(f"invalid report schema in {path}: test_conditions")
    if not np.all(np.isfinite(conditions)) or not np.allclose(
        conditions, expected
    ):
        raise ValueError(
            f"invalid report schema in {path}: expected 3x3 interior test grid"
        )
    train = np.asarray(result.get("train_conditions"), dtype=float)
    if train.shape != (16, 2) or not np.all(np.isfinite(train)):
        raise ValueError(
            f"invalid report schema in {path}: expected 16 training conditions"
        )
    expected_train = np.asarray(
        [(a, b) for a in levels for b in levels if a in (0.1, 0.7) or b in (0.1, 0.7)]
    )
    if not np.allclose(train, expected_train):
        raise ValueError(f"invalid perimeter training grid in {path}")
    for metric in ("tvd", "correlation_rmse"):
        for method in METHODS:
            value = result["models"][method]["test"][metric]
            mean = float(value["mean"])
            ci = np.asarray(value["ci95"], dtype=float)
            if not np.isfinite(mean) or ci.shape != (2,) or not np.all(np.isfinite(ci)):
                raise ValueError(f"invalid finite summary in {path}: {method}/{metric}")
            if not ci[0] <= mean <= ci[1]:
                raise ValueError(f"invalid ordered ci95 in {path}: {method}/{metric}")
    for method in ("cat", "bond"):
        values = result["models"][method]["test"]["tvd"].get("condition_means")
        if values is None or len(values) != len(conditions):
            raise ValueError(f"invalid report schema in {path}: condition_means")
    return result


def _metric(report: dict, method: str, metric: str) -> tuple[float, float, float]:
    value = report["models"][method]["test"][metric]
    lo, hi = map(float, value["ci95"])
    return float(value["mean"]), lo, hi


def _style(ax: mpl.axes.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(bottom=0)


def _summary_panel(
    ax: mpl.axes.Axes, reports: list[dict], metric: str, ylabel: str
) -> None:
    x = np.arange(len(METHODS), dtype=float)
    offsets = [-0.16, 0.16]
    for size_idx, report in enumerate(reports):
        for i, method in enumerate(METHODS):
            mean, lo, hi = _metric(report, method, metric)
            xx = x[i] + offsets[size_idx]
            marker = "o" if size_idx == 0 else "s"
            if method.endswith("encoder_only"):
                ax.scatter(xx, mean, color=COLORS[i], marker=marker, s=22, zorder=3)
            else:
                ax.errorbar(
                    xx,
                    mean,
                    yerr=[[mean - lo], [hi - mean]],
                    fmt=marker,
                    color=COLORS[i],
                    ms=4,
                    capsize=2.5,
                    lw=1.0,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                )
    ax.set_xticks(x, [label.replace("/", "/\n") for label in LABELS])
    ax.set_ylabel(ylabel)
    _style(ax)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.20)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color="#555555", linestyle="None", label="n=8"),
            Line2D([], [], marker="s", color="#555555", linestyle="None", label="n=12"),
        ],
        loc="upper left",
        ncol=2,
        handletextpad=0.2,
        columnspacing=0.7,
    )


def _condition_panel(ax: mpl.axes.Axes, reports: list[dict]) -> None:
    train = np.asarray(reports[0]["train_conditions"], dtype=float)
    test = np.asarray(reports[0]["test_conditions"], dtype=float)
    ax.scatter(
        train[:, 0],
        train[:, 1],
        marker="o",
        s=25,
        color="#777777",
        label="training (16)",
    )
    ax.scatter(
        test[:, 0],
        test[:, 1],
        marker="s",
        s=28,
        facecolor="white",
        edgecolor=OKABE_ITO[0],
        label="held-out (9)",
    )
    ax.set_xlabel("$a$")
    ax.set_ylabel("$b$")
    ax.set_title("Condition split")
    ax.set_xlim(0.05, 0.75)
    ax.set_ylim(0.05, 0.75)
    ax.set_xticks([0.1, 0.22, 0.38, 0.54, 0.7])
    ax.set_yticks([0.1, 0.22, 0.38, 0.54, 0.7])
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), handletextpad=0.2)
    ax.minorticks_off()


def make_figure(n8: dict, n12: dict, output: Path) -> list[Path]:
    fig = plt.figure(figsize=(170 * MM, 110 * MM), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 2))
    split = fig.add_subplot(grid[:, 0])
    tvd = fig.add_subplot(grid[0, 1])
    correlation = fig.add_subplot(grid[1, 1])
    _summary_panel(tvd, [n8, n12], "tvd", "Joint TVD")
    _summary_panel(
        correlation, [n8, n12], "correlation_rmse", "Connected-bond RMSE"
    )
    _condition_panel(split, [n8, n12])
    for ax, label in zip((split, tvd, correlation), ("a", "b", "c"), strict=True):
        if label:
            ax.text(
                -0.12,
                1.04,
                label,
                transform=ax.transAxes,
                fontweight="bold",
                va="bottom",
            )
    paths = [
        output / "ashkin_teller_paper_summary.pdf",
        output / "ashkin_teller_paper_summary.png",
    ]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n8_report", type=Path)
    parser.add_argument("n12_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n8_report.resolve() == args.n12_report.resolve():
        raise SystemExit("n8_report and n12_report must be distinct files")
    n8, n12 = _load(args.n8_report), _load(args.n12_report)
    if set(n8["seeds"]) != set(range(2701, 2711)):
        raise ValueError("n8 report does not contain expected seeds 2701..2710")
    if set(n12["seeds"]) != set(range(3711, 3721)):
        raise ValueError("n12 report does not contain expected seeds 3711..3720")
    args.output.mkdir(parents=True, exist_ok=True)
    files = make_figure(n8, n12, args.output)
    provenance = {
        "inputs": {"n8": str(args.n8_report), "n12": str(args.n12_report)},
        "input_sha256": {"n8": _hash(args.n8_report), "n12": _hash(args.n12_report)},
        "aggregation": (
            "plots report means over nine held-out conditions then ten seeds, "
            "separately by size; condition panel shows the shared frozen split"
        ),
        "uncertainty": (
            "error bars are report-provided paired-seed bootstrap ci95 intervals"
        ),
        "models": METHODS,
        "files": {path.name: _hash(path) for path in files},
    }
    (args.output / "figure-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

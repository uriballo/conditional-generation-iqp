#!/usr/bin/env python3
"""Plot post-hoc structural diagnostics for an Ashkin--Teller result."""

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
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "axes.xmargin": 0.02,
        "axes.ymargin": 0.04,
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


def _summary(
    result: dict, method: str, metric: str
) -> tuple[float, np.ndarray, tuple[float, float]]:
    entry = result["models"][method][metric]
    return (
        float(entry["mean"]),
        np.asarray(entry.get("seed_means", []), float),
        tuple(entry["ci95"]),
    )


def _save(fig: mpl.figure.Figure, output: Path) -> list[Path]:
    paths = [
        output / "ashkin_teller_structure_diagnostics.pdf",
        output / "ashkin_teller_structure_diagnostics.png",
    ]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def make_figure(result: dict, output: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 85 * MM), layout="constrained")
    x = np.arange(len(METHODS), dtype=float)
    jitter = np.linspace(-0.11, 0.11, 10)
    labels = [label.replace("/", "/\n") for label in LABELS]
    for ax, metric, ylabel in zip(
        axes,
        ("delta_spatial", "delta_dependence"),
        (
            "TVD increase after shell averaging",
            "TVD increase after removing dependence",
        ),
        strict=True,
    ):
        for i, method in enumerate(METHODS):
            mean, seeds, ci = _summary(result, method, metric)
            if method.endswith("encoder_only"):
                ax.scatter(x[i], mean, s=25, color=COLORS[i], zorder=3)
            else:
                ax.scatter(
                    x[i] + jitter[: len(seeds)],
                    seeds,
                    s=9,
                    alpha=0.28,
                    color=COLORS[i],
                    zorder=2,
                )
                ax.errorbar(
                    x[i],
                    mean,
                    yerr=[[mean - ci[0]], [ci[1] - mean]],
                    fmt="o",
                    ms=5,
                    color=COLORS[i],
                    capsize=3,
                    lw=1.2,
                    zorder=4,
                )
        ax.axhline(0, color="#555555", lw=0.8, ls="--", zorder=1)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.5, len(METHODS) - 0.5)
    title = result.get("label", "").replace("-", " ")
    fig.suptitle(f"Ashkin--Teller structural diagnostics ({title})", fontsize=10)
    return _save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    files = make_figure(result, args.output)
    provenance = {
        "input": str(args.input),
        "input_sha256": _hash(args.input),
        "label": result.get("label"),
        "models": METHODS,
        "aggregation": (
            "provided means and paired-seed 95% intervals; deterministic "
            "inputs are single points"
        ),
        "uncertainty": (
            "seed means and ci95 fields from the diagnostic result; no "
            "condition-level uncertainty plotted"
        ),
        "files": {path.name: _hash(path) for path in files},
    }
    (args.output / "structure-figure-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

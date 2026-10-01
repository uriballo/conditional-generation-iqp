"""Render phase-lattice scaling figures from an exported JSON payload."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

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

SIZES = (64, 128, 256, 512, 1024, 2048)
STEPS = tuple(range(0, 5001, 500))
TRAIN_BETAS = (0.6, 0.8, 1.2, 1.4)
HELD_BETAS = (0.95, 1.0, 1.05, 0.4, 1.6)


def _curve(ax, x, values, color, label):
    values = np.asarray(values)
    median = np.median(values, axis=0)
    lo, hi = np.quantile(values, (0.25, 0.75), axis=0)
    ax.plot(x, median, "o-", color=color, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.16)


def _axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Optimizer update")
    ax.set_xticks((0, 1000, 2000, 3000, 4000, 5000))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))


def _validate(payload: dict) -> tuple[dict, dict]:
    ancilla = payload.get("ancilla")
    scaling = payload.get("scaling")
    if not isinstance(ancilla, dict) or not isinstance(scaling, dict):
        raise ValueError("payload requires ancilla and scaling objects")
    matrix = np.asarray(ancilla.get("mixed_ancilla_per_seed"), dtype=float)
    widths = ancilla.get("ancilla_widths")
    if (
        matrix.shape != (5, 3)
        or not np.isfinite(matrix).all()
        or widths != [0, 1, 3, 6, 10]
    ):
        raise ValueError("invalid ancilla matrix or widths")
    data = scaling.get("trajectory_data", scaling.get("scaling"))
    if not isinstance(data, dict):
        raise ValueError("scaling requires trajectory_data")
    if tuple(data.get("sizes", ())) != SIZES or tuple(data.get("steps", ())) != STEPS:
        raise ValueError(
            "sizes and checkpoint cadence must match the complete protocol"
        )
    if list(data.get("seeds", ())) != [100, 101, 102]:
        raise ValueError("seeds must be 100, 101, 102")
    if data.get("partial") is not False or data.get("missing_sizes") != []:
        raise ValueError("export must be complete")
    if set(data.get("curves", {})) != {str(n) for n in SIZES}:
        raise ValueError("curves must contain exactly six sizes")
    for n in SIZES:
        item = data.get("curves", {}).get(str(n))
        if not isinstance(item, dict) or set(item) != {"training", "held_out"}:
            raise ValueError(f"missing size {n}")
        for grid, betas in (("training", TRAIN_BETAS), ("held_out", HELD_BETAS)):
            cell = item.get(grid)
            if not isinstance(cell, dict) or tuple(cell.get("betas", ())) != betas:
                raise ValueError(f"invalid {grid} condition grid for n={n}")
            values = np.asarray(cell.get("per_seed"), dtype=float)
            overall = np.asarray(cell.get("overall_per_seed"), dtype=float)
            if values.shape != (3, 11, len(betas)) or overall.shape != (3, 11):
                raise ValueError(f"invalid {grid} array shape for n={n}")
            if not np.isfinite(values).all() or not np.isfinite(overall).all():
                raise ValueError(f"non-finite {grid} values for n={n}")
            if not np.allclose(overall, values.mean(axis=2), rtol=0, atol=1e-15):
                raise ValueError(f"overall_per_seed mismatch for n={n}, {grid}")
    return ancilla, data


def render_ancilla(ancilla: dict, output: Path):
    fig, ax = plt.subplots(figsize=(85 * MM, 70 * MM), layout="constrained")
    _curve(
        ax,
        ancilla["ancilla_widths"],
        np.asarray(ancilla["mixed_ancilla_per_seed"]).T,
        OKABE_ITO[0],
        "Mixed visible-ancilla coupling",
    )
    ax.set_title("16 visible qubits", fontsize=9)
    ax.set_xlabel("Ancilla qubits $r$")
    ax.set_ylabel("Exact bitstring TVD")
    ax.set_xticks((0, 1, 3, 6, 10))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"))
    plt.close(fig)


def render_scaling(data: dict, output: Path, *, grid: str = "held_out"):
    if grid not in ("training", "held_out"):
        raise ValueError("grid must be training or held_out")
    fig, scale = plt.subplots(figsize=(85 * MM, 70 * MM), layout="constrained")
    for n in data["sizes"]:
        color = OKABE_ITO[list(SIZES).index(n)]
        _curve(
            scale,
            STEPS,
            data["curves"][str(n)][grid]["overall_per_seed"],
            color,
            rf"$n={n}$",
        )
    label = "Held-out" if grid == "held_out" else "Training"
    scale.set_title(f"Lattice scaling ({label.lower()})", fontsize=9)
    scale.set_ylabel(f"{label} sampled MMD$^2$")
    _axes(scale)
    scale.legend(fontsize=7, ncol=2)
    if data["partial"]:
        scale.set_title("Lattice scaling (interim)", fontsize=9)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return {"scaling": data}


def render_diagnostics(data: dict, output: Path):
    with PdfPages(output.with_suffix(".pdf")) as pdf:
        for grid, title in [
            ("training", "Training conditions"),
            ("held_out", "Held-out conditions"),
        ]:
            ncols = 1 if len(data["sizes"]) == 1 else 2
            nrows = (len(data["sizes"]) + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(170 * MM, (55 * nrows + 20) * MM),
                squeeze=False,
                layout="constrained",
            )
            handles = []
            for index, n in enumerate(data["sizes"]):
                ax = axes.flat[index]
                item = data["curves"][str(n)][grid]
                values = np.asarray(item["per_seed"])
                for j, beta in enumerate(item["betas"]):
                    _curve(
                        ax, STEPS, values[:, :, j], OKABE_ITO[j], rf"$\beta={beta:g}$"
                    )
                ax.set_title(rf"$n={n}$", fontsize=9)
                ax.set_ylabel("Sampled MMD$^2$")
                _axes(ax)
                handles, labels = ax.get_legend_handles_labels()
            for ax in list(axes.flat)[len(data["sizes"]) :]:
                ax.set_visible(False)
            suffix = "; interim" if data["partial"] else ""
            fig.suptitle(title + "; median/IQR over three seeds" + suffix, fontsize=10)
            fig.legend(
                handles,
                labels,
                loc="outside lower center",
                ncol=len(handles),
                fontsize=8,
            )
            pdf.savefig(fig)
            fig.savefig(output.parent / (output.name + "_" + grid + ".png"), dpi=300)
            plt.close(fig)


def render_train_comparison(data: dict, output: Path):
    ncols = 1 if len(data["sizes"]) == 1 else 2
    nrows = (len(data["sizes"]) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(170 * MM, (55 * nrows + 15) * MM),
        squeeze=False,
        layout="constrained",
    )
    for index, n in enumerate(data["sizes"]):
        ax = axes.flat[index]
        for grid, label, color in (
            ("training", "Training conditions", OKABE_ITO[0]),
            ("held_out", "Held-out conditions", OKABE_ITO[1]),
        ):
            _curve(
                ax,
                STEPS,
                data["curves"][str(n)][grid]["overall_per_seed"],
                color,
                label,
            )
        ax.set_title(rf"$n={n}$", fontsize=9)
        ax.set_ylabel("Sampled MMD$^2$")
        _axes(ax)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(data["sizes"]) :]:
        ax.set_visible(False)
    title = "Independent train/held-out MMD; seed median/IQR"
    if data["partial"]:
        title += "; interim"
    fig.suptitle(title, fontsize=10)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).with_name("figure_data.json")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    ancilla, scaling = _validate(json.loads(args.data.read_text()))
    args.output_root.mkdir(parents=True, exist_ok=True)
    render_ancilla(ancilla, args.output_root / "curie_weiss_ancilla")
    render_scaling(scaling, args.output_root / "curie_weiss_scaling")
    render_scaling(
        scaling, args.output_root / "curie_weiss_scaling_train", grid="training"
    )
    render_diagnostics(scaling, args.output_root / "per_condition_mmd")
    render_train_comparison(scaling, args.output_root / "train_held_out_mmd")


if __name__ == "__main__":
    main()

"""Render publication drafts from the frozen sixteen-qubit symmetry data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
BLUE = "#0072B2"
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.25,
        "lines.markersize": 3,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "axes.xmargin": 0.015,
        "axes.ymargin": 0.03,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "mathtext.fontset": "cm",
    }
)

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "paper/data/switching16/response.npz"
SUMMARY = ROOT / "paper/data/switching16/response-summary.json"
OUT = ROOT / "paper/data/switching16"
FIGURES = ROOT / "paper/figures"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(
    panels: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]], stem: str
) -> tuple[Path, Path]:
    data = np.load(INPUT)
    summary = json.loads(SUMMARY.read_text())
    beta = data["betas"]
    train = np.asarray(summary["fitting_betas"])
    fig = plt.figure(figsize=(85 * MM, 116 * MM), layout="constrained")
    grid = fig.add_gridspec(4, 1, height_ratios=[0.24, 1, 1, 0.12], hspace=0.10)
    legend_ax = fig.add_subplot(grid[0])
    legend_ax.axis("off")
    axes = [fig.add_subplot(grid[1]), fig.add_subplot(grid[2], sharex=None)]
    axes[1].sharex(axes[0])
    rug_ax = fig.add_subplot(grid[3], sharex=axes[0])
    rug_ax.set_yticks([])
    rug_ax.set_xticks([])
    rug_ax.spines[:].set_visible(False)
    rug_ax.set_ylim(0, 1)
    for number, (target, runs, encoder, ylabel, title) in enumerate(panels):
        ax = axes[number]
        ax.axvspan(1.2, 2.4, color="#eeeeee", zorder=0, linewidth=0)
        ax.axvline(1.2, color="#888888", linewidth=0.7, linestyle="-.", zorder=1)
        mean = runs.mean(axis=0)
        sd = runs.std(axis=0, ddof=1)
        ax.plot(
            beta,
            target,
            color="#222222",
            linestyle="--",
            label="Gibbs target",
            zorder=3,
        )
        ax.fill_between(
            beta, mean - sd, mean + sd, color=BLUE, alpha=0.16, linewidth=0, zorder=2
        )
        ax.plot(beta, mean, color=BLUE, label="Learned response", zorder=4)
        ax.plot(
            beta,
            encoder,
            color="#777777",
            linestyle=":",
            label="Encoder only",
            zorder=3,
        )
        ax.set_ylabel(f"{ylabel}\n{title}")
        ax.set_xlim(0, 2.4)
        ax.tick_params(labelbottom=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.01,
            0.92,
            f"({chr(97 + number)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        if number == 0:
            ax.text(
                0.60,
                1.04,
                "Within the\ntraining range",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                color="#555555",
            )
            ax.text(
                1.80,
                1.04,
                "Beyond the\ntraining range",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                color="#555555",
            )
    axes[1].set_xticks([0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4])
    rug_ax.plot(
        train,
        np.full_like(train, 0.55),
        marker="v",
        linestyle="none",
        color="#111111",
        markersize=4,
        clip_on=False,
    )
    rug_ax.set_xlim(0, 2.4)
    rug_ax.set_xlabel(r"Inverse temperature $\beta$", labelpad=1)
    handles = [
        mpl.lines.Line2D([], [], color="#222222", linestyle="--", linewidth=1.25),
        mpl.lines.Line2D([], [], color=BLUE, linewidth=1.25),
        mpl.lines.Line2D([], [], color="#777777", linestyle=":", linewidth=1.25),
    ]
    labels = ["Gibbs target", "Learned response", "Encoder only"]
    training = mpl.lines.Line2D(
        [], [], color="#111111", marker="v", linestyle="none", markersize=4
    )
    legend_ax.legend(
        [*handles, training],
        [*labels, "Training conditions"],
        ncol=2,
        loc="center",
        columnspacing=1.2,
        handlelength=2.2,
    )
    pdf, png = FIGURES / f"{stem}.pdf", FIGURES / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = np.load(INPUT)
    summary = json.loads(SUMMARY.read_text())
    assert data["balance"].shape == (5, 121) and data["fluctuation"].shape == (5, 121)
    assert np.allclose(data["betas"], np.linspace(0, 2.4, 121))
    assert summary["arrays_sha256"] == digest(INPUT)
    panels = [
        (
            data["target_balance"],
            data["balance"],
            data["encoder_balance"],
            r"$\langle E\rangle/16$",
            "Mean triangle balance",
        ),
        (
            data["target_balanced_probability"],
            data["balanced_probability"],
            data["encoder_balanced_probability"],
            r"$\Pr(E=16)$",
            "Probability of\ncomplete balance",
        ),
    ]
    pdf, png = render(panels, "conditional_symmetry_current")
    renderer_hash = digest(Path(__file__))
    receipt = {
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": digest(INPUT),
        "summary": str(SUMMARY.relative_to(ROOT)),
        "summary_sha256": digest(SUMMARY),
        "analysis_source_sha256": summary["source_sha256"],
        "renderer_source_sha256": renderer_hash,
        "output_pdf": str(pdf.relative_to(ROOT)),
        "output_png": str(png.relative_to(ROOT)),
        "n_frozen_fits": 5,
        "grid_points": 121,
        "beta_range": [0, 2.4],
        "aggregation": "mean and sample SD (ddof=1) across five frozen fits",
        "observables": ["mean triangle balance", "fully balanced-graph probability"],
        "training_betas": summary["fitting_betas"],
        "extrapolation_boundary": 1.2,
        "post_selection": True,
        "no_retraining": True,
    }
    receipt["output_pdf_sha256"] = digest(pdf)
    receipt["output_png_sha256"] = digest(png)
    (OUT / "FIGURE_CONDITIONAL.json").write_text(json.dumps(receipt, indent=2) + "\n")
    fluct_panels = [
        (
            data["target_balance"],
            data["balance"],
            data["encoder_balance"],
            r"$\langle E\rangle/16$",
            "Mean triangle balance",
        ),
        (
            data["target_fluctuation"],
            data["fluctuation"],
            data["encoder_fluctuation"],
            r"$\beta^2\operatorname{Var}(E)/16$",
            "Energy fluctuations",
        ),
    ]
    pdf, png = render(fluct_panels, "conditional_symmetry_fluctuation")
    receipt["output_pdf"] = str(pdf.relative_to(ROOT))
    receipt["output_png"] = str(png.relative_to(ROOT))
    receipt["output_pdf_sha256"] = digest(pdf)
    receipt["output_png_sha256"] = digest(png)
    receipt["observables"] = ["mean triangle balance", "fluctuation statistic"]
    receipt["note"] = (
        "The fluctuation statistic is shown as an appendix diagnostic; it is not "
        "interpreted as a verified heat capacity."
    )
    (OUT / "FIGURE_FLUCTUATION.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

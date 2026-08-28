"""Generate the canonical overview of the three manuscript datasets."""

from __future__ import annotations

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
    OKABE_ITO_COLORS,
    apply_journal_style,
    save_figure,
)

from iqp_cond.curie_weiss_data import exact_curie_weiss_target  # noqa: E402
from iqp_cond.gaussian_data import gaussian_target  # noqa: E402
from iqp_cond.physical_representability import _torch_float_tensor  # noqa: E402

CONDITIONS = np.linspace(0.0, 1.0, 5)
BETAS = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
MFC_ENERGIES = np.array([50, 75, 125, 175, 200])
MFC_ETA_BIN_COUNT = 16


def profiles() -> tuple[np.ndarray, np.ndarray]:
    """Return normalized Gaussian and n=16 Curie--Weiss target profiles."""
    gaussian = np.asarray(
        [gaussian_target(19, 0.5, float(c)).shell_probabilities for c in CONDITIONS]
    )
    curie = np.asarray(
        [exact_curie_weiss_target(16, float(beta)).profile for beta in BETAS]
    )
    return gaussian, curie


def mfc_profiles() -> np.ndarray:
    """Return MFC pseudorapidity PMFs using shared all-energy quantile bins."""
    archive = ROOT / ".scratch" / "mfc-zenodo"
    values_by_energy = {}
    all_values = []
    for energy in range(50, 201, 25):
        values = np.asarray(
            _torch_float_tensor(archive / f"{energy}_GeV.pt")[:, 4, 2], dtype=float
        )
        values_by_energy[energy] = values
        all_values.append(values)
    edges = np.quantile(
        np.concatenate(all_values), np.linspace(0, 1, MFC_ETA_BIN_COUNT + 1)
    )
    edges[0] = -np.inf
    edges[-1] = np.inf
    rows = []
    for energy in MFC_ENERGIES:
        bins = np.digitize(values_by_energy[int(energy)], edges[1:-1])
        rows.append(
            np.bincount(bins, minlength=MFC_ETA_BIN_COUNT) / bins.size
        )
    return np.asarray(rows)


def figure() -> plt.Figure:
    gaussian, curie = profiles()
    mfc = mfc_profiles()
    fig, axes = plt.subplots(
        5,
        3,
        figsize=(DOUBLE_COL_WIDTH, 105 * MM),
        sharey=True,
        gridspec_kw={"wspace": 0.08, "hspace": 0.12},
        constrained_layout=True,
    )
    ymax = 0.3
    yticks = np.linspace(0, ymax, 4)
    ytick_labels = [f"{value:.1f}" for value in yticks]
    datasets = (
        (gaussian, np.arange(gaussian.shape[1]), "Gaussian targets", "Hamming weight"),
        (
            curie,
            np.linspace(-1, 1, curie.shape[1]),
            "Curie--Weiss targets",
            "magnetization",
        ),
        (mfc, np.arange(MFC_ETA_BIN_COUNT), "MFC targets", r"$\eta$ quantile bin"),
    )
    for row_idx, row_axes in enumerate(axes):
        for col, (ax, (data, x, title, xlabel)) in enumerate(
            zip(row_axes, datasets, strict=True)
        ):
            ax.fill_between(
                x, 0, data[row_idx], alpha=0.28, color=OKABE_ITO_COLORS[row_idx]
            )
            ax.plot(x, data[row_idx], color=OKABE_ITO_COLORS[row_idx], lw=1.2)
            ax.set_ylim(0, ymax)
            ax.set_yticks(yticks, ytick_labels)
            ax.grid(axis="y", alpha=0.25)
            if row_idx < 4:
                ax.tick_params(labelbottom=False)
            label = (
                f"c = {CONDITIONS[row_idx]:g}"
                if col == 0
                else rf"$\beta$ = {BETAS[row_idx]:g}"
                if col == 1
                else rf"$E_{{in}}$ = {MFC_ENERGIES[row_idx]} GeV"
            )
            ax.text(
                0.04,
                0.92,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "none",
                    "pad": 1.5,
                },
            )
            if col > 0:
                ax.tick_params(labelleft=False)
            if row_idx == 4:
                ax.set_xlabel(xlabel)
                if col == 1:
                    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
                if col == 2:
                    ax.set_xlim(-0.5, MFC_ETA_BIN_COUNT - 0.5)
                    ax.set_xticks([0, 4, 8, 12, 15])
            if row_idx == 0:
                ax.set_title(title)
    fig.supylabel("probability")
    return fig


def main() -> None:
    apply_journal_style()
    save_figure(figure(), ROOT / "paper" / "figures" / "all_datasets")


if __name__ == "__main__":
    main()

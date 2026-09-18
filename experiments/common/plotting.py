"""Shared Matplotlib publication style and export helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "DOUBLE_COL_WIDTH",
    "MM",
    "OKABE_ITO_COLORS",
    "SINGLE_COL_WIDTH",
    "apply_style",
    "save_figure",
]

MM = 1.0 / 25.4
SINGLE_COL_WIDTH = 85.0 * MM
DOUBLE_COL_WIDTH = 170.0 * MM

OKABE_ITO_COLORS = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#009E73",  # Bluish Green
    "#E69F00",  # Orange
    "#CC79A7",  # Reddish Purple
    "#56B4E9",  # Sky Blue
    "#F0E442",  # Yellow
    "#000000",  # Black
]

_PDF_CREATED = datetime(2026, 8, 10, tzinfo=UTC)


def apply_style() -> None:
    """Apply publication-quality rcParams."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.prop_cycle": mpl.cycler("color", OKABE_ITO_COLORS[:6]),
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
            "errorbar.capsize": 3,
            "axes.xmargin": 0.02,
            "axes.ymargin": 0.05,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "text.usetex": False,
            "mathtext.fontset": "cm",
        },
    )


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    """Save a figure as PNG and PDF next to ``base_path``."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    stem = base_path.with_suffix("")
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(
        stem.with_suffix(".pdf"),
        metadata={"Creator": "iqp-cond", "CreationDate": _PDF_CREATED},
    )
    plt.close(fig)

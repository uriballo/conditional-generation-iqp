#!/usr/bin/env python3
"""Plot the exact conditional grid-Ising target family."""

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
    DOUBLE_COL_WIDTH,
    MM,
    apply_journal_style,
    save_figure,
)

from iqp_cond.grid_ising import (  # noqa: E402
    enumerate_spins,
    lattice_edges,
    sufficient_statistics,
)

SHAPE = (4, 4)
BETAS = (0.2, 0.35, 0.44, 0.55, 0.8)
FIELDS = (-0.1, 0.0, 0.1)


def _profile(
    values: np.ndarray, probability: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    support = np.unique(values)
    mass = np.asarray([np.sum(probability[values == value]) for value in support])
    return support, mass


def _probability_from_statistics(
    beta: float,
    field: float,
    interaction: np.ndarray,
    magnetization: np.ndarray,
) -> np.ndarray:
    logits = beta * interaction + field * magnetization
    weights = np.exp(logits - np.max(logits))
    return weights / np.sum(weights)


def make_plot(output: Path) -> None:
    """Create a four-panel view of temperature, field, and energy responses."""
    spins = enumerate_spins(SHAPE)
    interaction, magnetization = sufficient_statistics(spins, SHAPE)
    n_spins = spins.shape[1]
    n_edges = len(lattice_edges(SHAPE))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COL_WIDTH, 118 * MM),
        layout="constrained",
    )

    ax = axes[0, 0]
    for index, beta in enumerate(BETAS):
        probability = _probability_from_statistics(
            beta, 0.0, interaction, magnetization
        )
        support, mass = _profile(magnetization, probability)
        ax.plot(
            support / n_spins,
            mass,
            marker="o",
            color=colors[index],
            label=rf"$\beta={beta:g}$",
        )
    ax.set(xlabel=r"Magnetization $M/n$", ylabel=r"Probability $P(M)$")
    ax.legend(ncol=2, handlelength=1.4)

    ax = axes[0, 1]
    for index, field in enumerate(FIELDS):
        probability = _probability_from_statistics(
            0.44, field, interaction, magnetization
        )
        support, mass = _profile(magnetization, probability)
        ax.plot(
            support / n_spins,
            mass,
            marker="o",
            color=colors[index],
            label=rf"$h={field:g}$",
        )
    ax.set(
        xlabel=r"Magnetization $M/n$",
        ylabel=r"Probability $P(M)$",
        title=r"Field response at $\beta=0.44$",
    )
    ax.legend(handlelength=1.4)

    beta_grid = np.linspace(0.2, 0.8, 49)
    field_grid = np.linspace(-0.15, 0.15, 49)
    response = np.empty((beta_grid.size, field_grid.size))
    for beta_index, beta in enumerate(beta_grid):
        for field_index, field in enumerate(field_grid):
            probability = _probability_from_statistics(
                beta, field, interaction, magnetization
            )
            response[beta_index, field_index] = probability @ magnetization / n_spins
    ax = axes[1, 0]
    mesh = ax.pcolormesh(
        field_grid, beta_grid, response, shading="auto", cmap="coolwarm"
    )
    ax.axhline(0.4406868, color="black", linestyle=":", linewidth=1.0)
    ax.set(xlabel=r"Dimensionless field $h$", ylabel=r"Inverse temperature $\beta$")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label(r"$\mathbb{E}[M]/n$")

    ax = axes[1, 1]
    aligned_fraction = (interaction + n_edges) / (2 * n_edges)
    for index, beta in enumerate(BETAS):
        probability = _probability_from_statistics(
            beta, 0.0, interaction, magnetization
        )
        support, mass = _profile(aligned_fraction, probability)
        ax.plot(
            support,
            mass,
            marker="o",
            color=colors[index],
            label=rf"$\beta={beta:g}$",
        )
    ax.set(
        xlabel="Fraction of aligned lattice edges",
        ylabel="Probability",
        title=r"Local-order response at $h=0$",
    )

    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.13, 1.05, f"({label})", transform=ax.transAxes)
    save_figure(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "grid_ising_exploration" / "grid_ising",
    )
    args = parser.parse_args()
    apply_journal_style()
    make_plot(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Regenerate the evidence figures used by the stopped-campaign draft."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#0072B2"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
GRAY = "#666666"
CREATED = datetime(2026, 8, 10, tzinfo=UTC)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"Creator": "iqp-cond", "CreationDate": CREATED},
    )
    figure.savefig(
        output / f"{stem}.png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "iqp-cond"},
    )
    plt.close(figure)


def plot_response_support(output: Path) -> None:
    """Illustrate structural and observable-specific scalar supports."""
    full = np.arange(-4, 5, dtype=float)
    observable = np.asarray([-3, -1, 1, 3], dtype=float)
    figure, axis = plt.subplots(figsize=(6.6, 2.45), constrained_layout=True)
    axis.scatter(
        full, np.ones_like(full), s=64, color=GRAY, label=r"$\Omega_{\rm full}$"
    )
    axis.scatter(
        observable,
        np.zeros_like(observable),
        s=74,
        marker="D",
        color=BLUE,
        label=r"$\Omega_a$ after active-span/$Y$ filter",
    )
    for frequency in full:
        axis.vlines(frequency, 0.90, 1.10, color=GRAY, linewidth=0.8)
    for frequency in observable:
        axis.vlines(frequency, -0.10, 0.10, color=BLUE, linewidth=0.9)
    axis.set_yticks([0, 1], ["observable", "full probability"])
    axis.set_xticks(full.astype(int))
    axis.set_xlabel(r"condition frequency $\omega$")
    axis.set_ylim(-0.35, 1.35)
    axis.spines[["left", "right", "top"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.legend(loc="upper center", frameon=False, ncols=2)
    _save(figure, output, "response_support")


def plot_calibration(calibration_path: Path, output: Path) -> None:
    """Plot exact target-only interval diagnostics and the chosen grid."""
    calibration = _load_json(calibration_path)
    records = calibration["candidate_records"]
    if not isinstance(records, list):
        raise ValueError("candidate_records must be a list")
    labels = [
        f"[{row['interval'][0]:.2f}, {row['interval'][1]:.2f}]" for row in records
    ]
    medians = np.asarray([row["median_adjacent_tvd"] for row in records], dtype=float)
    minima = np.asarray([row["min_adjacent_tvd"] for row in records], dtype=float)
    maxima = np.asarray([row["max_adjacent_tvd"] for row in records], dtype=float)
    passed = np.asarray([row["passed"] for row in records], dtype=bool)
    selected = tuple(calibration["selected_interval"])
    selected_index = next(
        index for index, row in enumerate(records) if tuple(row["interval"]) == selected
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    positions = np.arange(len(records))
    colors = np.where(passed, BLUE, GRAY)
    axes[0].vlines(positions, minima, maxima, color=colors, linewidth=2.0)
    axes[0].scatter(positions, medians, color=colors, s=30, zorder=3)
    axes[0].scatter(
        [selected_index],
        [medians[selected_index]],
        marker="*",
        s=150,
        color=ORANGE,
        edgecolor="black",
        linewidth=0.4,
        zorder=4,
        label="selected",
    )
    axes[0].axhline(0.40, color=PURPLE, linestyle="--", linewidth=1, label="max rule")
    axes[0].set_ylabel("adjacent-target TVD")
    axes[0].set_xticks(positions, labels, rotation=50, ha="right")
    axes[0].set_title("Exact interval screen")
    axes[0].legend(frameon=False, fontsize=8)

    training = np.asarray(calibration["training_betas"], dtype=float)
    held_out = np.asarray(calibration["heldout_betas"], dtype=float)
    axes[1].scatter(
        training, np.ones_like(training), s=62, color=BLUE, label="Training"
    )
    axes[1].scatter(
        held_out,
        np.zeros_like(held_out),
        s=72,
        marker="D",
        color=ORANGE,
        label="Held-out midpoint",
    )
    axes[1].axvline(
        1 / 3, color=PURPLE, linestyle="--", linewidth=1.2, label=r"$\beta=1/3$"
    )
    axes[1].set_yticks([0, 1], ["held out", "training"])
    axes[1].set_xlabel(r"inverse temperature $\beta$")
    axes[1].set_ylim(-0.45, 1.45)
    axes[1].set_title("Preparatory frozen grid")
    axes[1].legend(frameon=False, fontsize=8, loc="center")
    axes[1].tick_params(axis="y", length=0)
    _save(figure, output, "ising_calibration")


def _validation_ess(summary_path: Path) -> np.ndarray:
    summary = _load_json(summary_path)
    records = summary["records"]
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    return np.asarray(
        [record["validation_audit"]["min_energy_ess"] for record in records],
        dtype=float,
    )


def plot_sampler_audits(results_root: Path, output: Path) -> None:
    """Plot the complete failed sampler-development attempts."""
    paths = [
        results_root / "sampler-attempts" / "thinning-10" / "summary.json",
        results_root / "sampler-attempts" / "thinning-16" / "summary.json",
    ]
    values = [_validation_ess(path) for path in paths]
    if any(len(item) != 40 for item in values):
        raise ValueError("each complete sampler pilot must contain 40 cells")

    figure, axis = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
    for index, (label, ess) in enumerate(
        zip(("10 sweeps", "16 sweeps"), values, strict=True)
    ):
        jitter = np.linspace(-0.12, 0.12, len(ess))
        colors = np.where(ess >= 100.0, BLUE, ORANGE)
        axis.scatter(index + jitter, ess, color=colors, s=24, alpha=0.85, label=label)
        axis.hlines(
            np.median(ess), index - 0.22, index + 0.22, color="black", linewidth=2
        )
    axis.axhline(100.0, color=PURPLE, linestyle="--", linewidth=1.3, label="ESS gate")
    axis.set_xticks([0, 1], ["thinning 10", "thinning 16"])
    axis.set_ylabel("minimum per-chain validation energy ESS")
    axis.set_title("Both complete sampler pilots failed 2 of 40 cells")
    axis.set_xlim(-0.45, 1.45)
    axis.legend(frameon=False, fontsize=8, ncols=3, loc="upper center")
    _save(figure, output, "sampler_pilots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/quantum-campaign-v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_response_support(args.output)
    plot_calibration(args.results_root / "calibration.json", args.output)
    plot_sampler_audits(args.results_root, args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create publication figures for the complete Curie--Weiss n=8 block."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from journal_style import (
    OKABE_ITO_COLORS,
    SINGLE_COL_WIDTH,
    apply_journal_style,
    save_figure,
)

from iqp_cond.exact_tvd_results import load_records

ROOT = Path("results/exact-tvd-ancilla-v1")
SUMMARY = ROOT / "curie-n8-complete-summary.json"
CONFIG = Path("configs/exact_tvd_ancilla_v1.json")
OUT = Path("artifacts/curie_weiss_n8_exact_tvd")
ANCILLAS = (0, 1, 3, 6, 10)


def _summary(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size != 10:
        raise ValueError(f"expected ten production seeds, received {array.size}")
    return tuple(float(value) for value in np.quantile(array, (0.5, 0.25, 0.75)))


def _write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty figure table")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _finish_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def _plot_heldout_ladder(seed_rows: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for ancillas in ANCILLAS:
        values = [
            float(row["conditioned_tvd"])
            for row in seed_rows
            if row["group"] == "held_out" and row["ancillas"] == ancillas
        ]
        median, q1, q3 = _summary(values)
        rows.append({"ancillas": ancillas, "median": median, "q1": q1, "q3": q3})
    _write_csv("heldout_ladder", rows)

    x = np.asarray([row["ancillas"] for row in rows], dtype=float)
    median = np.asarray([row["median"] for row in rows], dtype=float)
    q1 = np.asarray([row["q1"] for row in rows], dtype=float)
    q3 = np.asarray([row["q3"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 62 / 25.4), layout="constrained")
    ax.plot(x, median, "o-", color=OKABE_ITO_COLORS[0])
    ax.fill_between(x, q1, q3, color=OKABE_ITO_COLORS[0], alpha=0.18, linewidth=0)
    ax.set(xlabel="Number of ancillas $r$", ylabel="Held-out exact TVD", xticks=x)
    _finish_axes(ax)
    save_figure(fig, OUT / "heldout_ladder")


def _plot_tvd_vs_beta(condition_rows: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    conditions = sorted({float(row["condition"]) for row in condition_rows})
    for ancillas in ANCILLAS:
        for beta in conditions:
            matching = [
                row
                for row in condition_rows
                if row["ancillas"] == ancillas
                and np.isclose(float(row["condition"]), beta)
            ]
            median, q1, q3 = _summary(float(row["conditioned_tvd"]) for row in matching)
            rows.append(
                {
                    "beta": beta,
                    "ancillas": ancillas,
                    "split": matching[0]["split"],
                    "median": median,
                    "q1": q1,
                    "q3": q3,
                }
            )
    _write_csv("tvd_vs_beta", rows)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 67 / 25.4), layout="constrained")
    for index, ancillas in enumerate(ANCILLAS):
        selected = [row for row in rows if row["ancillas"] == ancillas]
        x = np.asarray([row["beta"] for row in selected], dtype=float)
        median = np.asarray([row["median"] for row in selected], dtype=float)
        q1 = np.asarray([row["q1"] for row in selected], dtype=float)
        q3 = np.asarray([row["q3"] for row in selected], dtype=float)
        color = OKABE_ITO_COLORS[index]
        ax.plot(x, median, "-", color=color, label=rf"$r={ancillas}$")
        ax.fill_between(x, q1, q3, color=color, alpha=0.10, linewidth=0)
        for split, facecolor in (("training", "none"), ("held_out", color)):
            mask = np.asarray([row["split"] == split for row in selected])
            ax.scatter(
                x[mask],
                median[mask],
                marker="o",
                facecolors=facecolor,
                edgecolors=color,
                linewidths=0.8,
                zorder=3,
            )
    ax.axvline(1.0, color="0.45", linestyle="--", linewidth=0.8, zorder=0)
    ax.set(xlabel="Inverse temperature $\\beta$", ylabel="Exact TVD")
    ax.legend(ncol=2, handlelength=1.4, columnspacing=0.9)
    _finish_axes(ax)
    save_figure(fig, OUT / "tvd_vs_beta")


def _plot_heldout_comparison(seed_rows: list[dict[str, object]]) -> None:
    labels = {
        "conditioned_tvd": "Shared conditional",
        "interpolation_tvd": "Independent interpolation",
        "oracle_tvd": "Target-aware independent",
    }
    rows: list[dict[str, object]] = []
    for ancillas in ANCILLAS:
        selected = [
            row
            for row in seed_rows
            if row["group"] == "held_out" and row["ancillas"] == ancillas
        ]
        for metric, label in labels.items():
            median, q1, q3 = _summary(float(row[metric]) for row in selected)
            rows.append(
                {
                    "ancillas": ancillas,
                    "model": label,
                    "median": median,
                    "q1": q1,
                    "q3": q3,
                }
            )
    _write_csv("heldout_comparison", rows)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 67 / 25.4), layout="constrained")
    for index, label in enumerate(labels.values()):
        selected = [row for row in rows if row["model"] == label]
        x = np.asarray([row["ancillas"] for row in selected], dtype=float)
        median = np.asarray([row["median"] for row in selected], dtype=float)
        q1 = np.asarray([row["q1"] for row in selected], dtype=float)
        q3 = np.asarray([row["q3"] for row in selected], dtype=float)
        color = OKABE_ITO_COLORS[index]
        ax.plot(x, median, "o-", color=color, label=label)
        ax.fill_between(x, q1, q3, color=color, alpha=0.14, linewidth=0)
    ax.set(
        xlabel="Number of ancillas $r$",
        ylabel="Held-out exact TVD",
        xticks=ANCILLAS,
    )
    ax.legend(handlelength=1.4)
    _finish_axes(ax)
    save_figure(fig, OUT / "heldout_comparison")


def _shell_masses(probabilities: np.ndarray, n_visible: int) -> np.ndarray:
    indices = np.arange(probabilities.size, dtype=np.uint32)
    weights = np.fromiter(
        (int(index).bit_count() for index in indices),
        dtype=np.int64,
        count=indices.size,
    )
    return np.bincount(weights, weights=probabilities, minlength=n_visible + 1)


def _plot_critical_profile(config: dict[str, object]) -> None:
    records = load_records(ROOT / "evaluations", config, require_complete=False)
    selected = [
        record
        for record in records
        if record.identity["dataset"] == "curie_weiss"
        and int(record.identity["visible_qubits"]) == 8
        and record.identity["family"] == "conditioned"
        and int(record.identity["ancillas"]) in {0, 3, 10}
    ]
    curves: dict[int, list[np.ndarray]] = {0: [], 3: [], 10: []}
    target: np.ndarray | None = None
    for record in selected:
        probabilities, target_probabilities, _ = record.at(1.0)
        ancillas = int(record.identity["ancillas"])
        curves[ancillas].append(_shell_masses(probabilities, 8))
        candidate = _shell_masses(target_probabilities, 8)
        if target is None:
            target = candidate
        elif not np.allclose(target, candidate):
            raise ValueError("critical-point targets differ across production seeds")
    if target is None:
        raise ValueError("critical-point target was not found")

    rows: list[dict[str, object]] = []
    magnetization = (2 * np.arange(9) - 8) / 8
    for shell, value in enumerate(target):
        rows.append(
            {
                "magnetization": magnetization[shell],
                "model": "Exact target",
                "median": float(value),
                "q1": float(value),
                "q3": float(value),
            }
        )
    for ancillas, seed_curves in curves.items():
        array = np.asarray(seed_curves, dtype=np.float64)
        if array.shape != (10, 9):
            raise ValueError(f"expected ten profile curves for r={ancillas}")
        quantiles = np.quantile(array, (0.5, 0.25, 0.75), axis=0)
        for shell in range(9):
            rows.append(
                {
                    "magnetization": magnetization[shell],
                    "model": f"r={ancillas}",
                    "median": float(quantiles[0, shell]),
                    "q1": float(quantiles[1, shell]),
                    "q3": float(quantiles[2, shell]),
                }
            )
    _write_csv("critical_profile", rows)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 67 / 25.4), layout="constrained")
    ax.plot(magnetization, target, "k--", marker="o", label="Exact target")
    for index, ancillas in enumerate((0, 3, 10)):
        model_rows = [row for row in rows if row["model"] == f"r={ancillas}"]
        median = np.asarray([row["median"] for row in model_rows], dtype=float)
        q1 = np.asarray([row["q1"] for row in model_rows], dtype=float)
        q3 = np.asarray([row["q3"] for row in model_rows], dtype=float)
        color = OKABE_ITO_COLORS[index]
        ax.plot(magnetization, median, "o-", color=color, label=rf"$r={ancillas}$")
        ax.fill_between(magnetization, q1, q3, color=color, alpha=0.14, linewidth=0)
    ax.set(xlabel="Magnetization $m$", ylabel="Probability mass at $\\beta=1$")
    ax.legend(ncol=2, handlelength=1.4, columnspacing=0.9)
    _finish_axes(ax)
    save_figure(fig, OUT / "critical_profile")


def _plot_magnetization_ridge(config: dict[str, object]) -> None:
    """Show the approach to the ordered, bimodal regime across held-out betas."""
    records = load_records(ROOT / "evaluations", config, require_complete=False)
    records = [
        record
        for record in records
        if record.identity["dataset"] == "curie_weiss"
        and int(record.identity["visible_qubits"]) == 8
        and record.identity["family"] == "conditioned"
        and int(record.identity["ancillas"]) == 10
    ]
    betas = (0.6, 0.8, 1.0, 1.2, 1.4)
    magnetization = (2 * np.arange(9) - 8) / 8
    curves: dict[float, list[np.ndarray]] = {beta: [] for beta in betas}
    targets: dict[float, np.ndarray] = {}
    for record in records:
        for beta in betas:
            probabilities, target_probabilities, _ = record.at(beta)
            curves[beta].append(_shell_masses(probabilities, 8))
            target = _shell_masses(target_probabilities, 8)
            if beta in targets and not np.allclose(targets[beta], target):
                raise ValueError(f"target differs across seeds at beta={beta}")
            targets[beta] = target
    if any(len(values) != 10 for values in curves.values()):
        raise ValueError("expected ten r=10 curves at every requested held-out beta")

    rows: list[dict[str, object]] = []
    for beta in betas:
        values = np.asarray(curves[beta])
        quantiles = np.quantile(values, (0.5, 0.25, 0.75), axis=0)
        for index, m in enumerate(magnetization):
            target_value = float(targets[beta][index])
            rows.append(
                {
                    "beta": beta,
                    "magnetization": float(m),
                    "model": "Exact target",
                    "median": target_value,
                    "q1": target_value,
                    "q3": target_value,
                }
            )
            rows.append(
                {
                    "beta": beta,
                    "magnetization": float(m),
                    "model": "Conditioned r=10",
                    "median": float(quantiles[0, index]),
                    "q1": float(quantiles[1, index]),
                    "q3": float(quantiles[2, index]),
                }
            )
    _write_csv("magnetization_ridge", rows)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 112 / 25.4), layout="constrained")
    color = OKABE_ITO_COLORS[0]
    for index, beta in enumerate(betas):
        values = np.asarray(curves[beta])
        quantiles = np.quantile(values, (0.5, 0.25, 0.75), axis=0)
        offset = index * 1.05
        ax.plot(magnetization, targets[beta] + offset, "k--", linewidth=1.0)
        ax.plot(magnetization, quantiles[0] + offset, "o-", color=color)
        ax.fill_between(
            magnetization,
            quantiles[1] + offset,
            quantiles[2] + offset,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.text(
            1.02,
            offset,
            rf"$\beta={beta:g}$",
            va="center",
            transform=ax.get_yaxis_transform(),
        )
    ax.set(xlabel="Magnetization $m$", yticks=[])
    ax.plot([], [], "k--", label="Exact target")
    ax.plot([], [], "o-", color=color, label="Conditioned $r=10$")
    ax.legend(loc="upper left", handlelength=1.4)
    _finish_axes(ax)
    save_figure(fig, OUT / "magnetization_ridge")


def main() -> None:
    apply_journal_style()
    data = json.loads(SUMMARY.read_text())
    seed_rows = [
        row
        for row in data["seed_rows"]
        if row["dataset"] == "curie_weiss" and row["visible_qubits"] == 8
    ]
    condition_rows = [
        row
        for row in data["condition_rows"]
        if row["dataset"] == "curie_weiss" and row["visible_qubits"] == 8
    ]
    config = json.loads(CONFIG.read_text())
    _plot_heldout_ladder(seed_rows)
    _plot_tvd_vs_beta(condition_rows)
    _plot_heldout_comparison(seed_rows)
    _plot_critical_profile(config)
    _plot_magnetization_ridge(config)
    print(f"wrote Curie--Weiss n=8 figures and tables to {OUT}")


if __name__ == "__main__":
    main()

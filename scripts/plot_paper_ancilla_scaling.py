#!/usr/bin/env python3
"""Render the paper's completed Gaussian/CW ancilla-scaling summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from journal_style import (
    DOUBLE_COL_WIDTH,
    OKABE_ITO_COLORS,
    apply_journal_style,
    save_figure,
)

ANCILLAS = (0, 1, 3, 6, 10)
SIZES = (8, 12)


def _values(rows, dataset, n, r, group, field):
    selected = [
        x
        for x in rows
        if x["dataset"] == dataset
        and int(x["visible_qubits"]) == n
        and int(x["ancillas"]) == r
        and x["group"] == group
    ]
    seeds = sorted(int(x["seed"]) for x in selected)
    if seeds != list(range(100, 110)):
        raise ValueError(
            f"expected unique seeds 100--109 for {dataset} n={n} r={r} "
            f"{group}, found {seeds}"
        )
    values = [float(x[field]) for x in selected]
    if not all(np.isfinite(values)):
        raise ValueError(f"nonfinite {field} value for {dataset} n={n} r={r} {group}")
    return np.asarray(values)


def _scaling(payload, output, dataset):
    rows = payload["seed_rows"]
    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_WIDTH, 62 / 25.4), sharey=True, layout="constrained"
    )
    for ax, n in zip(axes, SIZES, strict=True):
        for i, (field, label, marker, linestyle) in enumerate(
            (
                ("conditioned_tvd", "Conditioned", "o", "-"),
                ("interpolation_tvd", "Independent interpolation", "s", "--"),
                ("oracle_tvd", "Target-aware oracle", "^", ":"),
            )
        ):
            med, lo, hi = [], [], []
            for r in ANCILLAS:
                v = _values(rows, dataset, n, r, "held_out", field)
                med.append(np.median(v))
                lo.append(np.quantile(v, 0.25))
                hi.append(np.quantile(v, 0.75))
            color = OKABE_ITO_COLORS[i]
            ax.plot(
                ANCILLAS,
                med,
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=label,
            )
            ax.fill_between(ANCILLAS, lo, hi, color=color, alpha=0.13, linewidth=0)
        ax.set(title=rf"$n={n}$", xlabel="Ancillas $r$", xticks=ANCILLAS)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out exact TVD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=8)
    save_figure(fig, output / f"{dataset}_results")


def _condition_values(rows, dataset, n, r, condition, split, field="conditioned_tvd"):
    selected = [
        x
        for x in rows
        if x["dataset"] == dataset
        and int(x["visible_qubits"]) == n
        and int(x["ancillas"]) == r
        and x["split"] == split
        and np.isclose(float(x["condition"]), condition)
    ]
    seeds = sorted(int(x["seed"]) for x in selected)
    if seeds != list(range(100, 110)):
        raise ValueError(
            f"expected ten unique seeds for {dataset} n={n} "
            f"condition={condition}, found {seeds}"
        )
    values = [float(x[field]) for x in selected]
    if not all(np.isfinite(values)):
        raise ValueError("nonfinite condition response value")
    return np.asarray(values)


def _condition_response(payload, output, dataset):
    rows = payload["condition_rows"]
    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_WIDTH, 65 / 25.4), sharey=True, layout="constrained"
    )
    conditions = sorted(
        {float(x["condition"]) for x in rows if x["dataset"] == dataset}
    )
    if len(conditions) != (13 if dataset == "gaussian" else 11):
        raise ValueError(f"expected manifest condition count, found {len(conditions)}")
    for ax, n in zip(axes, SIZES, strict=True):
        for i, r in enumerate(ANCILLAS):
            color = OKABE_ITO_COLORS[i]
            med, lo, hi, plotted = [], [], [], []
            for condition in conditions:
                split = next(
                    (
                        x["split"]
                        for x in rows
                        if x["dataset"] == dataset
                        and int(x["visible_qubits"]) == n
                        and np.isclose(float(x["condition"]), condition)
                    ),
                    None,
                )
                if split is None:
                    continue
                values = _condition_values(rows, dataset, n, r, condition, split)
                plotted.append(condition)
                med.append(np.median(values))
                lo.append(np.quantile(values, 0.25))
                hi.append(np.quantile(values, 0.75))
            ax.plot(plotted, med, color=color, linestyle="-", label=rf"$r={r}$")
            ax.fill_between(plotted, lo, hi, color=color, alpha=0.10, linewidth=0)
            for split, marker, face in (
                ("training", "o", color),
                ("held_out", "s", "none"),
            ):
                points = [
                    c
                    for c in plotted
                    if any(
                        x["dataset"] == dataset
                        and int(x["visible_qubits"]) == n
                        and np.isclose(float(x["condition"]), c)
                        and x["split"] == split
                        for x in rows
                    )
                ]
                y = [med[plotted.index(c)] for c in points]
                ax.scatter(
                    points,
                    y,
                    marker=marker,
                    facecolors=face,
                    edgecolors=color,
                    s=22,
                    zorder=3,
                )
        if dataset == "curie_weiss":
            ax.axvline(1.0, color="0.45", linestyle="--", linewidth=0.8)
        xlabel = (
            r"Inverse temperature $\beta$" if dataset == "curie_weiss" else "Condition"
        )
        ax.set(title=rf"$n={n}$", xlabel=xlabel)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Exact conditioned TVD")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5, fontsize=8)
    save_figure(fig, output / f"{dataset}_condition_response")


def _train_holdout(payload, output):
    rows = payload["seed_rows"]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COL_WIDTH, 105 / 25.4),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for ax, (dataset, n) in zip(
        axes.flat,
        (("gaussian", 8), ("gaussian", 12), ("curie_weiss", 8), ("curie_weiss", 12)),
        strict=True,
    ):
        for i, group in enumerate(("training", "held_out")):
            med, lo, hi = [], [], []
            for r in ANCILLAS:
                v = _values(rows, dataset, n, r, group, "conditioned_tvd")
                med.append(np.median(v))
                lo.append(np.quantile(v, 0.25))
                hi.append(np.quantile(v, 0.75))
            color = OKABE_ITO_COLORS[i]
            ax.plot(
                ANCILLAS, med, "o-", color=color, label=group.replace("_", " ").title()
            )
            ax.fill_between(ANCILLAS, lo, hi, color=color, alpha=0.13, linewidth=0)
        ax.set(
            title=f"{dataset.replace('_', ' ').title()}, $n={n}$",
            xticks=ANCILLAS,
            xlabel="Ancillas $r$",
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Exact TVD")
    axes[1, 0].set_ylabel("Exact TVD")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=8)
    save_figure(fig, output / "conditioned_training_vs_heldout")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    payload = json.loads(a.summary.read_text())
    apply_journal_style()
    a.output.mkdir(parents=True, exist_ok=True)
    for dataset in ("gaussian", "curie_weiss"):
        _scaling(payload, a.output, dataset)
        _condition_response(payload, a.output, dataset)
    _train_holdout(payload, a.output)


if __name__ == "__main__":
    main()

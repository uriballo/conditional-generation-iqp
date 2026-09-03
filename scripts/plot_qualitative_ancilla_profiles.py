"""Plot qualitative exact distributions for the completed n=8 campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from journal_style import (
    DOUBLE_COL_WIDTH,
    SINGLE_COL_WIDTH,
    apply_journal_style,
    save_figure,
)

from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.exact_tvd_results import load_records
from iqp_cond.gaussian_data import gaussian_target

ANCILLAS = (0, 3, 10)
SEEDS = tuple(range(100, 110))
GAUSSIAN_CONDITIONS = (0.0625, 0.4375, 0.9375)
CURIE_CONDITIONS = (0.6, 1.0, 1.4)


def _profile(probabilities: np.ndarray, n: int) -> np.ndarray:
    shell = np.array([int(x).bit_count() for x in range(1 << n)])
    return np.bincount(shell, weights=probabilities, minlength=n + 1)


def _rows(records, dataset: str, condition: float, coordinate: np.ndarray):
    selected = [
        r
        for r in records
        if r.identity["dataset"] == dataset
        and r.identity["visible_qubits"] == 8
        and r.identity["family"] == "conditioned"
        and r.identity["seed"] in SEEDS
        and np.isclose(r.conditions, condition).any()
    ]
    out = []
    for r in ANCILLAS:
        group = [x for x in selected if x.identity["ancillas"] == r]
        if len(group) != len(SEEDS):
            raise ValueError(
                f"expected 10 curves for {dataset}, {condition}, "
                f"r={r}; got {len(group)}"
            )
        profiles = []
        targets = []
        for rec in sorted(group, key=lambda x: x.identity["seed"]):
            p, t, _ = rec.at(condition)
            profiles.append(_profile(p, 8))
            targets.append(_profile(t, 8))
        if not np.allclose(targets, targets[0]):
            raise ValueError(f"targets differ across seeds for {dataset}, {condition}")
        values = np.asarray(profiles)
        for x, med, q1, q3 in zip(
            coordinate,
            np.median(values, 0),
            np.quantile(values, 0.25, 0),
            np.quantile(values, 0.75, 0),
            strict=True,
        ):
            out.append(
                {
                    "condition": condition,
                    "coordinate": float(x),
                    "model": f"r={r}",
                    "median": float(med),
                    "q1": float(q1),
                    "q3": float(q3),
                }
            )
        if r == ANCILLAS[0]:
            target = targets[0]
            for x, y in zip(coordinate, target, strict=True):
                out.append(
                    {
                        "condition": condition,
                        "coordinate": float(x),
                        "model": "exact target",
                        "median": float(y),
                        "q1": float(y),
                        "q3": float(y),
                    }
                )
    return out


def make_plots(evaluations: Path, output: Path) -> None:
    config = json.loads(Path("configs/exact_tvd_ancilla_v1.json").read_text())
    records = load_records(evaluations, config, require_complete=False)
    apply_journal_style()
    for dataset, conditions in (
        ("gaussian", GAUSSIAN_CONDITIONS),
        ("curie_weiss", CURIE_CONDITIONS),
    ):
        for condition in conditions:
            coordinate = np.arange(9, dtype=float)
            if dataset == "curie_weiss":
                coordinate = (2 * coordinate - 8) / 8
                target = exact_curie_weiss_target(8, condition).profile
            else:
                target = gaussian_target(8, 0.5, condition)
                target = target.shell_probabilities
            rows = _rows(records, dataset, condition, coordinate)
            condition_label = str(condition).replace(".", "p")
            path = output / f"{dataset}_n8_condition_{condition_label}"
            path.with_suffix(".csv").parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix(".csv").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "condition",
                        "coordinate",
                        "model",
                        "median",
                        "q1",
                        "q3",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)
            fig, ax = plt.subplots(
                figsize=(SINGLE_COL_WIDTH, 66 / 25.4), layout="constrained"
            )
            for model, color, style in (
                (r"$r=0$", "#0072B2", "o-"),
                (r"$r=3$", "#D55E00", "s-"),
                (r"$r=10$", "#009E73", "^-"),
            ):
                data = [x for x in rows if x["model"] == model.replace("$", "")]
                y = np.array([x["median"] for x in data])
                lo = np.array([x["q1"] for x in data])
                hi = np.array([x["q3"] for x in data])
                ax.fill_between(
                    coordinate, lo, hi, color=color, alpha=0.18, linewidth=0
                )
                ax.plot(coordinate, y, style, color=color, label=model)
            ax.plot(coordinate, target, "k--", label="Exact target")
            ax.set_xlabel(
                "Magnetization $m$"
                if dataset == "curie_weiss"
                else "Hamming weight $K$"
            )
            ax.set_ylabel("Probability mass")
            if dataset == "curie_weiss":
                ax.set_xticks(np.linspace(-1.0, 1.0, 5))
            else:
                ax.set_xticks(coordinate)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(
                ncol=4,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                borderaxespad=0,
                columnspacing=0.8,
                handlelength=2.0,
            )
            save_figure(fig, path)

    coordinate = (2 * np.arange(9, dtype=float) - 8) / 8
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COL_WIDTH, 58 / 25.4),
        sharey=True,
        layout="constrained",
    )
    handles, labels = [], []
    for panel_index, (ax, condition) in enumerate(
        zip(axes, CURIE_CONDITIONS, strict=True)
    ):
        rows = _rows(records, "curie_weiss", condition, coordinate)
        target = exact_curie_weiss_target(8, condition).profile
        for model, color, style in (
            (r"$r=0$", "#0072B2", "o-"),
            (r"$r=3$", "#D55E00", "s-"),
            (r"$r=10$", "#009E73", "^-"),
        ):
            data = [x for x in rows if x["model"] == model.replace("$", "")]
            y = np.array([x["median"] for x in data])
            lo = np.array([x["q1"] for x in data])
            hi = np.array([x["q3"] for x in data])
            ax.fill_between(coordinate, lo, hi, color=color, alpha=0.18, linewidth=0)
            line = ax.plot(coordinate, y, style, color=color, label=model)[0]
            if len(handles) < 3:
                handles.append(line)
                labels.append(model)
        target_line = ax.plot(coordinate, target, "k--", label="Exact target")[0]
        if len(handles) == 3:
            handles.append(target_line)
            labels.append("Exact target")
        ax.set_title(rf"({chr(97 + panel_index)}) $\beta={condition:g}$", pad=4)
        ax.set_xticks(np.linspace(-1.0, 1.0, 5))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Magnetization $m$")
    axes[0].set_ylabel("Probability mass")
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0,
        columnspacing=1.0,
        handlelength=2.0,
    )
    save_figure(fig, output / "curie_weiss_n8_magnetization_profiles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluations",
        type=Path,
        default=Path("results/exact-tvd-ancilla-v1/evaluations"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("paper/qualitative_ancilla_profiles")
    )
    args = parser.parse_args()
    make_plots(args.evaluations, args.output)

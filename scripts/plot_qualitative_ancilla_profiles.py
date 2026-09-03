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
from iqp_cond.exact_tvd_results import interpolate_probabilities, load_records
from iqp_cond.gaussian_data import gaussian_target

ANCILLAS = (0, 3, 10)
SEEDS = tuple(range(100, 110))
GAUSSIAN_CONDITIONS = (0.0625, 0.4375, 0.9375)
CURIE_CONDITIONS = (0.6, 1.0, 1.4)
def _profile(probabilities: np.ndarray, n: int) -> np.ndarray:
    shell = np.array([int(x).bit_count() for x in range(1 << n)])
    return np.bincount(shell, weights=probabilities, minlength=n + 1)


def _rows(records, dataset: str, condition: float, coordinate: np.ndarray, n: int = 8):
    selected = [
        r
        for r in records
        if r.identity["dataset"] == dataset
        and r.identity["visible_qubits"] == n
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
            profiles.append(_profile(p, n))
            targets.append(_profile(t, n))
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


def _record_at(records, dataset, n, ancillas, seed, condition):
    matches = [
        r
        for r in records
        if r.identity["dataset"] == dataset
        and r.identity["visible_qubits"] == n
        and r.identity["ancillas"] == ancillas
        and r.identity["seed"] == seed
        and r.identity["family"] == "independent"
        and np.isclose(r.conditions, condition).any()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one independent record, found {len(matches)}")
    return matches[0].at(condition)[0]


def _interpolated_rows(records, dataset, n, condition, coordinate, grid):
    right_index = next(i for i, x in enumerate(grid) if x > condition)
    left, right = grid[right_index - 1], grid[right_index]
    weight = (condition - left) / (right - left)
    values = []
    for seed in SEEDS:
        p_left = _record_at(records, dataset, n, 10, seed, left)
        p_right = _record_at(records, dataset, n, 10, seed, right)
        values.append(_profile(interpolate_probabilities(p_left, p_right, weight), n))
    values = np.asarray(values)
    rows = []
    for x, med, q1, q3 in zip(
        coordinate,
        np.median(values, 0),
        np.quantile(values, 0.25, 0),
        np.quantile(values, 0.75, 0),
        strict=True,
    ):
        rows.append(
            {
                "dataset": dataset,
                "n": n,
                "condition": condition,
                "split": "held_out",
                "model": "adjacent independent interpolation",
                "coordinate": float(x),
                "median": float(med),
                "q1": float(q1),
                "q3": float(q3),
                "seed_count": len(SEEDS),
                "left_endpoint": left,
                "right_endpoint": right,
                "weight": weight,
            }
        )
    return rows


def make_plots(evaluations: Path, output: Path) -> None:
    config = json.loads(Path("configs/exact_tvd_ancilla_v1.json").read_text())
    records = load_records(evaluations, config, require_complete=False)
    gaussian_config = config["datasets"]["gaussian"]
    curie_config = config["datasets"]["curie_weiss"]
    gaussian_train = tuple(gaussian_config["training_conditions"])
    gaussian_heldout = tuple(gaussian_config["held_out_conditions"])
    curie_train = tuple(curie_config["training_conditions"])
    curie_heldout = tuple(curie_config["held_out_conditions"])
    if set(gaussian_train) & set(gaussian_heldout) or set(curie_train) & set(
        curie_heldout
    ):
        raise ValueError("training and held-out condition grids overlap")
    if not set((0.0, 0.5, 1.0)).issubset(gaussian_train):
        raise ValueError("Gaussian training panels are not training conditions")
    if not set(GAUSSIAN_CONDITIONS).issubset(gaussian_heldout):
        raise ValueError("Gaussian generalization panels are not held-out conditions")
    if not set(CURIE_CONDITIONS).issubset(curie_heldout):
        raise ValueError(
            "Curie-Weiss generalization panels are not held-out conditions"
        )
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

    def panel_figure(dataset, n, conditions, split, filename, models):
        x = np.arange(n + 1, dtype=float)
        if dataset == "curie_weiss":
            x = (2 * x - n) / n
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(DOUBLE_COL_WIDTH, 58 / 25.4),
            sharey=True,
            layout="constrained",
        )
        handles, labels = [], []
        csv_rows = []
        for index, (ax, condition) in enumerate(zip(axes, conditions, strict=True)):
            target = (
                exact_curie_weiss_target(n, condition).profile
                if dataset == "curie_weiss"
                else gaussian_target(n, 0.5, condition).shell_probabilities
            )
            base = _rows(records, dataset, condition, x, n)
            rows = [
                {
                    **row,
                    "dataset": dataset,
                    "n": n,
                    "split": split,
                    "seed_count": len(SEEDS),
                    "left_endpoint": "",
                    "right_endpoint": "",
                    "weight": "",
                }
                for row in base
                if row["model"] in {"r=0", "r=3", "r=10"}
            ]
            rows += [
                {
                    "dataset": dataset,
                    "n": n,
                    "condition": condition,
                    "split": split,
                    "model": "exact target",
                    "coordinate": float(xx),
                    "median": float(yy),
                    "q1": float(yy),
                    "q3": float(yy),
                    "seed_count": 0,
                    "left_endpoint": "",
                    "right_endpoint": "",
                    "weight": "",
                }
                for xx, yy in zip(x, target, strict=True)
            ]
            if models == "generalization":
                target_rows = [row for row in rows if row["model"] == "exact target"]
                rows = [row for row in rows if row["model"] == "r=10"]
                rows += _interpolated_rows(
                    records,
                    dataset,
                    n,
                    condition,
                    x,
                    gaussian_train if dataset == "gaussian" else curie_train,
                )
                rows += target_rows
            for row in rows:
                if row["model"] in {"r=0", "r=3", "r=10"}:
                    row["left_endpoint"] = row["right_endpoint"] = row["weight"] = ""
            csv_rows.extend(rows)
            styles = (
                [
                    ("r=0", "#0072B2", "o-"),
                    ("r=3", "#D55E00", "s-"),
                    ("r=10", "#009E73", "^-"),
                ]
                if models == "training"
                else [
                    ("r=10", "#009E73", "^-"),
                    ("adjacent independent interpolation", "#D55E00", "s-"),
                ]
            )
            for model, color, style in styles:
                data = [r for r in rows if r["model"] == model]
                y = np.array([r["median"] for r in data])
                lo = np.array([r["q1"] for r in data])
                hi = np.array([r["q3"] for r in data])
                ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
                line = ax.plot(x, y, style, color=color, label=model)[0]
                if index == 0:
                    handles.append(line)
                    labels.append(
                        "Conditioned, $r=10$"
                        if models == "generalization" and model == "r=10"
                        else model
                    )
            target_line = ax.plot(x, target, "k--", label="Exact target")[0]
            if index == 0:
                handles.append(target_line)
                labels.append("Exact target")
            if models == "generalization":
                labels[1] = "Adjacent independent interpolation"
            ax.set_title(
                rf"({chr(97 + index)}) $c={condition:g}$"
                if dataset == "gaussian"
                else rf"({chr(97 + index)}) $\beta={condition:g}$",
                pad=4,
            )
            ax.set_xlabel(
                "Hamming weight $K$" if dataset == "gaussian" else "Magnetization $m$"
            )
            ax.set_xticks(
                np.linspace(0, n, 5) if dataset == "gaussian" else np.linspace(-1, 1, 5)
            )
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Probability mass")
        fig.legend(
            handles,
            labels,
            ncol=len(labels),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            borderaxespad=0,
            handlelength=2,
        )
        path = output / filename
        model_count = 4 if models == "training" else 3
        expected_rows = (n + 1) * model_count * 3
        if len(csv_rows) != expected_rows:
            raise AssertionError(
                f"expected {expected_rows} rows, found {len(csv_rows)}"
            )
        with path.with_suffix(".csv").open("w", newline="") as f:
            fields = [
                "dataset",
                "n",
                "condition",
                "split",
                "model",
                "coordinate",
                "median",
                "q1",
                "q3",
                "seed_count",
                "left_endpoint",
                "right_endpoint",
                "weight",
            ]
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(csv_rows)
        save_figure(fig, path)

    panel_figure(
        "gaussian",
        8,
        tuple(x for x in gaussian_train if x in (0.0, 0.5, 1.0)),
        "training",
        "gaussian_n8_training_profiles",
        "training",
    )
    panel_figure(
        "gaussian",
        12,
        tuple(x for x in gaussian_train if x in (0.0, 0.5, 1.0)),
        "training",
        "gaussian_n12_training_profiles",
        "training",
    )
    panel_figure(
        "gaussian",
        8,
        tuple(x for x in gaussian_heldout if x in GAUSSIAN_CONDITIONS),
        "held_out",
        "gaussian_n8_generalization_profiles",
        "generalization",
    )
    panel_figure(
        "gaussian",
        12,
        tuple(x for x in gaussian_heldout if x in GAUSSIAN_CONDITIONS),
        "held_out",
        "gaussian_n12_generalization_profiles",
        "generalization",
    )
    panel_figure(
        "curie_weiss",
        8,
        tuple(x for x in curie_heldout if x in CURIE_CONDITIONS),
        "held_out",
        "curie_weiss_n8_generalization_profiles",
        "generalization",
    )


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

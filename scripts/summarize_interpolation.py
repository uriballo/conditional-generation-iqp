"""Aggregate held-out interpolation results and generate paper artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from journal_style import DOUBLE_COL_WIDTH, MM, apply_journal_style, save_figure

from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.interpolation import (
    HELDOUT_ANGLE_FRACTIONS,
    TRAIN_ANGLE_FRACTIONS,
    broadcast_angles,
    interpolation_targets,
)

METHOD_LABELS = {
    "dense": "Dense-grid interpolation",
    "nearest": "Oracle adjacent-angle lookup",
    "endpoint": "Endpoint-only interpolation",
}


def _median_iqr(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
    )


def _load_dense(directory: Path) -> list[dict[str, object]]:
    runs = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    seeds = {run["seed"] for run in runs}
    if seeds != set(range(10)):
        raise SystemExit(f"expected interpolation seeds 0-9; found {sorted(seeds)}")
    return runs


def _endpoint_rows(
    endpoint_directory: Path,
) -> list[dict[str, float | int | str]]:
    scalar_angles = pi * HELDOUT_ANGLE_FRACTIONS
    condition_angles = broadcast_angles(scalar_angles)
    targets = interpolation_targets(scalar_angles)
    bitstrings = enumerate_bitstrings(19)
    rows = []
    for seed in range(10):
        path = endpoint_directory / f"angle_complete_seed{seed}.npz"
        run = np.load(path)
        probabilities = exact_iqp_probabilities_many(
            run["best_params"],
            run["generators"],
            condition_angles,
        )
        for fraction, target, model_probs in zip(
            HELDOUT_ANGLE_FRACTIONS,
            targets,
            probabilities,
            strict=True,
        ):
            target_probs = target.bitstring_probabilities(bitstrings)
            model_shells = shell_probabilities(model_probs, bitstrings)
            rows.append(
                {
                    "seed": seed,
                    "method": "endpoint",
                    "angle_fraction_pi": float(fraction),
                    "shell_tvd": total_variation_distance(
                        target.shell_probabilities,
                        model_shells,
                    ),
                    "bitstring_tvd": total_variation_distance(
                        target_probs,
                        model_probs,
                    ),
                },
            )
    return rows


def _comparison_rows(
    dense_runs: list[dict[str, object]],
    dense_directory: Path,
    endpoint_directory: Path,
) -> list[dict[str, float | int | str]]:
    rows = _endpoint_rows(endpoint_directory)
    heldout_angles = pi * HELDOUT_ANGLE_FRACTIONS
    heldout_targets = interpolation_targets(heldout_angles)
    bitstrings = enumerate_bitstrings(19)
    for run in dense_runs:
        stored = np.load(
            dense_directory / f"interpolation_complete_seed{run['seed']}.npz",
        )
        adjacent_probabilities = exact_iqp_probabilities_many(
            stored["best_params"],
            stored["generators"],
            broadcast_angles(pi * TRAIN_ANGLE_FRACTIONS),
        )
        for point in run["points"]:
            if point["role"] != "heldout":
                continue
            heldout_index = int(
                np.flatnonzero(
                    np.isclose(
                        HELDOUT_ANGLE_FRACTIONS,
                        point["angle_fraction_pi"],
                    ),
                )[0],
            )
            target = heldout_targets[heldout_index]
            target_probs = target.bitstring_probabilities(bitstrings)
            adjacent_shell_tvds = []
            adjacent_bitstring_tvds = []
            for adjacent_index in (heldout_index, heldout_index + 1):
                adjacent_probs = adjacent_probabilities[adjacent_index]
                adjacent_shells = shell_probabilities(adjacent_probs, bitstrings)
                adjacent_shell_tvds.append(
                    total_variation_distance(
                        target.shell_probabilities,
                        adjacent_shells,
                    ),
                )
                adjacent_bitstring_tvds.append(
                    total_variation_distance(target_probs, adjacent_probs),
                )
            common = {
                "seed": run["seed"],
                "angle_fraction_pi": point["angle_fraction_pi"],
            }
            rows.extend(
                [
                    {
                        **common,
                        "method": "dense",
                        "shell_tvd": point["shell_tvd"],
                        "bitstring_tvd": point["bitstring_tvd"],
                    },
                    {
                        **common,
                        "method": "nearest",
                        "shell_tvd": min(adjacent_shell_tvds),
                        "bitstring_tvd": min(adjacent_bitstring_tvds),
                    },
                ],
            )
    return rows


def _per_seed_means(
    rows: list[dict[str, float | int | str]],
) -> dict[tuple[str, str], list[float]]:
    values = defaultdict(list)
    grouped = defaultdict(list)
    for row in rows:
        for metric in ("shell_tvd", "bitstring_tvd"):
            grouped[(row["method"], row["seed"], metric)].append(row[metric])
    for (method, _seed, metric), observations in grouped.items():
        values[(method, metric)].append(float(np.mean(observations)))
    return values


def _audit(
    rows: list[dict[str, float | int | str]],
) -> tuple[dict[str, object], str]:
    comparisons = []
    for metric in ("shell_tvd", "bitstring_tvd"):
        for fraction in HELDOUT_ANGLE_FRACTIONS:
            medians = {}
            for method in ("dense", "nearest"):
                values = [
                    row[metric]
                    for row in rows
                    if row["method"] == method
                    and np.isclose(row["angle_fraction_pi"], fraction)
                ]
                medians[method] = float(np.median(values))
            comparisons.append(
                {
                    "test": "continuous_conditioning",
                    "metric": metric,
                    "angle_fraction_pi": float(fraction),
                    "dense": medians["dense"],
                    "baseline": medians["nearest"],
                    "passed": medians["dense"] < medians["nearest"],
                },
            )

        seed_means = _per_seed_means(rows)
        dense = float(np.median(seed_means[("dense", metric)]))
        endpoint = float(np.median(seed_means[("endpoint", metric)]))
        comparisons.append(
            {
                "test": "dense_supervision",
                "metric": metric,
                "angle_fraction_pi": None,
                "dense": dense,
                "baseline": endpoint,
                "passed": dense < endpoint,
            },
        )
    audit = {
        "continuous_conditioning_passed": {
            metric: all(
                item["passed"]
                for item in comparisons
                if item["test"] == "continuous_conditioning"
                and item["metric"] == metric
            )
            for metric in ("shell_tvd", "bitstring_tvd")
        },
        "dense_supervision_passed": {
            metric: next(
                item["passed"]
                for item in comparisons
                if item["test"] == "dense_supervision" and item["metric"] == metric
            )
            for metric in ("shell_tvd", "bitstring_tvd")
        },
        "comparisons": comparisons,
    }
    lines = [
        "# Held-out interpolation audit",
        "",
        "## Frozen tests",
        "",
        "| Test | Metric | Angle / pi | Dense | Baseline | Result |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        angle = (
            "mean"
            if item["angle_fraction_pi"] is None
            else f"{item['angle_fraction_pi']:.5f}"
        )
        baseline_name = (
            "endpoint-only"
            if item["test"] == "dense_supervision"
            else "oracle adjacent lookup"
        )
        lines.append(
            f"| {item['test'].replace('_', ' ')} vs {baseline_name} | "
            f"{item['metric'].replace('_', ' ')} | {angle} | "
            f"{item['dense']:.4f} | {item['baseline']:.4f} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |",
        )
    return audit, "\n".join(lines) + "\n"


def _metrics_markdown(
    rows: list[dict[str, float | int | str]],
) -> str:
    seed_means = _per_seed_means(rows)
    lines = [
        "# Held-out interpolation metrics",
        "",
        "Median [interquartile range] of each seed's mean over four held-out angles.",
        "",
        "| Method | Profile TVD | Bitstring TVD |",
        "|---|---:|---:|",
    ]
    for method in ("dense", "nearest", "endpoint"):
        shell = _median_iqr(seed_means[(method, "shell_tvd")])
        bit = _median_iqr(seed_means[(method, "bitstring_tvd")])
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{shell[0]:.4f} [{shell[1]:.4f}, {shell[2]:.4f}] | "
            f"{bit[0]:.4f} [{bit[1]:.4f}, {bit[2]:.4f}] |",
        )
    return "\n".join(lines) + "\n"


def _latex_table(rows: list[dict[str, float | int | str]]) -> str:
    seed_means = _per_seed_means(rows)
    lines = []
    for method in ("dense", "nearest", "endpoint"):
        shell = _median_iqr(seed_means[(method, "shell_tvd")])
        bit = _median_iqr(seed_means[(method, "bitstring_tvd")])
        lines.append(
            f"{METHOD_LABELS[method]} & "
            f"{shell[0]:.3f} [{shell[1]:.3f},{shell[2]:.3f}] & "
            f"{bit[0]:.3f} [{bit[1]:.3f},{bit[2]:.3f}] \\\\",
        )
    return "\n".join(lines) + "\n"


def _plot(
    dense_runs: list[dict[str, object]],
    rows: list[dict[str, float | int | str]],
    output: Path,
) -> None:
    use_tex = apply_journal_style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COL_WIDTH, 60 * MM),
        layout="constrained",
    )
    dense_points = defaultdict(list)
    target_means = {}
    for run in dense_runs:
        for point in run["points"]:
            fraction = point["angle_fraction_pi"]
            dense_points[fraction].append(point["model_mean"])
            target_means[fraction] = point["target_mean"]
    fractions = np.asarray(sorted(dense_points))
    mean_stats = np.asarray([_median_iqr(dense_points[value]) for value in fractions])
    axes[0].plot(
        fractions,
        [target_means[value] for value in fractions],
        "o-",
        color="#000000",
        label="Target",
        markersize=3,
    )
    axes[0].plot(
        fractions,
        mean_stats[:, 0],
        "s--",
        color="#0072B2",
        label="Model",
        markersize=3,
    )
    axes[0].fill_between(
        fractions,
        mean_stats[:, 1],
        mean_stats[:, 2],
        color="#0072B2",
        alpha=0.15,
    )
    for fraction in HELDOUT_ANGLE_FRACTIONS:
        axes[0].axvline(fraction, color="#999999", alpha=0.2, linewidth=0.7)
    if use_tex:
        axes[0].set_title(r"(a) Conditional Mean")
        axes[0].set_xlabel(r"Angle $\alpha/\pi$")
        axes[0].set_ylabel(r"Hamming Weight")
    else:
        axes[0].set_title("(a) Conditional Mean")
        axes[0].set_xlabel("Angle alpha/pi")
        axes[0].set_ylabel("Hamming Weight")
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        fontsize=7.5,
        loc="upper right",
    )
    axes[0].spines[["top", "right"]].set_visible(False)

    colors = {"dense": "#0072B2", "nearest": "#D55E00", "endpoint": "#009E73"}
    styles = {"dense": "o-", "nearest": "^--", "endpoint": "s:"}
    for axis, metric, title, panel_letter in (
        (axes[1], "shell_tvd", "Held-Out Profile TVD", "(b)"),
        (axes[2], "bitstring_tvd", "Held-Out Bitstring TVD", "(c)"),
    ):
        for method in ("dense", "nearest", "endpoint"):
            medians = []
            lower = []
            upper = []
            for fraction in HELDOUT_ANGLE_FRACTIONS:
                values = [
                    row[metric]
                    for row in rows
                    if row["method"] == method
                    and np.isclose(row["angle_fraction_pi"], fraction)
                ]
                median, q1, q3 = _median_iqr(values)
                medians.append(median)
                lower.append(median - q1)
                upper.append(q3 - median)
            axis.errorbar(
                HELDOUT_ANGLE_FRACTIONS,
                medians,
                yerr=np.asarray([lower, upper]),
                fmt=styles[method],
                color=colors[method],
                linewidth=1.2,
                markersize=3,
                capsize=2,
                label=METHOD_LABELS[method].replace(" interpolation", ""),
            )
        if use_tex:
            axis.set_title(rf"{panel_letter} {title}")
            axis.set_xlabel(r"Held-Out $\alpha/\pi$")
        else:
            axis.set_title(f"{panel_letter} {title}")
            axis.set_xlabel("Held-Out alpha/pi")
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].set_ylabel("Profile TVD")
    axes[2].set_ylabel("Bitstring TVD")
    axes[1].set_ylim(0.02, 0.25)
    axes[2].set_ylim(0.30, 0.42)
    axes[1].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        fontsize=7.5,
        loc="upper left",
    )
    save_figure(figure, output)


def main() -> None:
    """Generate interpolation metrics, audit, table, and figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/interpolation/runs"),
    )
    parser.add_argument(
        "--endpoint-runs",
        type=Path,
        default=Path("results/runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/interpolation/summary"),
    )
    args = parser.parse_args()

    dense_runs = _load_dense(args.runs)
    rows = _comparison_rows(dense_runs, args.runs, args.endpoint_runs)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = _metrics_markdown(rows)
    audit, audit_markdown = _audit(rows)
    (args.output / "metrics.md").write_text(metrics)
    (args.output / "audit.md").write_text(audit_markdown)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output / "interpolation_table.tex").write_text(_latex_table(rows))
    _plot(dense_runs, rows, args.output / "interpolation.png")
    print(metrics)
    print(audit_markdown)


if __name__ == "__main__":
    main()

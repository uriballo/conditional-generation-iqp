"""Aggregate hamming-ball results and generate paper artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from journal_style import DOUBLE_COL_WIDTH, MM, apply_journal_style, save_figure

from iqp_cond.hamming_experiment import TRAIN_CENTER_COUNTS

METHOD_LABELS = {
    "dense": "Dense-grid interpolation",
    "oracle": "Oracle adjacent-angle lookup",
}
CENTER_COUNT_TOLERANCE = 1.0


def _median_iqr(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
    )


def _load_runs(directory: Path) -> list[dict[str, object]]:
    runs = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    seeds = {run["seed"] for run in runs}
    if not seeds:
        raise SystemExit(f"no hamming runs found in {directory}")
    return runs


def _comparison_rows(
    runs: list[dict[str, object]],
) -> list[dict[str, float | int | str]]:
    rows = []
    for run in runs:
        for point in run["points"]:
            if point["role"] != "heldout":
                continue
            common = {
                "seed": run["seed"],
                "angle_fraction_pi": point["angle_fraction_pi"],
                "effective_center_count": point["effective_center_count"],
                "target_effective_k": point["target_effective_k"],
            }
            rows.append(
                {
                    **common,
                    "method": "dense",
                    "bitstring_tvd": point["bitstring_tvd"],
                },
            )
            rows.append(
                {
                    **common,
                    "method": "oracle",
                    "bitstring_tvd": point["oracle_bitstring_tvd"],
                },
            )
    return rows


def _control_rows(
    runs: list[dict[str, object]],
) -> list[dict[str, float | int | str]]:
    rows = []
    for run in runs:
        for point in run["points"]:
            if point["role"] != "train":
                continue
            rows.append(
                {
                    "seed": run["seed"],
                    "target_k": point["target_k"],
                    "angle_fraction_pi": point["angle_fraction_pi"],
                    "effective_center_count": point["effective_center_count"],
                    "n_modes": point["n_modes"],
                },
            )
    return rows


def _per_seed_means(
    rows: list[dict[str, float | int | str]],
) -> dict[tuple[str, str], list[float]]:
    values = defaultdict(list)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["seed"], "bitstring_tvd")].append(
            row["bitstring_tvd"],
        )
    for (method, _seed, metric), observations in grouped.items():
        values[(method, metric)].append(float(np.mean(observations)))
    return values


def _audit(
    rows: list[dict[str, float | int | str]],
    control_rows: list[dict[str, float | int | str]],
) -> tuple[dict[str, object], str]:
    comparisons = []
    for fraction in sorted({row["angle_fraction_pi"] for row in rows}):
        medians = {}
        for method in ("dense", "oracle"):
            values = [
                row["bitstring_tvd"]
                for row in rows
                if row["method"] == method
                and np.isclose(row["angle_fraction_pi"], fraction)
            ]
            medians[method] = float(np.median(values))
        comparisons.append(
            {
                "test": "continuous_conditioning",
                "metric": "bitstring_tvd",
                "angle_fraction_pi": float(fraction),
                "dense": medians["dense"],
                "baseline": medians["oracle"],
                "passed": medians["dense"] < medians["oracle"],
            },
        )

    control = []
    medians_by_k = {}
    for target_k in TRAIN_CENTER_COUNTS:
        values = [
            row["effective_center_count"]
            for row in control_rows
            if row["target_k"] == target_k
        ]
        median = float(np.median(values))
        medians_by_k[target_k] = median
        control.append(
            {
                "test": "center_count_control",
                "target_k": target_k,
                "median_effective_k": median,
                "within_tolerance": abs(median - target_k) <= CENTER_COUNT_TOLERANCE,
            },
        )
    ordered = [medians_by_k[k] for k in TRAIN_CENTER_COUNTS]
    control.append(
        {
            "test": "center_count_monotonic",
            "target_k": None,
            "median_effective_k": None,
            "within_tolerance": all(
                later > earlier for earlier, later in itertools.pairwise(ordered)
            ),
        },
    )
    audit = {
        "continuous_conditioning_passed": all(item["passed"] for item in comparisons),
        "center_count_control_passed": all(
            item["within_tolerance"]
            for item in control
            if item["test"] == "center_count_control"
        ),
        "center_count_monotonic_passed": control[-1]["within_tolerance"],
        "comparisons": comparisons,
        "control": control,
    }
    lines = [
        "# Hamming-ball audit",
        "",
        "## Frozen tests",
        "",
        "| Test | Held-out angle / pi | Dense | Oracle | Result |",
        "|---|---|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| continuous conditioning vs oracle | {item['angle_fraction_pi']:.5f} | "
            f"{item['dense']:.4f} | {item['baseline']:.4f} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |",
        )
    lines.extend(
        [
            "",
            "| Test | Class K | Median effective K | Result |",
            "|---|---|---:|---:|",
        ],
    )
    for item in control:
        label = (
            "monotonic increase" if item["target_k"] is None else str(item["target_k"])
        )
        value = (
            "n/a"
            if item["median_effective_k"] is None
            else f"{item['median_effective_k']:.2f}"
        )
        lines.append(
            f"| {item['test'].replace('_', ' ')} | {label} | {value} | "
            f"{'PASS' if item['within_tolerance'] else 'FAIL'} |",
        )
    return audit, "\n".join(lines) + "\n"


def _metrics_markdown(
    rows: list[dict[str, float | int | str]],
    control_rows: list[dict[str, float | int | str]],
) -> str:
    seed_means = _per_seed_means(rows)
    lines = [
        "# Hamming-ball metrics",
        "",
        "Median [interquartile range] of each seed's mean bitstring TVD over "
        "three held-out midpoint angles.",
        "",
        "| Method | Bitstring TVD |",
        "|---|---:|",
    ]
    for method in ("dense", "oracle"):
        median, q1, q3 = _median_iqr(seed_means[(method, "bitstring_tvd")])
        lines.append(
            f"| {METHOD_LABELS[method]} | {median:.4f} [{q1:.4f}, {q3:.4f}] |",
        )
    lines.extend(
        [
            "",
            "Model effective center count at the trained classes (median over "
            "seeds; target K in parentheses).",
            "",
            "| Class K | Effective K | Modes |",
            "|---|---:|---:|",
        ],
    )
    for target_k in TRAIN_CENTER_COUNTS:
        effective = [
            row["effective_center_count"]
            for row in control_rows
            if row["target_k"] == target_k
        ]
        modes = [row["n_modes"] for row in control_rows if row["target_k"] == target_k]
        effective_median, effective_q1, effective_q3 = _median_iqr(effective)
        mode_median, mode_q1, mode_q3 = _median_iqr(modes)
        lines.append(
            f"| {target_k} | {effective_median:.2f} "
            f"[{effective_q1:.2f}, {effective_q3:.2f}] | "
            f"{mode_median:.1f} [{mode_q1:.1f}, {mode_q3:.1f}] |",
        )
    return "\n".join(lines) + "\n"


def _latex_table(rows: list[dict[str, float | int | str]]) -> str:
    seed_means = _per_seed_means(rows)
    lines = []
    for method in ("dense", "oracle"):
        median, q1, q3 = _median_iqr(seed_means[(method, "bitstring_tvd")])
        lines.append(
            f"{METHOD_LABELS[method]} & {median:.3f} [{q1:.3f},{q3:.3f}] \\\\",
        )
    return "\n".join(lines) + "\n"


def _plot(
    runs: list[dict[str, object]],
    rows: list[dict[str, float | int | str]],
    output: Path,
) -> None:
    use_tex = apply_journal_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL_WIDTH, 60 * MM),
        layout="constrained",
    )

    colors = {"dense": "#0072B2", "oracle": "#D55E00"}
    styles = {"dense": "o-", "oracle": "^--"}
    fractions = sorted({row["angle_fraction_pi"] for row in rows})
    for method in ("dense", "oracle"):
        medians = []
        lower = []
        upper = []
        for fraction in fractions:
            values = [
                row["bitstring_tvd"]
                for row in rows
                if row["method"] == method
                and np.isclose(row["angle_fraction_pi"], fraction)
            ]
            median, q1, q3 = _median_iqr(values)
            medians.append(median)
            lower.append(median - q1)
            upper.append(q3 - median)
        axes[0].errorbar(
            fractions,
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
        axes[0].set_title(r"(a) Held-Out Bitstring TVD")
        axes[0].set_xlabel(r"Held-Out $\alpha/\pi$")
        axes[0].set_ylabel(r"Bitstring TVD")
    else:
        axes[0].set_title("(a) Held-Out Bitstring TVD")
        axes[0].set_xlabel("Held-Out alpha/pi")
        axes[0].set_ylabel("Bitstring TVD")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    fractions_by_role = defaultdict(list)
    target_k_by_fraction = {}
    for run in runs:
        for point in run["points"]:
            fraction = point["angle_fraction_pi"]
            fractions_by_role[(point["role"], fraction)].append(
                point["effective_center_count"],
            )
            target_k_by_fraction[fraction] = point["target_effective_k"]
    all_fractions = np.asarray(sorted(target_k_by_fraction))
    train_fractions = np.asarray(sorted(fractions_by_role[("train",)]))
    axes[1].plot(
        all_fractions,
        [target_k_by_fraction[fraction] for fraction in all_fractions],
        "o-",
        color="#000000",
        label="Target",
        linewidth=1.5,
        markersize=3,
    )
    model_medians = [
        _median_iqr(fractions_by_role[("train", fraction)])[0]
        for fraction in train_fractions
    ]
    axes[1].plot(
        train_fractions,
        model_medians,
        "s--",
        color="#0072B2",
        label="Model",
        linewidth=1.5,
        markersize=3,
    )
    if use_tex:
        axes[1].set_title(r"(b) Effective Center Count")
        axes[1].set_xlabel(r"Angle $\alpha/\pi$")
        axes[1].set_ylabel(r"Effective $K$")
    else:
        axes[1].set_title("(b) Effective Center Count")
        axes[1].set_xlabel("Angle alpha/pi")
        axes[1].set_ylabel("Effective K")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    save_figure(figure, output)


def main() -> None:
    """Generate hamming metrics, audit, table, and figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/hamming/runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hamming/summary"),
    )
    args = parser.parse_args()

    runs = _load_runs(args.runs)
    rows = _comparison_rows(runs)
    control_rows = _control_rows(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = _metrics_markdown(rows, control_rows)
    audit, audit_markdown = _audit(rows, control_rows)
    (args.output / "metrics.md").write_text(metrics)
    (args.output / "audit.md").write_text(audit_markdown)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output / "hamming_table.tex").write_text(_latex_table(rows))
    _plot(runs, rows, args.output / "hamming.png")
    print(metrics)
    print(audit_markdown)


if __name__ == "__main__":
    main()

"""Summarize exact metrics and generate paper-facing figures and tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DISPLAY_NAMES = {
    "angle_heavy_hex": "Angle / heavy-hex",
    "angle_lattice": "Angle / lattice",
    "angle_complete": "Angle / complete",
    "no_condition_complete": "No condition",
    "basis_condition_complete": "Basis condition",
    "separate_a_complete": "Separate A",
    "separate_b_complete": "Separate B",
}


def _load(directory: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(directory.glob("*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def _condition_rows(
    runs: list[dict[str, object]],
) -> list[dict[str, object]]:
    flattened = []
    for run in runs:
        for condition in run["conditions"]:
            flattened.append(
                {
                    "model_name": run["model_name"],
                    "seed": run["seed"],
                    "n_parameters": run["n_parameters"],
                    **condition,
                },
            )
    return flattened


def _median_iqr(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
    )


def summary_table(rows: list[dict[str, object]]) -> str:
    """Return a Markdown table of seed-aggregated exact metrics."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_name"], row["condition"])].append(row)
    lines = [
        "| Model | Condition | Params | Bitstring TVD | Shell TVD | "
        "Mean error | Std. error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(grouped):
        model_name, condition = key
        group = grouped[key]
        bit = _median_iqr([item["bitstring_tvd"] for item in group])
        shell = _median_iqr([item["shell_tvd"] for item in group])
        mean = _median_iqr([item["mean_absolute_error"] for item in group])
        std = _median_iqr([item["std_absolute_error"] for item in group])
        lines.append(
            f"| {DISPLAY_NAMES[model_name]} | {condition} | "
            f"{group[0]['n_parameters']} | "
            f"{bit[0]:.4f} [{bit[1]:.4f}, {bit[2]:.4f}] | "
            f"{shell[0]:.4f} [{shell[1]:.4f}, {shell[2]:.4f}] | "
            f"{mean[0]:.3f} [{mean[1]:.3f}, {mean[2]:.3f}] | "
            f"{std[0]:.3f} [{std[1]:.3f}, {std[2]:.3f}] |",
        )
    return "\n".join(lines) + "\n"


def latex_table(rows: list[dict[str, object]]) -> str:
    """Return compact LaTeX rows for the paper table."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_name"], row["condition"])].append(row)
    order = [
        "angle_heavy_hex",
        "angle_lattice",
        "angle_complete",
        "no_condition_complete",
        "basis_condition_complete",
        "separate_a_complete",
        "separate_b_complete",
    ]
    lines = []
    for model_name in order:
        keys = [key for key in grouped if key[0] == model_name]
        for _, condition in sorted(keys):
            group = grouped[(model_name, condition)]
            bit = _median_iqr([item["bitstring_tvd"] for item in group])
            shell = _median_iqr([item["shell_tvd"] for item in group])
            mean = _median_iqr([item["mean_absolute_error"] for item in group])
            std = _median_iqr([item["std_absolute_error"] for item in group])
            model_label = DISPLAY_NAMES[model_name].replace("/", r"\,/\,")
            lines.append(
                f"{model_label} & {condition} & {group[0]['n_parameters']} & "
                f"{bit[0]:.3f} [{bit[1]:.3f},{bit[2]:.3f}] & "
                f"{shell[0]:.3f} [{shell[1]:.3f},{shell[2]:.3f}] & "
                f"{mean[0]:.2f} [{mean[1]:.2f},{mean[2]:.2f}] & "
                f"{std[0]:.2f} [{std[1]:.2f},{std[2]:.2f}] \\\\",
            )
        if keys:
            lines.append(r"\addlinespace[1pt]")
    return "\n".join([*lines[:-1], r"\bottomrule"]) + "\n"


def success_audit(rows: list[dict[str, object]]) -> tuple[str, dict[str, object]]:
    """Audit the preregistered condition-wise ordering criterion."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_name"], row["condition"])].append(row)

    comparisons = []
    passed = True
    for condition in ("A", "B"):
        for metric in ("shell_tvd", "bitstring_tvd"):
            medians = {
                model: _median_iqr(
                    [item[metric] for item in grouped[(model, condition)]],
                )[0]
                for model in (
                    "angle_complete",
                    "no_condition_complete",
                    "basis_condition_complete",
                )
            }
            comparison_passed = medians["angle_complete"] < min(
                medians["no_condition_complete"],
                medians["basis_condition_complete"],
            )
            passed &= comparison_passed
            comparisons.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "angle_complete": medians["angle_complete"],
                    "no_condition": medians["no_condition_complete"],
                    "basis_condition": medians["basis_condition_complete"],
                    "passed": comparison_passed,
                },
            )

    balanced = {}
    for model in (
        "angle_complete",
        "no_condition_complete",
        "basis_condition_complete",
    ):
        by_seed = defaultdict(list)
        for row in rows:
            if row["model_name"] == model:
                by_seed[row["seed"]].append(row)
        balanced[model] = {
            metric: _median_iqr(
                [
                    np.mean([item[metric] for item in seed_rows])
                    for seed_rows in by_seed.values()
                ],
            )[0]
            for metric in ("shell_tvd", "bitstring_tvd")
        }

    audit = {
        "passed": passed,
        "criterion": (
            "Angle/complete must have lower median shell TVD and bitstring TVD "
            "than both conditional baselines for A and B separately."
        ),
        "comparisons": comparisons,
        "balanced_condition_medians": balanced,
    }
    lines = [
        "# Preregistered success-criterion audit",
        "",
        f"**Overall result: {'PASS' if passed else 'FAIL'}.**",
        "",
        audit["criterion"],
        "",
        "| Condition | Metric | Angle/complete | No condition | "
        "Basis condition | Result |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            f"| {item['condition']} | {item['metric'].replace('_', ' ')} | "
            f"{item['angle_complete']:.4f} | {item['no_condition']:.4f} | "
            f"{item['basis_condition']:.4f} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |",
        )
    lines.extend(
        [
            "",
            "Balanced averages are descriptive diagnostics, not substitutes for "
            "the condition-wise criterion.",
            "",
            "| Model | Mean-condition shell TVD | Mean-condition bitstring TVD |",
            "|---|---:|---:|",
        ],
    )
    for model, values in balanced.items():
        lines.append(
            f"| {DISPLAY_NAMES[model]} | {values['shell_tvd']:.4f} | "
            f"{values['bitstring_tvd']:.4f} |",
        )
    return "\n".join(lines) + "\n", audit


def _best_representative(
    rows: list[dict[str, object]],
    model_name: str,
) -> dict[str, dict[str, object]]:
    candidates = defaultdict(dict)
    for row in rows:
        if row["model_name"] == model_name:
            candidates[row["seed"]][row["condition"]] = row
    scores = {
        seed: np.mean([item["shell_tvd"] for item in conditions.values()])
        for seed, conditions in candidates.items()
    }
    median_score = np.median(list(scores.values()))
    seed = min(scores, key=lambda item: abs(scores[item] - median_score))
    return candidates[seed]


def plot_weight_fit(rows: list[dict[str, object]], output: Path) -> None:
    """Plot target and representative angle-complete Hamming-weight laws."""
    representative = _best_representative(rows, "angle_complete")
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), sharey=True)
    for axis, condition in zip(axes, ("A", "B"), strict=True):
        row = representative[condition]
        weights = np.arange(len(row["target_shell_probabilities"]))
        axis.plot(
            weights,
            row["target_shell_probabilities"],
            "o-",
            color="#222222",
            label="Target",
            linewidth=1.5,
            markersize=3,
        )
        axis.plot(
            weights,
            row["model_shell_probabilities"],
            "s--",
            color="#0072B2",
            label="Angle-conditioned IQP",
            linewidth=1.5,
            markersize=3,
        )
        axis.set_title(f"Condition {condition}")
        axis.set_xlabel("Hamming weight")
        axis.set_xticks([0, 4, 8, 12, 16, 19])
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Probability mass")
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_comparisons(rows: list[dict[str, object]], output: Path) -> None:
    """Plot seed distributions for topology and conditioning comparisons."""
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    comparisons = (
        (
            axes[0],
            ["angle_heavy_hex", "angle_lattice", "angle_complete"],
            "Topology families",
        ),
        (
            axes[1],
            [
                "angle_complete",
                "no_condition_complete",
                "basis_condition_complete",
            ],
            "Conditioning ablation",
        ),
    )
    for axis, models, title in comparisons:
        data = []
        labels = []
        for model in models:
            per_seed = defaultdict(list)
            for row in rows:
                if row["model_name"] == model:
                    per_seed[row["seed"]].append(row["shell_tvd"])
            data.append([np.mean(values) for values in per_seed.values()])
            labels.append(DISPLAY_NAMES[model].replace("Angle / ", ""))
        axis.boxplot(data, tick_labels=labels, showfliers=False)
        axis.set_title(title)
        axis.set_ylabel("Mean condition shell TVD")
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Write aggregate tables and figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="?", default=Path("results/runs"))
    parser.add_argument("--output", type=Path, default=Path("results/summary"))
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="summarize partial model/seed matrices instead of requiring 10 seeds",
    )
    args = parser.parse_args()

    runs = _load(args.directory)
    seed_counts = {
        model: len({run["seed"] for run in runs if run["model_name"] == model})
        for model in DISPLAY_NAMES
    }
    incomplete = {model: count for model, count in seed_counts.items() if count != 10}
    if incomplete and not args.allow_incomplete:
        formatted = ", ".join(f"{model}={count}" for model, count in incomplete.items())
        raise SystemExit(f"expected 10 seeds per model; found {formatted}")

    rows = _condition_rows(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    table = summary_table(rows)
    (args.output / "metrics.md").write_text(table)
    (args.output / "results_table.tex").write_text(latex_table(rows))
    required_models = {
        "angle_complete",
        "no_condition_complete",
        "basis_condition_complete",
    }
    if required_models.issubset({row["model_name"] for row in rows}):
        audit_markdown, audit = success_audit(rows)
        (args.output / "success_criterion.md").write_text(audit_markdown)
        (args.output / "success_criterion.json").write_text(
            json.dumps(audit, indent=2) + "\n",
        )
    plot_weight_fit(rows, args.output / "weight_fit.png")
    plot_comparisons(rows, args.output / "comparisons.png")
    print(table)


if __name__ == "__main__":
    main()

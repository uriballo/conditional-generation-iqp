"""Aggregate the audited condition-count scaling matrix."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from journal_style import DOUBLE_COL_WIDTH, MM, apply_journal_style, save_figure

from iqp_cond.scaling import (
    SCALING_CONDITION_COUNTS,
    SCALING_TRACKS,
    aggregate_condition_runs,
    strict_comparison_counts,
)

METHOD_LABELS = {
    "shared_angle": "Shared Angle",
    "no_condition": "No-condition shared",
    "separate_bank": "Separate bank",
}
METHOD_ORDER = tuple(METHOD_LABELS)
METRICS = ("profile_tvd", "bitstring_tvd")


def _parse_ints(value: str) -> list[int]:
    if "-" in value:
        start, end = (int(part) for part in value.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def _load(
    directory: Path,
    expected_seeds: set[int],
    allow_incomplete: bool,
) -> list[dict[str, object]]:
    runs = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    if not runs:
        raise SystemExit(f"no scaling JSON files found in {directory}")
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["track"], run["n_conditions"], run["method"])].append(run)
    if not allow_incomplete:
        expected_groups = {
            (track, n_conditions, method)
            for track in SCALING_TRACKS
            for n_conditions in SCALING_CONDITION_COUNTS
            for method in METHOD_ORDER
        }
        incomplete = {
            key: sorted(int(run["seed"]) for run in grouped.get(key, []))
            for key in expected_groups
            if {int(run["seed"]) for run in grouped.get(key, [])}
            != expected_seeds
        }
        if incomplete:
            raise SystemExit(
                "incomplete scaling matrix; pass --allow-incomplete to inspect: "
                f"{incomplete}",
            )
    return runs


def _group(
    runs: list[dict[str, object]],
) -> dict[tuple[str, int, str], list[dict[str, object]]]:
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["track"], int(run["n_conditions"]), run["method"])].append(
            run,
        )
    return grouped


def _fmt(stats: dict[str, float], digits: int = 4) -> str:
    return (
        f"{stats['median']:.{digits}f} "
        f"[{stats['q1']:.{digits}f}, {stats['q3']:.{digits}f}]"
    )


def _summary_table(
    grouped: dict[tuple[str, int, str], list[dict[str, object]]],
) -> str:
    lines = [
        "# Condition-count scaling metrics",
        "",
        "Values are median [IQR] across seeds of per-seed Condition summaries.",
        "The mean, median, and worst Condition aggregations are retained.",
        "",
        "| Track | K | Method | Params | Mean profile TVD | Worst profile TVD | "
        "Median profile TVD | Mean bitstring TVD | Median bitstring TVD | "
        "Worst bitstring TVD |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for track in ("fixed_total", "fixed_per_condition"):
        for n_conditions in SCALING_CONDITION_COUNTS:
            for method in METHOD_ORDER:
                key = (track, n_conditions, method)
                if key not in grouped:
                    continue
                summary = aggregate_condition_runs(grouped[key])
                aggregate = summary["aggregate"]
                params = grouped[key][0]["n_parameters"]
                lines.append(
                    f"| {track} | {n_conditions} | {METHOD_LABELS[method]} | "
                    f"{params} | {_fmt(aggregate['mean']['profile_tvd'])} | "
                    f"{_fmt(aggregate['worst']['profile_tvd'])} | "
                    f"{_fmt(aggregate['median']['profile_tvd'])} | "
                    f"{_fmt(aggregate['mean']['bitstring_tvd'])} | "
                    f"{_fmt(aggregate['median']['bitstring_tvd'])} | "
                    f"{_fmt(aggregate['worst']['bitstring_tvd'])} |",
                )
    return "\n".join(lines) + "\n"


def _condition_table(
    grouped: dict[tuple[str, int, str], list[dict[str, object]]],
) -> str:
    lines = [
        "# Per-Condition scaling audit",
        "",
        "Values are median [IQR] across seeds; no Condition is pooled away.",
        "",
        "| Track | K | Method | Condition | alpha/pi | Profile TVD | "
        "Bitstring TVD |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for track in ("fixed_total", "fixed_per_condition"):
        for n_conditions in SCALING_CONDITION_COUNTS:
            for method in METHOD_ORDER:
                key = (track, n_conditions, method)
                if key not in grouped:
                    continue
                summary = aggregate_condition_runs(grouped[key])
                for row in summary["per_condition"]:
                    lines.append(
                        f"| {track} | {n_conditions} | {METHOD_LABELS[method]} | "
                        f"{row['condition']} | {row['angle_fraction_pi']:.6f} | "
                        f"{_fmt(row['profile_tvd'])} | "
                        f"{_fmt(row['bitstring_tvd'])} |",
                    )
    return "\n".join(lines) + "\n"


def _per_seed_table(
    grouped: dict[tuple[str, int, str], list[dict[str, object]]],
) -> str:
    lines = [
        "# Per-seed scaling metrics",
        "",
        "Each row keeps the mean, median, and worst Condition visible for both "
        "fidelity resolutions.",
        "",
        "| Track | K | Method | Seed | Mean profile | Median profile | Worst "
        "profile | Mean bitstring | Median bitstring | Worst bitstring |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for track in ("fixed_total", "fixed_per_condition"):
        for n_conditions in SCALING_CONDITION_COUNTS:
            for method in METHOD_ORDER:
                key = (track, n_conditions, method)
                if key not in grouped:
                    continue
                summary = aggregate_condition_runs(grouped[key])
                for row in summary["per_seed"]:
                    lines.append(
                        f"| {track} | {n_conditions} | {METHOD_LABELS[method]} | "
                        f"{row['seed']} | {row['mean_profile_tvd']:.4f} | "
                        f"{row['median_profile_tvd']:.4f} | "
                        f"{row['worst_profile_tvd']:.4f} | "
                        f"{row['mean_bitstring_tvd']:.4f} | "
                        f"{row['median_bitstring_tvd']:.4f} | "
                        f"{row['worst_bitstring_tvd']:.4f} |",
                    )
    return "\n".join(lines) + "\n"


def _audit(
    grouped: dict[tuple[str, int, str], list[dict[str, object]]],
) -> tuple[dict[str, object], str]:
    primary = "fixed_total"
    criterion_rows = []
    all_passed = True
    for metric in METRICS:
        for n_conditions in SCALING_CONDITION_COUNTS[1:]:
            shared_key = (primary, n_conditions, "shared_angle")
            no_condition_key = (primary, n_conditions, "no_condition")
            separate_key = (primary, n_conditions, "separate_bank")
            if not all(
                key in grouped
                for key in (shared_key, no_condition_key, separate_key)
            ):
                continue
            shared = aggregate_condition_runs(grouped[shared_key])
            no_condition = aggregate_condition_runs(grouped[no_condition_key])
            separate = aggregate_condition_runs(grouped[separate_key])
            shared_mean = shared["aggregate"]["mean"][metric]["median"]
            no_mean = no_condition["aggregate"]["mean"][metric]["median"]
            shared_worst = shared["aggregate"]["worst"][metric]["median"]
            no_worst = no_condition["aggregate"]["worst"][metric]["median"]
            mean_better = shared_mean <= 0.8 * no_mean
            worst_better = shared_worst <= no_worst + 0.05
            shared_k2 = aggregate_condition_runs(
                grouped[(primary, 2, "shared_angle")],
            )
            k2_mean = shared_k2["aggregate"]["mean"][metric]["median"]
            k2_worst = shared_k2["aggregate"]["worst"][metric]["median"]
            mean_degradation = shared_mean - k2_mean <= 0.10
            worst_degradation = shared_worst - k2_worst <= 0.15
            no_count = strict_comparison_counts(
                grouped[shared_key],
                grouped[no_condition_key],
                metric,
            )
            separate_count = strict_comparison_counts(
                grouped[shared_key],
                grouped[separate_key],
                metric,
            )
            failure_count = no_count["failures"] <= no_count["total"] / 2
            parameter_passed = grouped[shared_key][0]["n_parameters"] <= (
                0.5 * grouped[separate_key][0]["n_parameters"]
            )
            passed = all(
                (
                    mean_better,
                    worst_better,
                    mean_degradation,
                    worst_degradation,
                    failure_count,
                    parameter_passed,
                ),
            )
            all_passed &= passed
            criterion_rows.append(
                {
                    "n_conditions": n_conditions,
                    "metric": metric,
                    "shared_mean": shared_mean,
                    "no_condition_mean": no_mean,
                    "shared_worst": shared_worst,
                    "no_condition_worst": no_worst,
                    "k2_shared_mean": k2_mean,
                    "k2_shared_worst": k2_worst,
                    "shared_vs_no_condition_failures": no_count,
                    "shared_vs_separate_failures": separate_count,
                    "fidelity_gap_mean_vs_separate": (
                        shared["aggregate"]["mean"][metric]["median"]
                        - separate["aggregate"]["mean"][metric]["median"]
                    ),
                    "mean_better_by_20_percent": mean_better,
                    "worst_within_0_05": worst_better,
                    "mean_degradation_within_0_10": mean_degradation,
                    "worst_degradation_within_0_15": worst_degradation,
                    "failure_count_at_most_half": failure_count,
                    "parameter_advantage": parameter_passed,
                    "passed": passed,
                },
            )
    audit = {
        "protocol": "condition_count_scaling_v1",
        "primary_track": primary,
        "passed": bool(all_passed and len(criterion_rows) == 6),
        "criteria": {
            "shared_mean_at_least_20_percent_better_than_no_condition": True,
            "shared_worst_no_more_than_no_condition_plus_0_05": True,
            "mean_degradation_from_k2_no_more_than_0_10": True,
            "worst_degradation_from_k2_no_more_than_0_15": True,
            "strict_failure_count_at_most_half": True,
            "shared_uses_at_least_50_percent_fewer_parameters": True,
        },
        "comparisons": criterion_rows,
    }
    lines = [
        "# Condition-count scaling audit",
        "",
        f"**Primary fixed-total result: {'PASS' if audit['passed'] else 'FAIL'}.**",
        "",
        "All criteria are evaluated separately for profile and Bitstring TVD.",
        "Strict failures count individual seed/Condition comparisons.",
        "",
        "| K | Metric | Shared mean | No-condition mean | Shared worst | "
        "No-condition worst | Shared-vs-No failures | Shared-vs-Separate failures | "
        "Gap to Separate | Result |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in criterion_rows:
        no_count = row["shared_vs_no_condition_failures"]
        separate_count = row["shared_vs_separate_failures"]
        lines.append(
            f"| {row['n_conditions']} | {row['metric']} | "
            f"{row['shared_mean']:.4f} | {row['no_condition_mean']:.4f} | "
            f"{row['shared_worst']:.4f} | {row['no_condition_worst']:.4f} | "
            f"{no_count['failures']}/{no_count['total']} | "
            f"{separate_count['failures']}/{separate_count['total']} | "
            f"{row['fidelity_gap_mean_vs_separate']:+.4f} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |",
        )
    lines.extend(
        [
            "",
            "A failed Condition is retained in the per-Condition audit and is "
            "not hidden by the aggregate result.",
        ],
    )
    return audit, "\n".join(lines) + "\n"


def _latex_table(
    grouped: dict[tuple[str, int, str], list[dict[str, object]]],
) -> str:
    lines = []
    for n_conditions in SCALING_CONDITION_COUNTS:
        for method in METHOD_ORDER:
            key = ("fixed_total", n_conditions, method)
            if key not in grouped:
                continue
            aggregate = aggregate_condition_runs(grouped[key])["aggregate"]
            lines.append(
                f"{n_conditions} & {METHOD_LABELS[method]} & "
                f"{aggregate['mean']['profile_tvd']['median']:.3f} & "
                f"{aggregate['worst']['profile_tvd']['median']:.3f} & "
                f"{aggregate['mean']['bitstring_tvd']['median']:.3f} & "
                f"{aggregate['worst']['bitstring_tvd']['median']:.3f} "
                r"\\",
            )
    return "\n".join(lines) + "\n"


def _plot(
    grouped: dict[tuple[str, int, str], list[dict[str, object]]],
    output: Path,
) -> None:
    use_tex = apply_journal_style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COL_WIDTH, 55 * MM),
        layout="constrained",
    )
    colors = {
        "shared_angle": "#0072B2",
        "no_condition": "#D55E00",
        "separate_bank": "#009E73",
    }
    styles = {"shared_angle": "o-", "no_condition": "^--", "separate_bank": "s:"}
    for axis, metric, title in zip(
        axes[:2],
        METRICS,
        ("Profile TVD", "Bitstring TVD"),
        strict=True,
    ):
        for method in METHOD_ORDER:
            xs = []
            medians = []
            lower = []
            upper = []
            for n_conditions in SCALING_CONDITION_COUNTS:
                key = ("fixed_total", n_conditions, method)
                if key not in grouped:
                    continue
                aggregate = aggregate_condition_runs(grouped[key])["aggregate"][
                    "mean"
                ][metric]
                xs.append(n_conditions)
                medians.append(aggregate["median"])
                lower.append(aggregate["q1"])
                upper.append(aggregate["q3"])
            if not xs:
                continue
            axis.plot(
                xs,
                medians,
                styles[method],
                color=colors[method],
                linewidth=1.2,
                markersize=3,
                label=METHOD_LABELS[method],
            )
            axis.fill_between(
                xs,
                lower,
                upper,
                color=colors[method],
                alpha=0.10,
            )
        if use_tex:
            axis.set_title(title)
            axis.set_xlabel(r"Number of Conditions $K$")
            axis.set_ylabel("TVD (median [IQR])")
        else:
            axis.set_title(title)
            axis.set_xlabel("Number of Conditions K")
            axis.set_ylabel("TVD (median [IQR])")
        axis.set_xticks(SCALING_CONDITION_COUNTS)
        axis.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False, fontsize=7.5)
    parameter_axis = axes[2]
    for method in METHOD_ORDER:
        xs = []
        values = []
        for n_conditions in SCALING_CONDITION_COUNTS:
            key = ("fixed_total", n_conditions, method)
            if key not in grouped:
                continue
            values.append(grouped[key][0]["n_parameters"])
            xs.append(n_conditions)
        if xs:
            parameter_axis.plot(
                xs,
                values,
                styles[method],
                color=colors[method],
                linewidth=1.2,
                markersize=3,
                label=METHOD_LABELS[method],
            )
    if use_tex:
        parameter_axis.set_title("Trainable parameters")
        parameter_axis.set_xlabel(r"Number of Conditions $K$")
        parameter_axis.set_ylabel("Parameters")
    else:
        parameter_axis.set_title("Trainable parameters")
        parameter_axis.set_xlabel("Number of Conditions K")
        parameter_axis.set_ylabel("Parameters")
    parameter_axis.set_xticks(SCALING_CONDITION_COUNTS)
    parameter_axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, output)


def main() -> None:
    """Write scaling tables, audit, and the primary fidelity figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("results/scaling/runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/scaling/summary"),
    )
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    runs = _load(args.runs, set(_parse_ints(args.seeds)), args.allow_incomplete)
    grouped = _group(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.md").write_text(_summary_table(grouped))
    (args.output / "per_condition.md").write_text(_condition_table(grouped))
    (args.output / "per_seed.md").write_text(_per_seed_table(grouped))
    audit, audit_markdown = _audit(grouped)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output / "audit.md").write_text(audit_markdown)
    (args.output / "scaling_table.tex").write_text(_latex_table(grouped))
    _plot(grouped, args.output / "scaling.png")
    print((args.output / "metrics.md").read_text())
    print(audit_markdown)


if __name__ == "__main__":
    main()

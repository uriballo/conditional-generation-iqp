"""Aggregate Hopfield-beta results and generate audit artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from journal_style import (
    DOUBLE_COL_WIDTH,
    MM,
    SINGLE_COL_WIDTH,
    apply_journal_style,
    save_figure,
)

from iqp_cond.exact import enumerate_bitstrings, exact_iqp_probabilities_many
from iqp_cond.hopfield import (
    HELDOUT_BETAS,
    N_QUBITS,
    TRAIN_BETAS,
    broadcast_angles,
    hopfield_targets,
)

METHOD_LABELS = {
    "dense": "Dense conditional",
    "no_condition": "No-condition",
    "separate": "Separate models",
    "oracle": "Oracle adjacent-beta lookup",
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
    runs = [run for run in runs if run["model_name"].startswith("hopfield_complete")]
    seeds = {run["seed"] for run in runs}
    if not seeds:
        raise SystemExit(f"no hopfield runs found in {directory}")
    return runs


def _comparison_rows(
    dense_runs: list[dict[str, object]],
    runs_directory: Path,
) -> list[dict[str, float | int | str]]:
    """Build dense, oracle, no-condition, and separate comparison rows."""
    targets = hopfield_targets(np.concatenate([TRAIN_BETAS, HELDOUT_BETAS]))
    bitstrings = enumerate_bitstrings(N_QUBITS)
    target_probs = [target.probabilities for target in targets]
    rows: list[dict[str, float | int | str]] = []

    no_condition_runs = {
        run["seed"]: run
        for run in (
            json.loads(path.read_text())
            for path in sorted(runs_directory.glob("hopfield_no_condition*.json"))
        )
    }
    separate_rows = [
        json.loads(path.read_text())
        for path in sorted((runs_directory / "separate").glob("*.json"))
    ]

    for run in dense_runs:
        seed = run["seed"]
        stored = np.load(runs_directory / f"hopfield_complete_seed{seed}.npz")
        train_probabilities = exact_iqp_probabilities_many(
            stored["best_params"],
            stored["generators"],
            broadcast_angles(TRAIN_BETAS),
        )
        for local_index, beta in enumerate(TRAIN_BETAS):
            common = {
                "seed": seed,
                "beta": float(beta),
                "role": "train",
            }
            dense_point = run["points"][local_index]
            rows.append({
                **common,
                "method": "dense",
                "bitstring_tvd": dense_point["bitstring_tvd"],
                "shell_tvd": dense_point["shell_tvd"],
            })
            if seed in no_condition_runs:
                no_condition_point = next(
                    (
                        point
                        for point in no_condition_runs[seed]["points"]
                        if np.isclose(point["beta"], beta)
                    ),
                    None,
                )
                if no_condition_point:
                    rows.append({
                        **common,
                        "method": "no_condition",
                        "bitstring_tvd": no_condition_point["bitstring_tvd"],
                        "shell_tvd": no_condition_point["shell_tvd"],
                    })
            separate_match = next(
                (
                    run_match
                    for run_match in separate_rows
                    if run_match["seed"] == seed
                    and np.isclose(run_match["points"][0]["beta"], beta)
                ),
                None,
            )
            if separate_match:
                separate_point = separate_match["points"][0]
                rows.append({
                    **common,
                    "method": "separate",
                    "bitstring_tvd": separate_point["bitstring_tvd"],
                    "shell_tvd": separate_point["shell_tvd"],
                })
        for heldout_index, (beta, target_probs_row) in enumerate(
            zip(HELDOUT_BETAS, target_probs[4:], strict=True),
        ):
            dense_point = run["points"][len(TRAIN_BETAS) + heldout_index]
            adjacent_shell_tvds = []
            adjacent_bitstring_tvds = []
            for adjacent_index in (heldout_index, heldout_index + 1):
                adjacent_probs = train_probabilities[adjacent_index]
                adjacent_shell_tvds.append(
                    float(
                        0.5
                        * np.abs(
                            targets[adjacent_index].shell_probabilities
                            - np.bincount(
                                bitstrings.sum(axis=1),
                                weights=adjacent_probs,
                                minlength=N_QUBITS + 1,
                            ),
                        ).sum(),
                    ),
                )
                adjacent_bitstring_tvds.append(
                    float(
                        0.5 * np.abs(target_probs_row - adjacent_probs).sum(),
                    ),
                )
            common = {
                "seed": seed,
                "beta": float(beta),
                "role": "heldout",
            }
            rows.extend(
                [
                    {
                        **common,
                        "method": "dense",
                        "bitstring_tvd": dense_point["bitstring_tvd"],
                        "shell_tvd": dense_point["shell_tvd"],
                    },
                    {
                        **common,
                        "method": "oracle",
                        "bitstring_tvd": min(adjacent_bitstring_tvds),
                        "shell_tvd": min(adjacent_shell_tvds),
                    },
                ],
            )
    return rows


def _grouped_values(
    rows: list[dict[str, float | int | str]],
    method: str,
    role: str,
    metric: str,
) -> dict[float, list[float]]:
    values: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method and row["role"] == role:
            values[float(row["beta"])].append(row[metric])
    return values


def _audit(
    rows: list[dict[str, float | int | str]],
) -> tuple[dict[str, object], str]:
    comparisons = []
    for metric in ("bitstring_tvd", "shell_tvd"):
        for beta in TRAIN_BETAS:
            dense = [
                row[metric]
                for row in rows
                if row["method"] == "dense"
                and row["role"] == "train"
                and np.isclose(row["beta"], beta)
            ]
            no_condition = [
                row[metric]
                for row in rows
                if row["method"] == "no_condition"
                and np.isclose(row["beta"], beta)
            ]
            comparisons.append(
                {
                    "test": "memorization",
                    "metric": metric,
                    "beta": float(beta),
                    "role": "train",
                    "dense": float(np.median(dense)),
                    "baseline": float(np.median(no_condition)),
                    "passed": float(np.median(dense)) < float(
                        np.median(no_condition),
                    ),
                },
            )
        for beta in HELDOUT_BETAS:
            dense = [
                row[metric]
                for row in rows
                if row["method"] == "dense"
                and row["role"] == "heldout"
                and np.isclose(row["beta"], beta)
            ]
            oracle = [
                row[metric]
                for row in rows
                if row["method"] == "oracle"
                and np.isclose(row["beta"], beta)
            ]
            comparisons.append(
                {
                    "test": "continuous_conditioning",
                    "metric": metric,
                    "beta": float(beta),
                    "role": "heldout",
                    "dense": float(np.median(dense)),
                    "baseline": float(np.median(oracle)),
                    "passed": float(np.median(dense))
                    < float(np.median(oracle)),
                },
            )
    audit = {
        "memorization_passed": {
            metric: all(
                item["passed"]
                for item in comparisons
                if item["test"] == "memorization"
                and item["metric"] == metric
            )
            for metric in ("bitstring_tvd", "shell_tvd")
        },
        "continuous_conditioning_passed": {
            metric: all(
                item["passed"]
                for item in comparisons
                if item["test"] == "continuous_conditioning"
                and item["metric"] == metric
            )
            for metric in ("bitstring_tvd", "shell_tvd")
        },
        "comparisons": comparisons,
    }
    lines = [
        "# Hopfield-beta audit",
        "",
        "## Frozen tests",
        "",
        "| Test | Metric | Beta | Dense | Baseline | Result |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        baseline_name = (
            "oracle adjacent lookup"
            if item["test"] == "continuous_conditioning"
            else "no-condition"
        )
        lines.append(
            f"| {item['test'].replace('_', ' ')} vs {baseline_name} | "
            f"{item['metric'].replace('_', ' ')} | {item['beta']:.4f} | "
            f"{item['dense']:.4f} | {item['baseline']:.4f} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |",
        )
    return audit, "\n".join(lines) + "\n"


def _metrics_markdown(
    rows: list[dict[str, float | int | str]],
) -> str:
    lines = [
        "# Hopfield-beta metrics",
        "",
        "Median [interquartile range] over seeds. Memorization is the mean over "
        "the four training betas; interpolation is the mean over the three "
        "held-out betas.",
        "",
        "| Method | Train bitstring TVD | Train shell TVD | "
        "Held-out bitstring TVD | Held-out shell TVD |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("dense", "no_condition", "separate", "oracle"):
        stats = {}
        for role in ("train", "heldout"):
            for metric in ("bitstring_tvd", "shell_tvd"):
                values = defaultdict(list)
                for row in rows:
                    if row["method"] == method and row["role"] == role:
                        values[row["seed"]].append(row[metric])
                stats[(role, metric)] = (
                    _median_iqr([float(np.mean(v)) for v in values.values()])
                    if values
                    else (float("nan"), float("nan"), float("nan"))
                )
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{stats[('train', 'bitstring_tvd')][0]:.4f} "
            f"[{stats[('train', 'bitstring_tvd')][1]:.4f}, "
            f"{stats[('train', 'bitstring_tvd')][2]:.4f}] | "
            f"{stats[('train', 'shell_tvd')][0]:.4f} "
            f"[{stats[('train', 'shell_tvd')][1]:.4f}, "
            f"{stats[('train', 'shell_tvd')][2]:.4f}] | "
            f"{stats[('heldout', 'bitstring_tvd')][0]:.4f} "
            f"[{stats[('heldout', 'bitstring_tvd')][1]:.4f}, "
            f"{stats[('heldout', 'bitstring_tvd')][2]:.4f}] | "
            f"{stats[('heldout', 'shell_tvd')][0]:.4f} "
            f"[{stats[('heldout', 'shell_tvd')][1]:.4f}, "
            f"{stats[('heldout', 'shell_tvd')][2]:.4f}] |",
        )
    return "\n".join(lines) + "\n"


def _latex_table(rows: list[dict[str, float | int | str]]) -> str:
    lines = []
    for method in ("dense", "no_condition", "separate", "oracle"):
        stats = {}
        for role in ("train", "heldout"):
            for metric in ("bitstring_tvd", "shell_tvd"):
                values = defaultdict(list)
                for row in rows:
                    if row["method"] == method and row["role"] == role:
                        values[row["seed"]].append(row[metric])
                stats[(role, metric)] = (
                    _median_iqr([float(np.mean(v)) for v in values.values()])
                    if values
                    else (float("nan"), float("nan"), float("nan"))
                )
        lines.append(
            f"{METHOD_LABELS[method]} & "
            f"{stats[('train', 'bitstring_tvd')][0]:.3f} "
            f"[{stats[('train', 'bitstring_tvd')][1]:.3f},"
            f"{stats[('train', 'bitstring_tvd')][2]:.3f}] & "
            f"{stats[('train', 'shell_tvd')][0]:.3f} "
            f"[{stats[('train', 'shell_tvd')][1]:.3f},"
            f"{stats[('train', 'shell_tvd')][2]:.3f}] & "
            f"{stats[('heldout', 'bitstring_tvd')][0]:.3f} "
            f"[{stats[('heldout', 'bitstring_tvd')][1]:.3f},"
            f"{stats[('heldout', 'bitstring_tvd')][2]:.3f}] & "
            f"{stats[('heldout', 'shell_tvd')][0]:.3f} "
            f"[{stats[('heldout', 'shell_tvd')][1]:.3f},"
            f"{stats[('heldout', 'shell_tvd')][2]:.3f}] \\\\",
        )
    return "\n".join(lines) + "\n"


def _pattern_mass_markdown(
    dense_runs: list[dict[str, object]],
) -> str:
    targets = hopfield_targets(np.concatenate([TRAIN_BETAS, HELDOUT_BETAS]))
    lines = [
        "# Pattern-mass diagnostic",
        "",
        "Median [interquartile range] of the model mass on "
        r"$\{p_1,-p_1,p_2,-p_2\}$ and of the |model - target| mass error.",
        "",
        "| Beta | Target mass | Model mass | Mass error |",
        "|---|---:|---:|---:|",
    ]
    for target, beta in zip(
        targets,
        np.concatenate([TRAIN_BETAS, HELDOUT_BETAS]),
        strict=True,
    ):
        values = [
            point["model_pattern_mass"]
            for run in dense_runs
            for point in run["points"]
            if np.isclose(point["beta"], beta)
        ]
        errors = [
            point["pattern_mass_error"]
            for run in dense_runs
            for point in run["points"]
            if np.isclose(point["beta"], beta)
        ]
        model_median, model_q1, model_q3 = _median_iqr(values)
        error_median, error_q1, error_q3 = _median_iqr(errors)
        lines.append(
            f"| {beta:.4f} | {target.pattern_mass:.4f} | "
            f"{model_median:.4f} [{model_q1:.4f}, {model_q3:.4f}] | "
            f"{error_median:.4f} [{error_q1:.4f}, {error_q3:.4f}] |",
        )
    return "\n".join(lines) + "\n"


def _model_mass_stats(
    dense_runs: list[dict[str, object]],
    beta: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    masses = [
        point["model_pattern_mass"]
        for run in dense_runs
        for point in run["points"]
        if np.isclose(point["beta"], beta)
    ]
    errors = [
        point["pattern_mass_error"]
        for run in dense_runs
        for point in run["points"]
        if np.isclose(point["beta"], beta)
    ]
    return _median_iqr(masses), _median_iqr(errors)


def _plot_pattern_mass(
    dense_runs: list[dict[str, object]],
    runs_directory: Path,
    output: Path,
) -> None:
    use_tex = apply_journal_style()
    figure, axis = plt.subplots(
        figsize=(SINGLE_COL_WIDTH, 60 * MM), layout="constrained"
    )
    targets = hopfield_targets(np.concatenate([TRAIN_BETAS, HELDOUT_BETAS]))
    all_betas = np.concatenate([TRAIN_BETAS, HELDOUT_BETAS])
    axis.plot(
        all_betas,
        [target.pattern_mass for target in targets],
        "o-",
        color="#222222",
        label="Target",
        markersize=3,
    )
    for beta in HELDOUT_BETAS:
        axis.axvspan(
            float(beta / np.sqrt(2.0)),
            float(beta * np.sqrt(2.0)),
            color="#999999",
            alpha=0.12,
        )
    no_condition_runs = [
        json.loads(path.read_text())
        for path in sorted(runs_directory.glob("hopfield_no_condition*.json"))
    ]
    separate_runs = [
        json.loads(path.read_text())
        for path in sorted((runs_directory / "separate").glob("*.json"))
    ]
    baseline_runs = {}
    for run in no_condition_runs:
        baseline_runs[("no_condition", run["seed"], None)] = run
    for run in separate_runs:
        baseline_runs[("separate", run["seed"], run["points"][0]["beta"])] = run

    def _baseline_masses(key_prefix: str, betas: np.ndarray) -> list[list[float]]:
        masses = [[] for _ in betas]
        for key, run in baseline_runs.items():
            if not key[0] == key_prefix:
                continue
            beta = key[2] if key_prefix == "separate" else None
            for point in run["points"]:
                if beta is not None and not np.isclose(point["beta"], beta):
                    continue
                index = int(np.flatnonzero(np.isclose(betas, point["beta"]))[0])
                masses[index].append(point["model_pattern_mass"])
        return masses

    for method, color, style, betas in (
        ("dense", "#0072B2", "s--", all_betas),
        ("no_condition", "#D55E00", "^:", TRAIN_BETAS),
        ("separate", "#009E73", "D:", TRAIN_BETAS),
    ):
        if method == "dense":
            medians, iqrs = [], []
            for beta in betas:
                (median, q1, q3), _ = _model_mass_stats(dense_runs, float(beta))
                medians.append(median)
                iqrs.append((median - q1, q3 - median))
            yerr = np.asarray(iqrs).T
        else:
            masses = _baseline_masses(method, betas)
            medians = [float(np.median(m)) for m in masses]
            yerr = np.asarray(
                [
                    (
                        float(np.median(m)) - float(np.quantile(m, 0.25)),
                        float(np.quantile(m, 0.75)) - float(np.median(m)),
                    )
                    for m in masses
                ],
            ).T
        axis.errorbar(
            betas,
            medians,
            yerr=yerr,
            fmt=style,
            color=color,
            linewidth=1.2,
            markersize=3,
            capsize=2,
            label=METHOD_LABELS[method],
        )
    axis.set_xscale("log")
    if use_tex:
        axis.set_xlabel(r"$\beta$")
        axis.set_ylabel(r"Pattern State Mass")
        axis.set_title(r"Pattern Mass vs. $\beta$")
    else:
        axis.set_xlabel("beta")
        axis.set_ylabel("Pattern State Mass")
        axis.set_title("Pattern Mass vs. beta")
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, output)


def _plot_tvds(
    rows: list[dict[str, float | int | str]],
    output: Path,
) -> None:
    use_tex = apply_journal_style()
    figure, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_WIDTH, 60 * MM), layout="constrained"
    )
    for axis, metric, title, panel_letter in (
        (axes[0], "bitstring_tvd", "Bitstring TVD", "(a)"),
        (axes[1], "shell_tvd", "Profile TVD", "(b)"),
    ):
        for method, role in (
            ("dense", "train"),
            ("no_condition", "train"),
            ("separate", "train"),
        ):
            medians = []
            lower = []
            upper = []
            for beta in TRAIN_BETAS:
                values = [
                    row[metric]
                    for row in rows
                    if row["method"] == method
                    and row["role"] == role
                    and np.isclose(row["beta"], beta)
                ]
                median, q1, q3 = _median_iqr(values)
                medians.append(median)
                lower.append(median - q1)
                upper.append(q3 - median)
            axis.errorbar(
                TRAIN_BETAS,
                medians,
                yerr=np.asarray([lower, upper]),
                fmt={"dense": "o-", "no_condition": "^--", "separate": "s:"}[
                    method
                ],
                color={
                    "dense": "#0072B2",
                    "no_condition": "#D55E00",
                    "separate": "#009E73",
                }[method],
                linewidth=1.2,
                markersize=3,
                capsize=2,
                label=METHOD_LABELS[method],
            )
        for method, role in (("dense", "heldout"), ("oracle", "heldout")):
            medians = []
            lower = []
            upper = []
            for beta in HELDOUT_BETAS:
                values = [
                    row[metric]
                    for row in rows
                    if row["method"] == method
                    and row["role"] == role
                    and np.isclose(row["beta"], beta)
                ]
                median, q1, q3 = _median_iqr(values)
                medians.append(median)
                lower.append(median - q1)
                upper.append(q3 - median)
            axis.errorbar(
                HELDOUT_BETAS,
                medians,
                yerr=np.asarray([lower, upper]),
                fmt={("dense", "heldout"): "o--", ("oracle", "heldout"): "D:"}[
                    (method, role)
                ],
                color={
                    ("dense", "heldout"): "#0072B2",
                    ("oracle", "heldout"): "#CC79A7",
                }[(method, role)],
                linewidth=1.2,
                markersize=3,
                capsize=2,
                label=(
                    f"{METHOD_LABELS[method]} (held-out)"
                    if method == "dense"
                    else METHOD_LABELS[method]
                ),
            )
        axis.set_xscale("log")
        if use_tex:
            axis.set_title(rf"{panel_letter} {title}")
            axis.set_xlabel(r"$\beta$")
        else:
            axis.set_title(f"{panel_letter} {title}")
            axis.set_xlabel("beta")
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Bitstring TVD")
    axes[1].set_ylabel("Profile TVD")
    save_figure(figure, output)


def main() -> None:
    """Generate Hopfield metrics, audit, table, diagnostics, and figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/hopfield/runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hopfield/summary"),
    )
    args = parser.parse_args()
    dense_runs = _load_dense(args.runs)
    rows = _comparison_rows(dense_runs, args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = _metrics_markdown(rows)
    audit, audit_markdown = _audit(rows)
    (args.output / "metrics.md").write_text(metrics)
    (args.output / "audit.md").write_text(audit_markdown)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output / "hopfield_table.tex").write_text(_latex_table(rows))
    (args.output / "pattern_mass.md").write_text(
        _pattern_mass_markdown(dense_runs),
    )
    _plot_pattern_mass(dense_runs, args.runs, args.output / "pattern_mass.png")
    _plot_tvds(rows, args.output / "tvd_by_beta.png")
    print(metrics)
    print(audit_markdown)


if __name__ == "__main__":
    main()

"""Generate the five deterministic principal figures for the Gaussian paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from journal_style import DOUBLE_COL_WIDTH, apply_journal_style, save_figure

from iqp_cond.artifacts import read_immutable_json
from iqp_cond.gaussian_data import gaussian_target

COLORS = {
    "conditioned": "#0072B2",
    "no_condition": "#D55E00",
    "single_peak": "#009E73",
    "adjacent_lookup": "#CC79A7",
    "barycentric_mixture": "#E69F00",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/gaussian-foundations-v1/report/complete-report.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("paper/gaussian_figures"))
    return parser


def _load_report(path: Path) -> dict[str, object]:
    payload = read_immutable_json(path)
    if payload.get("completed_atomic_models") != 1_020:
        raise ValueError("principal figures require all 1,020 completed models")
    return payload


def _interval(values: list[float], seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, array.size, size=(10_000, array.size))].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _find_cell(
    cells: list[dict[str, object]],
    *,
    n_qubits: int,
    width_factor: float,
    peak_count: int | None = None,
    regime: str | None = None,
) -> dict[str, object]:
    matches = [
        cell
        for cell in cells
        if cell["n_qubits"] == n_qubits
        and np.isclose(cell["width_factor"], width_factor)
        and (peak_count is None or cell["peak_count"] == peak_count)
        and (regime is None or cell["data_regime"] == regime)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one report cell, found {len(matches)}")
    return matches[0]


def _condition_summary(
    cell: dict[str, object], role: str, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed_rows = cell["seed_rows"]
    conditions = np.asarray(
        [row["condition"] for row in seed_rows[0][role]], dtype=np.float64
    )
    estimates, lowers, uppers = [], [], []
    for index in range(conditions.size):
        values = [
            float(seed_row[role][index][method]["profile_tvd"])
            for seed_row in seed_rows
        ]
        estimate, lower, upper = _interval(values, 20_260_811 + index)
        estimates.append(estimate)
        lowers.append(lower)
        uppers.append(upper)
    return conditions, np.asarray(estimates), np.asarray(lowers), np.asarray(uppers)


def figure_pipeline(output: Path) -> None:
    # Panel (a) is typeset natively with Quantikz in paper/main.tex.  This
    # function generates only the data-driven panel embedded beside it.
    fig, ax = plt.subplots(figsize=(0.50 * DOUBLE_COL_WIDTH, 2.35))
    for condition in np.linspace(0.0, 1.0, 5):
        target = gaussian_target(19, 0.5, float(condition))
        line = ax.plot(
            np.arange(20),
            target.shell_probabilities,
            marker="o",
            lw=1.8,
            ms=4.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )[0]
        peak = int(np.argmax(target.shell_probabilities))
        ax.text(
            peak,
            target.shell_probabilities[peak] + 0.006,
            rf"$c = {condition:g}$",
            color=line.get_color(),
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )
    ax.set(
        xlabel=r"Hamming weight $k=|x|$",
        ylabel="shell probability",
        xlim=(-0.4, 19.4),
        ylim=(-0.004, 0.222),
    )
    ax.set_xticks([0, 5, 10, 15, 19])
    ax.grid(axis="y", color="#D8DDE2", lw=0.55, alpha=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(which="minor", left=False, bottom=False)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.20, top=0.94)
    save_figure(fig, output / "fig1_pipeline_targets")


def figure_headline(report: dict[str, object], output: Path) -> None:
    cell = _find_cell(report["factorial_cells"], n_qubits=19, width_factor=0.5)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.65), sharey=True)
    panels = [
        (
            "training",
            ["conditioned", "no_condition", "single_peak"],
            "training conditions",
        ),
        (
            "held_out",
            ["conditioned", "no_condition", "adjacent_lookup", "barycentric_mixture"],
            "held-out interpolation conditions",
        ),
    ]
    labels = {
        "conditioned": "conditioned",
        "no_condition": "no condition",
        "single_peak": "independent reference",
        "adjacent_lookup": "neighbor oracle",
        "barycentric_mixture": "barycentric mixture",
    }
    for panel_index, (role, methods, title) in enumerate(panels):
        ax = axes[panel_index]
        for method in methods:
            x, mean, lower, upper = _condition_summary(cell, role, method)
            ax.plot(x, mean, marker="o", color=COLORS[method], label=labels[method])
            ax.fill_between(x, lower, upper, color=COLORS[method], alpha=0.16)
        ax.set(xlabel="condition $c$", title=title, xlim=(-0.02, 1.02))
        ax.text(0.0, 1.02, "ab"[panel_index], transform=ax.transAxes, fontweight="bold")
    axes[0].set_ylabel("profile TVD")
    axes[0].legend(loc="upper center")
    axes[1].legend(loc="upper center", ncol=2)
    fig.tight_layout()
    save_figure(fig, output / "fig2_headline_profiles")


def figure_surface(report: dict[str, object], output: Path) -> None:
    cells = report["factorial_cells"]
    sizes = [8, 12, 16, 19]
    widths = [0.35, 0.5, 0.75]
    interval_names = [
        "training_no_condition_minus_conditioned_profile_tvd",
        "heldout_lookup_minus_conditioned_profile_tvd",
        "heldout_mixture_minus_conditioned_profile_tvd",
    ]
    titles = [
        "training\nno condition $-$ conditioned",
        "held-out interpolation\nneighbor oracle $-$ conditioned",
        "held-out interpolation\nmixture $-$ conditioned",
    ]
    matrices = []
    for name in interval_names:
        matrices.append(
            np.asarray(
                [
                    [
                        _find_cell(cells, n_qubits=n, width_factor=width)[
                            "paired_intervals"
                        ][name]["estimate"]
                        for width in widths
                    ]
                    for n in sizes
                ],
                dtype=np.float64,
            )
        )
    limit = max(abs(float(matrix.min())) for matrix in matrices)
    limit = max(limit, max(abs(float(matrix.max())) for matrix in matrices))
    fig, axes = plt.subplots(
        1, 3, figsize=(DOUBLE_COL_WIDTH, 2.55), constrained_layout=True
    )
    for index, (ax, matrix, title) in enumerate(
        zip(axes, matrices, titles, strict=True)
    ):
        image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        ax.set_xticks(range(3), ["0.35", "0.50", "0.75"])
        ax.set_yticks(range(4), sizes)
        ax.set(xlabel=r"width factor $s/\sqrt{n}$")
        ax.set_title(title, fontsize=9, pad=5)
        if index == 0:
            ax.set_ylabel("qubits $n$")
        ax.text(-0.12, 1.04, "abc"[index], transform=ax.transAxes, fontweight="bold")
    fig.colorbar(image, ax=axes, shrink=0.78, label="paired profile-TVD effect")
    save_figure(fig, output / "fig3_size_width_surface")


def figure_condition_count(report: dict[str, object], output: Path) -> None:
    cells = report["condition_count_cells"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.65))
    styles = {"fixed_per_peak": ("#0072B2", "o"), "fixed_total": ("#D55E00", "s")}
    for regime, (color, marker) in styles.items():
        regime_label = (
            "fixed per condition" if regime == "fixed_per_peak" else "fixed total"
        )
        means, worst = [], []
        for peak_count in [2, 3, 5, 9]:
            cell = _find_cell(
                cells,
                n_qubits=19,
                width_factor=0.5,
                peak_count=peak_count,
                regime=regime,
            )
            per_seed = [
                np.asarray(
                    [row["conditioned"]["profile_tvd"] for row in seed["held_out"]]
                )
                for seed in cell["seed_rows"]
            ]
            means.append(float(np.mean([values.mean() for values in per_seed])))
            worst.append(
                max(
                    float(np.mean([values[index] for values in per_seed]))
                    for index in range(per_seed[0].size)
                )
            )
        axes[0].plot(
            [2, 3, 5, 9],
            means,
            color=color,
            marker=marker,
            label=regime_label,
        )
        axes[0].plot([2, 3, 5, 9], worst, color=color, marker=marker, linestyle="--")
    axes[0].set(
        xlabel="training conditions $K$",
        ylabel="held-out profile TVD",
        xticks=[2, 3, 5, 9],
    )
    axes[0].legend(title="solid: mean; dashed: worst")
    parameters = 190
    axes[1].plot([2, 3, 5, 9], [parameters] * 4, marker="o", label="shared parameters")
    axes[1].plot([2, 3, 5, 9], [2_000] * 4, marker="s", label="shared updates")
    axes[1].plot(
        [2, 3, 5, 9],
        [2_000 * k for k in [2, 3, 5, 9]],
        marker="^",
        label="reference updates",
    )
    axes[1].set(
        xlabel="training conditions $K$",
        ylabel="count (log scale)",
        xticks=[2, 3, 5, 9],
        yscale="log",
    )
    axes[1].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=8,
        frameon=True,
    )
    for index, ax in enumerate(axes):
        ax.text(0.0, 1.02, "ab"[index], transform=ax.transAxes, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output / "fig4_condition_count")


def _topology_values(path: Path) -> dict[str, float]:
    text = path.read_text()
    result = {}
    for label, topology in (
        ("complete", "complete"),
        ("heavy-hex", "heavy-hex"),
        ("lattice", "lattice"),
    ):
        pattern = (
            rf"\| Angle / {re.escape(topology)} \| [AB] \| \d+ \| "
            r"[^|]+ \| ([0-9.]+)"
        )
        values = [float(value) for value in re.findall(pattern, text)]
        if len(values) != 2:
            raise ValueError(f"could not recover both {topology} profile-TVD values")
        result[label] = float(np.mean(values))
    return result


def _ancilla_values(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ancilla_rows = []
    control_rows = []
    for line in path.read_text().splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 4 or not re.fullmatch(r"\d+(?:\$\^\{\*\}\$)?", cells[0]):
            continue
        try:
            parameters = float(cells[2])
            profile_tvd = float(cells[3].split()[0])
        except ValueError:
            continue
        if "$" in cells[0]:
            control_rows.append((parameters, profile_tvd))
        else:
            ancilla_rows.append((float(cells[0]), parameters, profile_tvd))
    if len(ancilla_rows) != 6 or len(control_rows) != 5:
        raise ValueError("ancilla summary is incomplete")
    ancilla_array = np.asarray(ancilla_rows)
    control_array = np.asarray(control_rows)
    return (
        ancilla_array[:, 0],
        ancilla_array[:, 1],
        ancilla_array[:, 2],
        np.asarray([1, 2, 3, 5, 7], dtype=np.float64),
        control_array[:, 1],
    )


def _verify_legacy_evidence() -> None:
    evidence = read_immutable_json(
        Path("results/gaussian-foundations-v1/evidence/legacy-regeneration.json")
    )
    if evidence.get("exact_agreement") is not True:
        raise RuntimeError("archived Gaussian summaries lack exact agreement")
    for name, expected in evidence["verified_summary_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"archived Gaussian summary changed: {name}")


def figure_ablations(output: Path) -> None:
    _verify_legacy_evidence()
    topology = _topology_values(Path("results/summary/metrics.md"))
    encoding = json.loads(Path("results/encoding/summary/metrics.json").read_text())
    ancillas, parameters, ancilla_tvd, control_rungs, control_tvd = _ancilla_values(
        Path("results/ancilla/summary/metrics.md")
    )
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 2.65))
    axes[0].bar(
        topology.keys(), topology.values(), color=["#0072B2", "#E69F00", "#009E73"]
    )
    axes[0].set(ylabel="profile TVD", title="topology")
    encoding_values = [
        encoding["aggregate"][name]["shell_tvd"]["mean"]
        for name in ("angle", "symmetric_reupload")
    ]
    axes[1].bar(
        ["broadcast", "reupload"], encoding_values, color=["#0072B2", "#CC79A7"]
    )
    axes[1].set(title="condition encoding")
    axes[2].plot(
        ancillas,
        ancilla_tvd,
        marker="o",
        color="#0072B2",
        label="ancilla marginal",
    )
    axes[2].plot(
        control_rungs,
        control_tvd,
        marker="s",
        color="#D55E00",
        label="matched capacity",
    )
    axes[2].set(
        xlabel="ancillas $m$", ylabel="held-out profile TVD", title="ancilla scaling"
    )
    for x, y, count in zip(ancillas, ancilla_tvd, parameters, strict=True):
        axes[2].annotate(
            f"{int(count)}",
            (x, y),
            xytext=(8 if x == 0 else 0, -16 if x == 0 else 5),
            textcoords="offset points",
            ha="left" if x == 0 else "center",
            fontsize=7,
        )
    axes[2].legend(fontsize=7)
    for index, ax in enumerate(axes):
        ax.set_title(ax.get_title(), fontsize=10, pad=8)
        ax.text(
            -0.12,
            1.04,
            "abc"[index],
            transform=ax.transAxes,
            fontweight="bold",
        )
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_figure(fig, output / "fig5_ablations")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    apply_journal_style()
    report = _load_report(args.report)
    figure_pipeline(args.output)
    figure_headline(report, args.output)
    figure_surface(report, args.output)
    figure_condition_count(report, args.output)
    figure_ablations(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

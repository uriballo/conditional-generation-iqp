#!/usr/bin/env python3
"""Plot the seed-level metric audit for the strict phase-OOD encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.journal_style import (
        DOUBLE_COL_WIDTH,
        MM,
        OKABE_ITO_COLORS,
        apply_journal_style,
        save_figure,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from journal_style import (  # type: ignore[no-redef]
        DOUBLE_COL_WIDTH,
        MM,
        OKABE_ITO_COLORS,
        apply_journal_style,
        save_figure,
    )

ROOT = Path(__file__).parents[1]
DEFAULT_RESULTS = ROOT / "results" / "curie-weiss-phase-ood-prototype-v1"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "curie_weiss_phase_ood"
    / "phase_ood_metric_audit"
)
DEFAULT_GENERALIZATION_OUTPUT = (
    ROOT
    / "artifacts"
    / "curie_weiss_phase_ood"
    / "phase_ood_generalization"
)

METRICS = (
    "profile_tvd",
    "raw_tvd",
    "absolute_magnetization_error",
)
TEST_BETAS = (0.9, 1.0, 1.1)
MODEL_SPECS = (
    ("Affine", "critical-gap-r3.json", "affine_broadcast"),
    (
        r"$r=1$",
        "latent-input-linear-r1-critical-gap-10seed.json",
        "latent_input_linear",
    ),
    (
        r"$r=3$",
        "latent-input-linear-r3-critical-gap-10seed.json",
        "latent_input_linear",
    ),
    (
        r"$r=5$",
        "latent-input-linear-r5-critical-gap-10seed.json",
        "latent_input_linear",
    ),
    (
        "Visible-76",
        "visible-capacity-r5-critical-gap-10seed.json",
        "visible_capacity",
    ),
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def load_condition_records(
    results: Path = DEFAULT_RESULTS,
) -> list[dict[str, object]]:
    """Return one metric record per model, seed, and evaluated condition."""
    records: list[dict[str, object]] = []
    for model, filename, encoding in MODEL_SPECS:
        payload = _read(results / filename)
        selected = [
            row
            for row in payload["rows"]
            if row["encoding"] == encoding
        ]
        seeds = sorted({int(row["seed"]) for row in selected})
        if seeds != list(range(10)):
            raise ValueError(f"{model} must contain seeds 0--9, found {seeds}")
        for seed in seeds:
            rows = [row for row in selected if int(row["seed"]) == seed]
            if len(rows) != 9:
                raise ValueError(f"{model} seed {seed} must contain nine conditions")
            for row in rows:
                records.append(
                    {
                        "model": model,
                        "seed": seed,
                        "beta": float(row["beta"]),
                        "role": str(row["role"]),
                        **{metric: float(row[metric]) for metric in METRICS},
                    }
                )
    return records


def load_seed_records(results: Path = DEFAULT_RESULTS) -> list[dict[str, object]]:
    """Return one held-out metric record per model and optimizer seed."""
    condition_records = load_condition_records(results)
    records: list[dict[str, object]] = []
    for model, _, _ in MODEL_SPECS:
        for seed in range(10):
            rows = [
                row
                for row in condition_records
                if row["model"] == model
                and int(row["seed"]) == seed
                and row["role"] == "test"
            ]
            betas = tuple(sorted(float(row["beta"]) for row in rows))
            if betas != TEST_BETAS:
                raise ValueError(f"{model} seed {seed} has test betas {betas}")
            records.append(
                {
                    "model": model,
                    "seed": seed,
                    **{
                        metric: float(np.mean([float(row[metric]) for row in rows]))
                        for metric in METRICS
                    },
                }
            )
    return records


def metric_summary(
    records: list[dict[str, object]], metric: str
) -> dict[str, tuple[float, float, float]]:
    """Return mean, minimum, and maximum over seed-level held-out means."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric: {metric}")
    summary = {}
    for model, _, _ in MODEL_SPECS:
        values = np.asarray(
            [float(row[metric]) for row in records if row["model"] == model]
        )
        summary[model] = (
            float(values.mean()),
            float(values.min()),
            float(values.max()),
        )
    return summary


def _model_color(index: int) -> str:
    if index == 0:
        return OKABE_ITO_COLORS[1]
    if index == len(MODEL_SPECS) - 1:
        return OKABE_ITO_COLORS[4]
    return OKABE_ITO_COLORS[0]


def _metric_panel(
    ax: plt.Axes,
    records: list[dict[str, object]],
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    x_values = np.arange(len(MODEL_SPECS), dtype=float)
    seed_offsets = np.linspace(-0.13, 0.13, 10)
    summary = metric_summary(records, metric)
    means = []
    for index, (model, _, _) in enumerate(MODEL_SPECS):
        values = np.asarray(
            [float(row[metric]) for row in records if row["model"] == model]
        )
        mean, lower, upper = summary[model]
        means.append(mean)
        color = _model_color(index)
        ax.scatter(
            x_values[index] + seed_offsets,
            values,
            s=11,
            color=color,
            alpha=0.42,
            edgecolors="none",
            zorder=2,
        )
        marker = "D" if model == "Visible-76" else "o"
        ax.errorbar(
            x_values[index],
            mean,
            yerr=np.asarray([[mean - lower], [upper - mean]]),
            fmt=marker,
            markersize=5.2,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.2,
            capsize=2.5,
            zorder=4,
        )
        label_y = min(0.414, upper + 0.018)
        ax.text(
            x_values[index],
            label_y,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=color,
        )
    ax.plot(x_values[:4], means[:4], color=OKABE_ITO_COLORS[0], linewidth=1.2, zorder=1)
    ax.axvline(3.5, color="0.65", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_xticks(x_values, ("Affine", "1", "3", "5", "Visible\n76 par."))
    ax.set_xlabel("Ancilla count $r$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 0.43)
    ax.set_yticks(np.arange(0.0, 0.41, 0.1))
    ax.set_title(title, loc="left", fontsize=9.5, pad=5)
    ax.grid(axis="y", linestyle=":", color="0.82", linewidth=0.7)
    ax.text(
        4.0,
        0.012,
        "matched control",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="0.38",
    )


def _moment_scatter(ax: plt.Axes, records: list[dict[str, object]]) -> None:
    label_offsets = {
        "Affine": (3, 3),
        r"$r=1$": (3, -10),
        r"$r=3$": (3, -10),
        r"$r=5$": (-4, -13),
        "Visible-76": (3, 3),
    }
    for index, (model, _, _) in enumerate(MODEL_SPECS):
        selected = [row for row in records if row["model"] == model]
        x = np.asarray(
            [float(row["absolute_magnetization_error"]) for row in selected]
        )
        y = np.asarray([float(row["raw_tvd"]) for row in selected])
        color = _model_color(index)
        marker = "D" if model == "Visible-76" else "o"
        ax.scatter(
            x,
            y,
            s=13,
            marker=marker,
            color=color,
            alpha=0.38,
            edgecolors="none",
            zorder=2,
        )
        mean_x, mean_y = float(x.mean()), float(y.mean())
        ax.scatter(
            [mean_x],
            [mean_y],
            s=35,
            marker=marker,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )
        offset = label_offsets[model]
        ax.annotate(
            model,
            (mean_x, mean_y),
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            va="bottom" if offset[1] >= 0 else "top",
            fontsize=7.2,
            color=color,
        )
    ax.set_xlim(0.0, 0.1)
    ax.set_ylim(0.0, 0.43)
    ax.set_xticks(np.arange(0.0, 0.101, 0.02))
    ax.set_yticks(np.arange(0.0, 0.41, 0.1))
    ax.set_xlabel(r"Absolute error in $\mathbb{E}|m|$")
    ax.set_ylabel(r"Raw TVD $\downarrow$")
    ax.set_title(
        r"$\mathbf{c}$  One moment can mislead",
        loc="left",
        fontsize=9.5,
        pad=5,
    )
    ax.grid(linestyle=":", color="0.82", linewidth=0.7)


def build_figure(
    records: list[dict[str, object]], output: Path = DEFAULT_OUTPUT
) -> None:
    """Write the vector and raster metric-audit figure."""
    apply_journal_style()
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COL_WIDTH, 60 * MM),
        layout="constrained",
    )
    _metric_panel(
        axes[0],
        records,
        "profile_tvd",
        r"Profile TVD $\downarrow$",
        r"$\mathbf{a}$  Magnetization profile",
    )
    _metric_panel(
        axes[1],
        records,
        "raw_tvd",
        r"Raw TVD $\downarrow$",
        r"$\mathbf{b}$  Full distribution",
    )
    _moment_scatter(axes[2], records)
    save_figure(fig, output)


def _generalization_panel(
    ax: plt.Axes,
    records: list[dict[str, object]],
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    plotted_models = ("Affine", r"$r=5$", "Visible-76")
    styles = {
        "Affine": ("--", "o"),
        r"$r=5$": ("-", "s"),
        "Visible-76": (":", "D"),
    }
    betas = np.asarray(
        sorted({float(row["beta"]) for row in records}), dtype=float
    )
    largest_upper = 0.0
    for model in plotted_models:
        index = [spec[0] for spec in MODEL_SPECS].index(model)
        color = _model_color(index)
        means = []
        errors = []
        for beta in betas:
            values = np.asarray(
                [
                    float(row[metric])
                    for row in records
                    if row["model"] == model and float(row["beta"]) == beta
                ]
            )
            means.append(float(values.mean()))
            errors.append(float(values.std(ddof=1) / np.sqrt(values.size)))
        means_array = np.asarray(means)
        errors_array = np.asarray(errors)
        largest_upper = max(largest_upper, float(np.max(means_array + errors_array)))
        line_style, marker = styles[model]
        ax.fill_between(
            betas,
            np.maximum(0.0, means_array - errors_array),
            means_array + errors_array,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            betas,
            means_array,
            color=color,
            linestyle=line_style,
            linewidth=1.4,
            label={
                "Affine": "Affine (36)",
                r"$r=5$": r"Latent $r=5$ (76)",
                "Visible-76": "Visible-only (76)",
            }[model],
        )
        training = np.asarray(
            [
                any(
                    row["model"] == model
                    and float(row["beta"]) == beta
                    and row["role"] == "training"
                    for row in records
                )
                for beta in betas
            ]
        )
        ax.scatter(
            betas[training],
            means_array[training],
            marker=marker,
            s=22,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )
        ax.scatter(
            betas[~training],
            means_array[~training],
            marker=marker,
            s=25,
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            zorder=5,
        )
    ax.axvspan(0.85, 1.15, color="0.5", alpha=0.08, linewidth=0, zorder=0)
    ax.axvline(0.85, color="0.65", linestyle=":", linewidth=0.7, zorder=0)
    ax.axvline(1.15, color="0.65", linestyle=":", linewidth=0.7, zorder=0)
    ax.set_xlim(0.48, 1.52)
    ax.set_xticks((0.5, 0.7, 0.9, 1.1, 1.3, 1.5))
    ax.set_xlabel(r"Inverse temperature $\beta$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9.5, pad=5)
    ax.grid(axis="y", linestyle=":", color="0.82", linewidth=0.7)
    ax.set_ylim(0.0, largest_upper * 1.08)
    ax.locator_params(axis="y", nbins=5)


def build_generalization_figure(
    records: list[dict[str, object]],
    output: Path = DEFAULT_GENERALIZATION_OUTPUT,
) -> None:
    """Write the held-out critical-gap generalization figure."""
    apply_journal_style()
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COL_WIDTH, 58 * MM),
        layout="constrained",
    )
    _generalization_panel(
        axes[0],
        records,
        "profile_tvd",
        r"Profile TVD $\downarrow$",
        r"$\mathbf{a}$  Magnetization profile",
    )
    _generalization_panel(
        axes[1],
        records,
        "raw_tvd",
        r"Raw TVD $\downarrow$",
        r"$\mathbf{b}$  Full distribution",
    )
    _generalization_panel(
        axes[2],
        records,
        "absolute_magnetization_error",
        r"Absolute error in $\mathbb{E}|m|$",
        r"$\mathbf{c}$  Magnetization response",
    )
    axes[0].legend(loc="upper left", frameon=False, handlelength=2.4)
    axes[1].text(
        0.5,
        0.96,
        "held-out gap",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color="0.35",
    )
    axes[2].text(
        0.5,
        0.96,
        "hollow markers: held out",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        color="0.35",
    )
    save_figure(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--generalization-output",
        type=Path,
        default=DEFAULT_GENERALIZATION_OUTPUT,
    )
    args = parser.parse_args()
    build_figure(load_seed_records(args.results), args.output)
    build_generalization_figure(
        load_condition_records(args.results), args.generalization_output
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plot the currently available MFC pseudorapidity confirmatory results."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from iqp_cond.artifacts import read_immutable_json

try:
    from scripts.journal_style import (
        DOUBLE_COL_WIDTH,
        OKABE_ITO_COLORS,
        apply_journal_style,
        save_figure,
    )
except ModuleNotFoundError:
    from journal_style import (  # type: ignore[no-redef]
        DOUBLE_COL_WIDTH,
        OKABE_ITO_COLORS,
        apply_journal_style,
        save_figure,
    )

ENERGIES = (75, 100, 125, 150, 175)
RESULT_DIR = Path("results/mfc-pseudorapidity-confirmatory-v1/runs")
OUTPUT = Path("paper/mfc_confirmatory_figures/fig_mfc_pseudorapidity_interim")


def load_runs(result_dir: Path = RESULT_DIR) -> list[Mapping[str, Any]]:
    """Load a stable path snapshot, retaining only the requested IQP cell."""
    paths = tuple(sorted(Path(result_dir).glob("*.json")))
    out = []
    for path in paths:
        record = read_immutable_json(path)
        payload = record.get("payload", record)
        run = payload.get("run", {})
        if (
            run.get("campaign_version") != "mfc_pseudorapidity_confirmatory_v1"
            or run.get("stage") != "conditional"
            or run.get("bin_count") != 16
            or run.get("coding") != "binary"
            or run.get("held_out_energy_gev") not in ENERGIES
            or run.get("family") not in {"conditioned_iqp", "no_condition_iqp"}
        ):
            continue
        for key in (
            "target_physical_probabilities",
            "prediction_physical_probabilities",
        ):
            values = np.asarray(payload.get(key), dtype=float)
            if (
                values.shape != (16,)
                or not np.all(np.isfinite(values))
                or np.any(values < 0)
                or not np.isclose(values.sum(), 1.0, atol=1e-8)
            ):
                raise ValueError(f"malformed probability vector in {path}: {key}")
        out.append(payload)
    return out


def summarize(runs: list[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    seeds: dict[tuple[int, str], set[int]] = defaultdict(set)
    targets: dict[int, np.ndarray] = {}
    for payload in runs:
        run = payload["run"]
        energy = int(run["held_out_energy_gev"])
        family = str(run["family"])
        seed = int(run["seed"])
        key = (energy, family)
        if seed in seeds[key]:
            raise ValueError(f"duplicate {family} seed {seed} at {energy} GeV")
        seeds[key].add(seed)
        target = np.asarray(payload["target_physical_probabilities"], float)
        if energy in targets and not np.allclose(targets[energy], target, atol=1e-12):
            raise ValueError(f"inconsistent target at {energy} GeV")
        targets[energy] = target
        grouped[key].append(
            np.asarray(payload["prediction_physical_probabilities"], float)
        )
    result = {e: {"target": targets[e], "models": {}} for e in targets}
    for (e, family), values in grouped.items():
        arr = np.stack(values)
        result[e]["models"][family] = (
            len(values),
            np.quantile(arr, 0.25, axis=0),
            np.median(arr, axis=0),
            np.quantile(arr, 0.75, axis=0),
        )
    return result


def build_figure(summary: Mapping[int, Mapping[str, Any]]) -> plt.Figure:
    apply_journal_style()
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(DOUBLE_COL_WIDTH, 3.0),
        sharey=True,
        layout="constrained",
    )
    blue, vermillion = OKABE_ITO_COLORS[:2]
    x = np.arange(16)
    plotted_values = [np.asarray(cell["target"]) for cell in summary.values()]
    plotted_values.extend(
        np.asarray(statistic)
        for cell in summary.values()
        for model in cell["models"].values()
        for statistic in model[1:]
    )
    observed_maximum = max(
        (float(np.max(values)) for values in plotted_values), default=0.16
    )
    y_maximum = max(0.16, np.ceil(observed_maximum * 1.05 / 0.02) * 0.02)
    for ax, energy in zip(axes, ENERGIES, strict=True):
        ax.set_title(f"{energy} GeV", fontsize=10)
        cell = summary.get(energy)
        if cell is None:
            ax.text(
                0.5,
                0.5,
                "pending",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.35",
                style="italic",
            )
        else:
            ax.step(x, cell["target"], where="mid", color="black", ls="--", lw=1.1)
            count_lines: list[tuple[str, str]] = []
            for family, color, short_label in (
                ("conditioned_iqp", blue, "conditioned"),
                ("no_condition_iqp", vermillion, "no condition"),
            ):
                if family not in cell["models"]:
                    continue
                n, lo, med, hi = cell["models"][family]
                ax.fill_between(x, lo, hi, step="mid", color=color, alpha=0.18)
                ax.step(x, med, where="mid", color=color, lw=1.5)
                count_lines.append((f"{short_label}: {n}/10", color))
            for row, (label, color) in enumerate(count_lines):
                ax.text(
                    0.96,
                    0.96 - 0.065 * row,
                    label,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    color=color,
                )
        ax.set_xticks([0, 5, 10, 15])
        ax.set_xlabel(r"$\eta$ quantile bin")
        ax.set_ylim(0, y_maximum)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("probability mass")
    handles = [
        Line2D([], [], color="black", ls="--", label="target"),
        Patch(facecolor=blue, alpha=0.18, label="conditioned IQP, median and IQR"),
        Patch(
            facecolor=vermillion,
            alpha=0.18,
            label="no-condition IQP, median and IQR",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.07),
        frameon=False,
        fontsize=8,
    )
    axes[-1].text(
        0.96,
        0.04,
        "INTERIM RESULTS",
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.35",
        style="italic",
        transform=axes[-1].transAxes,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    save_figure(build_figure(summarize(load_runs(args.result_dir))), args.output)


if __name__ == "__main__":
    main()

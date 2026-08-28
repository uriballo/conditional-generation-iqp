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
        MM,
        OKABE_ITO_COLORS,
        apply_journal_style,
        save_figure,
    )
except ModuleNotFoundError:
    from journal_style import (  # type: ignore[no-redef]
        MM,
        OKABE_ITO_COLORS,
        apply_journal_style,
        save_figure,
    )

ENERGIES = (75, 100, 125, 150, 175)
RESULT_DIR = Path("results/mfc-pseudorapidity-confirmatory-v1/runs")
OUTPUT = Path("paper/mfc_confirmatory_figures/fig_mfc_pseudorapidity_interim")


def load_runs(result_dir: Path = RESULT_DIR) -> list[Mapping[str, Any]]:
    """Load a stable snapshot of the promoted conditional cell and baselines."""
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
            or run.get("family")
            not in {
                "conditioned_iqp",
                "no_condition_iqp",
                "separate_iqp",
                "categorical_mlp",
                "histogram_interpolation",
            }
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
    seeds: dict[tuple[int, str], set[int | None]] = defaultdict(set)
    targets: dict[int, np.ndarray] = {}
    for payload in runs:
        run = payload["run"]
        energy = int(run["held_out_energy_gev"])
        family = str(run["family"])
        seed = None if run.get("seed") is None else int(run["seed"])
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
    result = {e: {"target": targets[e], "models": {}, "tvd": {}} for e in targets}
    for (e, family), values in grouped.items():
        arr = np.stack(values)
        result[e]["models"][family] = (
            len(values),
            np.quantile(arr, 0.25, axis=0),
            np.median(arr, axis=0),
            np.quantile(arr, 0.75, axis=0),
        )
        tvds = np.asarray([0.5 * np.abs(value - targets[e]).sum() for value in values])
        result[e]["tvd"][family] = (
            len(values),
            float(np.quantile(tvds, 0.25)),
            float(np.median(tvds)),
            float(np.quantile(tvds, 0.75)),
        )
    return result


def build_figure(summary: Mapping[int, Mapping[str, Any]]) -> plt.Figure:
    apply_journal_style()
    fig = plt.figure(figsize=(170 * MM, 112 * MM), constrained_layout=True)
    grid = fig.add_gridspec(5, 2, width_ratios=(1.5, 1), wspace=0.28, hspace=0.10)
    axes = [fig.add_subplot(grid[0, 0])]
    axes.extend(
        fig.add_subplot(grid[i, 0], sharex=axes[0], sharey=axes[0]) for i in range(1, 5)
    )
    response_ax = fig.add_subplot(grid[:, 1])
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
        ax.set_title(rf"$E_{{in}}={energy}$ GeV", loc="left", fontsize=8, pad=2)
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
            # Connect values at the bin centres with straight segments.  This
            # keeps the discrete 16-bin result intact while avoiding the
            # blocky staircase rendering used for histogram edges.
            ax.fill_between(x, 0, cell["target"], color="0.82", alpha=0.65)
            ax.plot(x, cell["target"], color="0.15", lw=1.0, ls="--")
            for family, line_color in (
                ("conditioned_iqp", OKABE_ITO_COLORS[0]),
                ("no_condition_iqp", "0.45"),
            ):
                if family not in cell["models"]:
                    continue
                _n, lo, med, hi = cell["models"][family]
                if family == "conditioned_iqp":
                    ax.fill_between(
                        x,
                        lo,
                        hi,
                        color=OKABE_ITO_COLORS[0],
                        alpha=0.14,
                        linewidth=0,
                    )
                ax.plot(
                    x,
                    med,
                    color=line_color,
                    lw=1.1,
                    ls="--" if family == "no_condition_iqp" else "-",
                    marker="o" if family == "conditioned_iqp" else None,
                    markersize=2.5 if family == "conditioned_iqp" else None,
                )
            # ``n`` remains part of the summary for programmatic coverage checks;
            # it is intentionally not printed in the manuscript-style figure.
        ax.set_xticks([0, 5, 10, 15])
        if energy == ENERGIES[-1]:
            ax.set_xlabel(r"$\eta$ quantile bin")
        else:
            ax.tick_params(labelbottom=False)
        ax.set_ylim(0, y_maximum)
        ax.grid(axis="y", alpha=0.2)
    axes[2].set_ylabel("probability")
    response_ax.set_xlabel(r"held-out energy $E_{in}$ (GeV)")
    response_ax.set_ylabel("profile TVD")
    response_ax.set_title("held-out interpolation", fontsize=10, pad=8)
    response_ax.grid(axis="y", alpha=0.2)
    family_styles = {
        "conditioned_iqp": (OKABE_ITO_COLORS[0], "-", "conditioned IQP"),
        "no_condition_iqp": ("0.45", "--", "no condition"),
        "separate_iqp": (OKABE_ITO_COLORS[2], "-", "separate IQP"),
        "categorical_mlp": (OKABE_ITO_COLORS[1], "-", "categorical MLP"),
        "histogram_interpolation": (
            OKABE_ITO_COLORS[3],
            "-",
            "histogram interpolation",
        ),
    }
    for family, (color, style, label) in family_styles.items():
        xs, meds, los, his = [], [], [], []
        for energy in ENERGIES:
            if family in summary.get(energy, {}).get("tvd", {}):
                _n, lo, med, hi = summary[energy]["tvd"][family]
                xs.append(energy)
                meds.append(med)
                los.append(lo)
                his.append(hi)
        if xs:
            response_ax.plot(xs, meds, color=color, ls=style, marker="o", label=label)
            if family not in {"histogram_interpolation"} and len(xs) > 1:
                response_ax.fill_between(
                    xs, los, his, color=color, alpha=0.10, linewidth=0
                )
    response_ax.set_xticks(ENERGIES)
    handles = [
        Patch(facecolor="0.82", edgecolor="0.15", alpha=0.65, label="target"),
        Line2D([], [], color=OKABE_ITO_COLORS[0], marker="o", label="conditioned IQP"),
        Line2D([], [], color="0.45", ls="--", label="no condition"),
        Line2D([], [], color=OKABE_ITO_COLORS[2], label="separate IQP"),
        Line2D([], [], color=OKABE_ITO_COLORS[1], label="categorical MLP"),
        Line2D([], [], color=OKABE_ITO_COLORS[3], label="histogram interpolation"),
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        fontsize=7,
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

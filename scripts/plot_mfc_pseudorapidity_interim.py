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
        SINGLE_COL_WIDTH,
        apply_journal_style,
        save_figure,
    )
except ModuleNotFoundError:
    from journal_style import (  # type: ignore[no-redef]
        MM,
        OKABE_ITO_COLORS,
        SINGLE_COL_WIDTH,
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
        5,
        1,
        figsize=(SINGLE_COL_WIDTH, 105 * MM),
        sharey=True,
        sharex=True,
        gridspec_kw={"hspace": 0.12},
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
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
        energy_color = OKABE_ITO_COLORS[ENERGIES.index(energy)]
        ax.text(
            0.04,
            0.92,
            rf"$E_{{in}}={energy}$ GeV",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "none",
                "pad": 1.5,
            },
        )
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
            ax.fill_between(
                x, 0, cell["target"], step="mid", color=energy_color, alpha=0.28
            )
            ax.step(x, cell["target"], where="mid", color=energy_color, lw=1.1)
            for family, line_color in (
                ("conditioned_iqp", "0.12"),
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
                        step="mid",
                        color=energy_color,
                        alpha=0.14,
                        linewidth=0,
                    )
                ax.step(
                    x,
                    med,
                    where="mid",
                    color=line_color,
                    lw=1.1,
                    ls="--" if family == "no_condition_iqp" else "-",
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
    fig.supylabel("probability")
    handles = [
        Patch(
            facecolor=OKABE_ITO_COLORS[0],
            edgecolor=OKABE_ITO_COLORS[0],
            alpha=0.28,
            label="target",
        ),
        Line2D([], [], color="0.12", lw=1.1, label="conditioned"),
        Line2D([], [], color="0.45", ls="--", lw=1.1, label="no condition"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.025),
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

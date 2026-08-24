"""Generate deterministic Gaussian and stopped-Ising figures for the draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from journal_style import DOUBLE_COL_WIDTH, MM, apply_journal_style, save_figure


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gaussian",
        type=Path,
        default=Path("results/arxiv-foundations-v1/gaussian/paired-evidence.json"),
    )
    parser.add_argument(
        "--representability",
        type=Path,
        default=Path("results/arxiv-foundations-v1/evidence/representability.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("paper/foundations_figures")
    )
    return parser


def _payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())["payload"]


def _gaussian(payload: dict[str, object], output: Path) -> None:
    apply_journal_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL_WIDTH, 58 * MM),
        layout="constrained",
    )
    control = payload["conditional_control"]["intervals"]["shell_tvd"]
    improvement = np.asarray(
        [float(value) for value in control["paired_seed_values"].values()]
    )
    axes[0].axhline(0.0, color="#666666", linewidth=0.8)
    axes[0].scatter(np.arange(improvement.size), improvement, color="#0072B2", s=18)
    axes[0].errorbar(
        10.7,
        control["estimate"],
        yerr=np.asarray(
            [
                [control["estimate"] - control["lower"]],
                [control["upper"] - control["estimate"]],
            ]
        ),
        fmt="D",
        color="#D55E00",
        capsize=3,
        label="paired mean [95% CI]",
    )
    axes[0].set_title("(a) Training-condition control")
    axes[0].set_xlabel("Optimizer seed")
    axes[0].set_ylabel("Profile-TVD improvement")
    axes[0].legend(frameon=False, fontsize=7)

    intervals = payload["held_out_interpolation"]["paired_intervals"]
    endpoint = intervals["shell_tvd_vs_endpoint"]
    lookup = intervals["shell_tvd_vs_nearest"]
    for x, item, label, color in (
        (0, endpoint, "vs endpoint-only", "#009E73"),
        (1, lookup, "vs adjacent lookup", "#CC79A7"),
    ):
        axes[1].errorbar(
            x,
            item["estimate"],
            yerr=np.asarray(
                [
                    [item["estimate"] - item["lower"]],
                    [item["upper"] - item["estimate"]],
                ]
            ),
            fmt="o",
            color=color,
            capsize=4,
            markersize=5,
            label=label,
        )
    axes[1].axhline(0.0, color="#666666", linewidth=0.8)
    axes[1].set_xticks((0, 1), ("endpoint", "lookup"))
    axes[1].set_title("(b) Aggregate held-out interpolation")
    axes[1].set_ylabel("Baseline minus dense profile TVD")
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, output / "gaussian_evidence.png")


def _representability(payload: dict[str, object], output: Path) -> None:
    apply_journal_style()
    figure, axis = plt.subplots(
        figsize=(DOUBLE_COL_WIDTH, 55 * MM), layout="constrained"
    )
    rows = payload["rows"]
    scopes = ("train_beta_0", "train_beta_2", "train_beta_4")
    colors = {606: "#0072B2", 707: "#D55E00"}
    markers = {0: "o", 1: "s"}
    for row in rows:
        x = scopes.index(row["condition_scope"])
        axis.scatter(
            x + (row["optimizer_seed"] - 0.5) * 0.08,
            row["tvd"],
            color=colors[row["coupling_seed"]],
            marker=markers[row["optimizer_seed"]],
            s=28,
            alpha=0.9,
        )
    axis.axhline(
        payload["thresholds"]["median_tvd_max"],
        color="#009E73",
        linestyle="--",
        label="median threshold 0.25",
    )
    axis.axhline(
        payload["thresholds"]["cell_tvd_max"],
        color="#CC79A7",
        linestyle=":",
        label="cell threshold 0.35",
    )
    axis.set_xticks((0, 1, 2), ("low", "anchor", "high"))
    axis.set_xlabel("Training condition")
    axis.set_ylabel("Exact full-bitstring TVD")
    axis.set_title(
        "Stopped Ising representability screen: "
        f"median {payload['median_tvd']:.3f}, max {payload['maximum_tvd']:.3f}"
    )
    axis.legend(frameon=False, fontsize=7, ncols=2)
    axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, output / "ising_representability.png")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    _gaussian(_payload(args.gaussian), args.output)
    _representability(_payload(args.representability), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

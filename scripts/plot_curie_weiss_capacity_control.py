#!/usr/bin/env python3
"""Generate the matched-capacity figure and manuscript claim macros."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from iqp_cond.artifacts import read_immutable_json

BLUE = "#0072B2"
ORANGE = "#D55E00"
REGIONS = ("training", "held_out", "critical", "cold")


def _format_interval(row: dict[str, Any]) -> str:
    interval = row["interval"]
    return (
        f"{interval['estimate']:+.3f} "
        f"[{interval['lower']:+.3f}, {interval['upper']:+.3f}]"
    )


def _comparison(summary: dict[str, Any], metric: str, region: str) -> dict[str, Any]:
    return next(
        row
        for row in summary["comparisons"]
        if row["metric"] == metric and row["region"] == region
    )


def _write_claims(summary: dict[str, Any], destination: Path) -> None:
    macros = {
        "CapacityProfileHeldOut": ("profile_tvd", "held_out"),
        "CapacityRawHeldOut": ("bitstring_tvd", "held_out"),
        "CapacityProfileCritical": ("profile_tvd", "critical"),
        "CapacityRawCritical": ("bitstring_tvd", "critical"),
        "CapacityProfileCold": ("profile_tvd", "cold"),
        "CapacityRawCold": ("bitstring_tvd", "cold"),
    }
    lines = [r"\newif\ifcapacitycontrolresults", r"\capacitycontrolresultstrue"]
    for name, (metric, region) in macros.items():
        value = _format_interval(_comparison(summary, metric, region))
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")


def _control_curves(results: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    payloads = [
        read_immutable_json(results / f"seed-{seed}.json") for seed in range(10)
    ]
    betas = np.asarray([row["beta"] for row in payloads[0]["evaluation_rows"]])
    values = {
        metric: np.asarray(
            [
                [row[metric] for row in payload["evaluation_rows"]]
                for payload in payloads
            ]
        )
        for metric in ("profile_tvd", "bitstring_tvd")
    }
    return betas, values


def _ancilla_curves(report_path: Path) -> dict[str, np.ndarray]:
    report = read_immutable_json(report_path)
    rows = report["response_rows"]
    return {
        metric: np.asarray(
            [
                [
                    next(
                        row["conditioned"][metric]
                        for row in rows
                        if row["seed"] == seed
                        and row["ancillas"] == 5
                        and np.isclose(row["beta"], beta)
                    )
                    for beta in np.arange(0.5, 1.51, 0.1)
                ]
                for seed in range(10)
            ]
        )
        for metric in ("profile_tvd", "bitstring_tvd")
    }


def _plot(
    summary: dict[str, Any],
    results: Path,
    ancilla_report: Path,
    destination: Path,
) -> None:
    betas, control = _control_curves(results)
    ancilla = _ancilla_curves(ancilla_report)
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    held_out = np.asarray([index % 2 == 1 for index in range(len(betas))])

    for column, (metric, ylabel) in enumerate(
        (("profile_tvd", "profile TVD"), ("bitstring_tvd", "raw TVD"))
    ):
        axis = axes[0, column]
        for values, color, label in (
            (ancilla[metric], BLUE, "13 qubits, r=5 (91 phases)"),
            (control[metric], ORANGE, "8 qubits, 1/2/3-local (92 phases)"),
        ):
            mean = values.mean(axis=0)
            spread = values.std(axis=0, ddof=1)
            axis.plot(betas, mean, color=color, marker="o", label=label)
            axis.fill_between(
                betas, mean - spread, mean + spread, color=color, alpha=0.15
            )
            axis.scatter(
                betas[held_out],
                mean[held_out],
                facecolors="white",
                edgecolors=color,
                s=24,
                zorder=5,
            )
        axis.axvline(1.0, color="0.55", linestyle="--", linewidth=0.8)
        axis.set_xlabel(r"inverse temperature $\beta$")
        axis.set_ylabel(ylabel)
        axis.set_title("ab"[column], loc="left", fontweight="bold")

        effect_axis = axes[1, column]
        rows = [_comparison(summary, metric, region) for region in REGIONS]
        estimates = np.asarray([row["interval"]["estimate"] for row in rows])
        lower = np.asarray([row["interval"]["lower"] for row in rows])
        upper = np.asarray([row["interval"]["upper"] for row in rows])
        positions = np.arange(len(REGIONS))
        effect_axis.errorbar(
            positions,
            estimates,
            yerr=np.vstack((estimates - lower, upper - estimates)),
            color=BLUE,
            marker="o",
            linestyle="none",
            capsize=3,
        )
        effect_axis.axhline(0.0, color="black", linewidth=0.8)
        effect_axis.set_xticks(positions, ("train", "held out", "critical", "cold"))
        effect_axis.set_ylabel("visible-only TVD minus r=5 TVD")
        effect_axis.set_title("cd"[column], loc="left", fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/curie-weiss-capacity-control-v2/summary.json"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/curie-weiss-capacity-control-v2/results"),
    )
    parser.add_argument(
        "--ancilla-report",
        type=Path,
        default=Path(
            "results/curie-weiss-m5-foundations-v2/report/campaign-report.json"
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("paper/curie_weiss_ancilla_figures/matched_capacity_v2"),
    )
    parser.add_argument(
        "--claims",
        type=Path,
        default=Path("paper/generated/capacity_control_claims.tex"),
    )
    args = parser.parse_args()
    summary = read_immutable_json(args.summary)
    if (
        summary.get("status") != "complete"
        or len(summary.get("paired_seeds", [])) != 10
    ):
        raise RuntimeError("matched-capacity summary is not complete")
    _write_claims(summary, args.claims)
    _plot(summary, args.results, args.ancilla_report, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

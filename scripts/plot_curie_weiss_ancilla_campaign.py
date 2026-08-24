#!/usr/bin/env python3
"""Generate the two deterministic ancilla Curie--Weiss paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from iqp_cond.artifacts import read_immutable_json
from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.curie_weiss_statistics import thermodynamic_metrics

BLUE = "#0072B2"
ORANGE = "#D55E00"


def _rows(report: dict, ancillas: int, beta: float) -> list[dict]:
    return [
        row
        for row in report["response_rows"]
        if row["ancillas"] == ancillas and np.isclose(row["beta"], beta)
    ]


def _band(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20_260_815)
    indices = rng.integers(0, values.shape[0], size=(10_000, values.shape[0]))
    draws = values[indices].mean(axis=1)
    return values.mean(axis=0), *np.quantile(draws, (0.025, 0.975), axis=0)


def _save(figure: plt.Figure, directory: Path, name: str) -> None:
    figure.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _temperature_response(report: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(7.1, 4.5))
    coordinate = np.linspace(-1.0, 1.0, 9)
    for column, beta in enumerate((0.5, 1.0, 1.5)):
        axis = axes[0, column]
        rows = _rows(report, 5, beta)
        target = exact_curie_weiss_target(8, beta)
        mean, lower, upper = _band(
            np.asarray([row["conditioned_profile"] for row in rows])
        )
        axis.plot(coordinate, target.profile, color="black", label="target")
        axis.plot(coordinate, mean, color=BLUE, label="conditioned, r=5")
        axis.fill_between(coordinate, lower, upper, color=BLUE, alpha=0.18)
        if beta != 1.0:
            axis.plot(
                coordinate,
                np.mean([row["temperature_reference_profile"] for row in rows], axis=0),
                color=ORANGE,
                linestyle=":",
                label="independent per-\u03b2 IQP",
            )
        axis.set_title(rf"$\beta={beta:.1f}$" + (" (held out)" if beta == 1.0 else ""))
        axis.set_xlabel("magnetization per spin")
        if column == 0:
            axis.set_ylabel("probability")
    betas = np.asarray([0.5 + 0.1 * index for index in range(11)])
    for column, (metric, label) in enumerate(
        (
            ("absolute_magnetization", r"$\langle |m|\rangle$"),
            ("susceptibility", r"susceptibility $\chi$"),
            ("binder_cumulant", r"Binder cumulant $U$"),
        )
    ):
        model, lower, upper, reference, target_values = [], [], [], [], []
        for beta in betas:
            values = np.asarray(
                [row["conditioned"][metric] for row in _rows(report, 5, float(beta))]
            )
            point, low, high = _band(values[:, None])
            model.append(point.item())
            lower.append(low.item())
            upper.append(high.item())
            if any(np.isclose(beta, held) for held in (0.6, 0.8, 1.0, 1.2, 1.4)):
                reference.append(np.nan)
            else:
                reference.append(
                    np.mean(
                        [
                            row["temperature_reference"][metric]
                            for row in _rows(report, 5, float(beta))
                        ]
                    )
                )
            target = exact_curie_weiss_target(8, float(beta))
            target_values.append(
                thermodynamic_metrics(target, target.probabilities).to_dict()[metric]
            )
        axis = axes[1, column]
        axis.plot(betas, target_values, color="black", label="target")
        axis.plot(betas, model, color=BLUE, label="conditioned, r=5")
        axis.fill_between(betas, lower, upper, color=BLUE, alpha=0.18)
        training_mask = np.isfinite(reference)
        axis.scatter(
            betas[training_mask],
            np.asarray(reference)[training_mask],
            color=ORANGE,
            marker="o",
            facecolors="none",
            s=22,
            label="independent per-\u03b2 IQP",
            zorder=4,
        )
        axis.axvline(1.0, color="0.55", linestyle="--", linewidth=0.8)
        axis.set_ylabel(label)
        axis.set_xlabel(r"inverse temperature $\beta$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    _save(figure, output, "curie_weiss_ancilla_temperature_response")


def _fidelity(report: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    betas = np.asarray([0.5 + 0.1 * index for index in range(11)])
    for axis, metric, label in (
        (axes[0], "profile_tvd", "profile TVD"),
        (axes[1], "bitstring_tvd", "raw bitstring TVD"),
    ):
        means = [
            np.mean([row["conditioned"][metric] for row in _rows(report, 5, beta)])
            for beta in betas
        ]
        axis.plot(betas, means, color=BLUE, label="conditioned, r=5")
        unconditioned = [
            np.nan
            if any(np.isclose(beta, held) for held in (0.6, 0.8, 1.0, 1.2, 1.4))
            else np.mean(
                [row["temperature_reference"][metric] for row in _rows(report, 5, beta)]
            )
            for beta in betas
        ]
        axis.plot(
            betas,
            unconditioned,
            color=ORANGE,
            marker="o",
            linestyle=":",
            label="independent per-\u03b2 IQP",
        )
        axis.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
        axis.axvline(1.0, color="0.55", linestyle="--", linewidth=0.8)
        axis.set_ylabel(label)
        axis.set_xlabel(r"inverse temperature $\beta$")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    _save(figure, output, "curie_weiss_m5_fidelity")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "results/curie-weiss-m5-foundations-v2/report/campaign-report.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("paper/curie_weiss_ancilla_figures")
    )
    args = parser.parse_args(argv)
    report = read_immutable_json(args.report)
    if report.get("completed_runs") != 80 or report.get("headline_ancillas") != 5:
        raise RuntimeError("figures require all 80 immutable fixed-m=5 results")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    _temperature_response(report, args.output)
    _fidelity(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

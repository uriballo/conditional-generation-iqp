"""Generate the two deterministic Curie--Weiss paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from iqp_cond.artifacts import read_immutable_json
from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.curie_weiss_statistics import thermodynamic_metrics


def _rows(report: dict, n: int, beta: float) -> list[dict]:
    return [
        row
        for row in report["response_rows"]
        if row["n_qubits"] == n and np.isclose(row["beta"], beta)
    ]


def _bootstrap_band(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20_260_812)
    draws = values[rng.integers(0, values.size, size=(10_000, values.size))].mean(
        axis=1
    )
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/curie-weiss-foundations-v1/report/campaign-report.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("paper/curie_weiss_figures")
    )
    args = parser.parse_args(argv)
    report = read_immutable_json(args.report)
    if report.get("completed_runs") != 270:
        raise RuntimeError("figures require a complete Curie--Weiss report")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axes = plt.subplots(3, 3, figsize=(7.1, 6.0), sharex="col")
    for row_index, n in enumerate((8, 12, 16)):
        for column, beta in enumerate((0.5, 1.0, 1.5)):
            axis = axes[row_index, column]
            target = exact_curie_weiss_target(n, beta)
            records = _rows(report, n, beta)
            profiles = np.asarray([record["conditioned_profile"] for record in records])
            no_profiles = np.asarray(
                [record["no_condition_profile"] for record in records]
            )
            coordinate = np.linspace(-1, 1, n + 1)
            axis.plot(coordinate, target.profile, color="black", label="target")
            profile_bands = np.asarray(
                [_bootstrap_band(profiles[:, index]) for index in range(n + 1)]
            )
            axis.plot(
                coordinate,
                profile_bands[:, 0],
                color="#2b6cb0",
                label="conditioned",
            )
            axis.fill_between(
                coordinate,
                profile_bands[:, 1],
                profile_bands[:, 2],
                color="#2b6cb0",
                alpha=0.2,
            )
            axis.plot(
                coordinate,
                no_profiles.mean(axis=0),
                color="#c05621",
                linestyle="--",
                label="no condition",
            )
            if beta == 1.0:
                mixture_profiles = np.asarray(
                    [record["adjacent_mixture_profile"] for record in records]
                )
                axis.plot(
                    coordinate,
                    mixture_profiles.mean(axis=0),
                    color="#009e73",
                    linestyle=":",
                    label="adjacent mixture",
                )
            if row_index == 0:
                axis.set_title(
                    rf"$\beta={beta:.1f}$" + (" (held out)" if beta == 1.0 else "")
                )
            if column == 0:
                axis.set_ylabel(f"n={n}\nprobability")
            if row_index == 2:
                axis.set_xlabel("magnetization m")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(args.output / "curie_weiss_profiles.pdf")
    figure.savefig(args.output / "curie_weiss_profiles.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(4, 3, figsize=(7.1, 7.2), sharex=True)
    for column, n in enumerate((8, 12, 16)):
        betas = np.asarray(
            sorted(
                {row["beta"] for row in report["response_rows"] if row["n_qubits"] == n}
            )
        )
        conditioned_means, conditioned_lower, conditioned_upper = [], [], []
        baseline_means = {
            name: [] for name in ("no_condition", "adjacent_oracle", "adjacent_mixture")
        }
        response = {
            name: []
            for name in ("absolute_magnetization", "susceptibility", "binder_cumulant")
        }
        response_lower = {name: [] for name in response}
        response_upper = {name: [] for name in response}
        for beta in betas:
            records = _rows(report, n, float(beta))
            values = np.asarray(
                [record["conditioned"]["profile_tvd"] for record in records]
            )
            point, lower, upper = _bootstrap_band(values)
            conditioned_means.append(point)
            conditioned_lower.append(lower)
            conditioned_upper.append(upper)
            baseline_means["no_condition"].append(
                np.mean([record["no_condition"]["profile_tvd"] for record in records])
            )
            for baseline in ("adjacent_oracle", "adjacent_mixture"):
                available = [
                    record[baseline]["profile_tvd"]
                    for record in records
                    if record[baseline] is not None
                ]
                baseline_means[baseline].append(
                    np.nan if not available else np.mean(available)
                )
            for name in response:
                point, lower, upper = _bootstrap_band(
                    np.asarray([record["conditioned"][name] for record in records])
                )
                response[name].append(point)
                response_lower[name].append(lower)
                response_upper[name].append(upper)
        training = np.isin(betas, (0.5, 0.7, 0.9, 1.1, 1.3, 1.5))
        axes[0, column].fill_between(
            betas, conditioned_lower, conditioned_upper, color="#2b6cb0", alpha=0.2
        )
        axes[0, column].plot(
            betas, conditioned_means, color="#2b6cb0", label="conditioned"
        )
        axes[0, column].scatter(
            betas[training], np.asarray(conditioned_means)[training], color="#2b6cb0"
        )
        axes[0, column].scatter(
            betas[~training],
            np.asarray(conditioned_means)[~training],
            facecolors="white",
            edgecolors="#2b6cb0",
            label="held out",
        )
        axes[0, column].plot(
            betas,
            baseline_means["no_condition"],
            linestyle="--",
            color="#c05621",
            label="no condition",
        )
        axes[0, column].plot(
            betas,
            baseline_means["adjacent_oracle"],
            marker="s",
            linestyle="none",
            color="#6b46c1",
            label="adjacent oracle",
        )
        axes[0, column].plot(
            betas,
            baseline_means["adjacent_mixture"],
            marker="^",
            linestyle="none",
            color="#009e73",
            label="adjacent mixture",
        )
        axes[0, column].axvline(1.0, color="0.5", linestyle="--")
        axes[0, column].set_title(f"n={n}")
        axes[0, column].set_ylabel("profile TVD")
        for row, (name, label) in enumerate(
            (
                ("absolute_magnetization", r"$\langle |m|\rangle$"),
                ("susceptibility", r"$\chi$"),
                ("binder_cumulant", r"$U$"),
            ),
            start=1,
        ):
            targets = [
                thermodynamic_metrics(
                    exact_curie_weiss_target(n, float(beta)),
                    exact_curie_weiss_target(n, float(beta)).probabilities,
                ).to_dict()[name]
                for beta in betas
            ]
            axes[row, column].plot(
                betas, response[name], color="#c05621", label="conditioned"
            )
            axes[row, column].fill_between(
                betas,
                response_lower[name],
                response_upper[name],
                color="#c05621",
                alpha=0.18,
            )
            axes[row, column].scatter(
                betas[training],
                np.asarray(response[name])[training],
                color="#c05621",
            )
            axes[row, column].scatter(
                betas[~training],
                np.asarray(response[name])[~training],
                facecolors="white",
                edgecolors="#c05621",
            )
            axes[row, column].plot(betas, targets, color="black", label="target")
            axes[row, column].axvline(1.0, color="0.5", linestyle="--")
            axes[row, column].set_ylabel(label)
        axes[3, column].set_xlabel(r"inverse temperature $\beta$")
    axes[1, 0].legend(frameon=False)
    axes[0, 0].legend(frameon=False, fontsize=6)
    figure.tight_layout()
    figure.savefig(args.output / "curie_weiss_response.pdf")
    figure.savefig(args.output / "curie_weiss_response.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(7.1, 3.7), sharex=True)
    for column, n in enumerate((8, 12, 16)):
        betas = np.asarray(
            sorted(
                {row["beta"] for row in report["response_rows"] if row["n_qubits"] == n}
            )
        )
        training = np.isin(betas, (0.5, 0.7, 0.9, 1.1, 1.3, 1.5))
        for row_index, (name, label) in enumerate(
            (
                ("energy_per_spin", r"energy per spin $E/n$"),
                ("heat_capacity_per_spin", r"heat capacity $C/n$"),
            )
        ):
            model, lower, upper, target = [], [], [], []
            for beta in betas:
                records = _rows(report, n, float(beta))
                point, low, high = _bootstrap_band(
                    np.asarray([record["conditioned"][name] for record in records])
                )
                model.append(point)
                lower.append(low)
                upper.append(high)
                exact_target = exact_curie_weiss_target(n, float(beta))
                target.append(
                    thermodynamic_metrics(
                        exact_target, exact_target.probabilities
                    ).to_dict()[name]
                )
            axis = axes[row_index, column]
            axis.fill_between(betas, lower, upper, color="#c05621", alpha=0.18)
            axis.plot(betas, model, color="#c05621", label="conditioned")
            axis.scatter(betas[training], np.asarray(model)[training], color="#c05621")
            axis.scatter(
                betas[~training],
                np.asarray(model)[~training],
                facecolors="white",
                edgecolors="#c05621",
            )
            axis.plot(betas, target, color="black", label="target")
            axis.axvline(1.0, color="0.5", linestyle="--")
            axis.set_ylabel(label)
            if row_index == 0:
                axis.set_title(f"n={n}")
            if row_index == 1:
                axis.set_xlabel(r"inverse temperature $\beta$")
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output / "curie_weiss_thermodynamics.pdf")
    figure.savefig(args.output / "curie_weiss_thermodynamics.png", dpi=200)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

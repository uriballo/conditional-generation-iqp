"""Plot the frozen Curie--Weiss confirmatory result used in the manuscript."""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from iqp_cond.artifacts import read_immutable_json
from iqp_cond.curie_single_temperature import (
    SingleTemperatureManifest,
    plan_single_temperature_runs,
)

try:
    from scripts.journal_style import DOUBLE_COL_WIDTH, apply_journal_style, save_figure
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from journal_style import DOUBLE_COL_WIDTH, apply_journal_style, save_figure

MANIFEST = Path("configs/curie_weiss_confirmatory_v1.json")
DIRECT_MANIFEST = Path("configs/curie_weiss_single_temperature_v1.json")
DEFAULT_RESULT_DIRS = (
    Path(".scratch/curie-confirmatory-review/host1/runs"),
    Path(".scratch/curie-confirmatory-review/host2/runs"),
)
DEFAULT_OUTPUT = Path("paper/figures/curie_weiss_confirmatory_main")
DEFAULT_DIRECT_REFERENCE_DIRS = (
    Path(".scratch/curie-single-temperature-review-20260828/host1/results/runs"),
    Path(".scratch/curie-single-temperature-review-20260828/host2/results/runs"),
)
RunKey = tuple[int, str, int]
DirectKey = tuple[int, float, int]


def expected_run_keys(manifest: Mapping[str, Any]) -> set[RunKey]:
    """Return the exact coordinate grid declared by the frozen manifest."""
    return {
        (int(n_qubits), str(arm), int(seed))
        for n_qubits, arm, seed in itertools.product(
            manifest["system_sizes"], manifest["models"], manifest["seeds"]
        )
    }


def index_runs(
    payloads: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[RunKey, Mapping[str, Any]]:
    """Validate campaign provenance and index one complete result per coordinate."""
    found: dict[RunKey, Mapping[str, Any]] = {}
    manifest_digests: set[str] = set()
    expected_updates = int(manifest["training"]["updates"])
    campaign_version = str(manifest["campaign_version"])

    for payload in payloads:
        identity = payload["run_identity"]
        key = (
            int(identity["n_qubits"]),
            str(identity["arm"]),
            int(identity["seed"]),
        )
        if key in found:
            raise ValueError(f"conflicting duplicate run coordinate: {key}")
        if identity["campaign_version"] != campaign_version:
            raise ValueError(f"campaign mismatch for run coordinate: {key}")
        if payload.get("lightweight"):
            raise ValueError(f"lightweight result is not confirmatory: {key}")
        if payload.get("optimizer_updates") != expected_updates:
            raise ValueError(f"incomplete optimizer budget for run coordinate: {key}")
        found[key] = payload
        manifest_digests.add(str(payload["manifest_digest"]))

    expected = expected_run_keys(manifest)
    actual = set(found)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "manifest coverage mismatch: "
            f"found {len(actual)}/{len(expected)} coordinates; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    if len(manifest_digests) != 1:
        raise ValueError("results contain conflicting manifest digests")
    return found


def load_runs(
    result_dirs: Sequence[Path], manifest_path: Path = MANIFEST
) -> tuple[dict[RunKey, Mapping[str, Any]], Mapping[str, Any]]:
    """Read content-verified envelopes from all hosts and validate their union."""
    manifest = json.loads(manifest_path.read_text())
    paths = [
        path
        for directory in result_dirs
        for path in sorted(Path(directory).glob("*.json"))
    ]
    return index_runs((read_immutable_json(path) for path in paths), manifest), manifest


def index_direct_runs(
    payloads: Iterable[Mapping[str, Any]],
    manifest: SingleTemperatureManifest,
) -> dict[DirectKey, Mapping[str, Any]]:
    """Validate and index the immutable single-temperature direct references."""
    found: dict[DirectKey, Mapping[str, Any]] = {}
    expected_by_key = {
        (run.n_qubits, run.beta, run.seed): run
        for run in plan_single_temperature_runs(manifest)
    }
    for payload in payloads:
        identity = payload["run_identity"]
        key = (
            int(identity["n_qubits"]),
            float(identity["beta"]),
            int(identity["seed"]),
        )
        if key in found:
            raise ValueError(
                f"conflicting duplicate direct-reference coordinate: {key}"
            )
        expected_run = expected_by_key.get(key)
        if expected_run is None or identity != expected_run.to_dict():
            raise ValueError(f"direct-reference configuration mismatch: {key}")
        if (
            payload.get("manifest_digest") != manifest.digest
            or payload.get("optimizer_updates") != 2000
            or payload.get("lightweight") is not False
        ):
            raise ValueError(f"incomplete direct reference: {key}")
        found[key] = payload
    expected = set(expected_by_key)
    if set(found) != expected:
        missing = sorted(expected - set(found))[:3]
        extra = sorted(set(found) - expected)[:3]
        raise ValueError(
            f"direct-reference coverage mismatch: missing={missing}, extra={extra}"
        )
    return found


def load_direct_runs(
    directories: Sequence[Path] = DEFAULT_DIRECT_REFERENCE_DIRS,
    manifest_path: Path = DIRECT_MANIFEST,
) -> dict[DirectKey, Mapping[str, Any]]:
    manifest = SingleTemperatureManifest.from_json(manifest_path)
    paths = [
        path
        for directory in directories
        for path in sorted(Path(directory).glob("*.json"))
    ]
    return index_direct_runs(
        (read_immutable_json(path) for path in paths), manifest
    )


def quantiles(values: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the pointwise median and interquartile interval."""
    array = np.asarray(values, dtype=float)
    return (
        np.median(array, axis=0),
        np.quantile(array, 0.25, axis=0),
        np.quantile(array, 0.75, axis=0),
    )


def evaluation_row(payload: Mapping[str, Any], beta: float) -> Mapping[str, Any]:
    """Select the unique evaluation row at one temperature."""
    rows = [
        row
        for row in payload["metrics"]["evaluation_rows"]
        if np.isclose(float(row["beta"]), beta)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one evaluation row at beta={beta}, found {len(rows)}"
        )
    return rows[0]


def build_figure(
    runs: Mapping[RunKey, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    direct_runs: Mapping[DirectKey, Mapping[str, Any]] | None = None,
) -> plt.Figure:
    """Build the two-column, six-axis main-text figure."""
    apply_journal_style()
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(DOUBLE_COL_WIDTH, 7.0),
        sharex="col",
        gridspec_kw={"width_ratios": [1.08, 1.0]},
        layout="constrained",
    )
    figure.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.04, wspace=0.04)

    blue = "#0072B2"
    gray = "#777777"
    seeds = [int(seed) for seed in manifest["seeds"]]
    critical_betas = [float(beta) for beta in manifest["temperatures"]["critical"]]
    training_betas = np.asarray(manifest["temperatures"]["training"], dtype=float)
    held_out_betas = np.asarray(manifest["temperatures"]["held_out"], dtype=float)
    all_betas = np.sort(np.concatenate((training_betas, held_out_betas)))

    for row_index, beta in enumerate(critical_betas):
        rows = [
            evaluation_row(runs[(16, "full_joint_r5", seed)], beta) for seed in seeds
        ]
        target = np.asarray(rows[0]["target_profile"], dtype=float)
        median, lower, upper = quantiles([row["model_profile"] for row in rows])
        magnetization = np.linspace(-1.0, 1.0, target.size)
        axis = axes[row_index, 0]
        axis.fill_between(magnetization, 0.0, target, color="0.88", zorder=0)
        axis.plot(magnetization, target, color="black", linestyle="--", linewidth=1.2)
        axis.fill_between(magnetization, lower, upper, color=blue, alpha=0.22)
        axis.plot(magnetization, median, color=blue, marker="o", markersize=2.5)
        if direct_runs is not None:
            direct_rows = [
                direct_runs[(16, beta, seed)]["metrics"]["evaluation"] for seed in seeds
            ]
            direct_median, direct_lower, direct_upper = quantiles(
                [row["model_profile"] for row in direct_rows]
            )
            axis.fill_between(
                magnetization, direct_lower, direct_upper, color="#D55E00", alpha=0.18
            )
            axis.plot(
                magnetization,
                direct_median,
                color="#D55E00",
                linestyle=":",
                marker="s",
                markersize=2.0,
            )
        axis.text(
            0.03, 0.91, rf"$\beta={beta:.1f}$", transform=axis.transAxes, va="top"
        )
        axis.set_ylabel("probability")

    arms = (
        ("full_joint_r5", blue, "-"),
        ("no_condition_m5", gray, "--"),
    )
    for row_index, n_qubits in enumerate(manifest["system_sizes"]):
        axis = axes[row_index, 1]
        for arm, color, linestyle in arms:
            summaries = [
                quantiles(
                    [
                        evaluation_row(runs[(int(n_qubits), arm, seed)], float(beta))[
                            "profile_tvd"
                        ]
                        for seed in seeds
                    ]
                )
                for beta in all_betas
            ]
            median = np.asarray([summary[0] for summary in summaries], dtype=float)
            lower = np.asarray([summary[1] for summary in summaries], dtype=float)
            upper = np.asarray([summary[2] for summary in summaries], dtype=float)
            axis.fill_between(all_betas, lower, upper, color=color, alpha=0.14)
            axis.plot(all_betas, median, color=color, linestyle=linestyle)
            is_training = np.isin(all_betas, training_betas)
            axis.plot(all_betas[is_training], median[is_training], "o", color=color)
            axis.plot(
                all_betas[~is_training],
                median[~is_training],
                "o",
                markerfacecolor="white",
                markeredgecolor=color,
            )
        if direct_runs is None:
            continue
        direct_summaries = [
            quantiles(
                [
                    direct_runs[(int(n_qubits), float(beta), seed)]["metrics"][
                        "evaluation"
                    ]["profile_tvd"]
                    for seed in seeds
                ]
            )
            for beta in all_betas
        ]
        direct_median = np.asarray([summary[0] for summary in direct_summaries])
        direct_lower = np.asarray([summary[1] for summary in direct_summaries])
        direct_upper = np.asarray([summary[2] for summary in direct_summaries])
        axis.fill_between(
            all_betas, direct_lower, direct_upper, color="#D55E00", alpha=0.12
        )
        axis.plot(
            all_betas,
            direct_median,
            color="#D55E00",
            linestyle=":",
            marker="s",
            markersize=2.0,
        )
        axis.text(
            0.03, 0.91, rf"$n={int(n_qubits)}$", transform=axis.transAxes, va="top"
        )
        axis.set_ylabel("profile TVD")

    for axis in axes.flat:
        axis.grid(color="0.9", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0, 0].set_title(r"magnetization profiles ($n=16$)")
    axes[0, 1].set_title("temperature dependence")
    axes[2, 0].set_xlabel("magnetization $m$")
    axes[2, 1].set_xlabel(r"inverse temperature $\beta$")
    axes[2, 1].set_xticks(np.arange(0.5, 1.51, 0.2))
    axes[0, 0].text(-0.19, 1.14, "a", transform=axes[0, 0].transAxes, fontweight="bold")
    axes[0, 1].text(-0.19, 1.14, "b", transform=axes[0, 1].transAxes, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], color=blue, label="conditioned $r=5$"),
        Patch(
            facecolor="0.88", edgecolor="black", linestyle="--", label="exact target"
        ),
        Line2D(
            [0], [0], color="#D55E00", linestyle=":", marker="s", label="separate fit"
        ),
        Line2D([0], [0], marker="o", color="0.35", linestyle="none", label="training"),
        Line2D([0], [0], color=gray, linestyle="--", label="no-condition $r=5$"),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="0.35",
            linestyle="none",
            label="held out",
        ),
    ]
    figure.legend(handles=legend_handles, loc="outside upper center", ncol=3)
    return figure


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        action="append",
        type=Path,
        dest="result_dirs",
        help="Directory of immutable run envelopes; repeat once per host.",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--direct-reference-dir",
        action="append",
        type=Path,
        dest="direct_reference_dirs",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result_dirs = tuple(args.result_dirs or DEFAULT_RESULT_DIRS)
    runs, manifest = load_runs(result_dirs, args.manifest)
    direct_dirs = tuple(args.direct_reference_dirs or DEFAULT_DIRECT_REFERENCE_DIRS)
    save_figure(
        build_figure(runs, manifest, load_direct_runs(direct_dirs)), args.output
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

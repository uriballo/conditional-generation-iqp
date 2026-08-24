"""Aggregate sparse-grid interpolation results and generate paper artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from journal_style import DOUBLE_COL_WIDTH, MM, apply_journal_style, save_figure

from iqp_cond.ancilla import exact_ancilla_marginal_probabilities
from iqp_cond.exact import (
    enumerate_bitstrings,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.interpolation import broadcast_angles, interpolation_targets
from iqp_cond.sparse_interpolation import (
    SPARSE2_TRAIN_FRACTIONS,
    grid_fractions,
    oracle_adjacent_lookup,
)

GRID_LABELS = {
    2: "2-peak grid",
    3: "3-peak grid",
}
ANCILLA_COUNTS = (0, 2, 3, 5, 7)
SHARED_HELDOUT_FRACTIONS = np.asarray([2, 6], dtype=np.float64) / 32.0
ENDPOINT_RUN_PATTERN = "angle_complete_seed{seed}.npz"


def _median_iqr(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
    )


def _load_runs(
    directory: Path,
    n_peaks: int,
    n_ancillas: int,
    seeds: list[int],
) -> list[dict[str, object]]:
    pattern = f"sparse{n_peaks}_ancilla_{n_ancillas}_broadcast_all_complete_seed*.json"
    runs = [json.loads(path.read_text()) for path in sorted(directory.glob(pattern))]
    found = sorted(int(run["seed"]) for run in runs)
    if found != seeds:
        raise SystemExit(
            f"expected grid {n_peaks} ancillas {n_ancillas} seeds "
            f"{seeds}; found {found}",
        )
    return runs


def _run_npz_path(
    runs_directory: Path,
    n_peaks: int,
    n_ancillas: int,
    seed: int,
) -> Path:
    return (
        runs_directory
        / f"sparse{n_peaks}_ancilla_{n_ancillas}_broadcast_all_complete_seed{seed}.npz"
    )


def _stored_probabilities(
    npz_path: Path,
    n_ancillas: int,
    fractions: np.ndarray,
) -> np.ndarray:
    """Evaluate a stored model at angle fractions with ancilla marginalization."""
    run = np.load(npz_path)
    angles = broadcast_angles(pi * np.asarray(fractions, dtype=np.float64))
    return exact_ancilla_marginal_probabilities(
        run["best_params"],
        run["generators"],
        angles,
        n_data_qubits=19,
        n_ancillas=n_ancillas,
        scheme="broadcast_all",
    )


def _tvd_pairs(
    probs: np.ndarray,
    targets: tuple,
    bitstrings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Shell and bitstring TVD for each probability row against its target."""
    shell = np.empty(probs.shape[0], dtype=np.float64)
    bitstring = np.empty(probs.shape[0], dtype=np.float64)
    for index, (target, row) in enumerate(zip(targets, probs, strict=True)):
        target_probs = target.bitstring_probabilities(bitstrings)
        shells = shell_probabilities(row, bitstrings)
        shell[index] = total_variation_distance(
            target.shell_probabilities,
            shells,
        )
        bitstring[index] = total_variation_distance(target_probs, row)
    return shell, bitstring


def _oracle_rows_for_seed(
    npz_path: Path,
    n_ancillas: int,
    n_peaks: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    """Oracle rows for one seed: stored outputs at adjacent training angles."""
    train_fractions, heldout_fractions = grid_fractions(n_peaks)
    targets = interpolation_targets(pi * heldout_fractions)
    bitstrings = enumerate_bitstrings(19)
    train_probabilities = _stored_probabilities(
        npz_path,
        n_ancillas,
        train_fractions,
    )
    oracle_shell, oracle_bit = oracle_adjacent_lookup(
        train_probabilities,
        train_fractions,
        heldout_fractions,
        targets,
        bitstrings,
    )
    return [
        {
            "seed": seed,
            "method": "oracle",
            "angle_fraction_pi": float(fraction),
            "shell_tvd": float(shell_tvd),
            "bitstring_tvd": float(bitstring_tvd),
        }
        for fraction, shell_tvd, bitstring_tvd in zip(
            heldout_fractions,
            oracle_shell,
            oracle_bit,
            strict=True,
        )
    ]


def _rows_for_cell(
    runs_directory: Path,
    endpoint_directory: Path,
    n_peaks: int,
    n_ancillas: int,
    seeds: list[int],
) -> list[dict[str, float | int | str]]:
    """All (sparse, oracle) rows for one grid x ancilla-count cell.

    Interpolated rows reuse the stored per-run evaluation; oracle rows
    re-evaluate the stored parameters only at the training angles, one process
    per seed.
    """
    _train_fractions, heldout_fractions = grid_fractions(n_peaks)
    targets = interpolation_targets(pi * heldout_fractions)
    bitstrings = enumerate_bitstrings(19)
    rows = []

    if n_peaks == 2 and n_ancillas == 0:
        model_paths = {
            seed: endpoint_directory / ENDPOINT_RUN_PATTERN.format(seed=seed)
            for seed in seeds
        }
        interpolated_probabilities = {
            seed: _stored_probabilities(path, 0, heldout_fractions)
            for seed, path in model_paths.items()
        }
        oracle_probabilities = {
            seed: _stored_probabilities(path, 0, SPARSE2_TRAIN_FRACTIONS)
            for seed, path in model_paths.items()
        }
        for seed in seeds:
            interpolated_shell, interpolated_bit = _tvd_pairs(
                interpolated_probabilities[seed],
                targets,
                bitstrings,
            )
            oracle_shell, oracle_bit = oracle_adjacent_lookup(
                oracle_probabilities[seed],
                SPARSE2_TRAIN_FRACTIONS,
                heldout_fractions,
                targets,
                bitstrings,
            )
            for index, fraction in enumerate(heldout_fractions):
                rows.extend(
                    [
                        {
                            "seed": seed,
                            "method": "interpolated",
                            "angle_fraction_pi": float(fraction),
                            "shell_tvd": float(interpolated_shell[index]),
                            "bitstring_tvd": float(interpolated_bit[index]),
                        },
                        {
                            "seed": seed,
                            "method": "oracle",
                            "angle_fraction_pi": float(fraction),
                            "shell_tvd": float(oracle_shell[index]),
                            "bitstring_tvd": float(oracle_bit[index]),
                        },
                    ],
                )
    else:
        runs = _load_runs(runs_directory, n_peaks, n_ancillas, seeds)
        model_paths = {
            int(run["seed"]): _run_npz_path(
                runs_directory,
                n_peaks,
                n_ancillas,
                int(run["seed"]),
            )
            for run in runs
        }
        rows.extend(_interpolated_rows_from_points(runs))
        jobs = [
            (path, n_ancillas, n_peaks, seed)
            for seed, path in model_paths.items()
        ]
        per_worker_gb = 4.5 * 2 ** (n_ancillas - 7)
        worker_cap = max(1, int(24.0 / per_worker_gb))
        max_workers = min(len(jobs), worker_cap)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_oracle_rows_for_seed, *job) for job in jobs
            ]
            for future in futures:
                rows.extend(future.result())
    return rows


def _interpolated_rows_from_points(
    runs: list[dict[str, object]],
) -> list[dict[str, float | int | str]]:
    rows = []
    for run in runs:
        for point in run["points"]:
            if point["role"] != "heldout":
                continue
            rows.append(
                {
                    "seed": int(run["seed"]),
                    "method": "interpolated",
                    "angle_fraction_pi": point["angle_fraction_pi"],
                    "shell_tvd": point["shell_tvd"],
                    "bitstring_tvd": point["bitstring_tvd"],
                },
            )
    return rows


def _per_seed_means(
    rows: list[dict[str, float | int | str]],
    fractions: np.ndarray | None = None,
) -> dict[tuple[str, str], list[float]]:
    grouped = defaultdict(list)
    for row in rows:
        if fractions is not None and not np.any(
            np.isclose(fractions, row["angle_fraction_pi"]),
        ):
            continue
        for metric in ("shell_tvd", "bitstring_tvd"):
            grouped[(row["method"], row["seed"], metric)].append(row[metric])
    values = defaultdict(list)
    for (method, _seed, metric), observations in grouped.items():
        values[(method, metric)].append(float(np.mean(observations)))
    return values


def _audit(
    cells: dict[tuple[int, int], list[dict[str, float | int | str]]],
) -> tuple[dict[str, object], str]:
    comparisons = []
    for n_peaks in (2, 3):
        _, heldout_fractions = grid_fractions(n_peaks)
        for n_ancillas in ANCILLA_COUNTS:
            if (n_peaks, n_ancillas) not in cells:
                continue
            rows = cells[(n_peaks, n_ancillas)]
            for metric in ("shell_tvd", "bitstring_tvd"):
                for fraction in heldout_fractions:
                    medians = {}
                    for method in ("interpolated", "oracle"):
                        values = [
                            row[metric]
                            for row in rows
                            if row["method"] == method
                            and np.isclose(row["angle_fraction_pi"], fraction)
                        ]
                        medians[method] = float(np.median(values))
                    comparisons.append(
                        {
                            "test": "continuous_conditioning",
                            "n_peaks": n_peaks,
                            "n_ancillas": n_ancillas,
                            "metric": metric,
                            "angle_fraction_pi": float(fraction),
                            "interpolated": medians["interpolated"],
                            "baseline": medians["oracle"],
                            "passed": medians["interpolated"]
                            < medians["oracle"],
                        },
                    )

    for n_ancillas in ANCILLA_COUNTS:
        if (3, n_ancillas) not in cells or (2, n_ancillas) not in cells:
            continue
        for metric in ("shell_tvd", "bitstring_tvd"):
            three = _per_seed_means(
                cells[(3, n_ancillas)],
                SHARED_HELDOUT_FRACTIONS,
            )
            two = _per_seed_means(
                cells[(2, n_ancillas)],
                SHARED_HELDOUT_FRACTIONS,
            )
            dense_median = float(np.median(three[("interpolated", metric)]))
            two_peak_median = float(np.median(two[("interpolated", metric)]))
            comparisons.append(
                {
                    "test": "matched_dense_supervision",
                    "n_peaks": None,
                    "n_ancillas": n_ancillas,
                    "metric": metric,
                    "angle_fraction_pi": None,
                    "interpolated": dense_median,
                    "baseline": two_peak_median,
                    "passed": dense_median < two_peak_median,
                },
            )

    audit = {
        "continuous_conditioning_passed": {
            f"{n_peaks}_{n_ancillas}_{metric}": all(
                item["passed"]
                for item in comparisons
                if item["test"] == "continuous_conditioning"
                and item["n_peaks"] == n_peaks
                and item["n_ancillas"] == n_ancillas
                and item["metric"] == metric
            )
            for n_peaks in (2, 3)
            for n_ancillas in ANCILLA_COUNTS
            for metric in ("shell_tvd", "bitstring_tvd")
            if (n_peaks, n_ancillas) in cells
        },
        "matched_dense_supervision_passed": {
            f"{n_ancillas}_{metric}": next(
                item["passed"]
                for item in comparisons
                if item["test"] == "matched_dense_supervision"
                and item["n_ancillas"] == n_ancillas
                and item["metric"] == metric
            )
            for n_ancillas in ANCILLA_COUNTS
            for metric in ("shell_tvd", "bitstring_tvd")
            if (3, n_ancillas) in cells and (2, n_ancillas) in cells
        },
        "comparisons": comparisons,
    }

    lines = [
        "# Sparse-grid interpolation audit",
        "",
        "## Frozen tests",
        "",
        "| Test | Grid | Ancillas | Metric | Angle / pi | "
        "Interpolated | Baseline | Result |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for item in comparisons:
        if item["test"] == "continuous_conditioning":
            grid = f"{item['n_peaks']}-peak"
            angle = f"{item['angle_fraction_pi']:.5f}"
        else:
            grid = "matched"
            angle = "mean"
        baseline_name = (
            "2-peak grid"
            if item["test"] == "matched_dense_supervision"
            else "oracle adjacent lookup"
        )
        lines.append(
            f"| {item['test'].replace('_', ' ')} vs {baseline_name} | "
            f"{grid} | {item['n_ancillas']} | {item['metric'].replace('_', ' ')} | "
            f"{angle} | {item['interpolated']:.4f} | {item['baseline']:.4f} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |",
        )
    return audit, "\n".join(lines) + "\n"


def _metrics_markdown(
    cells: dict[tuple[int, int], list[dict[str, float | int | str]]],
) -> str:
    lines = [
        "# Sparse-grid interpolation metrics",
        "",
        "Median [interquartile range] of each seed's mean over the held-out",
        "angles of its grid. The 2-peak m=0 row reuses the endpoint-only",
        "primary model.",
        "",
    ]
    for n_peaks in (2, 3):
        lines.append(f"## {GRID_LABELS[n_peaks]}")
        lines.append("")
        lines.append(
            "| Ancillas | Interpolated shell TVD | Oracle shell TVD | "
            "Interpolated bitstring TVD | Oracle bitstring TVD |",
        )
        lines.append("|---|---:|---:|---:|---:|")
        for n_ancillas in ANCILLA_COUNTS:
            if (n_peaks, n_ancillas) not in cells:
                continue
            rows = cells[(n_peaks, n_ancillas)]
            seed_means = _per_seed_means(rows)
            stats = {
                (method, metric): _median_iqr(seed_means[(method, metric)])
                for method in ("interpolated", "oracle")
                for metric in ("shell_tvd", "bitstring_tvd")
            }
            lines.append(
                f"| {n_ancillas} | "
                f"{stats[('interpolated', 'shell_tvd')][0]:.4f} "
                f"[{stats[('interpolated', 'shell_tvd')][1]:.4f}, "
                f"{stats[('interpolated', 'shell_tvd')][2]:.4f}] | "
                f"{stats[('oracle', 'shell_tvd')][0]:.4f} "
                f"[{stats[('oracle', 'shell_tvd')][1]:.4f}, "
                f"{stats[('oracle', 'shell_tvd')][2]:.4f}] | "
                f"{stats[('interpolated', 'bitstring_tvd')][0]:.4f} "
                f"[{stats[('interpolated', 'bitstring_tvd')][1]:.4f}, "
                f"{stats[('interpolated', 'bitstring_tvd')][2]:.4f}] | "
                f"{stats[('oracle', 'bitstring_tvd')][0]:.4f} "
                f"[{stats[('oracle', 'bitstring_tvd')][1]:.4f}, "
                f"{stats[('oracle', 'bitstring_tvd')][2]:.4f}] |",
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _latex_table(
    cells: dict[tuple[int, int], list[dict[str, float | int | str]]],
) -> str:
    """Compact paper table: per-grid, per-m medians for interpolated and oracle."""
    lines = []
    for n_peaks in (2, 3):
        lines.append(f"% {GRID_LABELS[n_peaks]}")
        for n_ancillas in ANCILLA_COUNTS:
            if (n_peaks, n_ancillas) not in cells:
                continue
            cell_rows = cells[(n_peaks, n_ancillas)]
            seed_means = _per_seed_means(cell_rows)
            stats = {
                (method, metric): _median_iqr(seed_means[(method, metric)])
                for method in ("interpolated", "oracle")
                for metric in ("shell_tvd", "bitstring_tvd")
            }
            lines.append(
                f"{n_peaks} & {n_ancillas} & "
                f"{stats[('interpolated', 'shell_tvd')][0]:.3f} & "
                f"{stats[('oracle', 'shell_tvd')][0]:.3f} & "
                f"{stats[('interpolated', 'bitstring_tvd')][0]:.3f} & "
                f"{stats[('oracle', 'bitstring_tvd')][0]:.3f} \\\\",
            )
    if lines:
        lines[-1] = lines[-1].removesuffix("\\\\").rstrip()
    return "\n".join(lines) + "\n"


def _bitstring_weights(n_qubits: int = 19) -> np.ndarray:
    """Hamming-weight of every bitstring, used to turn probs into means."""
    return enumerate_bitstrings(n_qubits).sum(axis=1)


def _model_mean_path(
    runs_directory: Path,
    endpoint_directory: Path,
    n_peaks: int,
    n_ancillas: int,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Model Hamming-weight mean versus angle fraction for one cell."""
    train_fractions, heldout_fractions = grid_fractions(n_peaks)
    fractions = np.concatenate([train_fractions, heldout_fractions])
    if n_peaks == 2 and n_ancillas == 0:
        means_per_seed = []
        for seed in seeds:
            endpoint_path = endpoint_directory / ENDPOINT_RUN_PATTERN.format(
                seed=seed,
            )
            probs = _stored_probabilities(endpoint_path, 0, fractions)
            means_per_seed.append(probs @ _bitstring_weights())
        return fractions, np.median(np.asarray(means_per_seed), axis=0)
    else:
        runs = _load_runs(runs_directory, n_peaks, n_ancillas, seeds)
        means_by_fraction = defaultdict(list)
        for run in runs:
            for point in run["points"]:
                means_by_fraction[point["angle_fraction_pi"]].append(
                    point["model_mean"],
                )
        ordered = np.asarray(sorted(means_by_fraction))
        means = np.asarray(
            [_median_iqr(means_by_fraction[value])[0] for value in ordered],
        )
        return ordered, means
    bitstrings = enumerate_bitstrings(19)
    return fractions, probs @ bitstrings.sum(axis=1)


def _plot(
    runs_directory: Path,
    endpoint_directory: Path,
    cells: dict[tuple[int, int], list[dict[str, float | int | str]]],
    seeds: list[int],
    output: Path,
) -> None:
    use_tex = apply_journal_style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COL_WIDTH, 60 * MM),
        layout="constrained",
    )

    colors = {
        (3, 7): "#0072B2",
        (2, 7): "#D55E00",
        (3, 0): "#009E73",
        (2, 0): "#CC79A7",
    }
    for (n_peaks, n_ancillas), color in colors.items():
        if (n_peaks, n_ancillas) not in cells:
            continue
        fractions, means = _model_mean_path(
            runs_directory,
            endpoint_directory,
            n_peaks,
            n_ancillas,
            seeds,
        )
        axes[0].plot(
            fractions,
            means,
            "s--",
            color=color,
            linewidth=1.2,
            markersize=3,
            label=f"{GRID_LABELS[n_peaks].replace(' grid', '')}, m={n_ancillas}",
        )

    train_fractions, _ = grid_fractions(3)
    fractions = np.concatenate([train_fractions, np.asarray([2, 6]) / 32.0])
    targets = interpolation_targets(pi * fractions)
    axes[0].plot(
        fractions,
        [target.actual_mean for target in targets],
        "o-",
        color="#000000",
        markersize=3,
        label="Target",
    )
    if use_tex:
        axes[0].set_title(r"(a) Conditional Mean")
        axes[0].set_xlabel(r"Angle $\alpha/\pi$")
    else:
        axes[0].set_title("(a) Conditional Mean")
        axes[0].set_xlabel("Angle alpha/pi")
    axes[0].set_ylabel("Hamming Weight")
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        fontsize=6.5,
        loc="upper right",
    )
    axes[0].spines[["top", "right"]].set_visible(False)

    colors = {
        ("interpolated", 3): "#0072B2",
        ("oracle", 3): "#56B4E9",
        ("interpolated", 2): "#D55E00",
        ("oracle", 2): "#E69F00",
    }
    styles = {
        ("interpolated", 3): "o-",
        ("oracle", 3): "^--",
        ("interpolated", 2): "s-",
        ("oracle", 2): "d--",
    }
    for axis, metric, title, panel in (
        (axes[1], "shell_tvd", "Held-Out Profile TVD", "(b)"),
        (axes[2], "bitstring_tvd", "Held-Out Bitstring TVD", "(c)"),
    ):
        for (method, n_peaks), color in colors.items():
            present = [
                n_ancillas
                for n_ancillas in ANCILLA_COUNTS
                if (n_peaks, n_ancillas) in cells
            ]
            if not present:
                continue
            stats = np.asarray(
                [
                    _median_iqr(
                        _per_seed_means(cells[(n_peaks, n_ancillas)])[
                            (method, metric)
                        ],
                    )
                    for n_ancillas in present
                ],
            )
            axis.errorbar(
                present,
                stats[:, 0],
                yerr=np.asarray(
                    [stats[:, 0] - stats[:, 1], stats[:, 2] - stats[:, 0]],
                ),
                fmt=styles[(method, n_peaks)],
                color=color,
                linewidth=1.2,
                markersize=3,
                capsize=2,
                label=f"{method.capitalize()} "
                f"{GRID_LABELS[n_peaks].replace(' grid', '')}",
            )
        if use_tex:
            axis.set_title(rf"{panel} {title}")
            axis.set_xlabel(r"Ancilla Qubits $m$")
        else:
            axis.set_title(f"{panel} {title}")
            axis.set_xlabel("Ancilla Qubits m")
        axis.set_xticks(list(ANCILLA_COUNTS))
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].set_ylabel("Profile TVD")
    axes[2].set_ylabel("Bitstring TVD")
    axes[1].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        fontsize=7.5,
        loc="upper right",
    )
    axes[2].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        fontsize=7.5,
        loc="upper right",
    )
    save_figure(figure, output)


def _save_cells(cells: dict[tuple[int, int], list[dict]], path: Path) -> None:
    payload = {
        f"{n_peaks}_{n_ancillas}": rows
        for (n_peaks, n_ancillas), rows in cells.items()
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _load_cells(path: Path) -> dict[tuple[int, int], list[dict]]:
    payload = json.loads(path.read_text())
    cells = {}
    for key, rows in payload.items():
        n_peaks, n_ancillas = (int(part) for part in key.split("_"))
        cells[(n_peaks, n_ancillas)] = rows
    return cells


def main() -> None:
    """Generate sparse-grid metrics, audit, and figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/sparse-interpolation/runs"),
    )
    parser.add_argument(
        "--endpoint-runs",
        type=Path,
        default=Path("results/runs"),
    )
    parser.add_argument(
        "--seeds",
        default="0-9",
        help="seed range to validate against the stored runs",
    )
    parser.add_argument(
        "--cells-cache",
        type=Path,
        default=Path("results/sparse-interpolation/summary/cells.json"),
        help=(
            "cache of computed per-cell rows; reused if present and newer "
            "than the runs"
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="recompute every cell from stored runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sparse-interpolation/summary"),
    )
    args = parser.parse_args()

    if "-" in args.seeds:
        start, end = (int(part) for part in args.seeds.split("-"))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(part) for part in args.seeds.split(",")]

    cells = {}
    if not args.no_cache and args.cells_cache.exists():
        newest_run = max(
            (path.stat().st_mtime for path in args.runs.glob("*.json")),
            default=0.0,
        )
        if args.cells_cache.stat().st_mtime >= newest_run:
            cells = _load_cells(args.cells_cache)
            print(f"reused cell cache {args.cells_cache}")
    if not cells:
        for n_peaks in (2, 3):
            for n_ancillas in ANCILLA_COUNTS:
                if n_peaks == 2 and n_ancillas == 0:
                    label = f"cell grid={n_peaks} m={n_ancillas} (endpoint reuse)"
                    print(label, flush=True)
                    cells[(n_peaks, n_ancillas)] = _rows_for_cell(
                        args.runs,
                        args.endpoint_runs,
                        n_peaks,
                        n_ancillas,
                        seeds,
                    )
                elif n_peaks == 3 or n_ancillas in (2, 3, 5, 7):
                    print(f"cell grid={n_peaks} m={n_ancillas}", flush=True)
                    cells[(n_peaks, n_ancillas)] = _rows_for_cell(
                        args.runs,
                        args.endpoint_runs,
                        n_peaks,
                        n_ancillas,
                        seeds,
                    )
                else:
                    continue
        args.cells_cache.parent.mkdir(parents=True, exist_ok=True)
        _save_cells(cells, args.cells_cache)
        print(f"saved cell cache {args.cells_cache}")

    args.output.mkdir(parents=True, exist_ok=True)
    metrics = _metrics_markdown(cells)
    audit, audit_markdown = _audit(cells)
    (args.output / "metrics.md").write_text(metrics)
    (args.output / "audit.md").write_text(audit_markdown)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output / "sparse_interpolation_table.tex").write_text(
        _latex_table(cells),
    )
    figure_path = args.output / "sparse_interpolation.png"
    _plot(args.runs, args.endpoint_runs, cells, seeds, figure_path)
    print(metrics)
    print(audit_markdown)


if __name__ == "__main__":
    main()

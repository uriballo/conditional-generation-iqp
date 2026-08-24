"""Self-explanatory showcase variants for sparse-grid interpolation.

Each figure answers one question:

1. What is the Target? (heatmap of the Gaussian peak family plus the
   angle-to-mean path, with trained/held-out conditions marked)
2. Did interpolation work? (a) held-out profiles: Target vs interpolated
   model vs Oracle; (b) quantitative TVD summary -- the headline pair
3. Does interpolation beat memorization? (interpolated vs Oracle lookup)
4. Do ancillas sharpen the interpolated profiles?
5. Does more supervision help?
6. Interpolation with and without ancillas (m=0 vs m=7)

Color semantics across all variants: black = Target, blue = interpolated
model, red = Oracle lookup, gold = held-out conditions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from journal_style import (
    DOUBLE_COL_WIDTH,
    MM,
    SINGLE_COL_WIDTH,
    apply_journal_style,
    save_figure,
)

from iqp_cond.interpolation import interpolation_targets

ANCILLA_COUNTS = (0, 2, 3, 5, 7)
SHARED_HELDOUT_FRACTIONS = np.asarray([2, 6], dtype=np.float64) / 32.0
THREE_PEAK_ANGLES = np.asarray([0, 2, 4, 6, 8], dtype=np.float64) / 32.0
TRAINED_3_FRACTIONS = THREE_PEAK_ANGLES[
    ~np.isin(THREE_PEAK_ANGLES, SHARED_HELDOUT_FRACTIONS)
]
TARGET_COLOR = "#000000"
MODEL_M7 = "#0072B2"
MODEL_M0 = "#56B4E9"
TWO_PEAK = "#D55E00"
ORACLE_COLOR = "#D62728"
ORACLE_2PEAK = "#F2A0A0"
GOLD = "#E6A817"
HELDOUT_TINT = "#f7e8c8"


def _median_iqr(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
    )


def _load_cells(cache_path: Path) -> dict[tuple[int, int], list[dict]]:
    payload = json.loads(cache_path.read_text())
    cells = {}
    for key, rows in payload.items():
        n_peaks, n_ancillas = (int(part) for part in key.split("_"))
        cells[(n_peaks, n_ancillas)] = rows
    return cells


def _sparse_run_points(
    runs_directory: Path,
    n_peaks: int,
    n_ancillas: int,
) -> dict[int, list[dict]]:
    """Per-seed lists of evaluation points for one grid x ancilla cell."""
    pattern = f"sparse{n_peaks}_ancilla_{n_ancillas}_broadcast_all_complete_seed*.json"
    runs = [
        json.loads(path.read_text())
        for path in sorted(runs_directory.glob(pattern))
    ]
    return {int(run["seed"]): run["points"] for run in runs}


def _model_profiles(
    runs_directory: Path,
    n_peaks: int,
    n_ancillas: int,
    fraction: float,
) -> np.ndarray:
    """Median and IQR of model shell profiles across seeds at one angle."""
    profiles = []
    for points in _sparse_run_points(runs_directory, n_peaks, n_ancillas).values():
        for point in points:
            if np.isclose(point["angle_fraction_pi"], fraction):
                profiles.append(np.asarray(point["model_shell_probabilities"]))
    if not profiles:
        raise SystemExit(f"no profiles for {n_peaks}-peak m={n_ancillas} {fraction}")
    return np.asarray(profiles)


def _variant1_target_family(output: Path) -> None:
    """What is the Target (two-column): probability heatmap plus mean path."""
    use_tex = apply_journal_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL_WIDTH, 68 * MM),
        layout="constrained",
        width_ratios=[1.15, 1.0],
    )
    heat_axis, mean_axis = axes
    heldout_mask = np.isin(THREE_PEAK_ANGLES, SHARED_HELDOUT_FRACTIONS)

    fractions = np.linspace(0.0, 0.25, 129)
    probabilities = np.asarray(
        [
            target.shell_probabilities
            for target in interpolation_targets(pi * fractions)
        ],
    )
    image = heat_axis.imshow(
        probabilities,
        aspect="auto",
        origin="lower",
        extent=[-0.5, 19.5, 0.0, 0.25],
        cmap="viridis",
        interpolation="nearest",
        zorder=1,
    )
    for fraction, heldout in zip(THREE_PEAK_ANGLES, heldout_mask, strict=True):
        marker = "*" if heldout else "o"
        heat_axis.plot(
            9.5 - 19.0 * fraction,
            fraction,
            marker,
            markersize=7 if heldout else 5.5,
            color=GOLD if heldout else "#111111",
            markeredgecolor="#000000",
            markeredgewidth=0.6,
            zorder=3,
        )
    heat_axis.plot(
        9.5 - 19.0 * fractions,
        fractions,
        color="#ffffff",
        linewidth=0.8,
        linestyle="--",
        alpha=0.85,
        zorder=2,
    )
    colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.035, pad=0.02)
    colorbar.set_label(
        r"Target probability $p(k\mid\alpha)$"
        if use_tex
        else "Target probability p(k|alpha)",
        fontsize=6,
    )
    colorbar.ax.tick_params(labelsize=5.5)
    colorbar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    if use_tex:
        heat_axis.set_xlabel(r"Output Hamming weight $k$")
        heat_axis.set_ylabel(r"Condition angle $\alpha/\pi$")
        heat_axis.set_title(r"(a) Target family $p(k\mid\alpha)$")
    else:
        heat_axis.set_xlabel("Output Hamming weight k")
        heat_axis.set_ylabel("Condition angle alpha/pi")
        heat_axis.set_title("(a) Target family p(k|alpha)")
    heat_axis.set_xticks(range(0, 20, 2))
    heat_axis.set_yticks([0.0, 0.0625, 0.125, 0.1875, 0.25])
    heat_axis.tick_params(labelsize=7)

    mean_axis.plot(
        fractions,
        9.5 - 19.0 * fractions,
        color="#555555",
        linewidth=1.2,
    )
    mean_axis.plot(
        THREE_PEAK_ANGLES[~heldout_mask],
        9.5 - 19.0 * THREE_PEAK_ANGLES[~heldout_mask],
        "o",
        color="#111111",
        markersize=5,
        markeredgecolor="#ffffff",
        markeredgewidth=0.5,
        label="Trained",
        zorder=3,
    )
    mean_axis.plot(
        THREE_PEAK_ANGLES[heldout_mask],
        9.5 - 19.0 * THREE_PEAK_ANGLES[heldout_mask],
        "*",
        color=GOLD,
        markersize=8,
        markeredgecolor="#000000",
        markeredgewidth=0.6,
        label="Held-out",
        zorder=3,
    )
    if use_tex:
        mean_axis.set_xlabel(r"Condition angle $\alpha/\pi$")
        mean_axis.set_ylabel(r"Peak center $\mu(\alpha)$")
        mean_axis.set_title(r"(b) Mean trajectory $\mu=9.5-19\,\alpha/\pi$")
    else:
        mean_axis.set_xlabel("Condition angle alpha/pi")
        mean_axis.set_ylabel("Peak center mu(alpha)")
        mean_axis.set_title("(b) Mean trajectory mu = 9.5 - 19*alpha/pi")
    mean_axis.set_xticks([0.0, 0.0625, 0.125, 0.1875, 0.25])
    mean_axis.tick_params(labelsize=7)
    mean_axis.legend(frameon=False, fontsize=6.5, loc="lower left")
    mean_axis.spines[["top", "right"]].set_visible(False)
    save_figure(figure, output)


def _oracle_profile(
    runs_directory: Path,
    fraction: float,
) -> np.ndarray:
    """Better of the two stored neighboring trained outputs, m=7."""
    target = interpolation_targets(pi * np.asarray([fraction]))[0]
    trained = np.sort(TRAINED_3_FRACTIONS)
    neighbors = (
        (trained[trained < fraction][-1], trained[trained > fraction][0])
        if len(trained[trained < fraction])
        else (trained[trained > fraction][0],)
    )
    best = None
    best_tvd = np.inf
    for neighbor in neighbors:
        med = np.median(_model_profiles(runs_directory, 3, 7, float(neighbor)), axis=0)
        tvd = 0.5 * np.abs(med - target.shell_probabilities).sum()
        if tvd < best_tvd:
            best, best_tvd = med, tvd
    return best


def _variant2_heldout_profiles(
    runs_directory: Path,
    cells: dict,
    output: Path,
) -> None:
    """Did interpolation work at unseen angles: Target vs model vs Oracle."""
    use_tex = apply_journal_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(DOUBLE_COL_WIDTH, 58 * MM),
        layout="constrained",
        sharey=True,
    )
    for axis, fraction in zip(axes, SHARED_HELDOUT_FRACTIONS, strict=True):
        axis.axvspan(0, 19, color=HELDOUT_TINT, zorder=0)
        target = interpolation_targets(pi * np.asarray([fraction]))[0]
        profiles = _model_profiles(runs_directory, 3, 7, float(fraction))
        med = np.median(profiles, axis=0)
        lower = np.quantile(profiles, 0.25, axis=0)
        upper = np.quantile(profiles, 0.75, axis=0)
        oracle = _oracle_profile(runs_directory, float(fraction))
        axis.fill_between(np.arange(20), lower, upper, color=MODEL_M7, alpha=0.25)
        axis.plot(
            np.arange(20),
            target.shell_probabilities,
            color=TARGET_COLOR,
            linewidth=1.6,
            label="Target",
        )
        axis.plot(
            np.arange(20),
            med,
            color=MODEL_M7,
            linewidth=1.4,
            label="Interpolated model",
        )
        axis.plot(
            np.arange(20),
            oracle,
            color=ORACLE_COLOR,
            linewidth=1.2,
            linestyle="--",
            label="Oracle lookup",
        )
        rows = cells.get((3, 7))
        interp_tvd = float(
            np.median(
                [
                    row["shell_tvd"]
                    for row in rows
                    if row["method"] == "interpolated"
                    and np.isclose(row["angle_fraction_pi"], fraction)
                ],
            ),
        )
        oracle_tvd = float(
            np.median(
                [
                    row["shell_tvd"]
                    for row in rows
                    if row["method"] == "oracle"
                    and np.isclose(row["angle_fraction_pi"], fraction)
                ],
            ),
        )
        axis.text(
            0.98,
            0.97,
            f"Interp TVD = {interp_tvd:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color=MODEL_M7,
        )
        axis.text(
            0.98,
            0.90,
            f"Oracle TVD = {oracle_tvd:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color=ORACLE_COLOR,
        )
        axis.axvline(target.actual_mean, color="#888888", linewidth=0.6, linestyle=":")
        axis.set_title(
            rf"Held-out $\alpha/\pi={fraction:.4f}$"
            if use_tex
            else f"Held-out alpha/pi={fraction:.4f}",
            fontsize=7.5,
        )
        axis.set_ylim(0, None)
        axis.set_xticks(range(0, 20, 2))
        axis.set_xlim(-0.5, 19.5)
        axis.grid(axis="x", color="#dddddd", linewidth=0.5, zorder=0)
        axis.tick_params(axis="x", labelsize=7)
        axis.set_xlabel(
            r"Output Hamming weight $k$" if use_tex else "Output Hamming weight k",
            fontsize=7.5,
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Shell probability")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    save_figure(figure, output)


def _variant2_tvd_summary(cells: dict, output: Path) -> None:
    """Quantitative summary: held-out profile TVD, interpolated vs Oracle."""
    use_tex = apply_journal_style()
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(SINGLE_COL_WIDTH, 62 * MM),
        layout="constrained",
    )
    fractions = list(SHARED_HELDOUT_FRACTIONS)
    methods = [
        ("Interpolated", "interpolated", 3, MODEL_M7),
        ("Oracle lookup", "oracle", 3, ORACLE_COLOR),
        ("Interpolated, 2-peak", "interpolated", 2, TWO_PEAK),
    ]
    x_positions = np.arange(len(fractions))
    width = 0.26
    for offset, (label, method, n_peaks, color) in zip(
        (-width, 0.0, width),
        methods,
        strict=True,
    ):
        rows = cells.get((n_peaks, 7))
        if rows is None:
            continue
        values = [
            float(
                np.median(
                    [
                        row["shell_tvd"]
                        for row in rows
                        if row["method"] == method
                        and np.isclose(row["angle_fraction_pi"], fraction)
                    ],
                ),
            )
            for fraction in fractions
        ]
        axis.bar(
            x_positions + offset,
            values,
            width,
            color=color,
            label=label,
        )
        for x_pos, value in zip(x_positions + offset, values, strict=True):
            axis.text(
                x_pos,
                value + 0.004,
                f"{value:.3f}",
                ha="center",
                fontsize=6,
            )
    axis.set_ylim(0, 0.30)
    if use_tex:
        axis.set_title(r"Held-out profile TVD ($m{=}7$): lower is better")
        axis.set_xticks(
            x_positions,
            [rf"$\alpha/\pi={fraction:.4f}$" for fraction in fractions],
        )
        axis.set_xlabel(r"Held-out condition angle $\alpha/\pi$")
    else:
        axis.set_title("Held-out profile TVD (m=7): lower is better")
        axis.set_xticks(
            x_positions,
            [f"alpha/pi={fraction:.4f}" for fraction in fractions],
        )
        axis.set_xlabel("Held-out condition angle alpha/pi")
    axis.set_ylabel("Median profile TVD")
    axis.tick_params(labelsize=7)
    axis.legend(frameon=False, fontsize=6.5, loc="upper right")
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.995,
        "Takeaway: unseen conditions are recovered more accurately "
        "than nearest-neighbor lookup",
        ha="center",
        fontsize=7.5,
        fontweight="bold",
    )
    save_figure(figure, output)


def _variant6_m0_vs_m7(
    runs_directory: Path,
    cells: dict,
    output: Path,
) -> None:
    """Do ancillas help at the held-out angles: m=0 versus m=7."""
    use_tex = apply_journal_style()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(SINGLE_COL_WIDTH, 118 * MM),
        layout="constrained",
        sharey=True,
    )
    x_label = (
        r"Output Hamming weight $k$" if use_tex else "Output Hamming weight k"
    )
    for row, fraction in enumerate(SHARED_HELDOUT_FRACTIONS):
        for col, n_ancillas in enumerate((0, 7)):
            axis = axes[row, col]
            axis.axvspan(0, 19, color=HELDOUT_TINT, zorder=0)
            target = interpolation_targets(pi * np.asarray([fraction]))[0]
            profiles = _model_profiles(
                runs_directory,
                3,
                n_ancillas,
                float(fraction),
            )
            med = np.median(profiles, axis=0)
            lower = np.quantile(profiles, 0.25, axis=0)
            upper = np.quantile(profiles, 0.75, axis=0)
            axis.fill_between(
                np.arange(20),
                lower,
                upper,
                color=MODEL_M7 if n_ancillas else MODEL_M0,
                alpha=0.22,
            )
            axis.plot(
                np.arange(20),
                target.shell_probabilities,
                color=TARGET_COLOR,
                linewidth=1.5,
                label="Target",
            )
            axis.plot(
                np.arange(20),
                med,
                color=MODEL_M7 if n_ancillas else MODEL_M0,
                linewidth=1.3,
                label="Model (median)",
            )
            axis.axvline(
                target.actual_mean,
                color="#888888",
                linewidth=0.6,
                linestyle=":",
            )
            rows = cells.get((3, n_ancillas))
            tvd = float(
                np.median(
                    [
                        row["shell_tvd"]
                        for row in rows
                        if row["method"] == "interpolated"
                        and np.isclose(row["angle_fraction_pi"], fraction)
                    ],
                ),
            )
            axis.text(
                0.98,
                0.97,
                f"Profile TVD = {tvd:.3f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=6.5,
                color=MODEL_M7 if n_ancillas else MODEL_M0,
                fontweight="bold",
            )
            axis.set_ylim(0, None)
            axis.set_xticks(range(0, 20, 2))
            axis.set_xlim(-0.5, 19.5)
            axis.grid(axis="x", color="#dddddd", linewidth=0.5, zorder=0)
            axis.tick_params(axis="x", labelsize=6.5)
            axis.set_xlabel(x_label, fontsize=7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_title(
                rf"$m={n_ancillas}$"
                + (" (no ancillas)" if n_ancillas == 0 else " (traced-out)")
                if use_tex
                else f"m = {n_ancillas}"
                + (" (no ancillas)" if n_ancillas == 0 else " (traced-out)"),
                fontsize=7.5,
            )
            if col == 0:
                axis.set_ylabel(
                    rf"Held-out $\alpha/\pi={fraction:.4f}$"
                    if use_tex
                    else f"Held-out alpha/pi={fraction:.4f}",
                    fontsize=7,
                )
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper right")
    figure.text(
        0.5,
        0.995,
        "Held-out angles, never trained (gold): ancillas visibly improve "
        "agreement with the Target",
        ha="center",
        fontsize=7,
    )
    save_figure(figure, output)


def _variant3_vs_oracle(cells: dict, output: Path) -> None:
    """Interpolation vs memorization: held-out TVD against the Oracle lookup."""
    use_tex = apply_journal_style()
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(SINGLE_COL_WIDTH, 110 * MM),
        layout="constrained",
    )
    styles = {
        ("interpolated", 3): (MODEL_M7, "o-"),
        ("oracle", 3): (ORACLE_COLOR, "o--"),
        ("interpolated", 2): (TWO_PEAK, "s-"),
        ("oracle", 2): (ORACLE_2PEAK, "s--"),
    }
    for axis, metric, label in (
        (axes[0], "shell_tvd", "Profile TVD"),
        (axes[1], "bitstring_tvd", "Bitstring TVD"),
    ):
        for (method, n_peaks), (color, fmt) in styles.items():
            rows = cells.get((n_peaks, 7))
            if rows is None:
                continue
            fractions = sorted(
                {
                    row["angle_fraction_pi"]
                    for row in rows
                    if row["method"] == method
                },
            )
            stats = np.asarray(
                [
                    _median_iqr(
                        [
                            row[metric]
                            for row in rows
                            if row["method"] == method
                            and np.isclose(row["angle_fraction_pi"], fraction)
                        ],
                    )
                    for fraction in fractions
                ],
            )
            axis.errorbar(
                fractions,
                stats[:, 0],
                yerr=np.asarray(
                    [stats[:, 0] - stats[:, 1], stats[:, 2] - stats[:, 0]],
                ),
                fmt=fmt,
                color=color,
                linewidth=1.2,
                markersize=3.5,
                capsize=2,
                label=(
                    f"{method.capitalize()} {n_peaks}-peak"
                    if method == "interpolated"
                    else f"Oracle {n_peaks}-peak"
                ),
            )
        for n_peaks in (2, 3):
            rows = cells.get((n_peaks, 7))
            if rows is None:
                continue
            fractions = sorted(
                {
                    row["angle_fraction_pi"]
                    for row in rows
                    if row["method"] == "interpolated"
                },
            )
            interpolated_med = [
                float(
                    np.median(
                        [
                            row[metric]
                            for row in rows
                            if row["method"] == "interpolated"
                            and np.isclose(row["angle_fraction_pi"], fraction)
                        ],
                    ),
                )
                for fraction in fractions
            ]
            oracle_med = [
                float(
                    np.median(
                        [
                            row[metric]
                            for row in rows
                            if row["method"] == "oracle"
                            and np.isclose(row["angle_fraction_pi"], fraction)
                        ],
                    ),
                )
                for fraction in fractions
            ]
            color = MODEL_M7 if n_peaks == 3 else TWO_PEAK
            axis.fill_between(
                fractions,
                interpolated_med,
                oracle_med,
                color=color,
                alpha=0.08,
            )
        axis.set_ylabel(label)
        axis.tick_params(axis="x", labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    if use_tex:
        for axis in axes:
            axis.set_xlabel(r"Held-out angle $\alpha/\pi$")
    else:
        for axis in axes:
            axis.set_xlabel("Held-out angle alpha/pi")
    axes[0].set_title("m = 7")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
    figure.text(
        0.5,
        0.995,
        "Oracle = better of the two stored neighboring trained outputs; "
        "shaded regions mark where interpolation beats memorization",
        ha="center",
        fontsize=7,
    )
    save_figure(figure, output)


def _variant4_ancilla_effect(
    runs_directory: Path,
    output: Path,
) -> None:
    """Do ancillas sharpen the interpolated profiles at held-out angles."""
    use_tex = apply_journal_style()
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(SINGLE_COL_WIDTH, 110 * MM),
        layout="constrained",
    )
    for axis, fraction in zip(axes, SHARED_HELDOUT_FRACTIONS, strict=True):
        targets = interpolation_targets(pi * np.asarray([fraction]))
        target = targets[0]
        for n_ancillas, color in ((0, MODEL_M0), (7, MODEL_M7)):
            profiles = _model_profiles(runs_directory, 3, n_ancillas, float(fraction))
            med = np.median(profiles, axis=0)
            lower = np.quantile(profiles, 0.25, axis=0)
            upper = np.quantile(profiles, 0.75, axis=0)
            axis.fill_between(
                np.arange(20),
                lower,
                upper,
                color=color,
                alpha=0.18,
            )
            axis.plot(
                np.arange(20),
                med,
                color=color,
                linewidth=1.3,
                label=f"Model m = {n_ancillas}",
            )
        axis.plot(
            np.arange(20),
            target.shell_probabilities,
            color=TARGET_COLOR,
            linewidth=1.6,
            label="Target",
        )
        if use_tex:
            axis.set_title(rf"Held-out $\alpha/\pi = {fraction:.4f}$")
        else:
            axis.set_title(f"Held-out alpha/pi = {fraction:.4f}")
        axis.set_ylabel("Shell probability")
        axis.set_xticks(range(0, 20, 2))
        axis.set_xlim(-0.5, 19.5)
        axis.grid(axis="x", color="#dddddd", linewidth=0.5, zorder=0)
        axis.tick_params(axis="x", labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    if use_tex:
        for axis in axes:
            axis.set_xlabel(r"Output Hamming weight $k$")
    else:
        for axis in axes:
            axis.set_xlabel("Output Hamming weight k")
    figure.text(
        0.5,
        0.995,
        "3-peak model at a held-out angle: ancillas move the model profile "
        "onto the Target",
        ha="center",
        fontsize=7,
    )
    save_figure(figure, output)


def _variant5_supervision_curve(
    cells: dict,
    interpolation_runs: Path,
    ancilla_runs: Path,
    output: Path,
) -> None:
    """Does more supervision help: held-out TVD versus number of training peaks."""
    use_tex = apply_journal_style()
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(SINGLE_COL_WIDTH, 110 * MM),
        layout="constrained",
    )
    sparse_means: dict[tuple[int, int, str], dict[int, float]] = defaultdict(dict)
    for (n_peaks, n_ancillas), rows in cells.items():
        for metric in ("shell_tvd", "bitstring_tvd"):
            grouped = defaultdict(list)
            for row in rows:
                if row["method"] != "interpolated":
                    continue
                grouped[row["seed"]].append(row[metric])
            sparse_means[(n_peaks, n_ancillas, metric)] = {
                seed: float(np.mean(values)) for seed, values in grouped.items()
            }

    dense_means: dict[tuple[int, str], dict[int, float]] = defaultdict(dict)
    for seed in range(10):
        run = json.loads(
            (interpolation_runs / f"interpolation_complete_seed{seed}.json")
            .read_text(),
        )
        for metric in ("shell_tvd", "bitstring_tvd"):
            values = [
                point[metric] for point in run["points"] if point["role"] == "heldout"
            ]
            dense_means[(0, metric)][seed] = float(np.mean(values))
    for n_ancillas in (2, 7):
        for seed in range(10):
            run = json.loads(
                (
                    ancilla_runs
                    / f"ancilla_{n_ancillas}_broadcast_all_complete_seed{seed}.json"
                ).read_text(),
            )
            dense_means[(n_ancillas, "shell_tvd")][seed] = run["heldout_shell_tvd"]
            dense_means[(n_ancillas, "bitstring_tvd")][seed] = (
                run["heldout_bitstring_tvd"]
            )

    colors = {0: "#0072B2", 2: "#D55E00", 7: "#009E73"}
    markers = {0: "o", 2: "s", 7: "D"}
    for axis, metric, label in (
        (axes[0], "shell_tvd", "Profile TVD"),
        (axes[1], "bitstring_tvd", "Bitstring TVD"),
    ):
        for n_ancillas, color in colors.items():
            x = []
            y = []
            iqr = []
            for n_peaks in (2, 3, 5):
                if n_peaks < 5:
                    values = sparse_means[(n_peaks, n_ancillas, metric)]
                else:
                    values = dense_means[(n_ancillas, metric)]
                med = _median_iqr(list(values.values()))
                x.append(n_peaks)
                y.append(med[0])
                iqr.append([med[0] - med[1], med[2] - med[0]])
            axis.errorbar(
                x,
                y,
                yerr=np.asarray(iqr).T,
                fmt=f"{markers[n_ancillas]}-",
                color=color,
                linewidth=1.2,
                markersize=4,
                capsize=2,
                label=f"Ancillas m = {n_ancillas}",
            )
        axis.set_xticks([2, 3, 5])
        axis.set_ylabel(label)
        axis.tick_params(axis="x", labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    if use_tex:
        for axis in axes:
            axis.set_xlabel(r"Number of trained peaks $K$")
    else:
        for axis in axes:
            axis.set_xlabel("Number of trained peaks K")
    figure.text(
        0.5,
        0.995,
        "Held-out TVD averaged over each grid's own held-out angles; "
        "K=5 uses the dense five-angle grid",
        ha="center",
        fontsize=7,
    )
    save_figure(figure, output)


def main() -> None:
    """Generate the five standalone showcase variants."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cells-cache",
        type=Path,
        default=Path("results/sparse-interpolation/summary/cells.json"),
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("results/sparse-interpolation/runs"),
    )
    parser.add_argument(
        "--interpolation-runs",
        type=Path,
        default=Path("results/interpolation/runs"),
    )
    parser.add_argument(
        "--ancilla-runs",
        type=Path,
        default=Path("results/ancilla/runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sparse-interpolation/variants"),
    )
    args = parser.parse_args()

    cells = _load_cells(args.cells_cache)
    args.output.mkdir(parents=True, exist_ok=True)

    _variant1_target_family(args.output / "variant1_target_family")
    _variant2_heldout_profiles(
        args.runs,
        cells,
        args.output / "variant2_heldout_profiles",
    )
    _variant2_tvd_summary(cells, args.output / "variant2_tvd_summary")
    _variant6_m0_vs_m7(args.runs, cells, args.output / "variant6_m0_vs_m7")
    _variant3_vs_oracle(cells, args.output / "variant3_vs_oracle")
    _variant4_ancilla_effect(args.runs, args.output / "variant4_ancilla_effect")
    _variant5_supervision_curve(
        cells,
        args.interpolation_runs,
        args.ancilla_runs,
        args.output / "variant5_supervision_curve",
    )
    print(f"variants written to {args.output}")


if __name__ == "__main__":
    main()

"""Pilot: 3-peak interpolation with narrower, more separated target peaks.

Subcommands:

  run   --width-scale 0.5 --ancillas 0,7 --seeds 0-4 --output DIR
        train sparse-3 models on the narrow targets and persist runs

  plot  --output DIR
        render the 2x2 held-out m=0/m=7 comparison for every width present
        and print a TVD table
"""

from __future__ import annotations

import argparse
import json
import time
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from journal_style import (
    MM,
    SINGLE_COL_WIDTH,
    apply_journal_style,
    save_figure,
)

from iqp_cond.interpolation import interpolation_targets
from iqp_cond.sparse_interpolation import (
    run_sparse_ancilla,
    save_sparse_result,
)
from iqp_cond.training import TrainingConfig

HELDOUT_FRACTIONS = np.asarray([2, 6], dtype=np.float64) / 32.0
TARGET_COLOR = "#000000"
MODEL_M7 = "#0072B2"
MODEL_M0 = "#56B4E9"
HELDOUT_TINT = "#f7e8c8"


def _parse_seeds(value: str) -> list[int]:
    if "-" in value:
        start, end = (int(part) for part in value.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def _width_suffix(width_scale: float) -> str:
    return "" if width_scale == 1.0 else f"_w{width_scale:g}"


def _run_main(args: argparse.Namespace) -> None:
    config = TrainingConfig(steps=args.steps)
    for n_ancillas in (int(x) for x in args.ancillas.split(",")):
        for seed in _parse_seeds(args.seeds):
            model_name = (
                f"sparse3_ancilla_{n_ancillas}_broadcast_all_complete"
                f"{_width_suffix(args.width_scale)}"
            )
            destination = args.output / f"{model_name}_seed{seed}.json"
            if args.skip_existing and destination.exists():
                print(f"skip {model_name} seed={seed}")
                continue
            started = time.monotonic()
            print(f"run {model_name} seed={seed}", flush=True)
            result, metrics = run_sparse_ancilla(
                n_peaks=3,
                n_ancillas=n_ancillas,
                seed=seed,
                config=config,
                width_scale=args.width_scale,
            )
            save_sparse_result(result, metrics, config, args.output)
            elapsed = time.monotonic() - started
            print(
                f"done {model_name} seed={seed} seconds={elapsed:.1f} "
                f"heldout_shell={metrics['heldout_shell_tvd']:.6f} "
                f"heldout_bit={metrics['heldout_bitstring_tvd']:.6f}",
                flush=True,
            )


def _model_profiles(
    runs_directory: Path,
    n_ancillas: int,
    width_scale: float,
    fraction: float,
) -> np.ndarray:
    pattern = (
        f"sparse3_ancilla_{n_ancillas}_broadcast_all_complete"
        f"{_width_suffix(width_scale)}_seed*.json"
    )
    profiles = []
    for path in sorted(runs_directory.glob(pattern)):
        run = json.loads(path.read_text())
        for point in run["points"]:
            if np.isclose(point["angle_fraction_pi"], fraction):
                profiles.append(np.asarray(point["model_shell_probabilities"]))
    if not profiles:
        raise SystemExit(f"no profiles for m={n_ancillas} w={width_scale} {fraction}")
    return np.asarray(profiles)


def _tvd_table(runs_directory: Path, width_scale: float) -> dict:
    table = {}
    for n_ancillas in (0, 7):
        for fraction in HELDOUT_FRACTIONS:
            pattern = (
                f"sparse3_ancilla_{n_ancillas}_broadcast_all_complete"
                f"{_width_suffix(width_scale)}_seed*.json"
            )
            values = []
            for path in sorted(runs_directory.glob(pattern)):
                run = json.loads(path.read_text())
                for point in run["points"]:
                    if np.isclose(point["angle_fraction_pi"], fraction):
                        values.append(point["shell_tvd"])
            table[(n_ancillas, float(fraction))] = float(np.median(values))
    return table


def _plot_main(args: argparse.Namespace) -> None:
    width_scales = sorted(
        {
            float(run["target_width_scale"])
            for path in sorted(args.output.glob("sparse3_ancilla_*_seed*.json"))
            if (run := json.loads(path.read_text()))["n_peaks"] == 3
        },
    )
    use_tex = apply_journal_style()
    figures_dir = args.output.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    x_label = (
        r"Output Hamming weight $k$" if use_tex else "Output Hamming weight k"
    )
    print(f"\n{'width':>7} {'m':>2} {'frac':>8} {'shell TVD':>10}")
    for width_scale in width_scales:
        table = _tvd_table(args.output, width_scale)
        for (n_ancillas, fraction), tvd in sorted(table.items()):
            print(
                f"{width_scale:>7g} {n_ancillas:>2} {fraction:>8.4f} {tvd:>10.4f}",
            )
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(SINGLE_COL_WIDTH, 118 * MM),
            layout="constrained",
            sharey=True,
        )
        for row, fraction in enumerate(HELDOUT_FRACTIONS):
            for col, n_ancillas in enumerate((0, 7)):
                axis = axes[row, col]
                axis.axvspan(0, 19, color=HELDOUT_TINT, zorder=0)
                target = interpolation_targets(
                    pi * np.asarray([fraction]),
                    width_scale=width_scale,
                )[0]
                profiles = _model_profiles(
                    args.output,
                    n_ancillas,
                    width_scale,
                    float(fraction),
                )
                med = np.median(profiles, axis=0)
                lower = np.quantile(profiles, 0.25, axis=0)
                upper = np.quantile(profiles, 0.75, axis=0)
                color = MODEL_M7 if n_ancillas else MODEL_M0
                axis.fill_between(np.arange(20), lower, upper, color=color, alpha=0.22)
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
                    color=color,
                    linewidth=1.3,
                    label="Model (median)",
                )
                axis.axvline(
                    target.actual_mean,
                    color="#888888",
                    linewidth=0.6,
                    linestyle=":",
                )
                tvd = table[(n_ancillas, float(fraction))]
                axis.text(
                    0.98,
                    0.97,
                    f"Profile TVD = {tvd:.3f}",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.5,
                    color=color,
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
            f"3-peak protocol, target width scale {width_scale:g}: "
            "held-out angles, never trained (gold)",
            ha="center",
            fontsize=7,
        )
        save_figure(figure, figures_dir / f"narrow_w{width_scale:g}_m0_vs_m7")


def main() -> None:
    """Run or plot the narrow-target pilot."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--width-scale", type=float, default=0.5)
    run_parser.add_argument("--ancillas", default="0,7")
    run_parser.add_argument("--seeds", default="0-4")
    run_parser.add_argument("--steps", type=int, default=2_000)
    run_parser.add_argument("--skip-existing", action="store_true")
    run_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sparse-interpolation-narrow/runs"),
    )
    run_parser.set_defaults(func=_run_main)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sparse-interpolation-narrow/runs"),
    )
    plot_parser.set_defaults(func=_plot_main)

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()

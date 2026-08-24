#!/usr/bin/env python3
"""Deterministic paper figures for the two-block and radial-shell screens."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from iqp_cond.asymmetric_mixture import asymmetric_mixture_target
from iqp_cond.exact import enumerate_bitstrings

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"

REPO = Path(__file__).resolve().parents[1]
TWO_BLOCK_ROWS = REPO / "results/two-block-screen-v1/rows"
RADIAL_ROWS = REPO / "results/radial-shell-screen-v1/rows"
OUT = REPO / "paper/symmetry_figures"

TWO_BLOCK_ENCODINGS = ("angle", "spin_flip_branch", "two_block_branch")
TWO_BLOCK_COLORS = {
    "angle": ORANGE,
    "spin_flip_branch": GREEN,
    "two_block_branch": BLUE,
}
TRAIN_BETAS = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
HELD_BETAS = (0.6, 0.8, 1.0, 1.2, 1.4)


def _load_rows(directory: Path, encoding: str) -> list[dict]:
    return [
        json.loads(p.read_text()) for p in (directory / encoding).glob("seed*.json")
    ]


def _mean_se(values: list[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def _save(figure: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _two_block_figure() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    for axis, metric, label in (
        (axes[0], "raw_tvd", "raw bitstring TVD"),
        (axes[1], "profile_tvd", "2D magnetization-profile TVD"),
    ):
        for encoding in TWO_BLOCK_ENCODINGS:
            rows = _load_rows(TWO_BLOCK_ROWS, encoding)
            betas = np.asarray(rows[0]["betas"])
            means = [np.mean([r[metric][i] for r in rows]) for i in range(len(betas))]
            ses = [
                np.std([r[metric][i] for r in rows], ddof=1) / np.sqrt(len(rows))
                for i in range(len(betas))
            ]
            means, ses = np.asarray(means), np.asarray(ses)
            axis.plot(betas, means, color=TWO_BLOCK_COLORS[encoding], label=encoding)
            axis.fill_between(
                betas,
                means - ses,
                means + ses,
                color=TWO_BLOCK_COLORS[encoding],
                alpha=0.18,
            )
            held = [b in HELD_BETAS for b in betas]
            axis.scatter(
                betas[held],
                means[held],
                color=TWO_BLOCK_COLORS[encoding],
                marker="o",
                facecolors="none",
                s=18,
                zorder=4,
            )
        axis.set_xlabel(r"inverse temperature $\beta$")
        axis.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    _save(figure, "two_block_screen")


def _radial_figure() -> None:
    # Import the screen script's helpers for a single-seed re-evaluation.
    spec = importlib.util.spec_from_file_location(
        "radial_screen", REPO / "scripts/run_radial_shell_screen.py"
    )
    radial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(radial)

    encodings = ("angle", "spin_flip_branch", "radial_shell")
    colors = {"angle": ORANGE, "spin_flip_branch": GREEN, "radial_shell": BLUE}
    names = {
        "angle": "angle",
        "spin_flip_branch": "two-branch (Z$_2$)",
        "radial_shell": "radial shell (S$_n$)",
    }

    figure, axes = plt.subplots(
        1, 2, figsize=(7.1, 2.7), gridspec_kw={"width_ratios": [1.0, 1.35]}
    )

    # (a) held-out raw vs profile TVD bars
    x = np.arange(len(encodings))
    width = 0.36
    for offset, metric, label, color in (
        (-width / 2, "raw_tvd", "raw TVD", ORANGE),
        (width / 2, "profile_tvd", "weight-profile TVD\n(= perm.-averaged raw)", BLUE),
    ):
        means, ses = [], []
        for encoding in encodings:
            rows = _load_rows(RADIAL_ROWS, encoding)
            vals = [np.mean(r[metric]) for r in rows]
            means.append(np.mean(vals))
            ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        axes[0].bar(
            x + offset, means, width, yerr=ses, color=color, alpha=0.85, label=label
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([names[e] for e in encodings], fontsize=6.5)
    axes[0].set_ylabel("held-out TVD")
    axes[0].set_ylim(0, 0.55)
    axes[0].legend(frameon=False, fontsize=6)
    axes[0].set_title("(a)", loc="left", fontsize=8)

    # (b) weight profiles at a held-out asymmetric condition
    c1, c2 = 0.5, 0.75
    training, validation = radial.build_datasets(0)
    ansatz = radial.complete_ansatz(radial.N)
    target = asymmetric_mixture_target(radial.N, c1, c2)
    target_profile = np.bincount(
        enumerate_bitstrings(radial.N).sum(axis=1),
        weights=target,
        minlength=radial.N + 1,
    )
    k = np.arange(radial.N + 1)
    axes[1].plot(k, target_profile, color="black", label="target", lw=1.6)
    for encoding in encodings:
        result = radial.train(
            f"plot_{encoding}",
            ansatz,
            radial.condition_angles(encoding, radial.train_grid()),
            training,
            validation,
            0,
            radial.training_config(None, False),
            encoding=radial.ENCODINGS[encoding],
        )
        angles = radial.condition_angles(encoding, [(c1, c2)])
        probs = radial.exact_iqp_probabilities_many(
            result.best_params,
            np.asarray(ansatz.generators, dtype=np.int8),
            angles,
            encoding=radial.ENCODINGS[encoding],
        )[0]
        axes[1].plot(
            k,
            radial.weight_profile(probs),
            color=colors[encoding],
            label=names[encoding],
            lw=1.2,
        )
    axes[1].set_xlabel("Hamming weight $k$")
    axes[1].set_ylabel("probability")
    axes[1].set_title(rf"(b) held-out $(c_1,c_2)=({c1},{c2})$", loc="left", fontsize=8)
    axes[1].legend(frameon=False, fontsize=6)

    figure.tight_layout()
    _save(figure, "radial_shell")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    _two_block_figure()
    _radial_figure()
    print(f"wrote figures to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

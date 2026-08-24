"""Publication figure generator for Experiment 1 (Matched-Capacity Control)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.matched_capacity import (
    ALL_BETAS,
    TRAINING_BETAS,
    VISIBLE_QUBITS,
    compute_physical_observables,
)

ROOT = Path(__file__).parents[1]
RESULTS_PATH = ROOT / "results" / "matched-capacity-v1" / "summary.json"
OUTPUT_FIG_PDF = (
    ROOT
    / "paper"
    / "curie_weiss_ancilla_figures"
    / "matched_capacity_comparison.pdf"
)
OUTPUT_FIG_PNG = (
    ROOT
    / "paper"
    / "curie_weiss_ancilla_figures"
    / "matched_capacity_comparison.png"
)


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"Results file not found: {RESULTS_PATH}. Run the experiment first.")
        return

    OUTPUT_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS_PATH.read_text())
    runs = data["runs"]

    models = ["ancilla_5", "visible_matched_91", "visible_2local_36"]
    labels = {
        "ancilla_5": "Latent Ancilla ($N=13, r=5$, 91 params, mixed)",
        "visible_matched_91": "Matched Pure Visible ($n=8, k\\leq 3$, 91 params)",
        "visible_2local_36": "Standard Pure Visible ($n=8, k\\leq 2$, 36 params)",
    }
    colors = {
        "ancilla_5": "#0072B2",  # blue
        "visible_matched_91": "#D55E00",  # orange
        "visible_2local_36": "#7F7F7F",  # gray
    }
    markers = {
        "ancilla_5": "o",
        "visible_matched_91": "s",
        "visible_2local_36": "^",
    }

    betas = np.asarray(ALL_BETAS)

    # Compute exact physical observables for target
    target_mag = []
    target_binder = []
    for b in betas:
        tgt = exact_curie_weiss_target(VISIBLE_QUBITS, float(b))
        obs = compute_physical_observables(tgt.probabilities, VISIBLE_QUBITS)
        target_mag.append(obs["abs_magnetization"])
        target_binder.append(obs["binder_cumulant"])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.5,
        }
    )

    _fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

    # Panel (a): Profile TVD
    ax = axes[0, 0]
    for m in models:
        recs = runs[m]
        # shape (n_seeds, n_betas)
        prof_mat = np.asarray(
            [[r["per_beta_profile_tvd"][str(b)] for b in betas] for r in recs]
        )
        mean = prof_mat.mean(axis=0)
        std = prof_mat.std(axis=0)
        ax.plot(
            betas,
            mean,
            label=labels[m],
            color=colors[m],
            marker=markers[m],
            markeredgewidth=1.0,
        )
        ax.fill_between(
            betas,
            np.maximum(0, mean - std),
            mean + std,
            color=colors[m],
            alpha=0.15,
        )

    ax.axvline(
        1.0,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=r"Critical Point $\beta_c=1.0$",
    )
    ax.set_ylabel("Profile TVD (lower is better)")
    ax.set_title(
        r"\textbf{a} Magnetization Profile TVD across Temperature",
        loc="left",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper left")

    # Panel (b): Raw Bitstring TVD
    ax = axes[0, 1]
    for m in models:
        recs = runs[m]
        raw_mat = np.asarray(
            [[r["per_beta_raw_tvd"][str(b)] for b in betas] for r in recs]
        )
        mean = raw_mat.mean(axis=0)
        std = raw_mat.std(axis=0)
        ax.plot(
            betas,
            mean,
            label=labels[m],
            color=colors[m],
            marker=markers[m],
            markeredgewidth=1.0,
        )
        ax.fill_between(
            betas,
            np.maximum(0, mean - std),
            mean + std,
            color=colors[m],
            alpha=0.15,
        )

    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_ylabel("Raw Bitstring TVD (lower is better)")
    ax.set_title(
        r"\textbf{b} Raw Bitstring TVD across Temperature",
        loc="left",
    )
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel (c): Absolute Magnetization <|M|>
    ax = axes[1, 0]
    ax.plot(
        betas,
        target_mag,
        label="Exact Gibbs Target",
        color="black",
        linestyle="--",
        linewidth=2.0,
        zorder=5,
    )
    for m in models:
        recs = runs[m]
        mag_mat = np.asarray(
            [[r["per_beta_magnetization"][str(b)] for b in betas] for r in recs]
        )
        mean = mag_mat.mean(axis=0)
        std = mag_mat.std(axis=0)
        ax.plot(
            betas,
            mean,
            label=labels[m],
            color=colors[m],
            marker=markers[m],
            markeredgewidth=1.0,
        )
        ax.fill_between(
            betas,
            mean - std,
            mean + std,
            color=colors[m],
            alpha=0.15,
        )

    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel(r"Inverse Temperature $\beta$")
    ax.set_ylabel(r"Order Parameter $\langle |M| \rangle$")
    ax.set_title(
        r"\textbf{c} Spontaneous Magnetization Order Parameter",
        loc="left",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper left")

    # Panel (d): Binder Cumulant U4
    ax = axes[1, 1]
    ax.plot(
        betas,
        target_binder,
        label="Exact Gibbs Target",
        color="black",
        linestyle="--",
        linewidth=2.0,
        zorder=5,
    )
    for m in models:
        recs = runs[m]
        binder_mat = np.asarray(
            [
                [r["per_beta_binder_cumulant"][str(b)] for b in betas]
                for r in recs
            ]
        )
        mean = binder_mat.mean(axis=0)
        std = binder_mat.std(axis=0)
        ax.plot(
            betas,
            mean,
            label=labels[m],
            color=colors[m],
            marker=markers[m],
            markeredgewidth=1.0,
        )
        ax.fill_between(
            betas,
            mean - std,
            mean + std,
            color=colors[m],
            alpha=0.15,
        )

    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel(r"Inverse Temperature $\beta$")
    ax.set_ylabel(
        r"Binder Cumulant $U_4$"
    )
    ax.set_title(
        r"\textbf{d} Binder Cumulant (Phase Transition Signature)",
        loc="left",
    )
    ax.grid(True, linestyle=":", alpha=0.6)

    # Highlight training vs held-out ticks
    for ax in axes.ravel():
        ax.set_xticks(betas)
        # Bold or custom ticks
        ax.set_xticklabels(
            [
                f"{b:.1f}" if b in TRAINING_BETAS else f"*{b:.1f}"
                for b in betas
            ]
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_PDF, bbox_inches="tight")
    plt.savefig(OUTPUT_FIG_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to:\n  - {OUTPUT_FIG_PDF}\n  - {OUTPUT_FIG_PNG}")


if __name__ == "__main__":
    main()

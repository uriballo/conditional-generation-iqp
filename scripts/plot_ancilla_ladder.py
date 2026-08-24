"""Publication figure generator for Experiment 3 (Ancilla Ladder Capacity Sweep)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from iqp_cond.ancilla_ladder import ALL_BETAS

ROOT = Path(__file__).parents[1]
RESULTS_PATH = ROOT / "results" / "ancilla-ladder-v1" / "summary.json"
OUTPUT_FIG_PDF = (
    ROOT
    / "paper"
    / "curie_weiss_ancilla_figures"
    / "ancilla_ladder_scaling.pdf"
)
OUTPUT_FIG_PNG = (
    ROOT
    / "paper"
    / "curie_weiss_ancilla_figures"
    / "ancilla_ladder_scaling.png"
)


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"Results file not found: {RESULTS_PATH}. Run the experiment first.")
        return

    OUTPUT_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS_PATH.read_text())
    summary = data["summary"]
    rungs = [int(k) for k in summary.keys()]
    runs = data["runs"]
    betas = np.asarray(ALL_BETAS)

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

    _fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel (a): Error scaling vs Ancilla Count r
    ax = axes[0]
    ho_prof_mean = [summary[str(r)]["heldout_profile_tvd_mean"] for r in rungs]
    ho_prof_std = [summary[str(r)]["heldout_profile_tvd_std"] for r in rungs]
    cold_prof_mean = [summary[str(r)]["cold_profile_tvd_mean"] for r in rungs]
    cold_prof_std = [summary[str(r)]["cold_profile_tvd_std"] for r in rungs]

    ax.errorbar(
        rungs,
        ho_prof_mean,
        yerr=ho_prof_std,
        label=r"Held-Out Profile TVD (Mean)",
        color="#0072B2",
        marker="o",
        capsize=3,
    )
    ax.errorbar(
        rungs,
        cold_prof_mean,
        yerr=cold_prof_std,
        label=r"Cold Endpoint TVD ($\beta=1.5$)",
        color="#D55E00",
        marker="s",
        capsize=3,
    )

    ax.set_xlabel("Number of Latent Ancillas $r$")
    ax.set_ylabel("Profile TVD (lower is better)")
    ax.set_title(
        r"\textbf{a} Fidelity Scaling vs Ancilla Count $r$",
        loc="left",
    )
    ax.set_xticks(rungs)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper right")

    # Panel (b): Von Neumann Entropy across Temperature
    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    for idx, r in enumerate(rungs):
        recs = runs[str(r)]
        ent_mat = np.asarray(
            [[rec["per_beta_entropy"][str(b)] for b in betas] for rec in recs]
        )
        mean_ent = ent_mat.mean(axis=0)
        color = cmap(idx / (len(rungs) - 1))
        ax.plot(
            betas,
            mean_ent,
            label=f"$r={r}$",
            color=color,
            marker="o",
            markersize=4,
        )

    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.6, label=r"$\beta_c=1.0$")
    ax.set_xlabel(r"Inverse Temperature $\beta$")
    ax.set_ylabel(r"Visible Entropy $S(\rho_{\mathrm{vis}})$ [nats]")
    ax.set_title(
        r"\textbf{b} Visible Mixed-State Entropy vs $\beta$",
        loc="left",
    )
    ax.set_xticks(betas)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper left", ncol=2)

    # Panel (c): Effective Rank vs Theoretical Bound
    ax = axes[2]
    eff_ranks = [summary[str(r)]["mean_effective_rank"] for r in rungs]
    theoretical_bounds = [2**r for r in rungs]

    ax.plot(
        rungs,
        theoretical_bounds,
        label=r"Theoretical Max Rank $2^r$",
        color="gray",
        linestyle="--",
        marker="^",
        alpha=0.8,
    )
    ax.plot(
        rungs,
        eff_ranks,
        label=r"Empirical Effective Rank $R_{\mathrm{eff}}$",
        color="#009E73",
        marker="D",
        linewidth=2.0,
    )

    ax.set_xlabel("Number of Latent Ancillas $r$")
    ax.set_ylabel("Visible Density Matrix Rank")
    ax.set_title(
        r"\textbf{c} Effective Rank Saturation",
        loc="left",
    )
    ax.set_yscale("log")
    ax.set_xticks(rungs)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_PDF, bbox_inches="tight")
    plt.savefig(OUTPUT_FIG_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to:\n  - {OUTPUT_FIG_PDF}\n  - {OUTPUT_FIG_PNG}")


if __name__ == "__main__":
    main()

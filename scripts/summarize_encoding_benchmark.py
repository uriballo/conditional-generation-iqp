"""Summarize matched-seed encoding benchmark results,
generate LaTeX tables and figures.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from iqp_cond.encoding import (
    ANGLE_ENCODING,
    BLOCK_ZZ_PI4_ENCODING,
    BLOCK_ZZ_PI8_ENCODING,
    CALIBRATED_REUPLOAD_ENCODING,
    SYMMETRIC_REUPLOAD_ENCODING,
)

ENCODING_DISPLAY_NAMES = {
    ANGLE_ENCODING: "Commuting angle ($L=1$)",
    SYMMETRIC_REUPLOAD_ENCODING: "Symmetric reupload ($L=2$)",
    CALIBRATED_REUPLOAD_ENCODING: "Calibrated noncommuting ($L=2$)",
    BLOCK_ZZ_PI8_ENCODING: "2-Qubit block $ZZ(\\pi/8)$ ($b=2$)",
    BLOCK_ZZ_PI4_ENCODING: "2-Qubit block $ZZ(\\pi/4)$ ($b=2$)",
}

ENCODING_SHORT_NAMES = {
    ANGLE_ENCODING: "Angle",
    SYMMETRIC_REUPLOAD_ENCODING: "Symm. reupload",
    CALIBRATED_REUPLOAD_ENCODING: "Noncommuting ($L=2$)",
    BLOCK_ZZ_PI8_ENCODING: "Block $ZZ(\\pi/8)$",
    BLOCK_ZZ_PI4_ENCODING: "Block $ZZ(\\pi/4)$",
}

ORDERED_ENCODINGS = [
    ANGLE_ENCODING,
    CALIBRATED_REUPLOAD_ENCODING,
    BLOCK_ZZ_PI8_ENCODING,
    BLOCK_ZZ_PI4_ENCODING,
]


def _bootstrap_ci(
    data: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, lower_ci, upper_ci) using percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    resamples = rng.choice(data, size=(n_resamples, n), replace=True)
    resampled_means = np.mean(resamples, axis=1)
    mean_val = float(np.mean(data))
    low = float(np.quantile(resampled_means, alpha / 2.0))
    high = float(np.quantile(resampled_means, 1.0 - alpha / 2.0))
    return mean_val, low, high


def _load_runs(directory: Path) -> dict[str, dict[int, dict[str, object]]]:
    runs: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        encoding = payload.get("encoding", "angle")
        seed = int(payload["seed"])
        runs[encoding][seed] = payload
    return runs


def _extract_seed_metrics(payload: dict[str, object]) -> dict[str, float]:
    heldout = [p for p in payload["points"] if p["role"] == "heldout"]
    return {
        "heldout_mmd": float(payload["heldout_mmd"]),
        "shell_tvd": float(np.mean([p["shell_tvd"] for p in heldout])),
        "bitstring_tvd": float(np.mean([p["bitstring_tvd"] for p in heldout])),
        "mean_error": float(np.mean([p["mean_absolute_error"] for p in heldout])),
        "std_error": float(np.mean([p["std_absolute_error"] for p in heldout])),
    }


def generate_summary(
    runs_dir: Path,
    summary_dir: Path,
    paper_dir: Path | None = None,
) -> dict[str, object]:
    runs = _load_runs(runs_dir)
    available_encodings = [
        enc for enc in ORDERED_ENCODINGS if enc in runs and len(runs[enc]) > 0
    ]
    if not available_encodings:
        raise ValueError(f"No valid runs found in {runs_dir}")

    # Find common seeds across available encodings
    seed_sets = [set(runs[enc].keys()) for enc in available_encodings]
    common_seeds = sorted(set.intersection(*seed_sets))
    if not common_seeds:
        # Fallback to any seeds
        common_seeds = sorted(set.union(*seed_sets))

    print(f"Summarizing encodings: {available_encodings} on seeds {common_seeds}")

    metrics_by_enc: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_seed_data: list[dict[str, object]] = []

    for seed in common_seeds:
        seed_entry: dict[str, object] = {"seed": seed}
        for enc in available_encodings:
            if seed in runs[enc]:
                m = _extract_seed_metrics(runs[enc][seed])
                seed_entry[enc] = m
                for k, v in m.items():
                    metrics_by_enc[enc][k].append(v)
        per_seed_data.append(seed_entry)

    # Compute aggregate stats with bootstrap CIs
    stats: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for enc in available_encodings:
        for metric_name, values_list in metrics_by_enc[enc].items():
            arr = np.asarray(values_list, dtype=np.float64)
            mean_v, low_ci, high_ci = _bootstrap_ci(arr)
            stats[enc][metric_name] = {
                "mean": mean_v,
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "ci_95_low": low_ci,
                "ci_95_high": high_ci,
            }

    # Paired differences relative to ANGLE_ENCODING
    paired_diffs: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    win_rates: dict[str, dict[str, float]] = defaultdict(dict)
    if ANGLE_ENCODING in available_encodings:
        for enc in available_encodings:
            if enc == ANGLE_ENCODING:
                continue
            for metric_name in ["shell_tvd", "bitstring_tvd", "heldout_mmd"]:
                base_vals = np.asarray(metrics_by_enc[ANGLE_ENCODING][metric_name])
                enc_vals = np.asarray(metrics_by_enc[enc][metric_name])
                diff = enc_vals - base_vals
                mean_d, low_d, high_d = _bootstrap_ci(diff)
                paired_diffs[enc][metric_name] = {
                    "mean_diff": mean_d,
                    "ci_95_low": low_d,
                    "ci_95_high": high_d,
                }
                win_rates[enc][metric_name] = float(np.mean(diff < 0))

    summary_payload = {
        "encodings": available_encodings,
        "matched_seeds": common_seeds,
        "n_seeds": len(common_seeds),
        "statistics": stats,
        "paired_differences_vs_angle": paired_diffs,
        "win_rates_vs_angle": win_rates,
        "per_seed": per_seed_data,
    }

    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "metrics.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n"
    )

    # Generate Markdown Summary
    md_lines = [
        "# Tractable Noncommuting and Block Encoding Benchmark",
        "",
        (
            f"Matched seeds: {', '.join(map(str, common_seeds))} "
            f"($N={len(common_seeds)}$). Lower is better."
        ),
        "",
        (
            "| Encoding | Family | Held-Out Profile TVD | Held-Out Bitstring TVD | "
            "$\\Delta$ Profile TVD vs Angle | Win Rate |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]
    for enc in available_encodings:
        s = stats[enc]
        name = ENCODING_DISPLAY_NAMES.get(enc, enc)
        shell_str = (
            f"{s['shell_tvd']['mean']:.4f} "
            f"[{s['shell_tvd']['ci_95_low']:.4f}, {s['shell_tvd']['ci_95_high']:.4f}]"
        )
        bit_str = (
            f"{s['bitstring_tvd']['mean']:.4f} "
            f"[{s['bitstring_tvd']['ci_95_low']:.4f}, "
            f"{s['bitstring_tvd']['ci_95_high']:.4f}]"
        )
        if enc == ANGLE_ENCODING:
            diff_str = "Baseline"
            win_str = "-"
            family_str = "Commuting ($L=1$)"
        else:
            d = paired_diffs[enc]["shell_tvd"]
            diff_str = (
                f"{d['mean_diff']:+.4f} "
                f"[{d['ci_95_low']:+.4f}, {d['ci_95_high']:+.4f}]"
            )
            n_wins = int(win_rates[enc]["shell_tvd"] * len(common_seeds))
            win_str = f"{n_wins}/{len(common_seeds)}"
            family_str = (
                "Noncommuting ($L=2$)"
                if "reupload" in enc
                else "Entangling Block ($b=2$)"
            )
        md_lines.append(
            f"| {name} | {family_str} | {shell_str} | {bit_str} | {diff_str} | "
            f"{win_str} |"
        )

    (summary_dir / "metrics.md").write_text("\n".join(md_lines) + "\n")

    # Generate LaTeX Table
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\hline\hline",
        (
            r"Encoding Family & Layers/Block & Held-Out Profile TVD & "
            r"Held-Out Bitstring TVD & Paired $\Delta$ Profile TVD \\"
        ),
        r"\hline",
    ]
    for enc in available_encodings:
        s = stats[enc]
        name = ENCODING_SHORT_NAMES.get(enc, enc)
        shell_str = (
            f"${s['shell_tvd']['mean']:.3f} \\pm {s['shell_tvd']['std']:.3f}$"
        )
        bit_str = (
            f"${s['bitstring_tvd']['mean']:.3f} \\pm {s['bitstring_tvd']['std']:.3f}$"
        )
        if enc == ANGLE_ENCODING:
            family_col = "$L=1$"
            diff_col = "Baseline"
        elif "reupload" in enc:
            family_col = "$L=2$"
            d = paired_diffs[enc]["shell_tvd"]
            diff_col = (
                f"${d['mean_diff']:+.3f} \\; "
                f"[{d['ci_95_low']:+.3f}, {d['ci_95_high']:+.3f}]$"
            )
        else:
            family_col = "$b=2$"
            d = paired_diffs[enc]["shell_tvd"]
            diff_col = (
                f"${d['mean_diff']:+.3f} \\; "
                f"[{d['ci_95_low']:+.3f}, {d['ci_95_high']:+.3f}]$"
            )
        tex_lines.append(
            f"{name} & {family_col} & {shell_str} & {bit_str} & {diff_col} \\\\"
        )
    tex_lines.extend(
        [
            r"\hline\hline",
            r"\end{tabular}",
            (
                r"\caption{Held-out generalization comparison across "
                r"single-qubit noncommuting reuploading ($L=2$) and 2-qubit "
                r"block-entangled encodings ($b=2$, $\kappa \in \{\pi/8, \pi/4\}$) "
                r"on the $n=19$ Gaussian interpolation benchmark across matched "
                r"seeds. All encodings retain variance bound "
                r"$\mathbb{E}[|R|^2] \le 1$. 95\% confidence intervals are "
                r"computed via $10^4$ percentile bootstrap resamples.}"
            ),
            r"\label{tab:encoding_benchmark}",
            r"\end{table*}",
            "",
        ]
    )

    latex_content = "\n".join(tex_lines)
    (summary_dir / "encoding_benchmark_table.tex").write_text(latex_content)
    if paper_dir is not None:
        gen_dir = paper_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "encoding_benchmark_table.tex").write_text(latex_content)

    # Generate Publication Figure
    try:
        plot_encoding_benchmark(
            runs,
            available_encodings,
            common_seeds,
            summary_dir / "encoding_benchmark.png",
        )
        if paper_dir is not None:
            fig_dir = paper_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            plot_encoding_benchmark(
                runs,
                available_encodings,
                common_seeds,
                fig_dir / "encoding_benchmark.pdf",
            )
    except Exception as err:
        print(f"Notice: Plot generation skipped ({err})")

    return summary_payload


def plot_encoding_benchmark(
    runs: dict[str, dict[int, dict[str, object]]],
    encodings: list[str],
    seeds: list[int],
    output_path: Path,
) -> None:
    import matplotlib.font_manager

    # Avoid macOS font profiler KeyError in sandboxed environments
    matplotlib.font_manager._get_macos_fonts = lambda: []
    matplotlib.font_manager.findSystemFonts = lambda *args, **kwargs: []

    import matplotlib.pyplot as plt
    try:
        from scripts.journal_style import (
            DOUBLE_COL_WIDTH,
            MM,
            apply_journal_style,
            save_figure,
        )
    except ImportError:
        from journal_style import (
            DOUBLE_COL_WIDTH,
            MM,
            apply_journal_style,
            save_figure,
        )

    apply_journal_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COL_WIDTH, 60 * MM), layout="constrained"
    )

    colors = {
        ANGLE_ENCODING: "#000000",
        CALIBRATED_REUPLOAD_ENCODING: "#0072B2",
        BLOCK_ZZ_PI8_ENCODING: "#009E73",
        BLOCK_ZZ_PI4_ENCODING: "#D55E00",
        SYMMETRIC_REUPLOAD_ENCODING: "#CC79A7",
    }
    markers = {
        ANGLE_ENCODING: "o",
        CALIBRATED_REUPLOAD_ENCODING: "s",
        BLOCK_ZZ_PI8_ENCODING: "^",
        BLOCK_ZZ_PI4_ENCODING: "D",
        SYMMETRIC_REUPLOAD_ENCODING: "v",
    }

    heldout_fractions = [1 / 32, 3 / 32, 5 / 32, 7 / 32]

    # Panel (a): Profile TVD per held-out angle
    for enc in encodings:
        angle_tvds: dict[float, list[float]] = defaultdict(list)
        for seed in seeds:
            if seed in runs[enc]:
                for pt in runs[enc][seed]["points"]:
                    if pt["role"] == "heldout":
                        angle_tvds[float(pt["angle_fraction_pi"])].append(
                            pt["shell_tvd"]
                        )

        sorted_fracs = sorted(angle_tvds.keys())
        means = [float(np.mean(angle_tvds[f])) for f in sorted_fracs]
        errs = [
            float(np.std(angle_tvds[f], ddof=1)) / np.sqrt(len(seeds))
            for f in sorted_fracs
        ]

        axes[0].errorbar(
            sorted_fracs,
            means,
            yerr=errs,
            fmt=f"{markers.get(enc, 'o')}-",
            color=colors.get(enc, "#333333"),
            label=ENCODING_SHORT_NAMES.get(enc, enc),
            linewidth=1.2,
            markersize=4,
            capsize=2,
        )

    # Panel (b): Raw Bitstring TVD per held-out angle
    for enc in encodings:
        angle_bit_tvds: dict[float, list[float]] = defaultdict(list)
        for seed in seeds:
            if seed in runs[enc]:
                for pt in runs[enc][seed]["points"]:
                    if pt["role"] == "heldout":
                        angle_bit_tvds[float(pt["angle_fraction_pi"])].append(
                            pt["bitstring_tvd"]
                        )

        sorted_fracs = sorted(angle_bit_tvds.keys())
        means = [float(np.mean(angle_bit_tvds[f])) for f in sorted_fracs]
        errs = [
            float(np.std(angle_bit_tvds[f], ddof=1)) / np.sqrt(len(seeds))
            for f in sorted_fracs
        ]

        axes[1].errorbar(
            sorted_fracs,
            means,
            yerr=errs,
            fmt=f"{markers.get(enc, 'o')}-",
            color=colors.get(enc, "#333333"),
            label=ENCODING_SHORT_NAMES.get(enc, enc),
            linewidth=1.2,
            markersize=4,
            capsize=2,
        )

    axes[0].set_ylabel("Profile TVD")
    axes[1].set_ylabel("Raw Bitstring TVD")

    for _i, ax in enumerate(axes):
        ax.set_xlabel(r"Held-Out Angle $\alpha/\pi$")
        ax.set_xticks(heldout_fractions)
        ax.set_xticklabels(["1/32", "3/32", "5/32", "7/32"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[0].set_title("(a) Profile Generalization")
    axes[1].set_title("(b) Bitstring Generalization")

    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        fontsize=7,
    )
    save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("results/encoding_benchmark/runs"),
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("results/encoding_benchmark/summary"),
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper"),
    )
    args = parser.parse_args()
    summary = generate_summary(args.runs_dir, args.summary_dir, args.paper_dir)
    print(f"Generated summary for {summary['n_seeds']} matched seeds.")


if __name__ == "__main__":
    main()

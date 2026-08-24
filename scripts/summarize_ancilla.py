"""Summarize ancilla-assisted conditional IQP Born machine results."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Configure journal publication-quality style defaults
use_tex = shutil.which("latex") is not None

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.prop_cycle": mpl.cycler(
        "color",
        [
            "#0072B2",  # Blue
            "#D55E00",  # Vermillion
            "#009E73",  # Bluish Green
            "#E69F00",  # Orange
            "#CC79A7",  # Reddish Purple
            "#56B4E9",  # Sky Blue
        ],
    ),
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "lines.markersize": 4,
    "errorbar.capsize": 3,
    "axes.xmargin": 0.02,
    "axes.ymargin": 0.05,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    **(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
        }
        if use_tex
        else {"text.usetex": False, "mathtext.fontset": "cm"}
    ),
})

mm = 1 / 25.4


def _median_iqr(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
    )


def summarize_ancilla_runs(
    ancilla_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    control_dir: Path | None = None,
) -> None:
    """Summarize ancilla runs and generate comparisons against 0-ancilla baseline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ancilla_runs = [
        json.loads(p.read_text()) for p in sorted(ancilla_dir.glob("*.json"))
    ]
    baseline_runs = [
        json.loads(p.read_text()) for p in sorted(baseline_dir.glob("*.json"))
    ]
    control_runs = (
        [json.loads(p.read_text()) for p in sorted(control_dir.glob("*.json"))]
        if control_dir is not None and control_dir.exists()
        else []
    )

    grouped_by_ancilla = defaultdict(list)
    for run in ancilla_runs:
        grouped_by_ancilla[run["n_ancillas"]].append(run)

    grouped_by_control_params = defaultdict(list)
    for run in control_runs:
        grouped_by_control_params[run["n_parameters"]].append(run)

    lines = [
        "# Ancilla-Assisted Conditional IQP Study Summary",
        "",
        "| Ancillas ($m$) | Qubits ($N$) | Params | Held-out Profile TVD | "
        "Held-out Bitstring TVD | End. A Profile TVD | End. B Profile TVD |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    latex_lines = []

    # Baseline m=0 entry
    if baseline_runs:
        heldout_shells = []
        heldout_bits = []
        end_a_shells = []
        end_b_shells = []
        for r in baseline_runs:
            heldout_points = [p for p in r["points"] if p["role"] == "heldout"]
            heldout_shells.append(np.mean([p["shell_tvd"] for p in heldout_points]))
            heldout_bits.append(np.mean([p["bitstring_tvd"] for p in heldout_points]))
            end_a_shells.append(r["points"][0]["shell_tvd"])
            end_b_shells.append(r["points"][4]["shell_tvd"])

        hs = _median_iqr(heldout_shells)
        hb = _median_iqr(heldout_bits)
        ea = _median_iqr(end_a_shells)
        eb = _median_iqr(end_b_shells)
        lines.append(
            f"| 0 | 19 | 190 | {hs[0]:.4f} [{hs[1]:.4f}, {hs[2]:.4f}] | "
            f"{hb[0]:.4f} [{hb[1]:.4f}, {hb[2]:.4f}] | "
            f"{ea[0]:.4f} [{ea[1]:.4f}, {ea[2]:.4f}] | "
            f"{eb[0]:.4f} [{eb[1]:.4f}, {eb[2]:.4f}] |",
        )
        latex_lines.append(
            f"0 & 19 & 190 & {hb[0]:.3f} & "
            f"{hs[0]:.3f} & "
            f"{ea[0]:.3f} & "
            f"{eb[0]:.3f} \\\\",
        )

    for m_val in sorted(grouped_by_ancilla):
        runs = grouped_by_ancilla[m_val]
        hs = _median_iqr([r["heldout_shell_tvd"] for r in runs])
        hb = _median_iqr([r["heldout_bitstring_tvd"] for r in runs])
        ea = _median_iqr([r["train_shell_tvd_endpoint_a"] for r in runs])
        eb = _median_iqr([r["train_shell_tvd_endpoint_b"] for r in runs])
        n_params = runs[0]["n_parameters"]
        total_qubits = runs[0]["total_qubits"]
        lines.append(
            f"| {m_val} | {total_qubits} | {n_params} | "
            f"{hs[0]:.4f} [{hs[1]:.4f}, {hs[2]:.4f}] | "
            f"{hb[0]:.4f} [{hb[1]:.4f}, {hb[2]:.4f}] | "
            f"{ea[0]:.4f} [{ea[1]:.4f}, {ea[2]:.4f}] | "
            f"{eb[0]:.4f} [{eb[1]:.4f}, {eb[2]:.4f}] |",
        )
        latex_lines.append(
            f"{m_val} & {total_qubits} & {n_params} & "
            f"{hb[0]:.3f} & "
            f"{hs[0]:.3f} & "
            f"{ea[0]:.3f} & "
            f"{eb[0]:.3f} \\\\",
        )

    for n_params in sorted(grouped_by_control_params):
        runs = grouped_by_control_params[n_params]
        hs = _median_iqr([r["heldout_shell_tvd"] for r in runs])
        hb = _median_iqr([r["heldout_bitstring_tvd"] for r in runs])
        ea = _median_iqr([r["train_shell_tvd_endpoint_a"] for r in runs])
        eb = _median_iqr([r["train_shell_tvd_endpoint_b"] for r in runs])
        lines.append(
            f"| 0$^{{*}}$ | 19 | {n_params} | "
            f"{hs[0]:.4f} [{hs[1]:.4f}, {hs[2]:.4f}] | "
            f"{hb[0]:.4f} [{hb[1]:.4f}, {hb[2]:.4f}] | "
            f"{ea[0]:.4f} [{ea[1]:.4f}, {ea[2]:.4f}] | "
            f"{eb[0]:.4f} [{eb[1]:.4f}, {eb[2]:.4f}] |",
        )
    if grouped_by_control_params:
        lines.append("")
        lines.append(
            "$^{*}$Matched-capacity control: m=0 with complete 2-local "
            "plus 3-local generators.",
        )

    latex_lines.append(r"\bottomrule")
    summary_text = "\n".join(lines) + "\n"
    (output_dir / "metrics.md").write_text(summary_text)
    (output_dir / "ancilla_table.tex").write_text("\n".join(latex_lines) + "\n")
    print(summary_text)

    # Matched-capacity control table: params, control bit TVD, ancilla bit TVD
    control_lines = []
    matched_ancilla = {210: 1, 231: 2, 253: 3, 300: 5, 351: 7}
    for n_params in sorted(grouped_by_control_params):
        runs = grouped_by_control_params[n_params]
        control_bit = _median_iqr(
            [r["heldout_bitstring_tvd"] for r in runs],
        )
        m = matched_ancilla[n_params]
        ancilla_runs = grouped_by_ancilla.get(m, [])
        if not ancilla_runs:
            continue
        ancilla_bit = _median_iqr(
            [r["heldout_bitstring_tvd"] for r in ancilla_runs],
        )
        control_lines.append(
            f"{n_params} & {control_bit[0]:.3f} & {ancilla_bit[0]:.3f} & {m} \\\\",
        )
    if control_lines:
        (output_dir / "control_table.tex").write_text("\n".join(control_lines) + "\n")
    # Journal figure (170 mm width, double column)
    if grouped_by_ancilla and baseline_runs:
        ancilla_counts = [0, *sorted(grouped_by_ancilla.keys())]

        shell_stats = []
        bit_stats = []
        for m_val in ancilla_counts:
            if m_val == 0:
                h_s = [
                    np.mean(
                        [p["shell_tvd"] for p in r["points"] if p["role"] == "heldout"],
                    )
                    for r in baseline_runs
                ]
                h_b = [
                    np.mean(
                        [
                            p["bitstring_tvd"]
                            for p in r["points"]
                            if p["role"] == "heldout"
                        ],
                    )
                    for r in baseline_runs
                ]
            else:
                runs = grouped_by_ancilla[m_val]
                h_s = [r["heldout_shell_tvd"] for r in runs]
                h_b = [r["heldout_bitstring_tvd"] for r in runs]
            shell_stats.append(_median_iqr(h_s))
            bit_stats.append(_median_iqr(h_b))

        shell_arr = np.array(shell_stats)
        bit_arr = np.array(bit_stats)

        control_params = sorted(grouped_by_control_params)
        control_bit_stats = []
        for n_params in control_params:
            runs = grouped_by_control_params[n_params]
            control_bit_stats.append(
                _median_iqr([r["heldout_bitstring_tvd"] for r in runs]),
            )
        control_bit_arr = np.array(control_bit_stats)

        ladder_params = np.asarray(
            [
                190 if m == 0 else grouped_by_ancilla[m][0]["n_parameters"]
                for m in ancilla_counts
            ],
            dtype=float,
        )

        fig, axes = plt.subplots(
            1,
            2 if not grouped_by_control_params else 3,
            figsize=(170 * mm, 60 * mm),
            layout="constrained",
        )

        # Left panel: Bitstring TVD
        y_med = bit_arr[:, 0]
        y_err = np.array([y_med - bit_arr[:, 1], bit_arr[:, 2] - y_med])
        axes[0].errorbar(
            ancilla_counts,
            y_med,
            yerr=y_err,
            fmt="o-",
            color="#0072B2",
            label=r"Bitstring TVD",
        )
        if use_tex:
            axes[0].set_xlabel(r"Ancilla Qubits $m$")
            axes[0].set_ylabel(r"Held-Out Bitstring TVD")
            axes[0].set_title(r"(a) Bitstring TVD vs. Ancillas")
        else:
            axes[0].set_xlabel("Ancilla Qubits m")
            axes[0].set_ylabel("Held-Out Bitstring TVD")
            axes[0].set_title("(a) Bitstring TVD vs. Ancillas")

        axes[0].set_xticks(ancilla_counts)
        axes[0].spines[["top", "right"]].set_visible(False)

        # Right panel: Profile TVD
        y_med_s = shell_arr[:, 0]
        y_err_s = np.array([y_med_s - shell_arr[:, 1], shell_arr[:, 2] - y_med_s])
        axes[1].errorbar(
            ancilla_counts,
            y_med_s,
            yerr=y_err_s,
            fmt="s--",
            color="#D55E00",
            label=r"Profile TVD",
        )
        if use_tex:
            axes[1].set_xlabel(r"Ancilla Qubits $m$")
            axes[1].set_ylabel(r"Held-Out Profile TVD")
            axes[1].set_title(r"(b) Profile TVD vs. Ancillas")
        else:
            axes[1].set_xlabel("Ancilla Qubits m")
            axes[1].set_ylabel("Held-Out Profile TVD")
            axes[1].set_title("(b) Profile TVD vs. Ancillas")

        axes[1].set_xticks(ancilla_counts)
        axes[1].spines[["top", "right"]].set_visible(False)

        # Third panel: bitstring TVD vs. parameter count, ladder vs. m=0 control
        if grouped_by_control_params:
            y_med_c = control_bit_arr[:, 0]
            y_err_c = np.array(
                [y_med_c - control_bit_arr[:, 1], control_bit_arr[:, 2] - y_med_c],
            )
            axes[2].errorbar(
                ladder_params,
                bit_arr[:, 0],
                yerr=np.array(
                    [bit_arr[:, 0] - bit_arr[:, 1], bit_arr[:, 2] - bit_arr[:, 0]],
                ),
                fmt="o-",
                color="#0072B2",
                label=r"Ancilla ladder",
            )
            axes[2].errorbar(
                control_params,
                y_med_c,
                yerr=y_err_c,
                fmt="s--",
                color="#999999",
                label=r"$m{=}0$ 3-local control",
            )
            if use_tex:
                axes[2].set_xlabel(r"Shared Parameters")
                axes[2].set_ylabel(r"Held-Out Bitstring TVD")
                axes[2].set_title(r"(c) Matched-Capacity Control")
            else:
                axes[2].set_xlabel("Shared Parameters")
                axes[2].set_ylabel("Held-Out Bitstring TVD")
                axes[2].set_title("(c) Matched-Capacity Control")
            axes[2].legend(frameon=False, fontsize=8)
            axes[2].spines[["top", "right"]].set_visible(False)

        fig.savefig(output_dir / "ancilla_scaling.png")
        fig.savefig(output_dir / "ancilla_scaling.pdf")
        if use_tex:
            try:
                fig.savefig(output_dir / "ancilla_scaling.pgf", bbox_inches=None)
            except Exception as e:
                print(f"Notice: PGF export skipped ({e})")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("results/ancilla/runs"))
    parser.add_argument(
        "--baseline-runs",
        type=Path,
        default=Path("results/ancilla-baseline/runs"),
    )
    parser.add_argument(
        "--control-runs",
        type=Path,
        default=Path("results/ancilla-control/runs"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/ancilla/summary"))
    args = parser.parse_args()

    summarize_ancilla_runs(
        args.runs,
        args.baseline_runs,
        args.output,
        args.control_runs,
    )


if __name__ == "__main__":
    main()

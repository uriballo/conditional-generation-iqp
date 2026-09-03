#!/usr/bin/env python3
"""Plot the compact, single-column Curie--Weiss encoder ablation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from iqp_cond.artifacts import read_immutable_json, sha256_file
from iqp_cond.curie_encoder_ablation import AblationManifest, plan_runs

ARMS = ("product_generic_affine", "spin_flip_informed_piecewise")
ANCILLAS = (0, 10)
SEEDS = (100, 101, 102)
BETAS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
TRAINING = {0.5, 0.7, 0.9, 1.1, 1.3, 1.5}
MM = 1 / 25.4

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(
            "color",
            ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"],
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
        "axes.ymargin": 0.02,
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
            if shutil.which("latex")
            else {"text.usetex": False, "mathtext.fontset": "cm"}
        ),
    }
)


def load_records(config: Path) -> list[dict]:
    screen = json.loads(config.read_text())
    root = (config.parent.parent / screen["artifacts"]["root"]).resolve()
    parent_path = (config.parent.parent / screen["parent_manifest"]).resolve()
    parent = AblationManifest.from_json(parent_path)
    if parent.digest != screen["parent_manifest_digest"]:
        raise RuntimeError("parent manifest digest mismatch")
    expected = {
        (arm, n, r, seed): run
        for run in plan_runs(parent)
        if (arm := run.arm) in screen["arms"]
        and (n := run.visible_qubits) in screen["visible_qubits"]
        and (r := run.ancillas) in screen["ancillas"]
        and (seed := run.seed) in screen["seeds"]
    }
    records = []
    for _key, run in expected.items():
        path = root / "evaluations" / f"{run.run_id}.json"
        if not path.is_file():
            raise RuntimeError(f"missing evaluation envelope: {path}")
        record = read_immutable_json(path)
        if (
            record.get("run_id") != run.run_id
            or record.get("run_identity") != run.to_dict()
        ):
            raise RuntimeError(f"evaluation identity mismatch: {path}")
        if (
            record.get("manifest_digest") != parent.digest
            or record.get("stage") != "evaluation"
        ):
            raise RuntimeError(f"evaluation protocol mismatch: {path}")
        probability = root / record["probabilities_path"]
        if (
            not probability.is_file()
            or sha256_file(probability) != record["probabilities_sha256"]
        ):
            raise RuntimeError(f"probability artifact digest mismatch: {probability}")
        metrics = record.get("metrics")
        if not isinstance(metrics, list) or {
            float(row["condition"]) for row in metrics
        } != set(BETAS):
            raise RuntimeError(f"incomplete condition set: {path}")
        records.append(record)
    if len(records) != 24:
        raise RuntimeError(f"expected 24 selected evaluations, found {len(records)}")
    return records


def make_plot(records: list[dict], output: Path) -> None:
    grouped: dict[tuple[str, int], dict[float, list[float]]] = {}
    for record in records:
        ident = record["run_identity"]
        key = (ident["arm"], int(ident["ancillas"]))
        grouped.setdefault(key, {beta: [] for beta in BETAS})
        for row in record["metrics"]:
            grouped[key][float(row["condition"])].append(float(row["bitstring_tvd"]))
    fig, ax = plt.subplots(figsize=(85 * MM, 70 * MM), layout="constrained")
    colors = {ARMS[0]: "#0072B2", ARMS[1]: "#D55E00"}
    labels = {ARMS[0]: "Product", ARMS[1]: "Spin-flip"}
    for arm in ARMS:
        for r in ANCILLAS:
            med = []
            for beta in BETAS:
                values = np.asarray(grouped[(arm, r)][beta])
                med.append(np.median(values))
            style = "--" if r == 0 else "-"
            ax.plot(
                BETAS,
                med,
                color=colors[arm],
                linestyle=style,
                marker="o",
                markersize=3,
                markerfacecolor="white",
                markeredgecolor=colors[arm],
                label=f"{labels[arm]}, r={r}",
            )
            for beta, y in zip(BETAS, med, strict=True):
                ax.plot(
                    beta,
                    y,
                    marker="o",
                    markersize=3,
                    linestyle="none",
                    markerfacecolor=(colors[arm] if beta in TRAINING else "white"),
                    markeredgecolor=colors[arm],
                )
    ax.axvline(1.0, color="0.6", linewidth=0.7, zorder=0)
    ax.set(
        xlim=(0.5, 1.5),
        ylim=(0, None),
        xlabel=r"Inverse temperature $\beta$",
        ylabel="Exact bitstring TVD",
    )
    ax.set_xticks((0.5, 0.75, 1.0, 1.25, 1.5))
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=1, loc="upper left", handlelength=2.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/curie_encoder_ablation_n8_screen_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Overleaf/figures/curie_encoder_ablation_n8.pdf"),
    )
    args = parser.parse_args()
    make_plot(load_records(args.config), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

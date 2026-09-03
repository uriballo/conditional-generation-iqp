#!/usr/bin/env python3
"""Render compact exact-TVD campaign figures from a processed summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iqp_cond.exact import enumerate_bitstrings, shell_probabilities  # noqa: E402
from iqp_cond.exact_tvd_campaign import load_manifest  # noqa: E402
from iqp_cond.exact_tvd_results import load_records  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("summary", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--campaign", type=Path)
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/exact_tvd_ancilla_v1.json",
    )
    a = p.parse_args()
    payload = json.loads(a.summary.read_text())
    condition_rows = payload["condition_rows"]
    seed_rows = payload["seed_rows"]
    a.out.mkdir(parents=True, exist_ok=True)
    for name, field, title, xlabel in (
        (
            "ancilla_ladders",
            "conditioned_tvd",
            "Exact visible TVD by ancilla count",
            "ancillas",
        ),
        (
            "interpolation_curves",
            "interpolation_tvd",
            "Held-out interpolation TVD",
            "condition index",
        ),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        groups = {}
        rows = seed_rows if name == "ancilla_ladders" else condition_rows
        for row in rows:
            if name == "ancilla_ladders" and row["group"] != "held_out":
                continue
            if name == "interpolation_curves" and row["split"] != "held_out":
                continue
            value = row.get(field)
            key = (row["dataset"], row["visible_qubits"])
            if value is not None and value == value:
                groups.setdefault(key, {}).setdefault(
                    row["ancillas"] if name == "ancilla_ladders" else row["condition"],
                    [],
                ).append(value)
        for key, points in sorted(groups.items()):
            ax.plot(
                sorted(points),
                [sum(points[x]) / len(points[x]) for x in sorted(points)],
                marker="o",
                label=f"{key[0]} n={key[1]}",
            )
        ax.set(title=title, xlabel=xlabel, ylabel="TVD")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(a.out / f"{name}.png", dpi=180)
        plt.close(fig)
    if a.campaign:
        records = load_records(a.campaign / "evaluations", load_manifest(a.manifest))
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for axis, n_visible in zip(axes, (8, 12, 16), strict=True):
            selected = [
                record
                for record in records
                if record.identity["dataset"] == "curie_weiss"
                and record.identity["family"] == "conditioned"
                and int(record.identity["visible_qubits"]) == n_visible
            ]
            bits = enumerate_bitstrings(n_visible)
            target = selected[0].at(1.0)[1]
            axis.plot(
                np.arange(n_visible + 1),
                shell_probabilities(target, bits),
                color="black",
                linewidth=2,
                label="target",
            )
            for ancillas in (0, 1, 3, 6, 10):
                profiles = [
                    shell_probabilities(record.at(1.0)[0], bits)
                    for record in selected
                    if int(record.identity["ancillas"]) == ancillas
                ]
                axis.plot(
                    np.arange(n_visible + 1),
                    np.median(profiles, axis=0),
                    marker="o",
                    markersize=2,
                    label=f"r={ancillas}",
                )
            axis.set(title=f"n={n_visible}", xlabel="Hamming weight")
        axes[0].set_ylabel("Probability")
        axes[-1].legend(fontsize=7)
        fig.suptitle(r"Curie--Weiss critical profile at $\beta=1$")
        fig.tight_layout()
        fig.savefig(a.out / "curie_critical_profiles.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()

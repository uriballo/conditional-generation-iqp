#!/usr/bin/env python3
"""Plot an exploratory two-panel figure from a partial exact-TVD summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _band(
    rows: list[dict], field: str, ancillas: int
) -> tuple[float, float, float] | None:
    values = [
        float(row[field])
        for row in rows
        if int(row["ancillas"]) == ancillas
        and row.get(field) is not None
        and np.isfinite(row[field])
    ]
    if not values:
        return None
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.25)),
        float(np.quantile(values, 0.75)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", default="gaussian")
    parser.add_argument("--visible-qubits", type=int, default=8)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text())
    rows = [
        row
        for row in payload["seed_rows"]
        if row["dataset"] == args.dataset
        and int(row["visible_qubits"]) == args.visible_qubits
    ]
    if not rows:
        raise ValueError("summary contains no matching seed rows")
    ancillas = sorted({int(row["ancillas"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True)
    colors = {"conditioned_tvd": "C0", "interpolation_tvd": "C1", "oracle_tvd": "C2"}
    labels = {
        "conditioned_tvd": "conditioned",
        "interpolation_tvd": "adjacent-independent interpolation",
        "oracle_tvd": "target-aware oracle",
    }
    for field in colors:
        points = [
            _band([r for r in rows if r["group"] == "held_out"], field, a)
            for a in ancillas
        ]
        if not any(points):
            continue
        valid = [(a, p) for a, p in zip(ancillas, points, strict=True) if p is not None]
        x = np.asarray([a for a, _ in valid])
        y = np.asarray([p[0] for _, p in valid])
        lo = np.asarray([p[1] for _, p in valid])
        hi = np.asarray([p[2] for _, p in valid])
        axes[0].plot(x, y, marker="o", color=colors[field], label=labels[field])
        axes[0].fill_between(x, lo, hi, color=colors[field], alpha=0.16)
    axes[0].set_ylabel("Exact visible TVD")
    axes[0].set_title("Held-out performance")
    axes[0].legend(fontsize=8)

    for field, label, color in (
        ("generalization_advantage", "interpolation - conditioned", "C1"),
        ("sharing_cost", "conditioned - oracle", "C0"),
    ):
        points = [
            _band([r for r in rows if r["group"] == "held_out"], field, a)
            for a in ancillas
        ]
        valid = [(a, p) for a, p in zip(ancillas, points, strict=True) if p is not None]
        if not valid:
            continue
        x = np.asarray([a for a, _ in valid])
        y = np.asarray([p[0] for _, p in valid])
        lo = np.asarray([p[1] for _, p in valid])
        hi = np.asarray([p[2] for _, p in valid])
        axes[1].plot(x, y, marker="o", color=color, label=label)
        axes[1].fill_between(x, lo, hi, color=color, alpha=0.16)
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].set_title("Generalization and sharing costs")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("Ancillas")
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"Preliminary exact-TVD results: {args.dataset}, n={args.visible_qubits}"
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

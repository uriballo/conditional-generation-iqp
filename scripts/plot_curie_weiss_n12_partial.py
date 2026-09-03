"""Plot partial Curie--Weiss n=12 qualitative profiles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from journal_style import DOUBLE_COL_WIDTH, apply_journal_style, save_figure

from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.exact_tvd_results import interpolate_probabilities, load_records

SEEDS = tuple(range(100, 110))
BETAS = (0.6, 1.0, 1.4)
ANCILLAS = (0, 1, 3)
TRAINING = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
FIELDS = [
    "dataset",
    "n",
    "condition",
    "split",
    "model",
    "coordinate",
    "median",
    "q1",
    "q3",
    "seed_count",
    "left_endpoint",
    "right_endpoint",
    "weight",
]


def profile(p, n):
    return np.bincount(
        [int(x).bit_count() for x in range(1 << n)], weights=p, minlength=n + 1
    )


def conditioned(records, beta, r, n=12):
    group = [
        x
        for x in records
        if x.identity["dataset"] == "curie_weiss"
        and x.identity["visible_qubits"] == n
        and x.identity["ancillas"] == r
        and x.identity["family"] == "conditioned"
        and x.identity["seed"] in SEEDS
        and np.isclose(x.conditions, beta).any()
    ]
    if len(group) != 10:
        raise RuntimeError(
            f"missing conditioned records beta={beta}, r={r}: {len(group)}/10"
        )
    vals, targets = [], []
    for rec in sorted(group, key=lambda x: x.identity["seed"]):
        p, t, _ = rec.at(beta)
        vals.append(profile(p, n))
        targets.append(profile(t, n))
    if not np.allclose(targets, targets[0]):
        raise RuntimeError("inconsistent targets")
    return np.asarray(vals), targets[0]


def independent(records, beta, seed, n=12):
    group = [
        x
        for x in records
        if x.identity["dataset"] == "curie_weiss"
        and x.identity["visible_qubits"] == n
        and x.identity["ancillas"] == 3
        and x.identity["family"] == "independent"
        and x.identity["seed"] == seed
        and np.isclose(x.conditions, beta).any()
    ]
    if len(group) != 1:
        raise RuntimeError(f"missing independent record beta={beta}, seed={seed}")
    return group[0].at(beta)[0]


def rows_for(records, beta, model, split, n=12):
    x = (2 * np.arange(n + 1, dtype=float) - n) / n
    out = []
    left_endpoint = right_endpoint = weight = ""
    if model == "exact target":
        target = exact_curie_weiss_target(n, beta).profile
        for xx, y in zip(x, target, strict=True):
            out.append(
                {
                    "dataset": "curie_weiss",
                    "n": n,
                    "condition": beta,
                    "split": split,
                    "model": model,
                    "coordinate": xx,
                    "median": y,
                    "q1": y,
                    "q3": y,
                    "seed_count": 0,
                    "left_endpoint": "",
                    "right_endpoint": "",
                    "weight": "",
                }
            )
        return out
    if model.startswith("r="):
        vals, _ = conditioned(records, beta, int(model[2:]), n)
        med, q1, q3 = (
            np.median(vals, 0),
            np.quantile(vals, 0.25, 0),
            np.quantile(vals, 0.75, 0),
        )
        count = 10
    else:
        right = next(v for v in TRAINING if v > beta)
        left = TRAINING[TRAINING.index(right) - 1]
        w = (beta - left) / (right - left)
        left_endpoint, right_endpoint, weight = left, right, w
        vals = np.asarray(
            [
                profile(
                    interpolate_probabilities(
                        independent(records, left, s), independent(records, right, s), w
                    ),
                    n,
                )
                for s in SEEDS
            ]
        )
        med, q1, q3 = (
            np.median(vals, 0),
            np.quantile(vals, 0.25, 0),
            np.quantile(vals, 0.75, 0),
        )
        count = 10
    for xx, a, b, c in zip(x, med, q1, q3, strict=True):
        out.append(
            {
                "dataset": "curie_weiss",
                "n": n,
                "condition": beta,
                "split": split,
                "model": model,
                "coordinate": xx,
                "median": a,
                "q1": b,
                "q3": c,
                "seed_count": count,
                "left_endpoint": left_endpoint,
                "right_endpoint": right_endpoint,
                "weight": weight,
            }
        )
    return out


def main(evaluations, output):
    config = json.loads(Path("configs/exact_tvd_ancilla_v1.json").read_text())
    records = load_records(evaluations, config, require_complete=False)
    apply_journal_style()
    output.mkdir(parents=True, exist_ok=True)

    def make(filename, models, split):
        x = (2 * np.arange(13) - 12) / 12
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(DOUBLE_COL_WIDTH, 64 / 25.4),
            sharey=True,
            layout="constrained",
        )
        allrows = []
        colors = {
            "r=0": "#0072B2",
            "r=1": "#D55E00",
            "r=3": "#009E73",
            "adjacent independent interpolation": "#D55E00",
        }
        handles = []
        labels = {
            "r=0": r"$r=0$",
            "r=1": r"$r=1$",
            "r=3": r"$r=3$",
            "adjacent independent interpolation": "Independent interpolation",
        }
        for j, (ax, beta) in enumerate(zip(axes, BETAS, strict=True)):
            target = exact_curie_weiss_target(12, beta).profile
            for model in models:
                rr = rows_for(records, beta, model, split)
                allrows += rr
                if model == "exact target":
                    continue
                d = [z for z in rr if z["model"] == model]
                y = np.array([z["median"] for z in d])
                lo = np.array([z["q1"] for z in d])
                hi = np.array([z["q3"] for z in d])
                line = ax.plot(x, y, "o-", color=colors[model], label=labels[model])[0]
                ax.fill_between(x, lo, hi, color=colors[model], alpha=0.18, linewidth=0)
                if j == 0:
                    handles.append(line)
            t = ax.plot(x, target, "k--", label="Exact target")[0]
            if j == 0:
                handles.append(t)
            ax.set_title(rf"({chr(97 + j)})" + "\n" + rf"$\beta={beta:g}$", pad=3)
            ax.set_xticks((-1.0, 0.0, 1.0))
            ax.spines[["top", "right"]].set_visible(False)
        fig.supylabel("Probability mass")
        fig.supxlabel("Magnetization $m$")
        fig.legend(
            handles,
            [h.get_label() for h in handles],
            ncol=len(handles),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            frameon=False,
        )
        path = output / filename
        save_figure(fig, path)
        with path.with_suffix(".csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(allrows)

    make(
        "curie_weiss_n12_magnetization_profiles",
        ["r=0", "r=1", "r=3", "exact target"],
        "held_out",
    )
    make(
        "curie_weiss_n12_generalization_profiles",
        ["r=3", "adjacent independent interpolation", "exact target"],
        "held_out",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--evaluations",
        type=Path,
        default=Path("results/exact-tvd-ancilla-v1/evaluations"),
    )
    p.add_argument(
        "--output", type=Path, default=Path("artifacts/curie_weiss_n12_partial")
    )
    a = p.parse_args()
    main(a.evaluations, a.output)

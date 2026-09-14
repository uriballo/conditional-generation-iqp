#!/usr/bin/env python3
"""Verify all sealed data/ancilla arms and report paired-seed interactions."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

from iqp_cond.artifacts import read_immutable_json

SEEDS = tuple(range(4711, 4721))
SIZES = (100, 300, 1000)
METRICS = ("tvd", "correlation_rmse", "marginal_a_tvd", "marginal_b_tvd")
ARMS = (
    *((e, r) for e in ("cat", "bond") for r in (0, 2, 4)),
    ("cat_encoder_only", 0),
    ("bond_encoder_only", 0),
    ("log_composition", None),
    ("log_linear", None),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recompute(q, p):
    bits = (np.arange(4096)[:, None] >> np.arange(12)) & 1
    spins = 1 - 2 * bits
    a = spins[:, :6] * np.roll(spins[:, :6], -1, axis=1)
    b = spins[:, 6:] * np.roll(spins[:, 6:], -1, axis=1)

    def connected(v):
        return v @ (a * b) - (v @ a) * (v @ b)

    return {
        "tvd": 0.5 * np.abs(q - p).sum(1),
        "correlation_rmse": np.sqrt(((connected(q) - connected(p)) ** 2).mean(1)),
        "marginal_a_tvd": 0.5
        * np.abs(q.reshape(-1, 64, 64).sum(1) - p.reshape(-1, 64, 64).sum(1)).sum(1),
        "marginal_b_tvd": 0.5
        * np.abs(q.reshape(-1, 64, 64).sum(2) - p.reshape(-1, 64, 64).sum(2)).sum(1),
    }


def row_index(rows, seeds=SEEDS):
    expected = {(s, size, e, r) for s in seeds for size in SIZES for e, r in ARMS}
    indexed = {}
    for row in rows:
        key = (row["seed"], row["train_size"], row["model"], row["n_ancillas"])
        if key in indexed:
            raise ValueError("duplicate row identity")
        if key not in expected:
            raise ValueError("unexpected row identity")
        for role, count in [("train", 16), ("test", 9)]:
            if set(row[role]) != set(METRICS):
                raise ValueError("wrong metric set")
            for v in row[role].values():
                if np.asarray(v).shape != (count,) or not np.isfinite(v).all():
                    raise ValueError("invalid metric array")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("missing row identities")
    for e in ("cat_encoder_only", "bond_encoder_only"):
        for role in ("train", "test"):
            for metric in METRICS:
                first = indexed[(seeds[0], 100, e, 0)][role][metric]
                for s in seeds:
                    for size in SIZES:
                        np.testing.assert_allclose(
                            indexed[(s, size, e, 0)][role][metric],
                            first,
                            atol=1e-12,
                            rtol=0,
                        )
    return indexed


def verify(directory):
    m = read_immutable_json(directory / "manifest.json")
    seal = read_immutable_json(directory / "predictions-seal.json")
    ev = read_immutable_json(directory / "evaluation.json")
    if not (
        ev["manifest_sha256"]
        == seal["manifest_sha256"]
        == digest(directory / "manifest.json")
        and ev["seal_sha256"] == digest(directory / "predictions-seal.json")
    ):
        raise RuntimeError("manifest/seal/evaluation linkage mismatch")
    for file, key, source in [
        ("source.tar.gz", "source_archive_sha256", m),
        ("data.npz", "data_sha256", m),
        ("evaluation-targets.npz", "evaluation_targets_sha256", ev),
    ]:
        if digest(directory / file) != source[key]:
            raise RuntimeError("changed " + file)
    if (
        m["seeds"] != list(SEEDS)
        or m["train_sizes"] != list(SIZES)
        or m["ancillas"] != [0, 2, 4]
        or m["expected_iqp_fits"] != 180
        or m["expected_classical_bundles"] != 30
    ):
        raise RuntimeError("unexpected frozen design")
    with tarfile.open(directory / "source.tar.gz") as archive:
        for name, expected_hash in m["source_sha256"].items():
            member = archive.extractfile(name)
            if (
                member is None
                or hashlib.sha256(member.read()).hexdigest() != expected_hash
            ):
                raise RuntimeError("source member mismatch: " + name)
    expected = set()
    for seed in SEEDS:
        for size in SIZES:
            names = [
                f"{e}-s{size}-r{r}-{seed}" for e in ("cat", "bond") for r in (0, 2, 4)
            ]
            names.append(f"classical-s{size}-{seed}")
            expected.update(n + suffix for n in names for suffix in (".json", ".npz"))
    if (
        set(seal["files"]) != expected
        or seal["iqp_fits"] != 180
        or seal["classical_bundles"] != 30
    ):
        raise RuntimeError("incomplete prediction seal")
    for file, h in seal["files"].items():
        if digest(directory / file) != h:
            raise RuntimeError("changed sealed file " + file)
    index = row_index(ev["rows"])
    with np.load(directory / "data.npz") as d:
        train_target = d["train_targets"]
    with np.load(directory / "evaluation-targets.npz") as d:
        test_target = d["test_targets"]
    controls = {}
    for (seed, size, e, r), row in index.items():
        classical = e in ("log_composition", "log_linear")
        zero = e.endswith("_encoder_only")
        encoder = e.removesuffix("_encoder_only")
        name = (
            f"classical-s{size}-{seed}"
            if classical
            else f"{encoder}-s{size}-r{r}-{seed}"
        )
        meta = read_immutable_json(directory / (name + ".json"))
        if (
            meta["archive_sha256"] != digest(directory / (name + ".npz"))
            or meta["manifest_sha256"] != ev["manifest_sha256"]
        ):
            raise RuntimeError("bundle linkage mismatch")
        with np.load(directory / (name + ".npz")) as d:
            for role, target in [("train", train_target), ("test", test_target)]:
                key = (
                    e + "_" + role
                    if classical
                    else (
                        "encoder_only_" + role
                        if zero
                        else (
                            "train_probabilities"
                            if role == "train"
                            else "test_predictions"
                        )
                    )
                )
                q = d[key]
                if q.shape != target.shape or not np.isfinite(q).all() or (q < 0).any():
                    raise RuntimeError("invalid probabilities")
                np.testing.assert_allclose(q.sum(1), 1, atol=1e-11, rtol=0)
                for flip in (63, 4032):
                    np.testing.assert_allclose(
                        q, q[:, np.arange(4096) ^ flip], atol=1e-11, rtol=0
                    )
                if zero:
                    if (e, role) in controls:
                        np.testing.assert_allclose(
                            q, controls[(e, role)], atol=1e-12, rtol=0
                        )
                    else:
                        controls[(e, role)] = q.copy()
                for metric, value in recompute(q, target).items():
                    np.testing.assert_allclose(
                        row[role][metric], value, atol=1e-12, rtol=0
                    )
    return ev["rows"], {
        "manifest_sha256": ev["manifest_sha256"],
        "seal_sha256": ev["seal_sha256"],
        "evaluation_sha256": digest(directory / "evaluation.json"),
        "report_sha256": digest(__file__),
    }


def summarize(rows, seeds=SEEDS):
    index = row_index(rows, seeds)
    draws = np.random.default_rng(950702).integers(0, len(seeds), (10000, len(seeds)))

    def stat(values):
        values = np.asarray(values, dtype=float)
        return {
            "mean": float(values.mean()),
            "seed_values": values.tolist(),
            "ci95": np.quantile(values[draws].mean(1), [0.025, 0.975]).tolist(),
        }

    raw = {}
    curves = []
    for e, r in ARMS:
        for size in SIZES:
            result = {}
            for metric in METRICS:
                v = np.array([index[(s, size, e, r)]["test"][metric] for s in seeds])
                raw[(e, r, size, metric)] = v.mean(1)
                result[metric] = {
                    **stat(v.mean(1)),
                    "condition_means": v.mean(0).tolist(),
                    "worst_condition": stat(v.max(1)),
                    "maximum_seed_condition": float(v.max()),
                }
            curves.append(
                {"model": e, "n_ancillas": r, "train_size": size, "metrics": result}
            )
    contrasts = []

    def add(kind, e, r, metric, values, size=None):
        contrasts.append(
            {
                "kind": kind,
                "model": e,
                "n_ancillas": r,
                "metric": metric,
                "train_size": size,
                **stat(values),
            }
        )

    for e, r in ARMS:
        if e.endswith("_encoder_only"):
            continue
        for metric in METRICS:
            add(
                "training_100_minus_1000",
                e,
                r,
                metric,
                raw[(e, r, 100, metric)] - raw[(e, r, 1000, metric)],
            )
            for low, high in [(100, 300), (300, 1000)]:
                add(
                    f"training_{low}_minus_{high}",
                    e,
                    r,
                    metric,
                    raw[(e, r, low, metric)] - raw[(e, r, high, metric)],
                )
    for e in ("cat", "bond"):
        for metric in METRICS:
            for r in (0, 2, 4):
                for size in SIZES:
                    add(
                        "input_gain",
                        e,
                        r,
                        metric,
                        raw[(e + "_encoder_only", 0, size, metric)]
                        - raw[(e, r, size, metric)],
                        size,
                    )
            for r in (2, 4):
                benefits = {
                    size: raw[(e, 0, size, metric)] - raw[(e, r, size, metric)]
                    for size in SIZES
                }
                for size, v in benefits.items():
                    add("ancilla_benefit", e, r, metric, v, size)
                add(
                    "interaction_1000_minus_100",
                    e,
                    r,
                    metric,
                    benefits[1000] - benefits[100],
                )
                for low, high in [(100, 300), (300, 1000)]:
                    add(
                        f"interaction_{high}_minus_{low}",
                        e,
                        r,
                        metric,
                        benefits[high] - benefits[low],
                    )
    for r in (0, 2, 4):
        for size in SIZES:
            for metric in METRICS:
                add(
                    "bond_minus_cat",
                    "bond",
                    r,
                    metric,
                    raw[("bond", r, size, metric)] - raw[("cat", r, size, metric)],
                    size,
                )
    return {
        "seeds": list(seeds),
        "curves": curves,
        "contrasts": contrasts,
        "uncertainty": "10000 shared paired seed bootstrap indices; nine conditions averaged within seed; descriptive percentile95%",
        "fixed_validation_per_condition": 1000,
        "fixed_updates": 500,
    }


def plot(summary, out):
    # Importing the shared report applies the repository's publication rcParams.
    import matplotlib.pyplot as plt
    import report_reference_estimator_study  # noqa: F401

    curves = {
        (v["model"], v["n_ancillas"], v["train_size"]): v for v in summary["curves"]
    }
    fig, axes = plt.subplots(
        3, 2, figsize=(170 / 25.4, 175 / 25.4), layout="constrained"
    )
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for col, e in enumerate(("cat", "bond")):
        for row, metric in enumerate(("tvd", "correlation_rmse")):
            ax = axes[row, col]
            for i, (model, r, label, style) in enumerate(
                [
                    (e, 0, "0 ancillas", "-"),
                    (e, 2, "2 ancillas", "--"),
                    (e, 4, "4 ancillas", ":"),
                    ("log_composition", None, "Log composition", "-."),
                    ("log_linear", None, "Log-linear", "--"),
                ]
            ):
                vals = [curves[(model, r, size)]["metrics"][metric] for size in SIZES]
                ax.plot(
                    SIZES,
                    [v["mean"] for v in vals],
                    color=colors[i],
                    linestyle=style,
                    marker="o",
                    label=label,
                )
                ax.fill_between(
                    SIZES,
                    [v["ci95"][0] for v in vals],
                    [v["ci95"][1] for v in vals],
                    color=colors[i],
                    alpha=0.12,
                )
            control = curves[(e + "_encoder_only", 0, 100)]["metrics"][metric]["mean"]
            ax.axhline(control, color=".4", linestyle=":", label="Input only")
            ax.set_xscale("log")
            ax.set_xticks(SIZES, [str(x) for x in SIZES])
            ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title("Symmetry-only" if e == "cat" else "Bond-informed")
            if col == 0:
                ax.set_ylabel("Exact joint TVD" if row == 0 else "Connected-bond RMSE")
        ax = axes[2, col]
        for r, color, style in [(2, colors[1], "--"), (4, colors[2], ":")]:
            vals = [
                next(
                    v
                    for v in summary["contrasts"]
                    if v["kind"] == "ancilla_benefit"
                    and v["model"] == e
                    and v["n_ancillas"] == r
                    and v["metric"] == "tvd"
                    and v["train_size"] == size
                )
                for size in SIZES
            ]
            ax.plot(
                SIZES,
                [v["mean"] for v in vals],
                color=color,
                linestyle=style,
                marker="o",
            )
            ax.fill_between(
                SIZES,
                [v["ci95"][0] for v in vals],
                [v["ci95"][1] for v in vals],
                color=color,
                alpha=0.12,
            )
        ax.axhline(0, color=".4", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks(SIZES, [str(x) for x in SIZES])
        ax.set_xlabel("Training examples per condition")
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.set_ylabel("Paired ancilla TVD benefit")
    for row in range(3):
        limits = [ax.get_ylim() for ax in axes[row]]
        for col, ax in enumerate(axes[row]):
            ax.set_ylim(min(v[0] for v in limits), max(v[1] for v in limits))
            ax.text(
                -0.17,
                1.02,
                f"({chr(97 + 2 * row + col)})",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=8)
    fig.savefig(out / "data_ancilla_learning_curves.pdf")
    fig.savefig(out / "data_ancilla_learning_curves.png", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    rows, provenance = verify(args.directory)
    summary = summarize(rows)
    summary["provenance"] = provenance
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Data-size by ancilla-count study",
        "",
        "All 180 IQP fits and 30 classical bundles verified. Exact evaluation on the existing nine interior conditions. Fixed 1,000 validation examples per condition and 500 IQP updates; the x-axis is training data, not total labeled access. All uncertainty is descriptive paired-seed percentile95%, without multiplicity correction.",
        "",
        "| Model | Ancillas | Training / condition | Mean TVD [95% interval] | Mean connected-bond RMSE |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for v in summary["curves"]:
        m = v["metrics"]["tvd"]
        c = v["metrics"]["correlation_rmse"]
        lines.append(
            f"| {v['model']} | {v['n_ancillas']} | {v['train_size']} | {m['mean']:.5f} [{m['ci95'][0]:.5f}, {m['ci95'][1]:.5f}] | {c['mean']:.5f} |"
        )
    lines += [
        "",
        "Primary interaction: change in the four-ancilla TVD benefit from 100 to 1,000 training examples. Positive means ancillas help more with more training data.",
        "",
    ]
    for v in summary["contrasts"]:
        if (
            v["kind"] == "interaction_1000_minus_100"
            and v["n_ancillas"] == 4
            and v["metric"] == "tvd"
        ):
            lines.append(
                f"- {v['model']}: {v['mean']:.5f} [{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}]."
            )
    lines += [
        "",
        "All per-seed contrasts, endpoint/adjacent-size changes, input-relative gains, marginals and worst-condition errors are retained in summary.json. Deterministic controls are not independent fits. No out-of-domain, optimal sample-complexity, parameter-matched or quantum-advantage claim follows from this study.",
        "",
    ]
    (args.output / "report.md").write_text("\n".join(lines))
    if not args.no_plots:
        plot(summary, args.output)
    print("VERIFIED 300 rows; wrote", args.output, flush=True)


if __name__ == "__main__":
    main()

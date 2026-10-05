"""Render numerical paper figures and tables from portable compact data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {
    "encoder_only": "Encoder only",
    "shared": "Shared conditional IQP across conditions",
    "independent_conditional": "Independent conditional IQP per condition",
    "independent_unconditional": "Independent unconditional IQP per condition",
}
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=COLORS),
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "mathtext.fontset": "cm",
    }
)


def _save(fig, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / (stem + ".pdf"), output / (stem + ".png")]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def _gaussian(data, output):
    fig, axes = plt.subplots(
        1, 3, figsize=(170 * MM, 58 * MM), layout="constrained", sharey=True
    )
    for ax, index in zip(axes, (10, 40, 70), strict=True):
        values = np.genfromtxt(data / f"profile_{index}.csv", delimiter=",", names=True)
        for column, style, color, label in [
            ("target", "--", "0.25", "Target"),
            ("trained", "-", COLORS[0], "Trained"),
            ("encoder", ":", "0.45", "Encoder only"),
        ]:
            ax.plot(values["k"], values[column], style, color=color, label=label)
        ax.set_title(rf"$c={(index + 1 / 3) / 80:.3f}$")
        ax.set_xlabel(r"Hamming weight $k$")
        ax.set_xlim(-0.25, 16.25)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Probability mass")
    axes[0].legend(fontsize=7)
    return _save(fig, output, "gaussian_translation_profiles")


def curie_tables(payload):
    rows = []
    for family in ("shared", "independent_conditional", "independent_unconditional"):
        metrics = payload["metrics"][family]
        for kind in ("profile", "full"):
            values = np.asarray(metrics[kind + "_per_seed"])
            mean = values.mean(0)
            sd = values.std(0, ddof=1) if len(values) > 1 else np.full(3, np.nan)
            np.testing.assert_allclose(mean, metrics[kind + "_mean"], atol=1e-12)
            if len(values) > 1:
                np.testing.assert_allclose(sd, metrics[kind + "_sd"], atol=1e-12)
            for i, beta in enumerate(payload["betas"]):
                rows.append(
                    dict(
                        model=MODEL_NAMES[family],
                        evaluation="Held-out" if family == "shared" else "Trained",
                        metric=kind + "_tvd",
                        beta=beta,
                        mean=float(mean[i]),
                        sample_sd=float(sd[i]),
                        seeds=len(values),
                    )
                )
    return rows


def _curie(data, output):
    payload = json.loads((data / "figure_data.json").read_text())
    targets = np.asarray(payload["records"]["shared"][0]["target_profile"])
    arms = payload["arms"]
    maximum = max(targets.max(), *(np.asarray(v["mean"]).max() for v in arms.values()))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(170 * MM, 120 * MM),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    for panel, (family, ax) in enumerate(zip(MODEL_NAMES, axes.flat, strict=True)):
        title = (
            MODEL_NAMES[family]
            .replace(" across conditions", "\nacross conditions")
            .replace(" per condition", "\nper condition")
        )
        ax.set_title(f"({chr(97 + panel)}) {title}", fontsize=9)
        for i, _beta in enumerate(payload["betas"]):
            y = np.asarray(arms[family]["mean"])[i]
            if family != "encoder_only" or i == 0:
                ax.plot(
                    (2 * np.arange(17) - 16) / 16,
                    y,
                    color="0.3" if family == "encoder_only" else COLORS[i],
                    marker="o",
                    ms=2.5,
                    lw=1.3,
                )
            ax.plot(
                (2 * np.arange(17) - 16) / 16, targets[i], "--", color=COLORS[i], lw=1.1
            )
        ax.set_ylim(0, maximum * 1.08)
        ax.set_xlim(-1.03, 1.03)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel(r"Magnetization per spin $m$", fontsize=10)
    fig.supylabel("Probability mass", fontsize=10)
    handles = [
        plt.Line2D([], [], color=COLORS[i], label=rf"$\beta={beta:.1f}$")
        for i, beta in enumerate(payload["betas"])
    ]
    handles.insert(0, plt.Line2D([], [], color="0.3", ls="--", label="Target"))
    fig.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=8)
    paths = _save(fig, output, "curie_weiss_model_comparison")
    rows = curie_tables(payload)
    table = output / "curie_weiss_tvd.csv"
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths.append(table)
    return paths


def _scaling(data, output):
    payload = json.loads((data / "figure_data.json").read_text())
    fig, ax = plt.subplots(figsize=(85 * MM, 66 * MM), layout="constrained")
    rows = []
    for n in payload["sizes"]:
        matrix = np.asarray(payload["curves"][str(n)]["held_out"]["overall_per_seed"])
        if matrix.shape != (len(payload["seeds"]), len(payload["steps"])):
            raise ValueError("scaling seed/step coverage differs")
        med = np.median(matrix, axis=0)
        lo, hi = np.quantile(matrix, (0.25, 0.75), axis=0)
        ax.plot(payload["steps"], med, lw=1.3, marker="o", ms=3, label=rf"$n={n}$")
        ax.fill_between(payload["steps"], lo, hi, alpha=0.13)
        rows.extend(
            dict(
                qubits=n,
                step=step,
                median=float(med[i]),
                q25=float(lo[i]),
                q75=float(hi[i]),
            )
            for i, step in enumerate(payload["steps"])
        )
    ax.set_xlabel("Optimizer updates")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.set_ylabel(r"Held-out sampled MMD$^2$")
    ax.legend(ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    paths = _save(fig, output, "curie_weiss_lattice_scaling_mmd")
    table = output / "scaling_mmd.csv"
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return [*paths, table]


def _symmetry(data, output):
    payload = json.loads((data / "symmetry.json").read_text())
    fig, ax = plt.subplots(figsize=(85 * MM, 55 * MM), layout="constrained")
    x = np.arange(8)
    ax.bar(x - 0.2, payload["encoded"], 0.4, label="Encoded input", color=COLORS[0])
    ax.bar(x + 0.2, payload["ordinary"], 0.4, label="Ordinary input", color="0.45")
    ax.set_xlabel(r"Measured bitstring $x$")
    ax.set_ylabel("Probability")
    ax.set_xticks(x, payload["labels"])
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, output, "input_induced_pair_flip")


def export_completed(root, destination):
    """Package newly completed reported jobs, never the checked-in paper data."""
    root = Path(root)
    destination = Path(destination)
    records = json.loads((root / "reported-results.json").read_text())
    by_kind = {
        kind: [r for r in records if r["experiment"] == kind]
        for kind in (
            "gaussian",
            "shared",
            "conditional",
            "unconditional",
            "scaling",
            "symmetry",
        )
    }
    available = []
    if by_kind["shared"] and by_kind["conditional"] and by_kind["unconditional"]:
        target = None
        payload = dict(
            betas=[0.4, 1.0, 1.6],
            seeds=sorted(r["seed"] for r in by_kind["shared"]),
            arms={},
            records={},
            metrics={},
        )
        for kind, family in [
            ("shared", "shared"),
            ("conditional", "independent_conditional"),
            ("unconditional", "independent_unconditional"),
        ]:
            seed_profiles = []
            profile_errors = []
            full_errors = []
            items = []
            for seed in payload["seeds"]:
                cases = sorted(
                    [r for r in by_kind[kind] if r["seed"] == seed],
                    key=lambda r: r["job"]["beta"] or 0,
                )
                expected = 1 if kind == "shared" else 3
                if len(cases) != expected:
                    raise ValueError("incomplete Curie-Weiss seed/temperature coverage")
                profiles = []
                targets = []
                ptvd = []
                ftvd = []
                for r in cases:
                    with np.load(Path(r["directory"]) / "checkpoint.npz") as ck:
                        weights = np.bitwise_count(np.arange(65536, dtype=np.uint32))
                        q = np.asarray(ck["held_out_probabilities"])
                        t = np.asarray(ck["held_out_target_probabilities"])
                        profiles.extend(
                            np.asarray(
                                [
                                    np.bincount(weights, weights=row, minlength=17)
                                    for row in q
                                ]
                            ).tolist()
                        )
                        targets.extend(
                            np.asarray(
                                [
                                    np.bincount(weights, weights=row, minlength=17)
                                    for row in t
                                ]
                            ).tolist()
                        )
                    ptvd.extend(r["held_out_profile_tvd_per_condition"])
                    ftvd.extend(r["held_out_tvd_per_condition"])
                if target is None:
                    target = np.asarray(targets)
                np.testing.assert_allclose(targets, target, atol=1e-12)
                seed_profiles.append(profiles)
                profile_errors.append(ptvd)
                full_errors.append(ftvd)
                items.append(
                    dict(
                        seed=seed,
                        profile=profiles,
                        target_profile=targets,
                        profile_tvd=ptvd,
                        full_tvd=ftvd,
                    )
                )
            payload["records"][family] = items
            payload["arms"][family] = dict(mean=np.mean(seed_profiles, 0).tolist())
            metrics = {}
            for key, values in [("profile", profile_errors), ("full", full_errors)]:
                array = np.asarray(values)
                metrics[key + "_per_seed"] = array.tolist()
                metrics[key + "_mean"] = array.mean(0).tolist()
                metrics[key + "_sd"] = (
                    array.std(0, ddof=1).tolist() if len(array) > 1 else None
                )
            payload["metrics"][family] = metrics
        from math import comb

        uniform = np.asarray([comb(16, k) / 65536 for k in range(17)])
        payload["arms"]["encoder_only"] = dict(mean=np.tile(uniform, (3, 1)).tolist())
        p = destination / "curie-weiss"
        p.mkdir(parents=True, exist_ok=True)
        (p / "figure_data.json").write_text(json.dumps(payload, indent=2) + "\n")
        available.append("curie-weiss")
    if by_kind["gaussian"]:
        first = min(by_kind["gaussian"], key=lambda r: r["seed"])
        p = destination / "gaussian"
        p.mkdir(parents=True, exist_ok=True)
        from math import comb

        chance = np.sin(1.5 / 2) ** 2
        encoder = np.asarray(
            [comb(16, k) * chance**k * (1 - chance) ** (16 - k) for k in range(17)]
        )
        with np.load(Path(first["directory"]) / "checkpoint.npz") as ck:
            weights = np.bitwise_count(np.arange(65536, dtype=np.uint32))
            for index in (10, 40, 70):
                target = np.bincount(
                    weights,
                    weights=ck["held_out_target_probabilities"][index],
                    minlength=17,
                )
                trained = np.bincount(
                    weights, weights=ck["held_out_probabilities"][index], minlength=17
                )
                np.savetxt(
                    p / f"profile_{index}.csv",
                    np.column_stack([np.arange(17), target, trained, encoder]),
                    delimiter=",",
                    header="k,target,trained,encoder",
                    comments="",
                )
        available.append("gaussian")
    if by_kind["scaling"]:
        sizes = sorted({r["job"]["size"] for r in by_kind["scaling"]})
        seeds = sorted({r["seed"] for r in by_kind["scaling"]})
        steps = list(range(0, 5001, 500))
        if records[0].get("smoke"):
            steps = [0, 2, 4]
        curves = {}
        for n in sizes:
            curves[str(n)] = {}
            for grid in ("training", "held_out"):
                values = []
                per_condition = []
                for seed in seeds:
                    cases = [
                        r
                        for r in by_kind["scaling"]
                        if r["job"]["size"] == n and r["seed"] == seed
                    ]
                    if len(cases) != 1:
                        raise ValueError("incomplete scaling grid")
                    rows = {
                        r["step"]: r
                        for r in cases[0]["held_out_evaluation"]["rows"]
                        if r["grid"] == grid
                    }
                    values.append([rows[step]["mean"] for step in steps])
                    per_condition.append(
                        [rows[step]["mean_per_condition"] for step in steps]
                    )
                curves[str(n)][grid] = dict(
                    overall_per_seed=values, per_seed=per_condition
                )
        payload = dict(sizes=sizes, seeds=seeds, steps=steps, curves=curves)
        p = destination / "scaling"
        p.mkdir(parents=True, exist_ok=True)
        (p / "figure_data.json").write_text(json.dumps(payload, indent=2) + "\n")
        available.append("scaling")
    if by_kind["symmetry"]:
        order = [0, 3, 4, 7, 2, 1, 6, 5]
        r = by_kind["symmetry"][0]
        payload = dict(
            labels=["000", "110", "001", "111", "010", "100", "011", "101"],
            encoded=[r["encoded"][i] for i in order],
            ordinary=[r["ordinary"][i] for i in order],
        )
        p = destination / "symmetry"
        p.mkdir(parents=True, exist_ok=True)
        (p / "symmetry.json").write_text(json.dumps(payload, indent=2) + "\n")
        available.append("symmetry")
    return available


def render_all(data_root=None, output_root=None, experiment="all"):
    data = Path(data_root) if data_root else ROOT / "paper/data"
    output = Path(output_root) if output_root else ROOT / "results/paper/figures"
    available = ["gaussian", "curie-weiss", "scaling", "symmetry"]
    if (data / "reported-results.json").exists():
        available = export_completed(data, output / "data")
        data = output / "data"
    renderers = {
        "gaussian": _gaussian,
        "curie-weiss": _curie,
        "scaling": _scaling,
        "symmetry": _symmetry,
    }
    selected = available if experiment == "all" else [experiment]
    paths = []
    for name in selected:
        if name not in available:
            raise ValueError("requested experiment has no completed data")
        paths.extend(renderers[name](data / name, output))
    return paths

"""Render numerical paper figures and tables from portable compact data."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {
    "encoder_only": "Encoder only",
    "shared": "Shared conditional IQP",
    "independent_conditional": "Independent conditional IQPs",
    "independent_unconditional": "Independent unconditional IQPs",
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


def _provenance(data, output, stem, inputs, aggregation):
    path = output / (stem + ".provenance.json")
    path.write_text(
        json.dumps(
            {
                "renderer": "paper_experiments/figures.py",
                "renderer_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "inputs": {
                    name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                    for name in inputs
                },
                "primary_metric": "full_tvd",
                "secondary_metric": "profile_tvd",
                "selection_metric": "profile_tvd",
                "aggregation": aggregation,
                "selection_changed": False,
            },
            indent=2,
        )
        + "\n"
    )
    return path


def _gaussian_arrays(payload):
    conditions = np.asarray(payload["conditions"], dtype=float)
    matrices = {}
    seeds = None
    for control in ("conditional", "encoder_only"):
        rows = sorted(
            (r for r in payload["rows"] if r["control"] == control),
            key=lambda r: r["seed"],
        )
        current_seeds = [r["seed"] for r in rows]
        if not rows or len(set(current_seeds)) != len(rows):
            raise ValueError("Gaussian controls require unique non-empty seeds")
        if seeds is not None and seeds != current_seeds:
            raise ValueError("Gaussian controls have different seed coverage")
        seeds = current_seeds
        for metric, key in (("full", "bitstring_tvd"), ("profile", "profile_tvd")):
            values = np.asarray([r[key] for r in rows], dtype=float)
            if (
                values.shape != (len(rows), len(conditions))
                or not np.isfinite(values).all()
            ):
                raise ValueError("Gaussian TVD values do not cover every condition")
            matrices[control, metric] = values
    return conditions, seeds, matrices


def gaussian_tables(payload):
    """Aggregate each run over conditions, then take the median across seeds."""
    _, seeds, matrices = _gaussian_arrays(payload)
    rows = [
        dict(
            model=label,
            full_tvd=float(np.median(matrices[control, "full"].mean(axis=1))),
            profile_tvd=float(np.median(matrices[control, "profile"].mean(axis=1))),
            seeds=len(seeds),
        )
        for control, label in (
            ("conditional", "Conditional IQP"),
            ("encoder_only", "Encoder only"),
        )
    ]
    return rows


def _native_gaussian(data, output):
    source = ROOT / "paper" / "figure_sources" / "gaussian_intro.tex"
    if (
        not source.exists()
        or not shutil.which("pdflatex")
        or not shutil.which("pdftoppm")
    ):
        return None
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gaussian-figure-") as directory:
        work = Path(directory)
        tex = source.read_text().replace(
            "../data/gaussian/", str(data.resolve()).replace("\\", "/") + "/"
        )
        tex_path = work / "gaussian_translation_profiles.tex"
        tex_path.write_text(tex)
        for _ in range(2):
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    tex_path.name,
                ],
                cwd=work,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        pdf = output / "gaussian_translation_profiles.pdf"
        shutil.copy2(work / "gaussian_translation_profiles.pdf", pdf)
        prefix = work / "gaussian_translation_profiles"
        subprocess.run(
            ["pdftoppm", "-png", "-r", "300", "-singlefile", str(pdf), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        png = output / "gaussian_translation_profiles.png"
        shutil.copy2(prefix.with_suffix(".png"), png)
    return [pdf, png]


def _gaussian(data, output):
    with mpl.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "mathtext.fontset": "cm",
        }
    ):
        return _gaussian_render(data, output)


def _gaussian_render(data, output):
    output.mkdir(parents=True, exist_ok=True)
    payload = json.loads((data / "results.json").read_text())
    conditions, seeds, matrices = _gaussian_arrays(payload)
    encoder = matrices["encoder_only", "full"]
    np.testing.assert_allclose(
        encoder, np.broadcast_to(encoder[0], encoder.shape), atol=1e-12
    )
    mid = json.loads((data / "midpoint_profiles.json").read_text())
    midpoint = np.asarray(mid["conditions"], float)
    profiles = np.asarray([r["profiles"] for r in mid["records"]], float)
    targets = np.asarray(mid["target_profiles"], float)
    encoder_profile = np.asarray(mid["encoder_profile"], float)
    if (
        targets.shape != (4, 17)
        or encoder_profile.shape != (17,)
        or not np.isfinite(targets).all()
        or not np.isfinite(encoder_profile).all()
        or np.min(targets) < 0
        or np.min(encoder_profile) < 0
        or not np.allclose(targets.sum(axis=1), 1, atol=2e-12, rtol=0)
        or not np.isclose(encoder_profile.sum(), 1, atol=2e-12, rtol=0)
    ):
        raise ValueError("invalid Gaussian target or encoder profiles")
    if not np.array_equal(midpoint, np.array([0.125, 0.375, 0.625, 0.875])):
        raise ValueError("Gaussian midpoint conditions changed")
    if profiles.ndim != 3 or profiles.shape[1:] != (4, 17):
        raise ValueError("Gaussian midpoint profile shape changed")
    if [r["seed"] for r in mid["records"]] != seeds:
        raise ValueError("Gaussian midpoint seed coverage changed")
    row_hashes = {
        r["seed"]: r["checkpoint_sha256"]
        for r in payload["rows"]
        if r["control"] == "conditional"
    }
    if any(r["checkpoint_sha256"] != row_hashes.get(r["seed"]) for r in mid["records"]):
        raise ValueError("Gaussian midpoint checkpoint coverage changed")
    if (
        not np.isfinite(profiles).all()
        or np.min(profiles) < 0
        or not np.allclose(profiles.sum(2), 1, atol=2e-12, rtol=0)
    ):
        raise ValueError("invalid Gaussian midpoint profiles")
    k = np.arange(profiles.shape[-1])
    model = profiles.mean(0)
    means = (
        np.asarray(
            [r["profile"] for r in payload["rows"] if r["control"] == "conditional"]
        )
        @ k
    )
    target_mean = np.exp(-((k[None, :] - (5 + 6 * conditions[:, None])) ** 2) / 8)
    target_mean /= target_mean.sum(1, keepdims=True)
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(170 * MM, 68 * MM),
        gridspec_kw={"width_ratios": [2, 1]},
        layout="constrained",
    )
    for color, j in zip(
        (COLORS[0], COLORS[1], COLORS[2], COLORS[4]), range(4), strict=True
    ):
        left.plot(k, model[j], color=color)
        left.plot(k, targets[j], color=color, ls="--")
    left.plot(k, encoder_profile, color="0.5", ls=":")
    left.set(
        xlim=(0, 16),
        ylim=(0, 0.24),
        xticks=[0, 4, 8, 12, 16],
        xlabel="Hamming weight $k$",
        ylabel="Probability mass",
        title="(a) Profiles",
    )
    left.spines[["top", "right"]].set_visible(False)
    median = np.median(means, axis=0)
    lo = np.quantile(means, 0.25, axis=0)
    hi = np.quantile(means, 0.75, axis=0)
    right.plot(conditions, target_mean @ k, color="0.35", ls="--")
    right.plot(
        conditions,
        np.median(
            np.asarray(
                [
                    r["profile"]
                    for r in payload["rows"]
                    if r["control"] == "encoder_only"
                ]
            ),
            axis=0,
        )
        @ k,
        color="0.5",
        ls=":",
    )
    right.fill_between(conditions, lo, hi, color="0.65", alpha=0.25, lw=0)
    right.plot(conditions, median, color="0.1")
    training = np.array([0, 0.25, 0.5, 0.75, 1])
    training_profiles = np.exp(-((k[None, :] - (5 + 6 * training[:, None])) ** 2) / 8)
    training_profiles /= training_profiles.sum(1, keepdims=True)
    right.plot(training, training_profiles.dot(k), "o", color="0.5", ms=5.5)
    right.set(
        xlim=(-0.025, 1.025),
        ylim=(4.4, 11.2),
        xticks=[0, 0.5, 1],
        xlabel="Condition $c$",
        ylabel=r"$E[K]$",
        title="(b) Conditional response",
    )
    right.spines[["top", "right"]].set_visible(False)
    handles = (
        [mpl.lines.Line2D([], [], color="0.35", ls="--", label="Target")]
        + [
            mpl.lines.Line2D([], [], color=x, label=rf"$c={y:.3f}$")
            for x, y in zip(
                (COLORS[0], COLORS[1], COLORS[2], COLORS[4]), midpoint, strict=True
            )
        ]
        + [
            mpl.lines.Line2D([], [], color="0.5", ls=":", label="Encoder"),
            mpl.lines.Line2D(
                [],
                [],
                marker="o",
                color="0.5",
                ls="none",
                markersize=5.5,
                label="Training c",
            ),
        ]
    )
    fig.legend(
        # Matplotlib fills columns; this order preserves row-wise reading.
        handles=[handles[index] for index in (0, 4, 1, 5, 2, 6, 3)],
        loc="outside upper center",
        ncol=4,
        fontsize=9,
        columnspacing=0.7,
    )
    midpoint_csv = output / "midpoint_profiles.csv"
    with midpoint_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "k",
                *[item for j in range(4) for item in (f"target_{j}", f"model_{j}")],
                "encoder",
            ]
        )
        for row in range(17):
            writer.writerow(
                [
                    row,
                    *[
                        item
                        for j in range(4)
                        for item in (targets[j, row], model[j, row])
                    ],
                    encoder_profile[row],
                ]
            )
    response_csv = output / "response.csv"
    with response_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["c", "target", "model", "lower", "upper", "encoder"])
        encoder_mean = (
            np.median(
                np.asarray(
                    [
                        r["profile"]
                        for r in payload["rows"]
                        if r["control"] == "encoder_only"
                    ]
                ),
                axis=0,
            )
            @ k
        )
        for j, value in enumerate(conditions):
            writer.writerow(
                [value, target_mean[j] @ k, median[j], lo[j], hi[j], encoder_mean[j]]
            )
    training_csv = output / "training_response.csv"
    with training_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["c", "target"])
        for value, profile in zip(training, training_profiles, strict=True):
            writer.writerow([value, profile @ k])
    paths = _save(fig, output, "gaussian_translation_profiles")
    paths.extend([midpoint_csv, response_csv, training_csv])
    rows = gaussian_tables(payload)
    best_full = min(row["full_tvd"] for row in rows)
    best_profile = min(row["profile_tvd"] for row in rows)
    for row in rows:
        row["best_full"] = np.isclose(row["full_tvd"], best_full, atol=1e-12, rtol=0)
        row["best_profile"] = np.isclose(
            row["profile_tvd"], best_profile, atol=1e-12, rtol=0
        )
    table = output / "gaussian_tvd.csv"
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "full_tvd", "profile_tvd", "seeds"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    paths.append(table)
    table = output / "gaussian_tvd.tex"

    def _gaussian_row(row):
        full = (
            rf"\boldsymbol{{{row['full_tvd']:.4f}}}"
            if row["best_full"]
            else f"{row['full_tvd']:.4f}"
        )
        profile = (
            rf"\boldsymbol{{{row['profile_tvd']:.4f}}}"
            if row["best_profile"]
            else f"{row['profile_tvd']:.4f}"
        )
        return rf"{row['model']} & ${full}$ & ${profile}$ \\"

    selection_label = (
        "development full-distribution TVD"
        if payload.get("selection_metric") in ("full_tvd", "mean_exact_development_tvd")
        else "development-profile TVD"
    )
    table.write_text(
        "\\begin{table}[!htbp]\n\\centering\n\\small\n"
        "\\caption{Exact Gaussian full-distribution and secondary profile TVD. "
        f"Each value is the median across {len(seeds)} runs of the mean error "
        f"over {len(conditions)} held-out conditions. "
        f"Selection uses {selection_label}; bold entries mark the lowest "
        "value in each column.}\n"
        "\\label{tab:gaussian_tvd}\n\\begin{tabular}{lcc}\n\\toprule\n"
        r"Model & Full TVD $\downarrow$ & Profile TVD $\downarrow$ \\"
        + "\n\\midrule\n"
        + "\n".join(_gaussian_row(r) for r in rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    paths.append(table)
    provenance_path = _provenance(
        data,
        output,
        "gaussian_translation_profiles",
        ["results.json", "midpoint_profiles.json"],
        "Profile panels show mean probabilities across five midpoint runs; "
        "the response shows median per-run expected Hamming weight with IQR; "
        "target and encoder curves are exact.",
    )
    provenance = json.loads(provenance_path.read_text())
    provenance["selection_metric"] = (
        "full_tvd"
        if payload.get("selection_metric") in ("full_tvd", "mean_exact_development_tvd")
        else "profile_tvd"
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    paths.append(provenance_path)
    return paths


def curie_tables(payload):
    rows = []
    for family in ("shared", "independent_conditional", "independent_unconditional"):
        metrics = payload["metrics"][family]
        for kind in ("full", "profile"):
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
                        evaluation="Held-out" if family == "shared" else "Fitted",
                        metric=kind + "_tvd",
                        beta=beta,
                        mean=float(mean[i]),
                        sample_sd=float(sd[i]),
                        seeds=len(values),
                    )
                )
    return rows


def _curie_encoder_tvd(payload):
    # Both the exact target and uniform encoder are uniform within each shell.
    target = np.asarray(payload["records"]["shared"][0]["target_profile"])
    encoder = np.asarray(payload["arms"]["encoder_only"]["mean"])
    return 0.5 * np.abs(target - encoder).sum(axis=1)


def _curie_tex(payload, kind):
    rows = curie_tables(payload)
    encoder = _curie_encoder_tvd(payload)
    display_label = "full-distribution" if kind == "full" else "Hamming-profile"
    families = tuple(MODEL_NAMES)
    means = np.stack(
        [
            encoder,
            *[np.asarray(payload["metrics"][f][kind + "_mean"]) for f in families[1:]],
        ]
    )
    best = means.min(axis=0)
    lines = []
    for j, family in enumerate(families):
        cells = [
            MODEL_NAMES[family],
            "Baseline" if j == 0 else "Held-out" if family == "shared" else "Fitted",
        ]
        for i, beta in enumerate(payload["betas"]):
            if j == 0:
                value = f"{encoder[i]:.4f}"
            else:
                row = next(
                    r
                    for r in rows
                    if r["model"] == MODEL_NAMES[family]
                    and r["metric"] == kind + "_tvd"
                    and r["beta"] == beta
                )
                value = f"{row['mean']:.4f}"
                if np.isfinite(row["sample_sd"]):
                    value += rf" \pm {row['sample_sd']:.4f}"
            is_best = np.isclose(means[j, i], best[i], atol=1e-12, rtol=0)
            cells.append("$" + (rf"\boldsymbol{{{value}}}" if is_best else value) + "$")
        lines.append(" & ".join(cells) + r" \\")
    return (
        "\\begin{table*}[!htbp]\n\\centering\n\\small\n"
        f"\\caption{{Exact {display_label} TVD for sixteen-qubit Curie--Weiss models. "
        "Trained entries give means $\\pm$ sample standard deviations across "
        f"{len(payload['seeds'])} runs from the confirmation cohort used for model "
        "selection, reused for reporting; the encoder is deterministic. Shared "
        "reporting temperatures "
        "are excluded from fitting and current selection, while independent circuits "
        "fit at each reporting temperature. Bold entries mark the lowest mean error, "
        "including the encoder.}\n" + f"\\label{{tab:curie_weiss_{kind}_tvd}}\n"
        "\\setlength{\\tabcolsep}{5pt}\n\\renewcommand{\\arraystretch}{1.12}\n"
        "\\begin{tabular}{llccc}\n\\toprule\n"
        + "Model & Evaluation & "
        + rf"\multicolumn{{3}}{{c}}{{{display_label.capitalize()} TVD $\downarrow$}} \\"
        + "\n"
        + r"\cmidrule(lr){3-5}"
        + "\n"
        + " & & "
        + " & ".join(rf"$\beta={b:.1f}$" for b in payload["betas"])
        + r" \\"
        + "\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


def _curie(data, output):
    payload = json.loads((data / "figure_data.json").read_text())
    targets = np.asarray(payload["records"]["shared"][0]["target_profile"])
    arms = payload["arms"]
    families = (
        "encoder_only",
        "shared",
        "independent_conditional",
        "independent_unconditional",
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(170 * MM, 120 * MM),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    x = (2 * np.arange(17) - 16) / 16
    maximum = max(targets.max(), *(np.asarray(v["mean"]).max() for v in arms.values()))
    for panel, (family, ax) in enumerate(zip(families, axes.flat, strict=True)):
        title = {
            "encoder_only": "Encoder only",
            "shared": "Shared conditional IQP\nacross conditions",
            "independent_conditional": "Independent conditional IQP\nper condition",
            "independent_unconditional": "Independent unconditional IQP\nper condition",
        }[family]
        ax.set_title(f"({chr(97 + panel)}) {title}", fontsize=9)
        for i, _beta in enumerate(payload["betas"]):
            if family != "encoder_only" or i == 0:
                ax.plot(
                    x,
                    np.asarray(arms[family]["mean"])[i],
                    color=COLORS[i] if family != "encoder_only" else "0.3",
                    marker="o",
                    ms=2.5,
                    lw=1.3,
                )
            ax.plot(x, targets[i], "--", color=COLORS[i], lw=1.1)
        ax.set_ylim(0, maximum * 1.08)
        ax.set_xlim(-1.03, 1.03)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Probability mass")
    axes[1, 0].set_ylabel("Probability mass")
    fig.supxlabel(r"Magnetization per spin $m$", fontsize=10)
    handles = [plt.Line2D([], [], color="0.3", ls="--", label="Target")]
    handles.extend(
        plt.Line2D([], [], color=COLORS[i], label=rf"$\beta={beta:.2f}$")
        for i, beta in enumerate(payload["betas"])
    )
    fig.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=8)
    paths = _save(fig, output, "curie_weiss_model_comparison")
    rows = curie_tables(payload)
    table = output / "curie_weiss_tvd.csv"
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths.append(table)
    for kind in ("full", "profile"):
        table = output / f"curie_weiss_{kind}_tvd.tex"
        table.write_text(_curie_tex(payload, kind))
        paths.append(table)
    paths.append(
        _provenance(
            data,
            output,
            "curie_weiss_model_comparison",
            ["figure_data.json"],
            "Mean raw profiles; full-TVD tables: mean/sample SD of "
            "individual-seed errors. Encoder is deterministic.",
        )
    )
    if (data / "response_sweep.json").exists():
        from .physical_response import render as render_physical_response

        paths.extend(render_physical_response(data, output))
    return paths


def _scaling(data, output):
    payload = json.loads((data / "figure_data.json").read_text())
    fig, ax = plt.subplots(figsize=(85 * MM, 66 * MM), layout="constrained")
    rows = []
    for n in payload["sizes"]:
        matrix = np.asarray(payload["curves"][str(n)]["held_out"]["overall_per_seed"])
        if matrix.shape != (len(payload["seeds"]), len(payload["steps"])):
            raise ValueError("scaling seed/step coverage differs")
        if not np.isfinite(matrix).all():
            raise ValueError("scaling curves contain non-finite values")
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
    ax.set_ylabel(r"Held-out MMD$^2$")
    ax.legend(ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    paths = _save(fig, output, "curie_weiss_lattice_scaling_mmd")
    table = output / "scaling_mmd.csv"
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = output / "curie_weiss_lattice_scaling_mmd.provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "renderer": "paper_experiments/figures.py",
                "renderer_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "input_sha256": hashlib.sha256(
                    (data / "figure_data.json").read_bytes()
                ).hexdigest(),
                "primary_metric": "held_out_sampled_mmd_squared",
                "aggregation": "median and interquartile range across seeds",
                "encoder": payload.get("model_metadata", {}).get("encoder"),
                "ansatz": payload.get("model_metadata", {}).get("ansatz_by_size"),
            },
            indent=2,
        )
        + "\n"
    )
    return [*paths, table, provenance]


def export_completed(root, destination):
    """Package newly completed reported jobs, never the checked-in paper data."""
    root = Path(root)
    destination = Path(destination)
    records = json.loads((root / "reported-results.json").read_text())
    supported = {"gaussian", "scaling"}
    if any(record["experiment"] not in supported for record in records):
        raise ValueError(
            "reported export supports Gaussian translation and scaling only; "
            "use frozen compact data for historical Curie-Weiss replay"
        )
    by_kind = {
        kind: [record for record in records if record["experiment"] == kind]
        for kind in sorted(supported)
    }
    available = []
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
        rows = []
        conditions = None
        weights = np.bitwise_count(np.arange(65536, dtype=np.uint32))
        shell_counts = np.bincount(weights, minlength=17)
        encoder_probabilities = (encoder / shell_counts)[weights]
        for record in sorted(by_kind["gaussian"], key=lambda r: r["seed"]):
            with np.load(Path(record["directory"]) / "checkpoint.npz") as ck:
                grid = np.asarray(ck["evaluation_conditions"])
                if conditions is None:
                    conditions = grid
                else:
                    np.testing.assert_allclose(grid, conditions, atol=1e-12)
                targets = np.asarray(ck["held_out_target_probabilities"])
                target_profiles = np.asarray(
                    [np.bincount(weights, weights=v, minlength=17) for v in targets]
                )
                for control, probabilities in (
                    ("conditional", np.asarray(ck["held_out_probabilities"])),
                    (
                        "encoder_only",
                        np.broadcast_to(encoder_probabilities, targets.shape),
                    ),
                ):
                    profiles = np.asarray(
                        [
                            np.bincount(weights, weights=v, minlength=17)
                            for v in probabilities
                        ]
                    )
                    rows.append(
                        dict(
                            control=control,
                            seed=record["seed"],
                            checkpoint_sha256=hashlib.sha256(
                                (
                                    Path(record["directory"]) / "checkpoint.npz"
                                ).read_bytes()
                            ).hexdigest(),
                            selection_metric=record.get(
                                "selection_metric", "profile_tvd"
                            ),
                            profile=profiles.tolist(),
                            target_profile=target_profiles.tolist(),
                            profile_tvd=(
                                0.5 * np.abs(profiles - target_profiles).sum(1)
                            ).tolist(),
                            bitstring_tvd=(
                                0.5 * np.abs(probabilities - targets).sum(1)
                            ).tolist(),
                        )
                    )
        selection_metrics = {row["selection_metric"] for row in rows}
        if len(selection_metrics) != 1:
            raise ValueError("Gaussian runs have inconsistent selection metrics")
        (p / "results.json").write_text(
            json.dumps(
                dict(
                    conditions=conditions.tolist(),
                    selection_metric=selection_metrics.pop(),
                    rows=rows,
                    aggregation=(
                        "Median across seeds of mean error over reporting conditions"
                    ),
                ),
                indent=2,
            )
            + "\n"
        )
        from .gaussian import export_midpoint_profiles

        export_midpoint_profiles(
            {
                r["seed"]: Path(r["directory"]) / "checkpoint.npz"
                for r in by_kind["gaussian"]
            },
            p / "midpoint_profiles.json",
        )
        available.append("gaussian")
    if by_kind["scaling"]:
        sizes = sorted({r["job"]["size"] for r in by_kind["scaling"]})
        seeds = sorted({r["seed"] for r in by_kind["scaling"]})
        steps = list(range(0, 5001, 500))
        if records[0].get("smoke"):
            steps = [0, 2, 4]
        curves = {}
        encoders = set()
        encoder_scales = set()
        ansatz_by_size = {}
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
                    record = cases[0]
                    config = record.get("job", {}).get("config", {})
                    encoders.add(config.get("encoder", record.get("encoder")))
                    encoder_scales.add(config.get("encoder_scale"))
                    ansatz = record.get("ansatz", record.get("job", {}).get("ansatz"))
                    if ansatz is not None:
                        ansatz_by_size.setdefault(str(n), set()).add(
                            json.dumps(ansatz, sort_keys=True)
                        )
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
        if len(encoders) > 1:
            raise ValueError("scaling records use mixed encoders")
        if len(encoder_scales) > 1:
            raise ValueError("scaling records use mixed encoder scales")
        payload = dict(
            sizes=sizes,
            seeds=seeds,
            steps=steps,
            curves=curves,
            model_metadata=dict(
                encoder=next(iter(encoders), None),
                encoder_scale=next(iter(encoder_scales), None),
                ansatz_by_size={
                    n: [json.loads(value) for value in sorted(values)]
                    for n, values in ansatz_by_size.items()
                },
                job_ids=sorted(r["job"]["job_id"] for r in by_kind["scaling"]),
            ),
        )
        p = destination / "scaling"
        p.mkdir(parents=True, exist_ok=True)
        (p / "figure_data.json").write_text(json.dumps(payload, indent=2) + "\n")
        available.append("scaling")
    return available


def render_all(data_root=None, output_root=None, experiment="all"):
    data = Path(data_root) if data_root else ROOT / "paper/data"
    output = Path(output_root) if output_root else ROOT / "results/paper/figures"
    followup_data = data / "curie-weiss-followup" / "figure_data.json"
    if experiment == "curie-weiss-followup":
        experiment = "curie-weiss"
    if experiment == "curie-weiss" and not followup_data.exists():
        raise ValueError(
            "current Curie-Weiss figures require completed final data; "
            "earlier appendix evidence uses curie-weiss-historical"
        )
    gaussian_lattice = data / "gaussian_lattice"
    has_lattice = (gaussian_lattice / "SUMMARY.json").exists()
    available = [name for name in ("gaussian", "scaling") if (data / name).is_dir()]
    if has_lattice:
        available = [name for name in available if name != "gaussian"]
        available.append("gaussian-lattice")
    if (data / "gaussian").is_dir():
        available.append("gaussian-historical")
    if followup_data.exists():
        available.append("curie-weiss")
    if (data / "curie-weiss").is_dir():
        available.append("curie-weiss-historical")
    if (data / "reported-results.json").exists():
        available = export_completed(data, output / "data")
        data = output / "data"
    phase = data / "gaussian2d_phase"
    switching = data / "switching16"
    if phase.exists():
        available.append("gaussian2d-phase")
    if switching.exists():
        available.append("switching16")
    renderers = {
        "gaussian": _gaussian,
        "gaussian-historical": _gaussian,
        "curie-weiss-historical": _curie,
        "scaling": _scaling,
    }
    if has_lattice:
        from paper.figure_sources.gaussian_lattice import main as render_lattice

        renderers["gaussian-lattice"] = lambda data_dir, output_dir: (
            render_lattice(["--results", str(data_dir), "--output", str(output_dir)])
            or [
                output_dir / name
                for name in (
                    "gaussian_translation_profiles.pdf",
                    "gaussian_translation_profiles.png",
                    "gaussian2d_lattice_heatmaps.pdf",
                    "gaussian2d_lattice_heatmaps.png",
                    "gaussian2d_lattice_mean_response.pdf",
                    "gaussian2d_lattice_mean_response.png",
                    "gaussian2d_lattice_width_response.pdf",
                    "gaussian2d_lattice_width_response.png",
                )
                if (output_dir / name).exists()
            ]
        )
    if "curie-weiss" in available:
        from .cw_followup_reporting import render as render_followup

        renderers["curie-weiss"] = render_followup
    if "gaussian2d-phase" in available:
        from paper.figure_sources.gaussian2d_phase import main as render_phase

        renderers["gaussian2d-phase"] = lambda data_dir, output_dir: (
            render_phase(["--results", str(data_dir), "--output", str(output_dir)])
            or list(output_dir.glob("gaussian2d_phase_*"))
        )
    if "switching16" in available:
        from paper.data.switching16.render import main as render_switching_main

        def render_switching(data_dir, output_dir):
            return render_switching_main(data_dir, output_dir)

        renderers["switching16"] = render_switching
    if experiment == "gaussian" and has_lattice:
        experiment = "gaussian-lattice"
    selected = (
        [
            name
            for name in available
            if name
            not in {"curie-weiss-historical", "gaussian-historical", "switching16"}
            and not (has_lattice and name == "gaussian2d-phase")
        ]
        if experiment == "all"
        else [experiment]
    )
    paths = []
    for name in selected:
        if name not in available:
            raise ValueError("requested experiment has no completed data")
        input_name = {
            "curie-weiss": "curie-weiss-followup",
            "curie-weiss-historical": "curie-weiss",
            "gaussian2d-phase": "gaussian2d_phase",
            "gaussian-lattice": "gaussian_lattice",
            "gaussian-historical": "gaussian",
        }.get(name, name)
        paths.extend(renderers[name](data / input_name, output))
    return paths

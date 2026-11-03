"""Render the paper's numerical figures and tables from the compact data bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
COLORS = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
ROOT = Path(__file__).resolve().parents[1]
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.titlesize": 10,
        "axes.prop_cycle": mpl.cycler(color=COLORS),
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "axes.xmargin": 0.02,
        "axes.ymargin": 0.02,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        **(
            {
                "text.usetex": True,
                "text.latex.preamble": (
                    r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{lmodern} "
                    r"\renewcommand{\familydefault}{\sfdefault} "
                    r"\usepackage{sansmath} \sansmath"
                ),
            }
            if shutil.which("latex")
            else {"text.usetex": False, "mathtext.fontset": "cm"}
        ),
    }
)


def _save(fig, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / (stem + ".pdf"), output / (stem + ".png")]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
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
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        ax.plot(payload["steps"], mean, lw=1.2, marker="o", ms=2.5, label=rf"$n={n}$")
        ax.fill_between(payload["steps"], mean - sd, mean + sd, alpha=0.13)
        rows.extend(
            dict(
                qubits=n,
                step=step,
                mean=float(mean[i]),
                sd=float(sd[i]),
            )
            for i, step in enumerate(payload["steps"])
        )
    ax.set_xlabel("Optimizer updates")
    # A logarithmic axis separates the sizes after the initial drop.
    ax.set_yscale("log")
    ax.set_yticks([0.002, 0.005, 0.01, 0.02, 0.04])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylabel(r"Test MMD$^2$")
    ax.legend(ncol=2, fontsize=8, columnspacing=1.0, handlelength=1.6)
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
                "aggregation": "mean and sample standard deviation across seeds",
                "encoder": payload.get("model_metadata", {}).get("encoder"),
                "ansatz": payload.get("model_metadata", {}).get("ansatz_by_size"),
            },
            indent=2,
        )
        + "\n"
    )
    if (data / "physics" / "dense").is_dir():
        return [*paths, table, provenance, *_scaling_observables(data, output)]
    return [*paths, table, provenance]


def _curie_weiss_m2(n, beta):
    """Exact <m^2> of the n-spin Curie--Weiss target."""
    k = np.arange(n + 1)
    m = (2 * k - n) / n
    log_weight = (
        np.array(
            [
                math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                for i in k
            ]
        )
        + beta * n * m**2 / 2
    )
    weight = np.exp(log_weight - log_weight.max())
    return float((weight * m**2).sum() / weight.sum())


def _scaling_observables(data, output):
    """Classically estimated <m^2> of the selected scaling fits against the target."""
    if (data / "physics" / "jobs").is_dir():
        receipts = {
            job.name: json.loads((job / "provenance.json").read_text())["identity"][
                "fit_receipt_sha256"
            ]
            for job in (data / "physics" / "jobs").iterdir()
        }
    else:
        receipts = {
            fit.stem: hashlib.sha256(fit.read_bytes()).hexdigest()
            for fit in (data / "physics" / "fits").glob("*.json")
        }
    estimates, inputs, betas = {}, {}, None
    for path in sorted((data / "physics" / "dense").glob("*.json")):
        record = json.loads(path.read_text())
        if record["fit_receipt_sha256"] != receipts[record["job_id"]]:
            raise ValueError("dense observables do not belong to the final fits")
        if betas is None:
            betas = np.asarray(record["betas"], dtype=float)
        elif not np.array_equal(betas, record["betas"]):
            raise ValueError("dense observable temperatures differ")
        estimates.setdefault(record["size"], []).append(record["m2"])
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Two panels: (a) held-out MMD during training, (b) error in <m^2>.
    payload = json.loads((data / "figure_data.json").read_text())
    fig, (ax_mmd, ax) = plt.subplots(
        1, 2, figsize=(170 * MM, 64 * MM), layout="constrained"
    )
    for color, n in zip(COLORS, payload["sizes"], strict=False):
        matrix = np.asarray(payload["curves"][str(n)]["held_out"]["overall_per_seed"])
        mmd_mean, mmd_sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
        ax_mmd.plot(payload["steps"], mmd_mean, color=color, lw=1.2, marker="o", ms=2.5)
        ax_mmd.fill_between(
            payload["steps"],
            mmd_mean - mmd_sd,
            mmd_mean + mmd_sd,
            color=color,
            alpha=0.13,
        )
    ax_mmd.set_yscale("log")
    ax_mmd.set_yticks([0.002, 0.005, 0.01, 0.02, 0.04])
    ax_mmd.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax_mmd.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax_mmd.set(xlabel="Optimizer updates", ylabel=r"Test MMD$^2$")
    ax_mmd.spines[["top", "right"]].set_visible(False)
    ax_mmd.set_title("(a)", loc="left", fontsize=9)
    ax.set_title("(b)", loc="left", fontsize=9)
    rows = []
    for color, n in zip(COLORS[: len(estimates)], sorted(estimates), strict=True):
        values = np.asarray(estimates[n])
        if values.shape[0] != 3:
            raise ValueError("scaling observables require three fits per size")
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        target = np.array([_curie_weiss_m2(n, beta) for beta in betas])
        ax.plot(betas, mean - target, color=color, lw=1.3, label=rf"$n={n}$")
        ax.fill_between(
            betas, mean - sd - target, mean + sd - target, color=color, alpha=0.15
        )
        rows.extend(
            dict(
                qubits=n,
                beta=float(b),
                model_mean=float(mu),
                model_sd=float(s),
                target=float(t),
            )
            for b, mu, s, t in zip(betas, mean, sd, target, strict=True)
        )
    training = json.loads((data / "scaling_protocol.json").read_text())[
        "training_betas"
    ]
    for beta in training:
        ax.axvline(beta, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(r"$\beta$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set(ylabel=r"Error in $\langle m^2\rangle$")
    ax.axhline(0, color="k", lw=0.8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, fontsize=8)
    paths = _save(fig, output, "curie_weiss_lattice_scaling_observables")
    table = output / "scaling_observables.csv"
    with table.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = output / "curie_weiss_lattice_scaling_observables.provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "renderer_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "input_sha256": inputs,
                "estimator": "<m^2> = 1/n + (n-1)/n * mean distinct-pair <Z_i Z_j>; "
                "5 banks x 256 uniform pairs x 4096 reference draws per temperature",
                "aggregation": (
                    "mean over banks within a fit; mean and sample SD across fits"
                ),
                "checkpoint": "selected",
            },
            indent=2,
        )
        + "\n"
    )
    return [*paths, table, provenance]


def _gaussian_lattice(data, output):
    from .gaussian_lattice_figures import main as render_lattice

    diagonal = data.parent / "gaussian2d-diagonal"
    variants = data.parent / "gaussian2d-diagonal-variants"
    extra = []
    if (diagonal / "SUMMARY.json").exists():
        extra += ["--diagonal", str(diagonal)]
    if variants.is_dir():
        extra += ["--variants", str(variants)]
    return render_lattice(["--results", str(data), "--output", str(output), *extra])


def _curie_weiss(data, output):
    from .cw_followup_reporting import render as render_followup

    paths = render_followup(data, output)
    classical = data.parent / "curie-weiss-classical"
    if classical.is_dir():
        from .cw_classical_report import paper_artifacts

        paths += paper_artifacts(classical, data, output)
    return paths


# Experiment name, data directory and renderer, in replay order.
EXPERIMENTS = {
    "scaling": ("scaling", _scaling),
    "gaussian-lattice": ("gaussian_lattice", _gaussian_lattice),
    "curie-weiss": ("curie-weiss-followup", _curie_weiss),
}


def _available(data, name):
    directory = data / EXPERIMENTS[name][0]
    marker = {
        "scaling": directory,
        "gaussian-lattice": directory / "SUMMARY.json",
        "curie-weiss": directory / "figure_data.json",
    }[name]
    return marker.exists()


def render_all(data_root=None, output_root=None, experiment="all"):
    data = Path(data_root) if data_root else ROOT / "data"
    output = Path(output_root) if output_root else ROOT / "results/figures"
    if experiment == "all":
        selected = [name for name in EXPERIMENTS if _available(data, name)]
    elif experiment not in EXPERIMENTS or not _available(data, experiment):
        raise ValueError("requested experiment has no completed data")
    else:
        selected = [experiment]
    paths = []
    for name in selected:
        directory, renderer = EXPERIMENTS[name]
        paths.extend(renderer(data / directory, output))
    return paths

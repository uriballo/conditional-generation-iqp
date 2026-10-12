"""Publication figures for the Gaussian lattice and translation replay bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

MM = 1 / 25.4
OKABE = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
WHITE_BLUES = LinearSegmentedColormap.from_list(
    "white_blues", ["#FFFFFF", "#DEEBF7", "#3182BD", "#08519C"]
)
STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "axes.prop_cycle": mpl.cycler(color=OKABE),
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "axes.xmargin": 0.02,
    "axes.ymargin": 0.02,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    **(
        {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"}
        if shutil.which("latex")
        else {"text.usetex": False, "mathtext.fontset": "cm"}
    ),
}


def load(root: Path):
    summary = json.loads((root / "SUMMARY.json").read_text())
    datafile = root / "DIAGNOSTICS.npz"
    digest = hashlib.sha256(datafile.read_bytes()).hexdigest()
    if summary.get("diagnostic_npz_sha256") != digest:
        raise ValueError("DIAGNOSTICS.npz hash does not match SUMMARY.json")
    a = dict(np.load(datafile))
    required = {
        "grid_tvd",
        "grid_profile_tvd",
        "grid_u",
        "grid_v",
        "fitting_grid",
        "fitting_axes",
        "model_mean",
        "model_sd",
        "target_mean",
        "target_sd",
        "u_axis",
        "v_axis",
        "translation_profiles",
        "translation_target_profiles",
        "translation_profile_c",
        "translation_development_c",
        "translation_response",
        "translation_target_mean",
        "translation_reporting_c",
        "translation_fitting_c",
        "translation_fitting_target_mean",
    }
    if not required <= set(a):
        raise ValueError(f"diagnostics missing fields: {sorted(required - set(a))}")
    if summary.get("protocols") != ["grid", "axes"]:
        raise ValueError("protocol order must be ['grid', 'axes']")
    if a["grid_tvd"].shape != (2, 5, 41, 41) or a["grid_profile_tvd"].shape != (
        2,
        5,
        41,
        41,
    ):
        raise ValueError("unexpected grid shape")
    if a["model_mean"].shape != (2, 5, 3, 21) or a["model_sd"].shape != (2, 5, 3, 21):
        raise ValueError("unexpected response shape")
    return a, summary, digest


def save(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)


def lattice_heatmaps(a, out):
    values = a["grid_tvd"][1].mean(axis=0)
    vmax = float(np.ceil(values.max() * 100) / 100)
    vmax = max(vmax, 1e-12)
    fig, ax = plt.subplots(figsize=(85 * MM, 90 * MM), layout="constrained")
    image = ax.imshow(
        values,
        origin="lower",
        extent=(-1.025, 1.025, -1.025, 1.025),
        vmin=0,
        vmax=vmax,
        cmap=WHITE_BLUES,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title("Axis training", fontsize=9)
    ax.set_xlabel("Mean condition $u$")
    ax.set_ylabel("Width condition $v$")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    train = a["fitting_axes"]
    ax.scatter(
        train[:, 0], train[:, 1], marker="+", color="0.35", s=35,
        linewidths=1, label="Training"
    )
    hull = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [-1, 0]])
    ax.plot(
        hull[:, 0], hull[:, 1], "--", color="0.4", lw=1,
        label="Training hull",
    )
    ax.scatter(
        [0], [0], facecolors="white", edgecolors="0.2", s=34, zorder=4,
        label="Fixed origin"
    )
    fig.colorbar(
        image, ax=ax, label="Mean full TVD", shrink=0.68, ticks=np.linspace(0, vmax, 5)
    )
    fig.legend(
        loc="outside lower center", ncol=3, fontsize=8,
        columnspacing=1.0, handletextpad=0.4
    )
    save(fig, out, "gaussian2d_lattice_heatmaps")


def lattice_response(a, out):
    labels = ["Axis training", "Grid training"]
    for name, values, ylabel in (
        ("mean_response", a["model_mean"], "Mean Hamming weight"),
        ("width_response", a["model_sd"], "Standard deviation of $K$"),
    ):
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(170 * MM, 105 * MM),
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        for p, label in enumerate(labels):
            for j, fixed in enumerate([-1, 0, 1]):
                ax = axes[p, j]
                data = values[1 - p, :, j]
                ax.plot(
                    a["u_axis" if name == "mean_response" else "v_axis"],
                    np.median(data, axis=0),
                    color=OKABE[0],
                    label="Five-fit median" if p == j == 0 else None,
                )
                ax.fill_between(
                    a["u_axis" if name == "mean_response" else "v_axis"],
                    np.quantile(data, 0.25, axis=0),
                    np.quantile(data, 0.75, axis=0),
                    color=OKABE[0],
                    alpha=0.18,
                    linewidth=0,
                )
                target = a["target_mean" if name == "mean_response" else "target_sd"][j]
                ax.plot(
                    a["u_axis" if name == "mean_response" else "v_axis"],
                    target,
                    "k--",
                    label="Exact target" if p == j == 0 else None,
                )
                ax.axhline(
                    8 if name == "mean_response" else 2,
                    color="0.5",
                    ls=":",
                    label="Encoder-only" if p == j == 0 else None,
                )
                ax.set_title(
                    rf"$v={fixed}$" if name == "mean_response" else rf"$u={fixed}$"
                )
                if j == 0:
                    ax.set_ylabel(f"{label}\n{ylabel}")
                if p == 1:
                    ax.set_xlabel(
                        "Mean condition $u$"
                        if name == "mean_response"
                        else "Width condition $v$"
                    )
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
        )
        save(fig, out, f"gaussian2d_lattice_{name}")


def translation(a, out):
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 78 * MM), layout="constrained")
    k = np.arange(17)
    uniform = np.array([math.comb(16, int(i)) for i in k], dtype=float) / 2**16
    for i in range(4):
        axes[0].plot(
            np.arange(17),
            a["translation_profiles"][:, i].mean(0),
            color=OKABE[i],
            label=rf"$\langle K\rangle_{{\mathrm{{target}}}}"
            rf"\approx{(a['translation_target_profiles'][i] @ k):.2f}$",
        )
        axes[0].plot(
            np.arange(17), a["translation_target_profiles"][i], "--", color=OKABE[i]
        )
    axes[0].set(
        xlabel="Hamming weight $k$",
        ylabel="Probability",
        title="(a) Profiles",
        xlim=(0, 16),
        xticks=[0, 4, 8, 12, 16],
    )
    axes[0].plot(k, uniform, ":", color="0.5", label="Encoder-only")
    condition_handles = axes[0].get_legend_handles_labels()[0][:4]
    fig.legend(
        handles=condition_handles, loc="outside upper center", ncol=4, fontsize=9
    )
    mean = np.median(a["translation_response"], axis=0)
    lo, hi = np.quantile(a["translation_response"], [0.25, 0.75], axis=0)
    axes[1].plot(
        a["translation_reporting_c"], mean, color=OKABE[0], label="Five-fit median"
    )
    axes[1].fill_between(
        a["translation_reporting_c"], lo, hi, color=OKABE[0], alpha=0.18
    )
    axes[1].plot(
        a["translation_reporting_c"],
        a["translation_target_mean"],
        "k--",
        label="Exact target",
    )
    axes[1].axhline(8, color="0.5", ls=":", label="Encoder-only")
    axes[1].scatter(
        a["translation_fitting_c"],
        a["translation_fitting_target_mean"],
        marker="+",
        color="0.35",
        label="Training marker",
    )
    axes[1].scatter(
        [0.5],
        [8],
        facecolors="white",
        edgecolors="0.2",
        label="Fixed baseline",
    )
    axes[1].set(
        xlabel="Translation condition $c$",
        ylabel="Mean Hamming weight",
        title="(b) Mean response",
    )
    handles = [
        plt.Line2D([], [], color="0.25", label="Model"),
        plt.Line2D([], [], color="0.25", ls="--", label="Target"),
        plt.Line2D([], [], color="0.5", ls=":", label="Encoder only"),
        plt.Line2D([], [], color="0.35", marker="+", ls="", label="Fitting conditions"),
        plt.Line2D(
            [], [], color="0.2", marker="o", mfc="white", ls="", label="Fixed center"
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "gaussian_translation_profiles")


def tables(summary, out):
    """Recompute manuscript errors from per-condition, per-fit evidence."""
    results = summary["results"]

    def value(design, key, median=False):
        entries = np.asarray(
            [np.mean(row["selected"][key]) for row in results[design]["fits"]]
        )
        if median:
            return rf"${np.median(entries):.4f}$"
        return rf"${entries.mean():.4f}\pm{entries.std(ddof=1):.4f}$"

    def quartile_value(design, key):
        entries = np.asarray(
            [np.mean(row["selected"][key]) for row in results[design]["fits"]]
        )
        median = np.median(entries)
        lower, upper = np.quantile(entries, [0.25, 0.75])
        return rf"\shortstack{{${median:.4f}$\\$[{lower:.4f},{upper:.4f}]$}}"

    translation_tex = r"""\begin{table}[!htbp]
\centering
\small
""" + (
        r"\caption{Exact full-distribution and profile TVDs for conditional generation "
        r"of binomial distributions. Entries report the median across five independent "
        r"fits of the mean error over eighty reporting conditions; "
        r"brackets give the 25th--75th percentiles across fits. Lower is better.}" "\n"
        r"\label{tab:gaussian_tvd}" "\n"
        r"\begin{tabular}{lcc}\toprule" "\n"
        r"Model & Full TVD & Profile TVD\\\midrule" "\n"
    )
    translation_tex += (
        "Conditional IQP & "
        + quartile_value("translation", "full_tvd")
        + " & "
        + quartile_value("translation", "profile_tvd")
        + r"\\"
        + "\n"
    )
    encoder = results["translation"]["fits"][0]["encoder_mean_full_tvd"]
    translation_tex += (
        rf"Encoder only & ${encoder:.4f}$ & ${encoder:.4f}$\\"
        + "\n"
        + r"\bottomrule\end{tabular}"
        + "\n"
        + r"\end{table}"
        + "\n"
    )
    (out / "gaussian_tvd.tex").write_text(translation_tex)
    lattice_tex = r"""\begin{table*}[!t]
\caption{Two-coordinate reporting errors.
Values are means and sample standard deviations of per-fit mean errors
across five independent fits. Full TVD is the primary metric and selects
checkpoints on separate development conditions. The two fitting geometries
have different reporting sets, so columns do not form a paired accuracy
comparison. Lower is better.}
\label{tab:gaussian-two-dimensional}
\centering
\begin{tabular}{lcc}\toprule
 & Axis training, sixteen pairs & Grid training, thirty-two pairs\\\midrule
"""
    controls = [
        results[d]["fits"][0]["encoder_mean_full_tvd"] for d in ("axes", "grid")
    ]
    lattice_tex += (
        rf"Encoder only, full and profile TVD & ${controls[0]:.4f}$ "
        rf"& ${controls[1]:.4f}$\\" + "\n"
    )
    for key, label in [("full_tvd", "full TVD"), ("profile_tvd", "profile TVD")]:
        lattice_tex += (
            "Learned model, "
            + label
            + " & "
            + value("axes", key)
            + " & "
            + value("grid", key)
            + r"\\"
            + "\n"
        )
    lattice_tex += r"\bottomrule\end{tabular}" + "\n" + r"\end{table*}" + "\n"
    (out / "gaussian2d_lattice_tvd.tex").write_text(lattice_tex)


@mpl.rc_context(STYLE)
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/gaussian-centered-figure-replay")
    )
    args = parser.parse_args(argv)
    a, summary, digest = load(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    lattice_heatmaps(a, args.output)
    lattice_response(a, args.output)
    translation(a, args.output)
    tables(summary, args.output)
    hashes = {
        name: hashlib.sha256((args.results / name).read_bytes()).hexdigest()
        for name in ("SUMMARY.json", "DIAGNOSTICS.npz")
    }
    provenance = {
        "summary_sha256": hashes["SUMMARY.json"],
        "diagnostics_sha256": digest,
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "aggregation": (
            "five final fits: median and IQR for responses; "
            "mean across fits for heatmaps"
        ),
        "center_baseline": (
            "fixed encoder-only baseline at central condition, marked as an open circle"
        ),
        "smoothing": "none",
        "heatmap_protocol": "axes",
        "heatmap_interpretation": (
            "single axis-training heatmap with fitting conditions and training hull"
        ),
        "uncertainty": "IQR across five final fits only",
        "protocols": ["grid", "axes"],
        "translation_profile_indices": [0, 26, 53, 79],
        "translation_profile_conditions": a["translation_profile_c"].tolist(),
        "translation_profile_target_means": (
            a["translation_target_profiles"] @ np.arange(17)
        ).tolist(),
        "translation_profile_geometry": (
            "fixed held-out reporting subset; selection unchanged"
        ),
    }
    (args.output / "gaussian_lattice_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    return [
        args.output / name
        for name in (
            "gaussian_translation_profiles.pdf",
            "gaussian_translation_profiles.png",
            "gaussian2d_lattice_heatmaps.pdf",
            "gaussian2d_lattice_heatmaps.png",
            "gaussian2d_lattice_mean_response.pdf",
            "gaussian2d_lattice_mean_response.png",
            "gaussian2d_lattice_width_response.pdf",
            "gaussian2d_lattice_width_response.png",
            "gaussian_tvd.tex",
            "gaussian2d_lattice_tvd.tex",
            "gaussian_lattice_provenance.json",
        )
    ]


if __name__ == "__main__":
    main()

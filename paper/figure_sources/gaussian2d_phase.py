"""Portable publication plots for the phase-prior Gaussian campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
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
    diagnostics = root / "DIAGNOSTICS.npz"
    digest = hashlib.sha256(diagnostics.read_bytes()).hexdigest()
    if summary.get("diagnostic_npz_sha256") != digest:
        raise ValueError("DIAGNOSTICS.npz hash does not match SUMMARY.json")
    arrays = dict(np.load(diagnostics))
    required = {
        "model_mean",
        "model_sd",
        "fixed_mean",
        "fixed_sd",
        "target_mean",
        "target_sd",
        "u_axis",
        "v_axis",
        "grid_u",
        "grid_v",
        "grid_tvd",
        "fitting_rectangle",
        "fitting_axes",
    }
    if not required <= set(arrays):
        raise ValueError("diagnostics are missing required fields")
    if arrays["grid_tvd"].shape != (2, 5, 41, 41):
        raise ValueError("unexpected dense-grid shape")
    if np.max(arrays["grid_tvd"].mean(axis=1)) > 0.06 + 1e-12:
        raise ValueError("dense-grid values exceed the frozen colour scale")
    if summary.get("status") != "complete" or summary.get("protocols") != [
        "rectangle",
        "axes",
    ]:
        raise ValueError("incomplete or reordered campaign")
    return arrays, summary, digest


def save(fig, output: Path, stem: str):
    fig.savefig(output / f"gaussian2d_phase_{stem}.pdf")
    fig.savefig(output / f"gaussian2d_phase_{stem}.png", dpi=300)
    plt.close(fig)


def heatmaps(a, output):
    values = a["grid_tvd"].mean(axis=1)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(170 * MM, 72 * MM),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    for i, (ax, title, points) in enumerate(
        zip(
            axes,
            ("Separate variation", "Joint variation"),
            ("fitting_axes", "fitting_rectangle"),
            strict=True,
        )
    ):
        image = ax.imshow(
            values[1 if i == 0 else 0],
            origin="lower",
            extent=(4.925, 11.075, 1.59, 2.41),
            vmin=0,
            vmax=0.06,
            cmap="Greys",
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_box_aspect(1)
        train = a[points]
        mu, sigma = 8 + 3 * train[:, 0], 2 * (1 + 0.2 * train[:, 1])
        ax.scatter(mu, sigma, marker="+", color="#0072B2", s=35)
        if i == 0:
            hull = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [-1, 0]])
        else:
            hull = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
        ax.plot(
            8 + 3 * hull[:, 0], 2 * (1 + 0.2 * hull[:, 1]), "#0072B2", lw=1, ls="--"
        )
        ax.set_title(title)
        ax.set_xlabel(r"target center $\mu$")
    axes[0].set_ylabel(r"target width $\sigma_G$")
    axes[0].set_xlim(4.7, 11.3)
    axes[0].set_ylim(1.55, 2.45)
    fig.colorbar(image, ax=axes, label="mean full TVD", shrink=0.85)
    save(fig, output, "heatmaps")


@mpl.rc_context(STYLE)
def main(argv=None):
    paper_root = Path(__file__).resolve().parents[1]
    repo_root = paper_root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path, default=paper_root / "data/gaussian2d_phase"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "results/gaussian2d-phase-publication-replay",
    )
    args = parser.parse_args(argv)
    a, summary, digest = load(args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    labels = ["separate variation", "joint variation"]
    xw, xm = a["v_axis"], a["u_axis"]
    model_mean = np.median(a["model_mean"], axis=1)
    model_mean_iqr = np.quantile(a["model_mean"], [0.25, 0.75], axis=1)
    model_sd = np.median(a["model_sd"], axis=1)
    model_sd_iqr = np.quantile(a["model_sd"], [0.25, 0.75], axis=1)
    k = np.arange(17)
    r0 = np.exp(-((k - 8) ** 2) / 8)
    r0 /= r0.sum()
    encoder_sd = np.sqrt(r0 @ (k - 8) ** 2)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(170 * MM, 105 * MM),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    for p, label in enumerate(labels):
        for j, fixed_u in enumerate(summary["fixed_u"]):
            ax = axes[p, j]
            ax.plot(
                xw,
                model_sd[1 if p == 0 else 0, j],
                color="#0072B2",
                label="sampled model" if p == j == 0 else None,
            )
            ax.fill_between(
                xw,
                model_sd_iqr[0, 1 if p == 0 else 0, j],
                model_sd_iqr[1, 1 if p == 0 else 0, j],
                color="#0072B2",
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(
                xw,
                a["fixed_sd"][1 if p == 0 else 0, j],
                color="#D55E00",
                ls=":",
                label=r"fixed $\theta=3\pi/8$" if p == j == 0 else None,
            )
            ax.plot(
                xw, a["target_sd"][j], "k--", label="target" if p == j == 0 else None
            )
            ax.axhline(
                encoder_sd,
                color="0.45",
                ls="-.",
                label="encoder only" if p == j == 0 else None,
            )
            ax.set_title(rf"$u={fixed_u}$")
            if j == 0:
                ax.set_ylabel(f"{label}\nwidth")
            if p == 1:
                ax.set_xlabel("width condition $v$")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=4
    )
    save(fig, args.output, "width_response")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(85 * MM, 100 * MM),
        layout="constrained",
        sharex=True,
        sharey=True,
    )
    for p, label in enumerate(labels):
        idx = 1 if p == 0 else 0
        ax = axes[p]
        ax.plot(
            xm,
            model_mean[idx],
            color="#0072B2",
            label="sampled model" if p == 0 else None,
        )
        ax.fill_between(
            xm,
            model_mean_iqr[0, idx],
            model_mean_iqr[1, idx],
            color="#0072B2",
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            xm,
            a["fixed_mean"][idx],
            color="#D55E00",
            ls=":",
            label=r"fixed $\theta=3\pi/8$" if p == 0 else None,
        )
        ax.plot(xm, a["target_mean"], "k--", label="target" if p == 0 else None)
        ax.axhline(8, color="0.45", ls="-.", label="encoder only" if p == 0 else None)
        ax.set_title(label)
        ax.set_ylabel("mean")
    axes[-1].set_xlabel("mean condition $u$ at $v=0$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    save(fig, args.output, "mean_response")
    heatmaps(a, args.output)
    source_hashes = {
        name: hashlib.sha256((args.results / name).read_bytes()).hexdigest()
        for name in ("README.md", "SUMMARY.json", "DIAGNOSTICS.npz")
    }
    provenance = {
        "input_sha256": digest,
        "summary_sha256": hashlib.sha256(
            (args.results / "SUMMARY.json").read_bytes()
        ).hexdigest(),
        "source_copy_hashes": source_hashes,
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocols": ["separate variation", "joint variation"],
        "response_aggregation": "median and IQR across five final fits per protocol",
        "heatmaps": {
            "order": ["separate variation", "joint variation"],
            "axis_transform": "mu=8+3u; sigma_G=2(1+0.2v)",
            "aggregation": "arithmetic mean across five final fits per protocol",
            "pixel_edge_extent": {"mu": [4.925, 11.075], "sigma_G": [1.59, 2.41]},
            "vmin": 0,
            "vmax": 0.06,
            "smoothing": "none",
        },
        "fixed_control": summary.get("fixed_control", {"theta": 3 * np.pi / 8}),
        "note": (
            "Measurement distribution of the phase encoder is fixed; "
            "no necessity claim is made."
        ),
    }
    (args.output / "gaussian2d_phase_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

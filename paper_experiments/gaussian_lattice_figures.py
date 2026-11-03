"""Publication figures and table for the Hamming-weight family bundles."""

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
    "font.family": "sans-serif",
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
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{lmodern} \renewcommand{\familydefault}{\sfdefault} \usepackage{sansmath} \sansmath",
        }
        if shutil.which("latex")
        else {"text.usetex": False, "mathtext.fontset": "cm"}
    ),
}


def save(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)


def load(root: Path):
    """Verified bundle arrays; grid_tvd[fit, v, u] and responses[fit, slice, x]."""
    summary = json.loads((root / "SUMMARY.json").read_text())
    datafile = root / "DIAGNOSTICS.npz"
    digest = hashlib.sha256(datafile.read_bytes()).hexdigest()
    if summary.get("diagnostic_npz_sha256") != digest:
        raise ValueError(f"DIAGNOSTICS.npz hash does not match SUMMARY.json: {root}")
    a = dict(np.load(datafile))
    fits = len(a["grid_tvd"])
    if a["grid_tvd"].shape != (fits, 41, 41) or a["model_sd"].shape != (fits, 3, 21):
        raise ValueError(f"unexpected diagnostic shapes: {root}")
    return a, summary, digest


def load_diagonal(root: Path):
    a, summary, _ = load(root)
    return a, summary


def _heatmap_panel(ax, values, a, norm):
    image = ax.imshow(
        values,
        origin="lower",
        extent=(-1.025, 1.025, -1.025, 1.025),
        norm=norm,
        cmap=WHITE_BLUES,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xlabel("Mean condition $u$")
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    hull = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [-1, 0]])
    ax.plot(
        hull[:, 0],
        hull[:, 1],
        "--",
        color="0.15",
        lw=1.2,
        label="Training convex hull",
    )
    train = a["fitting_axes"]
    ax.scatter(
        train[:, 0],
        train[:, 1],
        marker="D",
        facecolors="#E69F00",
        edgecolors="0.1",
        s=28,
        linewidths=0.8,
        label="Training condition",
        zorder=6,
        clip_on=False,
    )
    return image


def lattice_heatmaps(a, out, diagonal=None):
    values = a["grid_tvd"].mean(axis=0)
    if diagonal is None:
        # A fixed range up to 0.125 places the errors on an absolute TVD scale.
        vmax = 0.125
        panels = [(values, None)]
        figsize = (85 * MM, 88 * MM)
    else:
        if not np.array_equal(diagonal["fitting_axes"], a["fitting_axes"]):
            raise ValueError("diagonal baseline uses different training conditions")
        # One logarithmic TVD scale for both models, so that the smaller errors of
        # input conditioning remain visible next to the baseline. Values below
        # vmin (near the origin, where input conditioning is exact) are clipped.
        vmax = 0.125
        panels = [
            (values, "(a) Input conditioning"),
            (diagonal["grid_tvd"].mean(axis=0), "(b) Diagonal conditioning"),
        ]
        figsize = (150 * MM, 74 * MM)
    if max(v.max() for v, _ in panels) > vmax:
        raise ValueError("heatmap values exceed the color range")
    if diagonal is None:
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
        ticks = np.linspace(0, vmax, 6)
    else:
        norm = mpl.colors.LogNorm(vmin=0.002, vmax=vmax)
        ticks = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    fig, axes = plt.subplots(
        1, len(panels), figsize=figsize, layout="constrained", squeeze=False
    )
    for ax, (panel, title) in zip(axes[0], panels, strict=True):
        image = _heatmap_panel(ax, panel, a, norm)
        if title:
            ax.set_title(title, fontsize=10)
    axes[0, 0].set_ylabel("Width condition $v$")
    for ax in axes[0, 1:]:
        ax.tick_params(labelleft=False)
    fig.colorbar(
        image,
        ax=axes[0].tolist(),
        label="Mean full TVD",
        shrink=0.68 if diagonal is None else 0.85,
        ticks=ticks,
        format=mpl.ticker.FuncFormatter(lambda x, _: f"{x:g}"),
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        fontsize=8,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    save(fig, out, "gaussian2d_lattice_heatmaps")


def heatmap_correlations(a):
    """Correlate axis-training grid error with target and training distances."""
    from paper_experiments.gaussian2d_lattice import target_profiles

    error = a["grid_tvd"].mean(axis=0).ravel()
    u, v = np.meshgrid(a["grid_u"], a["grid_v"])
    points = np.stack([u.ravel(), v.ravel()], axis=1)
    uniform = np.array([math.comb(16, k) for k in range(17)]) / 2**16
    from_uniform = abs(target_profiles(points) - uniform).sum(axis=1) / 2
    to_training = np.linalg.norm(points[:, None] - a["fitting_axes"][None], axis=2).min(
        axis=1
    )
    keep = np.linalg.norm(points, axis=1) > 0
    return {
        "target_distance_from_uniform": float(
            np.corrcoef(from_uniform[keep], error[keep])[0, 1]
        ),
        "distance_to_nearest_training": float(
            np.corrcoef(to_training[keep], error[keep])[0, 1]
        ),
    }


def lattice_response(a, out):
    """Mean and width slices of the axis-trained model, mean +/- sample SD."""
    fig, axes = plt.subplots(
        2, 3, figsize=(170 * MM, 105 * MM), layout="constrained", sharey="row"
    )
    rows = (
        (
            "model_mean",
            "target_mean",
            "u_axis",
            8,
            "Mean Hamming weight",
            "Mean condition $u$",
            "v",
        ),
        (
            "model_sd",
            "target_sd",
            "v_axis",
            2,
            "Standard deviation of $K$",
            "Width condition $v$",
            "u",
        ),
    )
    for r, (key, target_key, x_key, encoder, ylabel, xlabel, fixed_name) in enumerate(
        rows
    ):
        for j, fixed in enumerate([-1, 0, 1]):
            ax = axes[r, j]
            data = a[key][:, j]
            mean = data.mean(axis=0)
            sd = data.std(axis=0, ddof=1)
            first = r == j == 0
            ax.plot(
                a[x_key], mean, color=OKABE[0], label="Model mean" if first else None
            )
            ax.fill_between(
                a[x_key], mean - sd, mean + sd, color=OKABE[0], alpha=0.18, linewidth=0
            )
            ax.plot(
                a[x_key],
                a[target_key][j],
                "k--",
                label="Exact target" if first else None,
            )
            ax.axhline(
                encoder, color="0.5", ls=":", label="Encoder only" if first else None
            )
            ax.set_title(rf"${fixed_name}={fixed}$")
            ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3
    )
    save(fig, out, "gaussian2d_lattice_axes_response")


def uncertain(mean, sd):
    """Uncertainty to one significant figure (two if it starts with 1)."""
    leading = f"{sd:.1e}"
    places = max(-int(leading.split("e")[1]) + (leading[0] == "1"), 0)
    return rf"${mean:.{places}f}\pm{sd:.{places}f}$"


def exact(value):
    """At most three significant figures without trailing zeros."""
    return f"${float(f'{value:.3g}'):g}$"


VARIANTS = (
    ("lattice-quadratic", "quadratic map"),
    ("complete-affine", "all-to-all layer"),
    ("complete-quadratic", "all-to-all layer, quadratic map"),
)


def tables(summary, out, diagonal=None, variants=None):
    """Mixed-pair errors recomputed from per-condition, per-fit evidence."""
    fits = summary["results"]["axes"]["fits"]

    def value(key):
        entries = np.asarray([np.mean(row["selected"][key]) for row in fits])
        return uncertain(entries.mean(), entries.std(ddof=1))

    count = {10: "ten"}.get(len(fits), str(len(fits)))
    caption = (
        r"\caption{Two-coordinate errors after axis training, evaluated on the "
        r"sixteen mixed conditions. Input conditioning is the conditional IQP Born "
        r"machine; diagonal conditioning makes the phases of the same lattice "
        r"circuit, acting on $\ket{0}^{\otimes16}$, an affine function of the "
        r"condition"
        + (
            r", and its variants use a quadratic condition map, an all-to-all "
            r"diagonal layer, or both"
            if variants
            else ""
        )
        + r". Values are means and standard deviations of the per-seed mean "
        rf"full-distribution TVD across {count} seeds; the deterministic encoder-only row has "
        r"no spread. Lower is better.}"
    )
    lines = [
        r"\begin{table}[!t]",
        caption,
        r"\label{tab:gaussian-two-dimensional}",
        r"\centering",
        r"\begin{tabular}{lc}\toprule",
        r"Model & TVD\\\midrule",
        rf"Encoder only & {exact(fits[0]['encoder_mean_full_tvd'])}\\",
        f"Input conditioning (40 parameters) & {value('full_tvd')}" + r"\\",
    ]
    if diagonal is not None:
        row = diagonal["diagonal"]
        lines.append(
            f"Diagonal conditioning ({row['n_parameters']} parameters) & "
            + uncertain(row["mixed_full_tvd"]["mean"], row["mixed_full_tvd"]["sd"])
            + r"\\"
        )
    for name, label in VARIANTS if variants else ():
        row = json.loads((variants / name / "SUMMARY.json").read_text())["diagonal"]
        lines.append(
            f"Diagonal, {label} ({row['n_parameters']} parameters) & "
            + uncertain(row["mixed_full_tvd"]["mean"], row["mixed_full_tvd"]["sd"])
            + r"\\"
        )
    lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    (out / "gaussian2d_lattice_tvd.tex").write_text("\n".join(lines) + "\n")


@mpl.rc_context(STYLE)
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/gaussian-lattice-figure-replay")
    )
    parser.add_argument("--diagonal", type=Path, default=None)
    parser.add_argument("--variants", type=Path, default=None)
    args = parser.parse_args(argv)
    a, summary, digest = load(args.results)
    diagonal, diagonal_summary = (
        load_diagonal(args.diagonal) if args.diagonal else (None, None)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    lattice_heatmaps(a, args.output, diagonal)
    lattice_response(a, args.output)
    tables(summary, args.output, diagonal_summary, args.variants)
    n_fits = len(a["grid_tvd"])
    provenance = {
        "summary_sha256": hashlib.sha256(
            (args.results / "SUMMARY.json").read_bytes()
        ).hexdigest(),
        "diagonal_summary_sha256": (
            hashlib.sha256((args.diagonal / "SUMMARY.json").read_bytes()).hexdigest()
            if args.diagonal
            else None
        ),
        "diagnostics_sha256": digest,
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "aggregation": (
            f"{n_fits} final fits: mean and sample SD for responses; "
            "mean across fits for heatmaps"
        ),
        "smoothing": "none",
        "heatmap_error_correlations": heatmap_correlations(a),
        "uncertainty": f"sample SD across {n_fits} final fits only",
    }
    (args.output / "gaussian_lattice_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return [
        args.output / name
        for name in (
            "gaussian2d_lattice_heatmaps.pdf",
            "gaussian2d_lattice_heatmaps.png",
            "gaussian2d_lattice_axes_response.pdf",
            "gaussian2d_lattice_axes_response.png",
            "gaussian2d_lattice_tvd.tex",
            "gaussian_lattice_provenance.json",
        )
    ]


if __name__ == "__main__":
    main()

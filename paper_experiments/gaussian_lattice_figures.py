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


def load_diagonal(root: Path):
    summary = json.loads((root / "SUMMARY.json").read_text())
    datafile = root / "DIAGNOSTICS.npz"
    if (
        summary.get("diagnostic_npz_sha256")
        != hashlib.sha256(datafile.read_bytes()).hexdigest()
    ):
        raise ValueError("diagonal DIAGNOSTICS.npz hash does not match SUMMARY.json")
    d = dict(np.load(datafile))
    if d["grid_tvd"].shape != (5, 41, 41):
        raise ValueError("unexpected diagonal grid shape")
    return d, summary


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
    values = a["grid_tvd"][1].mean(axis=0)
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
        vmax = 0.3
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
        norm = mpl.colors.LogNorm(vmin=0.005, vmax=vmax)
        ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
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

    error = a["grid_tvd"][1].mean(axis=0).ravel()
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
    """Mean and width slices for one training design, mean +/- sample SD."""
    for protocol, stem in ((1, "axes"), (0, "grid")):
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
        for r, (
            key,
            target_key,
            x_key,
            encoder,
            ylabel,
            xlabel,
            fixed_name,
        ) in enumerate(rows):
            for j, fixed in enumerate([-1, 0, 1]):
                ax = axes[r, j]
                data = a[key][protocol, :, j]
                mean = data.mean(axis=0)
                sd = data.std(axis=0, ddof=1)
                first = r == j == 0
                ax.plot(
                    a[x_key],
                    mean,
                    color=OKABE[0],
                    label="Model mean" if first else None,
                )
                ax.fill_between(
                    a[x_key],
                    mean - sd,
                    mean + sd,
                    color=OKABE[0],
                    alpha=0.18,
                    linewidth=0,
                )
                ax.plot(
                    a[x_key],
                    a[target_key][j],
                    "k--",
                    label="Exact target" if first else None,
                )
                ax.axhline(
                    encoder,
                    color="0.5",
                    ls=":",
                    label="Encoder only" if first else None,
                )
                ax.set_title(rf"${fixed_name}={fixed}$")
                ax.set_xlabel(xlabel)
                if j == 0:
                    ax.set_ylabel(ylabel)
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(),
            loc="outside lower center",
            ncol=3,
        )
        save(fig, out, f"gaussian2d_lattice_{stem}_response")


# Profile panels show the held-out conditions nearest c = 1/16, 5/16, 11/16, 15/16,
# the points farthest from both training (multiples of 1/4) and development
# (odd multiples of 1/8) conditions.
PROFILE_TARGETS = np.array([1, 5, 11, 15]) / 16


def translation_profiles(root: Path, a):
    """Exact profiles at the panel conditions, replayed from the selected fits."""
    from paper_experiments.gaussian2d_lattice import target_profiles
    from paper_experiments.gaussian2d_lattice_report import evaluator

    thetas = np.load(root / "PARAMETERS.npz")["translation"]
    c = a["translation_reporting_c"]

    def evaluate(index):
        pairs = np.stack([2 * c[index] - 1, np.zeros(len(index))], axis=1)
        fits = [evaluator(t)(pairs, profiles=True)["profiles"] for t in thetas]
        return np.asarray(fits), target_profiles(pairs)

    stored = evaluate(np.array([0, 26, 53, 79]))[0]
    if not np.allclose(stored, a["translation_profiles"], atol=1e-12, rtol=0):
        raise ValueError("replayed profiles do not match DIAGNOSTICS.npz")
    index = np.abs(c[None, :] - PROFILE_TARGETS[:, None]).argmin(axis=1)
    profiles, targets = evaluate(index)
    return index, c[index], profiles, targets


def translation(a, out, root: Path):
    _, profile_c, profiles, target_profiles = translation_profiles(root, a)
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 78 * MM), layout="constrained")
    k = np.arange(17)
    uniform = np.array([math.comb(16, int(i)) for i in k], dtype=float) / 2**16
    for i in range(4):
        axes[0].plot(
            np.arange(17),
            profiles[:, i].mean(0),
            color=OKABE[i],
            label=rf"$c\approx{profile_c[i]:.3f}$ "
            rf"($\langle K\rangle_{{\mathrm{{target}}}}"
            rf"\approx{(target_profiles[i] @ k):.2f}$)",
        )
        axes[0].plot(np.arange(17), target_profiles[i], "--", color=OKABE[i])
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
        handles=condition_handles, loc="outside upper center", ncol=2, fontsize=9
    )
    mean = a["translation_response"].mean(axis=0)
    sd = a["translation_response"].std(axis=0, ddof=1)
    lo, hi = mean - sd, mean + sd
    axes[1].fill_between(
        a["translation_reporting_c"], lo, hi, color=OKABE[0], alpha=0.25, zorder=1
    )
    axes[1].plot(
        a["translation_reporting_c"],
        mean,
        color=OKABE[0],
        lw=2.2,
        label="Five-fit mean",
        zorder=2,
    )
    axes[1].plot(
        a["translation_reporting_c"],
        a["translation_target_mean"],
        "k--",
        lw=1.0,
        dashes=(4, 3),
        label="Exact target",
        zorder=3,
    )
    axes[1].axhline(8, color="0.5", ls=":", label="Encoder-only")
    axes[1].scatter(
        a["translation_fitting_c"],
        a["translation_fitting_target_mean"],
        marker="+",
        s=110,
        linewidths=1.8,
        color="0.15",
        label="Training marker",
        zorder=4,
        clip_on=False,
    )
    axes[1].scatter(
        [0.5],
        [8],
        s=60,
        facecolors="white",
        edgecolors="0.1",
        linewidths=1.4,
        label="Fixed baseline",
        zorder=5,
    )
    axes[1].set(
        xlabel="Condition $c$",
        ylabel="Mean Hamming weight",
        title="(b) Mean response",
    )
    handles = [
        plt.Line2D([], [], color="0.25", label="Model"),
        plt.Line2D([], [], color="0.25", ls="--", label="Target"),
        plt.Line2D([], [], color="0.5", ls=":", label="Encoder only"),
        plt.Line2D(
            [],
            [],
            color="0.15",
            marker="+",
            ms=9,
            mew=1.8,
            ls="",
            label="Training conditions",
        ),
        plt.Line2D(
            [],
            [],
            color="0.1",
            marker="o",
            mfc="white",
            ms=7,
            mew=1.4,
            ls="",
            label="Fixed center",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "gaussian_translation_profiles")


def tables(summary, out, diagonal=None):
    """Recompute manuscript errors from per-condition, per-fit evidence."""
    results = summary["results"]

    def value(design, key):
        entries = np.asarray(
            [np.mean(row["selected"][key]) for row in results[design]["fits"]]
        )
        return rf"${entries.mean():.4f}\pm{entries.std(ddof=1):.4f}$"

    translation_tex = r"""\begin{table}[!htbp]
\centering
\small
""" + (
        r"\caption{Exact full-distribution and profile TVDs for conditional generation "
        r"of binomial distributions. Entries report the mean error over eighty test "
        r"conditions, given as the mean and sample standard deviation across five "
        r"independent fits. Lower is better.}"
        "\n"
        r"\label{tab:gaussian_tvd}"
        "\n"
        r"\begin{tabular}{lcc}\toprule"
        "\n"
        r"Model & Full TVD & Profile TVD\\\midrule"
        "\n"
    )
    translation_tex += (
        "Conditional IQP & "
        + value("translation", "full_tvd")
        + " & "
        + value("translation", "profile_tvd")
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
    caption = r"""\caption{Two-coordinate errors after axis training, evaluated
on sixteen mixed condition pairs. Values are means and standard deviations of
per-fit mean errors across five independent fits; the deterministic encoder-only
row has no spread. Lower is better.}"""
    if diagonal is not None:
        caption = r"""\caption{Two-coordinate errors after axis training, evaluated
on sixteen mixed condition pairs. Input conditioning is the conditional IQP Born
machine; diagonal conditioning makes the phases of the same lattice circuit,
acting on $\ket{0}^{\otimes16}$, an affine function of the condition. Values are means and
standard deviations of per-fit mean errors across five independent fits; the
deterministic encoder-only row has no spread. Lower is better.}"""
    lattice_tex = (
        r"\begin{table}[!t]"
        + "\n"
        + caption
        + "\n"
        + r"""\label{tab:gaussian-two-dimensional}
\centering
\begin{tabular}{lc}\toprule
Quantity & TVD\\\midrule
"""
    )
    control = results["axes"]["fits"][0]["encoder_mean_full_tvd"]
    lattice_tex += rf"Encoder only, full and profile & ${control:.4f}$\\" + "\n"
    for key, label in [("full_tvd", "full"), ("profile_tvd", "profile")]:
        lattice_tex += (
            ("Learned model, " if diagonal is None else "Input conditioning, ")
            + label
            + " & "
            + value("axes", key)
            + r"\\"
            + "\n"
        )
    if diagonal is not None:
        rows = diagonal["diagonal"]
        for key, label in [
            ("mixed_full_tvd", "full"),
            ("mixed_profile_tvd", "profile"),
        ]:
            lattice_tex += (
                "Diagonal conditioning, "
                + label
                + rf" & ${rows[key]['mean']:.4f}\pm{rows[key]['sd']:.4f}$\\"
                + "\n"
            )
    lattice_tex += r"\bottomrule\end{tabular}" + "\n" + r"\end{table}" + "\n"
    (out / "gaussian2d_lattice_tvd.tex").write_text(lattice_tex)


@mpl.rc_context(STYLE)
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/gaussian-centered-figure-replay")
    )
    parser.add_argument("--diagonal", type=Path, default=None)
    args = parser.parse_args(argv)
    a, summary, digest = load(args.results)
    diagonal, diagonal_summary = (
        load_diagonal(args.diagonal) if args.diagonal else (None, None)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    lattice_heatmaps(a, args.output, diagonal)
    lattice_response(a, args.output)
    translation(a, args.output, args.results)
    tables(summary, args.output, diagonal_summary)
    hashes = {
        name: hashlib.sha256((args.results / name).read_bytes()).hexdigest()
        for name in ("SUMMARY.json", "DIAGNOSTICS.npz")
    }
    provenance = {
        "summary_sha256": hashes["SUMMARY.json"],
        "diagonal_summary_sha256": (
            hashlib.sha256((args.diagonal / "SUMMARY.json").read_bytes()).hexdigest()
            if args.diagonal
            else None
        ),
        "diagnostics_sha256": digest,
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "aggregation": (
            "five final fits: mean and sample SD for responses; "
            "mean across fits for heatmaps"
        ),
        "center_baseline": (
            "fixed encoder-only baseline at central condition, marked as an open circle"
        ),
        "smoothing": "none",
        "heatmap_protocol": "axes",
        "heatmap_error_correlations": heatmap_correlations(a),
        "heatmap_interpretation": (
            "single axis-training heatmap with fitting conditions and training hull"
        ),
        "uncertainty": "sample SD across five final fits only",
        "protocols": ["grid", "axes"],
        "translation_profile_rule": (
            "held-out reporting conditions nearest c = 1/16, 5/16, 11/16, 15/16; "
            "profiles replayed exactly from PARAMETERS.npz; selection unchanged"
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
            "gaussian2d_lattice_axes_response.pdf",
            "gaussian2d_lattice_axes_response.png",
            "gaussian2d_lattice_grid_response.pdf",
            "gaussian2d_lattice_grid_response.png",
            "gaussian_tvd.tex",
            "gaussian2d_lattice_tvd.tex",
            "gaussian_lattice_provenance.json",
        )
    ]


if __name__ == "__main__":
    main()

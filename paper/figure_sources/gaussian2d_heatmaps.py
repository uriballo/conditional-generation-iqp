"""Render dense Gaussian condition-space TVD heatmaps and summary tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "ytick.minor.size": 2,
        "legend.frameon": False,
        "axes.xmargin": 0.02,
        "axes.ymargin": 0.02,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        **(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
            }
            if shutil.which("latex")
            else {"text.usetex": False, "mathtext.fontset": "cm"}
        ),
    }
)


def digest(p):
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def xy(c):
    return 8 + 3 * c[:, 0], 2 * (1 + 0.2 * c[:, 1])


def main(results: Path, output: Path):
    npath, jpath = results / "DENSE_GRID.npz", results / "DENSE_GRID.json"
    z = np.load(npath)
    meta = json.loads(jpath.read_text())
    assert digest(npath) == meta["dense_arrays_sha256"]
    required = {
        "conditions",
        "axes_tvd",
        "rectangle_tvd",
        "encoder_tvd",
        "common_withheld",
        "inside_hull",
    }
    assert required <= set(z.files)
    c = z["conditions"]
    assert c.shape == (1681, 2)
    assert all(z[k].shape == (1681,) for k in ("common_withheld", "inside_hull"))
    assert z["axes_tvd"].shape == z["rectangle_tvd"].shape == (5, 1681)
    assert z["encoder_tvd"].shape == (1681,)
    assert max(np.max(z[k].mean(0)) for k in ("axes_tvd", "rectangle_tvd")) <= 0.18
    assert set(meta["protocols"]) >= {"axes", "rectangle"}
    out = output
    out.mkdir(parents=True, exist_ok=True)
    u = np.linspace(-1, 1, 41)
    v = np.linspace(-1, 1, 41)
    assert np.array_equal(c, np.array([(a, b) for b in v for a in u]))
    extent = (4.7, 11.3, 1.55, 2.45)
    norm = mpl.colors.Normalize(0, 0.18)
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(170 * MM, 64 * MM), layout="constrained")
    gs = fig.add_gridspec(
        3, 3, height_ratios=[0.055, 0.13, 1], width_ratios=[1, 1, 0.05]
    )
    legend_ax = fig.add_subplot(gs[0, :2])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([], [], marker="x", color="black", ls="", label="Training pair"),
            Line2D([], [], color=".3", ls="--", label="Training hull boundary"),
        ],
        loc="center",
        ncol=2,
        fontsize=8,
    )
    axs = [fig.add_subplot(gs[2, i]) for i in range(2)]
    hulls = [
        np.array([[-1, 0], [0, -1], [1, 0], [0, 1], [-1, 0]]),
        np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]),
    ]
    titles = [
        (
            "(a) Mean and width varied separately",
            "Outside boundary: joint extrapolation",
        ),
        (
            "(b) Mean and width varied jointly",
            r"Interpolation within a $3\times3$ training grid",
        ),
    ]
    arrays = [z["axes_tvd"].mean(0), z["rectangle_tvd"].mean(0)]
    train = [
        np.asarray(meta["protocols"][p]["training_uv"]) for p in ("axes", "rectangle")
    ]
    for col, (ax, arr, h, trs, (title, sub)) in enumerate(
        zip(axs, arrays, hulls, train, titles, strict=True)
    ):
        x_grid, y_grid = np.meshgrid(8 + 3 * u, 2 * (1 + 0.2 * v))
        im = ax.pcolormesh(
            x_grid,
            y_grid,
            arr.reshape(41, 41),
            cmap="Blues",
            norm=norm,
            shading="nearest",
            rasterized=True,
        )
        tx, ty = xy(trs)
        ax.scatter(tx, ty, marker="x", s=24, color="white", linewidths=2.4, zorder=4)
        ax.scatter(tx, ty, marker="x", s=24, color="black", linewidths=1.0, zorder=5)
        hx, hy = xy(h)
        ax.plot(hx, hy, "--", color="0.25", lw=1.15, zorder=4)
        assert trs.shape == (9, 2)
        ax.set(xlabel=r"Mean parameter $\mu$", xlim=extent[:2], ylim=extent[2:])
        ax.set_xticks([5, 8, 11])
        ax.set_yticks([1.6, 2.0, 2.4])
        if col == 0:
            ax.set_ylabel(r"Width parameter $\sigma_{\mathrm G}$")
        else:
            ax.tick_params(labelleft=False)
        ax.spines[["top", "right"]].set_visible(False)
        title_ax = fig.add_subplot(gs[1, col])
        title_ax.axis("off")
        title_ax.text(0.5, 0.82, title, ha="center", va="center", fontsize=9)
        title_ax.text(0.5, 0.18, sub, ha="center", va="center", fontsize=7.5)
    cax = fig.add_subplot(gs[2, 2])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.05, 0.10, 0.15, 0.18], format="%.2f")
    cb.set_label("Mean full TVD", fontsize=8.5)
    cb.ax.minorticks_off()
    fig.savefig(out / "gaussian2d_heatmaps.pdf")
    fig.savefig(out / "gaussian2d_heatmaps.png", dpi=300)
    plt.close(fig)
    common = z["common_withheld"].astype(bool)
    inside = z["inside_hull"].astype(bool)
    for protocol in ("axes", "rectangle"):
        for used in (
            meta["protocols"][protocol]["training_uv"]
            + meta["protocols"][protocol]["development_uv"]
        ):
            assert not np.any(
                np.all(np.isclose(c[common], used, rtol=0, atol=1e-10), axis=-1)
            )
    assert np.array_equal(inside, abs(c).sum(-1) <= 1 + 1e-10)
    table = {
        "encoder": z["encoder_tvd"],
        "axes": z["axes_tvd"],
        "rectangle": z["rectangle_tvd"],
    }
    rows = {}
    for name, a in table.items():
        groups = [common, common & inside, common & ~inside]
        vals = []
        for g in groups:
            per = a[:, g].mean(1) if a.ndim == 2 else np.array([a[g].mean()])
            vals.append(
                {
                    "count": int(g.sum()),
                    "mean": float(per.mean()),
                    "sample_sd": float(per.std(ddof=1)) if len(per) > 1 else 0.0,
                }
            )
        for group, value in zip(("all", "inside", "outside"), vals, strict=True):
            assert value["count"] == meta["table_counts"][group]
            assert abs(value["mean"] - meta["table"][name][group]["mean"]) < 1e-12
            if name != "encoder":
                assert (
                    abs(value["sample_sd"] - meta["table"][name][group]["sample_sd"])
                    < 1e-12
                )
        rows[name] = vals
    payload = {
        "inputs_sha256": {
            "DENSE_GRID.npz": digest(npath),
            "DENSE_GRID.json": digest(jpath),
        },
        "aggregation": "mean over five fits; sample SD across per-fit group means",
        "contours": [],
        "colormap": "Blues",
        "color_scale": [0, 0.18],
        "interpolation": ("nearest heatmap cells; no contours or smoothing"),
        "renderer_sha256": digest(Path(__file__)),
        "table": rows,
    }
    (out / "gaussian2d_heatmaps.provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    labels = [
        "Encoder only",
        "Mean and width varied separately",
        r"Mean and width varied jointly",
    ]
    caption = (
        "Mean full-distribution TVD on a common dense-grid domain. All $1652$ "
        "pairs are excluded from both designs' fitting and development sets. "
        "The inside and outside groups use the separate-variation training hull, "
        r"$|u|+|v|\leq1$, for every row. Trained values are means and sample "
        "standard deviations of five per-fit condition averages; the encoder-only "
        "baseline is deterministic. This is a post-selection evaluation of frozen fits."
    )
    tex = (
        r"\begin{table}[h!]" + "\n" + r"\caption{" + caption + "}\n"
        r"\label{tab:gaussian-two-dimensional}" + "\n" + r"\centering" + "\n"
        r"\begin{tabular}{lccc}\toprule" + "\n"
        r" & All unseen pairs & Inside axis hull & Outside axis hull\\" + "\n"
        r" & $1652$ pairs & $828$ pairs & $824$ pairs\\\midrule" + "\n"
    )
    md = (
        "| Method | All unseen pairs (1652) | Inside axis hull (828) "
        "| Outside axis hull (824) |\n|---|---:|---:|---:|\n"
    )
    for lab, key in zip(labels, ("encoder", "axes", "rectangle"), strict=True):
        cells = [
            f"{x['mean']:.4f}"
            if key == "encoder"
            else rf"{x['mean']:.4f} $\pm$ {x['sample_sd']:.4f}"
            for x in rows[key]
        ]
        tex += lab + " & " + " & ".join(cells) + r"\\" + "\n"
        md += "| " + lab + " | " + " | ".join(cells).replace(r"$\pm$", "±") + " |\n"
    tex += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n"
    (out / "table.tex").write_text(tex)
    (out / "table.md").write_text(md)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/gaussian2d-figures"))
    a = p.parse_args()
    main(a.results, a.output)

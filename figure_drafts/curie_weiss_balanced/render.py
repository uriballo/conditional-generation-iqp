"""Render the balanced Curie--Weiss phase comparison from an exported payload."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib as mpl
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
        "ytick.minor.size": 2,
        "xtick.major.width": 0.8,
        "ytick.minor.width": 0.6,
        "legend.frameon": False,
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
)

CONDITIONS = (0.4, 1.0, 1.6)
SEEDS = (410, 411, 412)
ARMS = ("encoder_only", "full", "independent", "plain_independent", "condition_blind")
PLOTTED_ARMS = ARMS[:4]
LABELS = (
    "Encoder only",
    "Shared conditional IQP",
    "Separate conditional IQPs",
    "Separate IQPs, no encoder",
    "Condition blind",
)


def _validate(payload: dict) -> dict:
    if payload.get("plotted_mode") != "raw":
        raise ValueError("payload plotted_mode must be raw")
    if payload.get("postprocessing") != "none; actual circuit probabilities":
        raise ValueError(
            "payload postprocessing must identify raw actual circuit probabilities"
        )
    if payload.get("seeds") != list(SEEDS) or payload.get("conditions") != list(
        CONDITIONS
    ):
        raise ValueError("seeds and conditions must be exactly 410..412 and .4, 1, 1.6")
    metadata = payload.get("metadata", payload)
    protocol = metadata.get("training_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("metadata training_protocol must be an object")
    for key, value in {
        "n_ops": 2048,
        "mc_samples": 2048,
        "bodyness_targets": [2.0, 4.0, 6.0],
    }.items():
        if protocol.get(key) != value:
            raise ValueError(
                f"training_protocol {key} does not match expected protocol"
            )
    for key, value in {"topology": "complete", "n_parameters": 136}.items():
        if metadata.get(key) != value:
            raise ValueError(f"metadata {key} does not match expected protocol")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("payload must contain exactly the five comparison arms")
    for arm in ARMS:
        expected_entries = 1 if arm == "encoder_only" else 3
        entries = arms[arm].get("raw") if isinstance(arms[arm], dict) else None
        if not isinstance(entries, list) or len(entries) != expected_entries:
            raise ValueError(f"{arm}/raw must contain {expected_entries} entries")
        for entry in entries:
            profiles = np.asarray(entry.get("profiles"), dtype=float)
            target = np.asarray(entry.get("target_profiles"), dtype=float)
            if profiles.shape != (3, 17) or target.shape != (3, 17):
                raise ValueError(f"invalid profile shape for {arm}/raw")
            for array in (profiles, target):
                if not np.isfinite(array).all() or np.any(array < 0):
                    raise ValueError(f"invalid profile values for {arm}/raw")
                np.testing.assert_allclose(array.sum(axis=1), 1, atol=1e-12, rtol=0)
            if "profile_tvd" in entry:
                np.testing.assert_allclose(
                    entry["profile_tvd"],
                    0.5 * np.abs(profiles - target).sum(axis=1),
                    atol=1e-12,
                    rtol=0,
                )
    return payload


def render_comparison(data: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(170 * MM, 112 * MM),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    maximum = 0.0
    for panel, (arm, label) in enumerate(zip(PLOTTED_ARMS, LABELS[:4], strict=True)):
        ax = axes.flat[panel]
        for j, beta in enumerate(CONDITIONS):
            entries = data["arms"][arm]["raw"]
            profiles = np.asarray([r["profiles"][j] for r in entries])
            target = np.asarray(entries[0]["target_profiles"][j])
            maximum = max(maximum, profiles.max(), target.max())
            if arm != "encoder_only" or j == 0:
                ax.plot(
                    np.arange(17),
                    profiles.mean(0),
                    color="0.3" if arm == "encoder_only" else OKABE_ITO[j],
                    lw=1.3,
                    marker="o",
                    ms=2.5,
                    label=f"beta={beta:g}",
                )
            ax.plot(np.arange(17), target, color=OKABE_ITO[j], ls="--", lw=1.1)
        ax.set_title(f"({chr(97 + panel)}) {label}", fontsize=9)
        ax.set_xlim(-0.25, 16.25)
        ax.set_xticks((0, 4, 8, 12, 16))
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat:
        ax.set_ylim(0, maximum * 1.08)
    axes[1, 0].set_xlabel("Hamming weight $k$")
    axes[1, 1].set_xlabel("Hamming weight $k$")
    axes[0, 0].set_ylabel("Probability mass")
    axes[1, 0].set_ylabel("Probability mass")
    handles = [
        plt.Line2D([], [], color=OKABE_ITO[j], label=rf"$\beta={beta:g}$")
        for j, beta in enumerate(CONDITIONS)
    ]
    handles += [plt.Line2D([], [], color="0.3", ls="--", label="Target")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=8)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).with_name("figure_data.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "../../paper/figures"
    )
    args = parser.parse_args()
    data = _validate(json.loads(args.data.read_text()))
    render_comparison(data, args.output / "curie_weiss_model_comparison")


if __name__ == "__main__":
    main()

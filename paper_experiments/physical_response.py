"""Publication renderer for Curie--Weiss physical-response observables."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .target import CurieWeissTarget

MM = 1 / 25.4
COLORS = {"product": "#0072B2", "conditional": "#E69F00", "unconditional": "#009E73"}
STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
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
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "text.usetex": False,
    "mathtext.fontset": "cm",
}


def _validate_profiles(profiles, shape):
    if (
        profiles.shape != shape
        or not np.all(np.isfinite(profiles))
        or np.any(profiles < -1e-12)
        or not np.allclose(profiles.sum(-1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("invalid shell profiles or profile shape")


def _observables(profiles, beta, n=16):
    p = np.asarray(profiles, float)
    if p.shape[-1] != n + 1:
        raise ValueError("profiles must end in n+1 shell probabilities")
    m = (2 * np.arange(n + 1) - n) / n
    a = p @ np.abs(m)
    c = beta**2 * n / 4 * (p @ m**4 - (p @ m**2) ** 2)
    return a, c


def physical_arrays(sweep, reported):
    n = sweep["n_visible"]
    beta = np.asarray(sweep["betas"], float)
    expected = np.arange(0, 2.0001, 0.02)
    if (
        n != 16
        or beta.shape != (101,)
        or not np.allclose(beta, expected, atol=1e-12, rtol=0)
    ):
        raise ValueError("expected n=16 and beta grid 0..2 step .02")
    seeds = sweep["seeds"]
    if len(seeds) != 5 or len(set(seeds)) != 5 or seeds != reported["seeds"]:
        raise ValueError("expected five unique seeds")
    product = np.asarray(sweep["arms"]["product"]["profiles_per_seed"], float)
    _validate_profiles(product, (5, 101, 17))
    if reported["betas"] != [0.4, 1.0, 1.6]:
        raise ValueError("unexpected reporting temperatures")
    indices = [int(np.flatnonzero(np.isclose(beta, b))[0]) for b in reported["betas"]]
    shared = {r["seed"]: r for r in reported["records"]["shared"]}
    identities = sweep["provenance"]["checkpoint_identities"]
    if (
        len(shared) != 5
        or len(reported["records"]["shared"]) != 5
        or set(shared) != set(seeds)
        or len(identities) != 5
        or {r["seed"] for r in identities} != set(seeds)
    ):
        raise ValueError("incomplete checkpoint identities")
    for identity in identities:
        if (
            identity["arm"] != "product"
            or identity["checkpoint_sha256"]
            != (shared[identity["seed"]]["checkpoint_sha256"])
        ):
            raise ValueError("response sweep checkpoint identity mismatch")
    for si, seed in enumerate(seeds):
        if not np.allclose(
            product[si, indices], shared[seed]["profile"], atol=1e-12, rtol=0
        ):
            raise ValueError("response sweep differs from reported profiles")
    absolute, heat = _observables(product, beta[None, :])
    target_profiles = np.array([CurieWeissTarget(n, b).probabilities for b in beta])
    stored_target = np.asarray(sweep["target"]["profiles"], float)
    _validate_profiles(stored_target, (101, 17))
    if not np.allclose(stored_target, target_profiles, atol=1e-12, rtol=0):
        raise ValueError("stored target differs from exact Curie-Weiss law")
    target_a, target_c = _observables(target_profiles, beta)
    encoder = np.array(
        [
            np.full(17, 1 / 2**n) * np.array([math.comb(n, k) for k in range(n + 1)])
            for _ in beta
        ]
    )
    encoder_a, encoder_c = _observables(encoder, beta)
    independent = {}
    for key, arm in (
        ("conditional", "independent_conditional"),
        ("unconditional", "independent_unconditional"),
    ):
        records = reported["records"][arm]
        lookup = {(r["seed"], r["fit_beta"]): r for r in records}
        marker_betas = [0.4, 1.0, 1.6]
        expected_keys = {(seed, b) for seed in seeds for b in marker_betas}
        if len(records) != 15 or set(lookup) != expected_keys:
            raise ValueError("independent records must cover each seed and marker beta")
        ib = np.array([[0.4, 1.0, 1.6]] * 5)
        profiles = np.array(
            [[lookup[(seed, b)]["profile"] for b in ib[0]] for seed in seeds], float
        )
        _validate_profiles(profiles, (5, 3, 17))
        independent_a, independent_c = _observables(profiles, ib)
        independent[key] = {
            "betas": ib[0],
            "absolute_mean": independent_a.mean(0),
            "absolute_sd": independent_a.std(0, ddof=1),
            "heat_capacity": independent_c.mean(0),
            "heat_capacity_sd": independent_c.std(0, ddof=1),
        }
    return {
        "beta": beta,
        "target": (target_a, target_c),
        "encoder": (encoder_a, encoder_c),
        "product": (
            absolute.mean(0),
            absolute.std(0, ddof=1),
            heat.mean(0),
            heat.std(0, ddof=1),
        ),
        "independent": independent,
    }


@mpl.rc_context(STYLE)
def render(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    sweep_path = data_dir / "response_sweep.json"
    sweep = json.loads(sweep_path.read_text())
    reported = json.loads((data_dir / "figure_data.json").read_text())
    arrays = physical_arrays(sweep, reported)
    beta = arrays["beta"]
    fig, axes = plt.subplots(
        2, 1, figsize=(85 * MM, 132 * MM), sharex=True, layout="constrained"
    )
    for ax, idx, ylabel in (
        (axes[0], 0, r"$\langle |m|\rangle$"),
        (axes[1], 1, r"$c_{\mathrm{fluc}}$ per spin"),
    ):
        target = arrays["target"][idx]
        enc = arrays["encoder"][idx]
        pm, ps, hm, hs = arrays["product"]
        mean, sd = (pm, ps) if idx == 0 else (hm, hs)
        ax.plot(beta, target, "--", color="0.35", label="Target")
        ax.plot(beta, mean, color=COLORS["product"], label="Conditional IQP")
        ax.fill_between(
            beta,
            mean - sd,
            mean + sd,
            color=COLORS["product"],
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(beta, enc, ":", color="0.4", label="Encoder only")
        ax.set_ylabel(ylabel)
        if idx == 0:
            ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("(a)" if idx == 0 else "(b)", loc="left", fontsize=10)
        for key, label, marker, color in (
            ("conditional", "Independent conditional", "^", COLORS["conditional"]),
            (
                "unconditional",
                "Independent unconditional",
                "s",
                COLORS["unconditional"],
            ),
        ):
            d = arrays["independent"][key]
            y = d["absolute_mean"] if idx == 0 else d["heat_capacity"]
            e = d["absolute_sd"] if idx == 0 else d["heat_capacity_sd"]
            ax.errorbar(
                d["betas"],
                y,
                yerr=e,
                color=color,
                marker=marker,
                mfc="white",
                linestyle="none",
                capsize=2,
                label=label,
            )
        if idx == 1:
            ax.set_ylim(bottom=0)
    axes[1].set_xlabel(r"$\beta$")
    axes[1].set_xlim(0, 2)
    axes[1].set_xticks([0, 0.5, 1, 1.5, 2])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "curie_weiss_physical_response.pdf",
        output_dir / "curie_weiss_physical_response.png",
    ]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    provenance = {
        "inputs": {
            name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in ("response_sweep.json", "figure_data.json")
        },
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "formulas": {
            "m": "(2k-n)/n",
            "absolute_mean": "E[|m|]",
            "heat_capacity": "beta^2*n/4*(E[m^4]-E[m^2]^2)",
        },
        "aggregation": "per-seed nonlinear observable, then mean and sample SD ddof=1",
        "grid_roles": sweep["provenance"].get("grid_roles", {}),
        "new_fits": False,
        "selection_changed": False,
        "checkpoint_identities": sweep["provenance"]["checkpoint_identities"],
        "training_betas": sweep["training_betas"],
        "development_betas": sweep["development_betas"],
        "grid": "101 exact evaluations, beta 0..2 step 0.02",
        "marker_scope": "separately fitted circuits at beta 0.4,1.0,1.6 only",
        "fluctuation_proxy_limitation": (
            "model heat capacity is an equilibrium fluctuation proxy, "
            "not a learned temperature derivative"
        ),
    }
    path = output_dir / "curie_weiss_physical_response.provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    return [*paths, path]

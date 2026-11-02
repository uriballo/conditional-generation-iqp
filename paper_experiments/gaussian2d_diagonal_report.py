"""Compact evidence for the diagonal-conditioning baseline, with a C-IQP comparison.

Reads a completed :mod:`gaussian2d_diagonal_run` campaign and writes, to a new
directory under ``results/``, the final-fit reporting metrics, the 41x41 full
and profile TVD sweeps (layout of ``DIAGNOSTICS.npz`` ``grid_tvd``: index
``[fit, v, u]``), axis responses and a side-by-side comparison with the bundled
conditional-IQP "axes" fits. It never writes to ``data/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import gaussian2d_lattice
from .gaussian2d_diagonal import BASES, metrics, n_phases

ROOT = Path(__file__).resolve().parents[1]
CIQP_BUNDLE = ROOT / "data/gaussian_lattice"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stats(values):
    values = np.asarray(values, dtype=float)
    sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return dict(mean=float(values.mean()), sd=sd, per_fit=values.tolist())


def _inside(pairs):
    return abs(np.asarray(pairs)).sum(1) <= 1 + 1e-12


AXIS_KEYS = (
    "training_axes",
    "training_u_axis",
    "training_v_axis",
    "development_u_axis",
    "development_v_axis",
    "corners",
    "origin",
)
CORNERS = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])


def axis_diagnostics(evaluate, conditions):
    """Mean full TVD on training axes (origin excluded), development axes, corners.

    ``evaluate(pairs)`` returns a metrics dict; ``conditions`` holds the
    protocol's "axes" fitting and development sets.
    """
    fitting = np.asarray(conditions["fitting"], dtype=float)
    development = np.asarray(conditions["development"], dtype=float)
    on_u = development[:, 1] == 0
    groups = dict(
        training_axes=fitting[~np.all(fitting == 0, axis=1)],
        training_u_axis=fitting[(fitting[:, 1] == 0) & (fitting[:, 0] != 0)],
        training_v_axis=fitting[(fitting[:, 0] == 0) & (fitting[:, 1] != 0)],
        development_u_axis=development[on_u],
        development_v_axis=development[~on_u],
        corners=CORNERS,
        origin=np.zeros((1, 2)),
    )
    if tuple(groups) != AXIS_KEYS:
        raise RuntimeError("axis diagnostic keys out of sync")
    return {
        name: float(np.mean(evaluate(pairs)["full_tvd"]))
        for name, pairs in groups.items()
    }


def ciqp_reference(bundle=CIQP_BUNDLE):
    """Published C-IQP axes numbers recomputed from the bundled summary."""
    summary = json.loads((bundle / "SUMMARY.json").read_text())
    fits = summary["results"]["axes"]["fits"]
    return dict(
        source=str(bundle / "SUMMARY.json"),
        source_sha256=sha(bundle / "SUMMARY.json"),
        selected_candidate=summary["results"]["axes"]["selected_candidate"],
        n_parameters=40,
        mixed_full_tvd=_stats([np.mean(f["selected"]["full_tvd"]) for f in fits]),
        inside_full_tvd=_stats([f["inside_mean_full_tvd"] for f in fits]),
        outside_full_tvd=_stats([f["outside_mean_full_tvd"] for f in fits]),
        mixed_profile_tvd=_stats([np.mean(f["selected"]["profile_tvd"]) for f in fits]),
        encoder_alone_full_tvd=_stats([f["encoder_mean_full_tvd"] for f in fits]),
        seconds=_stats([f["seconds"] for f in fits]),
        **_ciqp_axes(bundle, summary),
    )


def _ciqp_axes(bundle, summary):
    """Axis diagnostics and endpoint ratios replayed from bundled parameters."""
    if summary["results"]["axes"]["fits"][0].get("mean_endpoint_ratios") is None:
        return {}
    protocol = json.loads((bundle / "protocol.json").read_text())
    thetas = np.load(bundle / "PARAMETERS.npz")["axes"]
    rows = [
        axis_diagnostics(
            lambda pairs, t=theta: gaussian2d_lattice.metrics(t, pairs),
            protocol["protocols"]["axes"],
        )
        for theta in thetas
    ]
    fits = summary["results"]["axes"]["fits"]
    out = {key: _stats([r[key] for r in rows]) for key in rows[0]}
    for key in ("mean_endpoint_ratios", "width_endpoint_ratios"):
        out[key] = _stats_rows([f[key] for f in fits])
    return out


def _stats_rows(rows):
    rows = np.asarray(rows, dtype=float)
    return dict(
        mean=rows.mean(0).tolist(),
        sd=rows.std(0, ddof=1).tolist() if len(rows) > 1 else None,
        per_fit=rows.tolist(),
    )


def extract(campaign: Path, output: Path, *, grid_points=41, axis_points=21):
    campaign, output = Path(campaign), Path(output)
    if output.resolve().is_relative_to((ROOT / "data").resolve()):
        raise ValueError("refusing to write under data/")
    receipt = json.loads((campaign / "campaign_receipt.json").read_text())
    if receipt["status"] != "complete":
        raise ValueError("requires a completed campaign")
    source = json.loads((campaign / "source-manifest.json").read_text())
    for name, digest in source["sha256"].items():
        if sha(campaign / "source-workspace" / name) != digest:
            raise ValueError(f"frozen source integrity failure: {name}")
    protocol_name = receipt.get("protocol", "gaussian2d_diagonal_protocol.json")
    protocol = json.loads(
        (campaign / "source-workspace/paper_experiments" / protocol_name).read_text()
    )
    basis = protocol.get("basis", "affine")
    terms = BASES[basis]
    ansatz = protocol.get("ansatz", "lattice")
    phases = n_phases(ansatz)
    n_weights = len(terms) * phases
    finals = [r for r in receipt["records"] if r["stage"] == "final"]
    if len(finals) != len(protocol["seeds"]["final"]):
        raise ValueError("expected one final fit per final seed")
    grid = np.linspace(-1, 1, grid_points)
    axis = np.linspace(-1, 1, axis_points)
    mesh = np.asarray([(u, v) for v in grid for u in grid])
    mean_pairs = np.asarray([(u, v) for v in (-1, 0, 1) for u in axis])
    sd_pairs = np.asarray([(u, v) for u in (-1, 0, 1) for v in axis])
    fits, params, grid_full, grid_profile, means, sds = [], [], [], [], [], []
    for row in finals:
        path = campaign / "axes" / "final" / row["candidate"] / str(row["seed"])
        meta = json.loads((path / "fit.json").read_text())
        if sha(path / "fit.npz") != row["checkpoint_sha256"]:
            raise ValueError(f"fit integrity failure: {path}")
        data = np.load(path / "fit.npz")
        weights = data["best_params"]
        if weights.shape != (n_weights,):
            raise ValueError("unexpected weight shape")
        pairs = data["reporting_conditions"]
        report = metrics(weights, pairs)
        for key, value in meta["reporting_metrics"].items():
            if not np.allclose(report[key], value, atol=1e-10, rtol=0):
                raise ValueError(f"replay mismatch: {key}")
        inside = _inside(pairs)
        full = np.asarray(report["full_tvd"])
        unconditioned = weights.copy()
        unconditioned[phases:] = 0
        info = dict(
            seed=row["seed"],
            candidate=row["candidate"],
            best_step=meta["best_step"],
            seconds=meta["seconds"],
            development_mean_full_tvd=meta["best_score"],
            selected=report,
            mixed_mean_full_tvd=float(full.mean()),
            inside_mean_full_tvd=float(full[inside].mean()),
            outside_mean_full_tvd=float(full[~inside].mean()),
            mixed_mean_profile_tvd=float(np.mean(report["profile_tvd"])),
            terminal_mixed_mean_full_tvd=float(
                np.mean(metrics(data["final_params"], pairs)["full_tvd"])
            ),
            w0_only_mixed_mean_full_tvd=float(
                np.mean(metrics(unconditioned, pairs)["full_tvd"])
            ),
            weight_norms={
                f"w_{term}": float(np.linalg.norm(row))
                for term, row in zip(terms, weights.reshape(-1, phases), strict=True)
            },
            **axis_diagnostics(
                lambda pairs, w=weights: metrics(w, pairs),
                protocol["protocols"]["axes"],
            ),
        )
        sweep = metrics(weights, mesh)
        grid_full.append(np.reshape(sweep["full_tvd"], (grid_points, grid_points)))
        grid_profile.append(
            np.reshape(sweep["profile_tvd"], (grid_points, grid_points))
        )
        m = metrics(weights, mean_pairs)
        s = metrics(weights, sd_pairs)
        means.append(np.reshape(m["generated_mean"], (3, axis_points)))
        sds.append(np.reshape(s["generated_sd"], (3, axis_points)))
        target_mean = np.reshape(m["target_mean"], (3, axis_points))
        target_sd = np.reshape(s["target_sd"], (3, axis_points))
        info["mean_endpoint_ratios"] = (
            (means[-1][:, -1] - means[-1][:, 0])
            / (target_mean[:, -1] - target_mean[:, 0])
        ).tolist()
        info["width_endpoint_ratios"] = (
            (sds[-1][:, -1] - sds[-1][:, 0]) / (target_sd[:, -1] - target_sd[:, 0])
        ).tolist()
        fits.append(info)
        params.append(weights)
        print(f"evaluated diagonal axes {row['seed']}", flush=True)
    arrays = dict(
        grid_u=grid,
        grid_v=grid,
        u_axis=axis,
        v_axis=axis,
        target_mean=target_mean,
        target_sd=target_sd,
        fitting_axes=np.asarray(protocol["protocols"]["axes"]["fitting"]),
        reporting_axes=np.asarray(protocol["protocols"]["axes"]["reporting"]),
        grid_tvd=np.asarray(grid_full),
        grid_profile_tvd=np.asarray(grid_profile),
        model_mean=np.asarray(means),
        model_sd=np.asarray(sds),
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "DIAGNOSTICS.npz", **arrays)
    np.savez_compressed(output / "PARAMETERS.npz", axes=np.asarray(params))
    diagonal = dict(
        selected_candidate=receipt["winner"],
        basis=basis,
        **({} if ansatz == "lattice" else {"ansatz": ansatz}),
        n_parameters=n_weights,
        mixed_full_tvd=_stats([f["mixed_mean_full_tvd"] for f in fits]),
        inside_full_tvd=_stats([f["inside_mean_full_tvd"] for f in fits]),
        outside_full_tvd=_stats([f["outside_mean_full_tvd"] for f in fits]),
        mixed_profile_tvd=_stats([f["mixed_mean_profile_tvd"] for f in fits]),
        w0_only_full_tvd=_stats([f["w0_only_mixed_mean_full_tvd"] for f in fits]),
        seconds=_stats([f["seconds"] for f in fits]),
        mean_endpoint_ratios=_stats_rows([f["mean_endpoint_ratios"] for f in fits]),
        width_endpoint_ratios=_stats_rows([f["width_endpoint_ratios"] for f in fits]),
        **{key: _stats([f[key] for f in fits]) for key in AXIS_KEYS},
    )
    summary = dict(
        status="complete",
        smoke=receipt["smoke"],
        basis=basis,
        **({} if ansatz == "lattice" else {"ansatz": ansatz}),
        campaign=str(campaign),
        campaign_receipt_sha256=sha(campaign / "campaign_receipt.json"),
        protocol_sha256=receipt["protocol_sha256"],
        diagnostic_npz_sha256=sha(output / "DIAGNOSTICS.npz"),
        parameters_sha256=sha(output / "PARAMETERS.npz"),
        shortlist=receipt["shortlist"],
        confirmation_scores=receipt.get("confirmation_scores"),
        screen_scores={
            r["candidate"]: r["best_score"]
            for r in receipt["records"]
            if r["stage"] == "screen"
        },
        campaign_fit_seconds_total=receipt.get("fit_seconds_total"),
        campaign_session_seconds=receipt.get("session_seconds"),
        diagonal=diagonal,
        ciqp=ciqp_reference(),
        fits=fits,
        grid_layout="grid_tvd[fit, v_index, u_index], u and v on linspace(-1,1,41)",
        axis_response_layout=(
            "model_mean[fit, v in (-1,0,1), u_axis]; "
            "model_sd[fit, u in (-1,0,1), v_axis]"
        ),
        aggregation=(
            "condition mean per fit, then mean and sample SD across five fits; "
            "inside means |u|+|v|<=1"
        ),
    )
    (output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = extract(args.campaign, args.output)
    for label in ("diagonal", "ciqp"):
        row = summary[label]
        print(
            f"{label}: mixed {row['mixed_full_tvd']['mean']:.4f} "
            f"+/- {row['mixed_full_tvd']['sd']:.4f}; inside "
            f"{row['inside_full_tvd']['mean']:.4f} +/- "
            f"{row['inside_full_tvd']['sd']:.4f}; outside "
            f"{row['outside_full_tvd']['mean']:.4f} +/- "
            f"{row['outside_full_tvd']['sd']:.4f}; profile "
            f"{row['mixed_profile_tvd']['mean']:.4f} +/- "
            f"{row['mixed_profile_tvd']['sd']:.4f}"
        )


if __name__ == "__main__":
    main()

"""Compact, verified replay bundle of the input-conditioning Hamming-weight campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np

from .gaussian2d_hpo import load_campaign
from .gaussian2d_lattice import (
    CollectivePhaseEncoder,
    basis,
    characters,
    hadamard,
    target_profiles,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluator(theta):
    """Resolve the output of each reference weight once for an untied layer."""
    weights = basis()[1]
    phase = np.exp(1j * (characters() @ theta))
    vectors = (weights[None] == np.arange(17)[:, None]) * phase
    output = hadamard(vectors)
    counts = np.array([comb(16, k) for k in range(17)])

    def evaluate(conditions, *, profiles=False):
        fields = {
            k: []
            for k in (
                "full_tvd",
                "profile_tvd",
                "within_shell_residual",
                "generated_mean",
                "generated_sd",
                "target_mean",
                "target_sd",
            )
        }
        saved_profiles = []
        for start in range(0, len(conditions), 16):
            pairs = np.asarray(conditions[start : start + 16])
            shells = np.asarray(CollectivePhaseEncoder.from_conditions(pairs).shells)
            q = abs(shells @ output) ** 2
            if np.max(abs(q.sum(1) - 1)) > 1e-10:
                raise ValueError("output normalization failure")
            rho = np.stack(
                [np.bincount(weights, weights=row, minlength=17) for row in q]
            )
            target = target_profiles(pairs)
            p = target[:, weights] / counts[weights]
            mean = rho @ np.arange(17)
            sd = np.sqrt(np.maximum(rho @ np.arange(17) ** 2 - mean**2, 0))
            tm = target @ np.arange(17)
            ts = np.sqrt(np.maximum(target @ np.arange(17) ** 2 - tm**2, 0))
            values = (
                abs(q - p).sum(1) / 2,
                abs(rho - target).sum(1) / 2,
                abs(q - rho[:, weights] / counts[weights]).sum(1) / 2,
                mean,
                sd,
                tm,
                ts,
            )
            for key, value in zip(fields, values, strict=True):
                fields[key].extend(value.tolist())
            if profiles:
                saved_profiles.extend(rho)
        fields = {k: np.asarray(v) for k, v in fields.items()}
        if profiles:
            fields["profiles"] = np.asarray(saved_profiles)
        return fields

    return evaluate


def _endpoint_ratio(model, target):
    return ((model[:, -1] - model[:, 0]) / (target[:, -1] - target[:, 0])).tolist()


def extract(campaign: Path, output: Path):
    search, fits = load_campaign(campaign)
    if search["campaign"] != "input":
        raise ValueError("requires the input-conditioning campaign")
    protocol = search["protocol"]
    conditions = protocol["protocols"]["axes"]
    reporting = np.asarray(conditions["reporting"])
    arrays = dict(
        grid_u=np.linspace(-1, 1, 41),
        grid_v=np.linspace(-1, 1, 41),
        u_axis=np.linspace(-1, 1, 21),
        v_axis=np.linspace(-1, 1, 21),
        fitting_axes=np.asarray(conditions["fitting"]),
        reporting_axes=reporting,
    )
    mesh = np.asarray([(u, v) for v in arrays["grid_v"] for u in arrays["grid_u"]])
    mean_pairs = np.asarray([(u, v) for v in (-1, 0, 1) for u in arrays["u_axis"]])
    sd_pairs = np.asarray([(u, v) for u in (-1, 0, 1) for v in arrays["v_axis"]])
    inside = abs(reporting).sum(1) <= 1 + 1e-12
    encoder = evaluator(np.zeros(40))(reporting)
    rows, params, grid_full, grid_profile, means, sds = [], [], [], [], [], []
    for record, theta in fits:
        evaluate = evaluator(theta)
        report = evaluate(reporting)
        for key, value in record["reporting_metrics"].items():
            if key in report and not np.allclose(
                report[key], value, atol=1e-10, rtol=0
            ):
                raise ValueError(f"independent evaluator mismatch: {key}")
        diagnostic = evaluate(mesh)
        grid_full.append(diagnostic["full_tvd"].reshape(41, 41))
        grid_profile.append(diagnostic["profile_tvd"].reshape(41, 41))
        m, s = evaluate(mean_pairs), evaluate(sd_pairs)
        means.append(m["generated_mean"].reshape(3, 21))
        sds.append(s["generated_sd"].reshape(3, 21))
        arrays["target_mean"] = m["target_mean"].reshape(3, 21)
        arrays["target_sd"] = s["target_sd"].reshape(3, 21)
        rows.append(
            dict(
                repeat=record["job"]["repeat"],
                selected_start=record["selected_start"],
                best_step=record["selected_step"],
                development_mean_full_tvd=record["J"],
                seconds=record["wall_seconds"],
                selected={k: v.tolist() for k, v in report.items()},
                encoder_mean_full_tvd=float(encoder["full_tvd"].mean()),
                inside_mean_full_tvd=float(report["full_tvd"][inside].mean()),
                outside_mean_full_tvd=float(report["full_tvd"][~inside].mean()),
                mean_endpoint_ratios=_endpoint_ratio(means[-1], arrays["target_mean"]),
                width_endpoint_ratios=_endpoint_ratio(sds[-1], arrays["target_sd"]),
                width_decreasing_steps=(np.diff(sds[-1], axis=1) < -1e-10)
                .sum(1)
                .tolist(),
            )
        )
        params.append(theta)
        print(f"evaluated input repeat {record['job']['repeat']}", flush=True)
    arrays.update(
        grid_tvd=np.asarray(grid_full),
        grid_profile_tvd=np.asarray(grid_profile),
        model_mean=np.asarray(means),
        model_sd=np.asarray(sds),
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "DIAGNOSTICS.npz", **arrays)
    np.savez_compressed(output / "PARAMETERS.npz", axes=np.asarray(params))
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summary = dict(
        status="complete",
        protocols=["axes"],
        n_fits=len(fits),
        fixed_u=[-1, 0, 1],
        fixed_v=[-1, 0, 1],
        diagnostic_npz_sha256=sha(output / "DIAGNOSTICS.npz"),
        parameters_sha256=sha(output / "PARAMETERS.npz"),
        protocol_sha256=sha(output / "protocol.json"),
        search={k: v for k, v in search.items() if k != "protocol"},
        results=dict(
            axes=dict(
                fits=rows,
                selected_config=search["winner"]["config"],
                reporting_conditions=conditions["reporting"],
            )
        ),
        grid_layout="grid_tvd[fit, v_index, u_index], u and v on linspace(-1,1,41)",
        axis_response_layout=(
            "model_mean[fit, v in (-1,0,1), u_axis]; "
            "model_sd[fit, u in (-1,0,1), v_axis]"
        ),
        aggregation=(
            "condition mean per fit, then mean and sample SD across the final fits"
        ),
        limitations=[
            "central match is architectural, not learned",
            "exploratory task-informed family and encoder",
            "exact development targets used for configuration and checkpoint selection",
            "no claim that lattice complexity is necessary",
        ],
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
    extract(args.campaign, args.output)


if __name__ == "__main__":
    main()

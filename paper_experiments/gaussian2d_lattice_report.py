"""Extract compact, verified replay evidence from a completed Gaussian campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np

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


def extract(campaign: Path, output: Path):
    receipt = json.loads((campaign / "campaign_receipt.json").read_text())
    if (
        receipt["status"] != "complete"
        or receipt["smoke"]
        or len(receipt["records"]) != 31
    ):
        raise ValueError("requires the completed 31-fit campaign")
    source = json.loads((campaign / "source-manifest.json").read_text())
    for name, digest in source["sha256"].items():
        if sha(campaign / "source-workspace" / name) != digest:
            raise ValueError(f"frozen source integrity failure: {name}")
    protocol = json.loads(
        (
            campaign
            / "source-workspace/paper_experiments/gaussian2d_lattice_protocol.json"
        ).read_text()
    )
    arrays = dict(
        grid_u=np.linspace(-1, 1, 41),
        grid_v=np.linspace(-1, 1, 41),
        u_axis=np.linspace(-1, 1, 21),
        v_axis=np.linspace(-1, 1, 21),
    )
    mesh = np.asarray([(u, v) for v in arrays["grid_v"] for u in arrays["grid_u"]])
    mean_pairs = np.asarray([(u, v) for v in (-1, 0, 1) for u in arrays["u_axis"]])
    sd_pairs = np.asarray([(u, v) for u in (-1, 0, 1) for v in arrays["v_axis"]])
    summaries, params, records = {}, {}, []
    grid_full, grid_profile, model_mean, model_sd = [], [], [], []
    for design in ("grid", "axes", "translation"):
        selected = [
            r
            for r in receipt["records"]
            if r["design"] == design and r["stage"] == "final"
        ]
        if len(selected) != 5:
            raise ValueError(f"{design}: expected five final fits")
        fits, diagnostics, means, sds, rows, parameters = [], [], [], [], [], []
        for row in selected:
            path = campaign / design / "final" / row["candidate"] / str(row["seed"])
            meta = json.loads((path / "fit.json").read_text())
            if (
                sha(path / "fit.npz") != row["checkpoint_sha256"]
                or meta["status"] != "complete"
            ):
                raise ValueError(f"fit integrity failure: {path}")
            data = np.load(path / "fit.npz")
            theta = data["best_params"]
            evaluate = evaluator(theta)
            report = evaluate(data["reporting_conditions"])
            for key in report:
                if not np.allclose(
                    report[key], meta["reporting_metrics"][key], atol=1e-10, rtol=0
                ):
                    raise ValueError(f"independent evaluator mismatch: {design}/{key}")
            terminal = (
                evaluate(data["reporting_conditions"])
                if np.array_equal(theta, data["final_params"])
                else evaluator(data["final_params"])(data["reporting_conditions"])
            )
            encoder = evaluator(np.zeros(40))(data["reporting_conditions"])
            info = dict(
                seed=row["seed"],
                best_step=meta["best_step"],
                seconds=meta["seconds"],
                selected={k: v.tolist() for k, v in report.items()},
                terminal_mean_full_tvd=float(terminal["full_tvd"].mean()),
                encoder_mean_full_tvd=float(encoder["full_tvd"].mean()),
            )
            if design == "axes":
                inside = abs(data["reporting_conditions"]).sum(1) <= 1 + 1e-12
                info.update(
                    inside_mean_full_tvd=float(report["full_tvd"][inside].mean()),
                    outside_mean_full_tvd=float(report["full_tvd"][~inside].mean()),
                )
            rows.append(info)
            parameters.append(theta)
            records.append(
                dict(
                    design=design,
                    seed=row["seed"],
                    candidate=row["candidate"],
                    checkpoint_sha256=row["checkpoint_sha256"],
                )
            )
            if design != "translation":
                diagnostic = evaluate(mesh)
                fits.append(diagnostic["full_tvd"].reshape(41, 41))
                diagnostics.append(diagnostic["profile_tvd"].reshape(41, 41))
                m = evaluate(mean_pairs)
                s = evaluate(sd_pairs)
                means.append(m["generated_mean"].reshape(3, 21))
                sds.append(s["generated_sd"].reshape(3, 21))
                info["mean_endpoint_ratios"] = (
                    (
                        m["generated_mean"].reshape(3, 21)[:, -1]
                        - m["generated_mean"].reshape(3, 21)[:, 0]
                    )
                    / (
                        m["target_mean"].reshape(3, 21)[:, -1]
                        - m["target_mean"].reshape(3, 21)[:, 0]
                    )
                ).tolist()
                info["width_endpoint_ratios"] = (
                    (
                        s["generated_sd"].reshape(3, 21)[:, -1]
                        - s["generated_sd"].reshape(3, 21)[:, 0]
                    )
                    / (
                        s["target_sd"].reshape(3, 21)[:, -1]
                        - s["target_sd"].reshape(3, 21)[:, 0]
                    )
                ).tolist()
                info["width_decreasing_steps"] = (
                    (np.diff(s["generated_sd"].reshape(3, 21), axis=1) < -1e-10)
                    .sum(1)
                    .tolist()
                )
                arrays["target_mean"] = m["target_mean"].reshape(3, 21)
                arrays["target_sd"] = s["target_sd"].reshape(3, 21)
            else:
                profile_indices = np.array([0, 26, 53, 79])
                profile_conditions = data["reporting_conditions"][profile_indices]
                profiles = evaluate(profile_conditions, profiles=True)
                means.append(report["generated_mean"])
                fits.append(profiles["profiles"])
                arrays.update(
                    translation_target_profiles=target_profiles(profile_conditions),
                    translation_target_mean=report["target_mean"],
                    translation_development_c=(data["development_conditions"][:, 0] + 1)
                    / 2,
                    translation_reporting_c=(data["reporting_conditions"][:, 0] + 1)
                    / 2,
                    translation_profile_c=(profile_conditions[:, 0] + 1) / 2,
                    translation_fitting_c=(data["fitting_conditions"][:, 0] + 1) / 2,
                    translation_fitting_target_mean=target_profiles(
                        data["fitting_conditions"]
                    )
                    @ np.arange(17),
                )
            print(f"evaluated {design} {row['seed']}", flush=True)
        summaries[design] = dict(
            fits=rows,
            selected_candidate=receipt["winners"].get(design, "fixed"),
            reporting_conditions=protocol["protocols"][design]["reporting"],
        )
        params[design] = np.asarray(parameters)
        if design != "translation":
            grid_full.append(fits)
            grid_profile.append(diagnostics)
            model_mean.append(means)
            model_sd.append(sds)
            arrays[f"fitting_{design}"] = np.asarray(
                protocol["protocols"][design]["fitting"]
            )
        else:
            arrays.update(
                translation_profiles=np.asarray(fits),
                translation_response=np.asarray(means),
            )
    arrays.update(
        grid_tvd=np.asarray(grid_full),
        grid_profile_tvd=np.asarray(grid_profile),
        model_mean=np.asarray(model_mean),
        model_sd=np.asarray(model_sd),
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "DIAGNOSTICS.npz", **arrays)
    np.savez_compressed(output / "PARAMETERS.npz", **params)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summary = dict(
        status="complete",
        protocols=["grid", "axes"],
        n_fits=5,
        fixed_u=[-1, 0, 1],
        fixed_v=[-1, 0, 1],
        diagnostic_npz_sha256=sha(output / "DIAGNOSTICS.npz"),
        parameters_sha256=sha(output / "PARAMETERS.npz"),
        protocol_sha256=sha(output / "protocol.json"),
        campaign_receipt_sha256=sha(campaign / "campaign_receipt.json"),
        source=source,
        identity=receipt["identity"],
        fit_records=records,
        results=summaries,
        aggregation=(
            "condition mean per fit, then mean and sample SD across "
            "five independent fits; response median/IQR"
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

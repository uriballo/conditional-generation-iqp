"""Package audited finite-validation Gaussian results and replay them portably."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/gaussian2d-pair-training-20261009"
DATA = ROOT / "paper/data/gaussian2d_phase"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays_sha(values):
    h = hashlib.sha256()
    for key in sorted(values):
        h.update(key.encode())
        h.update(np.asarray(values[key]).tobytes())
    return h.hexdigest()


def model(theta, conditions, product=False):
    k = np.arange(17)
    x = np.diag(np.sqrt((k[:-1] + 1) * (16 - k[:-1])), 1)
    x += x.T
    eig, vec = eigh(x)
    base = np.sqrt(np.exp(-((k - 8) ** 2) / 8))
    base /= np.linalg.norm(base)
    if product:
        base = np.sqrt(np.array([comb(16, int(i)) for i in k]) / 2**16)
    c = np.asarray(conditions, dtype=float)
    amp = base * np.exp(-0.25j * c[:, 0, None] * (16 - 2 * k))
    amp *= np.exp(-1j * c[:, 1, None] * (k - 8) ** 2 / 40)
    unitary = (vec * np.exp(1j * float(theta) * eig)) @ vec.T
    return abs(amp @ unitary.T) ** 2


def target(conditions):
    k = np.arange(17)
    c = np.asarray(conditions, dtype=float)
    mu = 8 + 3 * c[:, 0, None]
    sigma = 2 * (1 + 0.2 * c[:, 1, None])
    p = np.exp(-((k - mu) ** 2) / (2 * sigma**2))
    return p / p.sum(axis=1, keepdims=True)


def tvd(q, p):
    return 0.5 * np.abs(q - p).sum(axis=1)


def conditions(root):
    path = root / (
        "source-workspace/.local_research/gaussian2d-phase-sampled-20261009/"
        "conditions.json"
    )
    return json.loads(path.read_text())["protocols"]


def calibration(fitting, means, product=False):
    k = np.arange(17)
    if product:
        ladder = 8 * np.cos(fitting[:, 1] / 40) ** 15
    else:
        base = np.sqrt(np.exp(-((k - 8) ** 2) / 8))
        base /= np.linalg.norm(base)
        adjacent = np.sqrt((k[:-1] + 1) * (16 - k[:-1])) * base[:-1] * base[1:]
        ladder = np.cos(fitting[:, 1, None] * (2 * (8 - k[:-1]) - 1) / 40) @ adjacent
    g = ladder * np.sin(fitting[:, 0] / 2)
    sine = float(g @ (means - 8) / (g @ g))
    return float((np.pi - np.arcsin(np.clip(sine, -1, 1))) / 2)


def heldout(fitting, reporting):
    return np.array(
        [
            not np.any(np.all(np.isclose(fitting, point, atol=1e-12, rtol=0), axis=1))
            for point in reporting
        ]
    )


def package():
    receipt = json.loads((RESULTS / "campaign_receipt.json").read_text())
    audit = json.loads((RESULTS / "AUDIT.json").read_text())
    summary = json.loads((RESULTS / "SUMMARY.json").read_text())
    manifest = RESULTS / "source-manifest.json"
    if receipt.get("status") != "complete" or len(receipt["records"]) != 20:
        raise RuntimeError("campaign is not complete")
    if not audit.get("passed") or audit.get("fit_count") != 20:
        raise RuntimeError("audit gate failed")
    if summary["source_manifest_sha256"] != sha(manifest):
        raise RuntimeError("summary manifest hash mismatch")
    if audit["summary_sha256"] != sha(RESULTS / "SUMMARY.json"):
        raise RuntimeError("summary differs from audited evidence")
    if audit["evaluation_sha256"] != sha(RESULTS / "EVALUATION.npz"):
        raise RuntimeError("evaluation differs from audited evidence")
    if any(
        value["source_manifest_sha256"] != sha(manifest) for value in (receipt, audit)
    ):
        raise RuntimeError("campaign/audit source identities differ")
    prot = conditions(RESULTS)
    rows, export = [], {}
    for row in summary["fits"]:
        rec = next(
            r
            for r in receipt["records"]
            if r["protocol"] == row["protocol"]
            and r["estimator"] == row["estimator"]
            and r["seed"] == row["seed"]
        )
        fit = ROOT / rec["receipt"]
        cp = fit.parent / "checkpoint.npz"
        fit_receipt = json.loads(fit.read_text())
        if fit_receipt["checkpoint_sha256"] != sha(cp):
            raise RuntimeError("checkpoint differs from audited receipt")
        if row["checkpoint_sha256"] != sha(cp):
            raise RuntimeError("checkpoint differs from summary")
        with np.load(cp) as z:
            fitting, pool = np.asarray(z["training_conditions"]), np.asarray(z["pool"])
        reporting = np.asarray(prot[row["protocol"]]["reporting"], dtype=float)
        mask = heldout(fitting, reporting)
        angles = row["theta"]
        means = pool.sum(-1).mean(-1)
        cal_g, cal_p = calibration(fitting, means), calibration(fitting, means, True)
        if (
            abs(cal_g - angles["mean_calibration_gaussian"]) > 1e-14
            or abs(cal_p - angles["mean_calibration_product"]) > 1e-14
        ):
            raise RuntimeError("OLS calibration mismatch")
        names = (
            "selected",
            "terminal",
            "mean_calibration_gaussian",
            "mean_calibration_product",
        )
        metrics = {}
        for name in names:
            q = model(angles[name], reporting, name.endswith("product"))
            metrics[name] = float(tvd(q, target(reporting))[mask].mean())
            if abs(metrics[name] - row["metrics"][name]["full_tvd_mean"]) > 1e-11:
                raise RuntimeError("TVD replay mismatch")
        key = f"{row['protocol']}_{row['estimator']}_{row['seed']}"
        export[key + "_reporting"] = reporting
        export[key + "_heldout"] = mask
        export[key + "_fitting"] = fitting
        export[key + "_fitting_means"] = means
        export[key + "_angles"] = np.array([angles[n] for n in names])
        rows.append(
            {
                "protocol": row["protocol"],
                "estimator": row["estimator"],
                "seed": row["seed"],
                "angles": {n: angles[n] for n in names},
                "tvd": metrics,
                "metrics": {n: row["metrics"][n] for n in ("selected", "terminal")},
            }
        )
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA / "finite_validation_parameters.npz", **export)
    payload = {
        "status": "passed",
        "fit_count": 20,
        "rows": rows,
        "source_manifest_sha256": sha(manifest),
        "summary_sha256": sha(RESULTS / "SUMMARY.json"),
        "audit_sha256": sha(RESULTS / "AUDIT.json"),
        "arrays_sha256": arrays_sha(export),
    }
    (DATA / "finite_validation.json").write_text(json.dumps(payload, indent=2) + "\n")
    provenance = {
        "inputs": {
            n: sha(RESULTS / n)
            for n in (
                "SUMMARY.json",
                "AUDIT.json",
                "campaign_receipt.json",
                "source-manifest.json",
            )
        },
        "export_sha256": sha(DATA / "finite_validation_parameters.npz"),
        "summary_export_sha256": sha(DATA / "finite_validation.json"),
        "replayer_sha256": sha(Path(__file__)),
    }
    (DATA / "finite_validation_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return payload


def replay(output):
    payload = json.loads((DATA / "finite_validation.json").read_text())
    arrays = dict(np.load(DATA / "finite_validation_parameters.npz"))
    if payload["arrays_sha256"] != arrays_sha(arrays):
        raise RuntimeError("exported array hash mismatch")
    provenance = json.loads((DATA / "finite_validation_provenance.json").read_text())
    if provenance["export_sha256"] != sha(DATA / "finite_validation_parameters.npz"):
        raise RuntimeError("exported file hash mismatch")
    if provenance["summary_export_sha256"] != sha(DATA / "finite_validation.json"):
        raise RuntimeError("exported summary hash mismatch")
    if len(payload["rows"]) != 20:
        raise RuntimeError("expected twenty exported fits")
    identities = {(r["protocol"], r["estimator"], r["seed"]) for r in payload["rows"]}
    if len(identities) != 20:
        raise RuntimeError("duplicate fit identities")
    max_tvd_error, max_angle_error = 0.0, 0.0
    for row in payload["rows"]:
        key = f"{row['protocol']}_{row['estimator']}_{row['seed']}"
        names = (
            "selected",
            "terminal",
            "mean_calibration_gaussian",
            "mean_calibration_product",
        )
        if not np.array_equal(
            arrays[key + "_angles"], [row["angles"][n] for n in names]
        ):
            raise RuntimeError("angle metadata differs from exported array")
        for product, name in ((False, names[2]), (True, names[3])):
            angle = calibration(
                arrays[key + "_fitting"], arrays[key + "_fitting_means"], product
            )
            error = abs(angle - row["angles"][name])
            max_angle_error = max(max_angle_error, error)
            if error > 1e-13:
                raise RuntimeError("portable finite-mean calibration mismatch")
        for name, theta in row["angles"].items():
            reporting = arrays[key + "_reporting"]
            value = tvd(
                model(theta, reporting, name.endswith("product")), target(reporting)
            )[arrays[key + "_heldout"]].mean()
            error = abs(value - row["tvd"][name])
            max_tvd_error = max(max_tvd_error, error)
            if error > 1e-11:
                raise RuntimeError("portable replay mismatch")
    output.mkdir(parents=True, exist_ok=True)
    (output / "finite_validation_replay.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "fit_count": 20,
                "maximum_tvd_replay_error": max_tvd_error,
                "maximum_calibration_angle_error": max_angle_error,
            },
            indent=2,
        )
        + "\n"
    )


def write_table(path):
    rows = json.loads((DATA / "finite_validation.json").read_text())["rows"]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Full-distribution TVD with finite-target validation "
        "and exact model moments. Values are means and sample standard "
        "deviations across five fits on each design's "
        "reporting set. The designs have different reporting conditions. "
        "Direct calibration uses fitting-pool means only, "
        "without validation data or iterative optimization.}",
        "\\label{tab:gaussian-phase-finite-validation}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Model & Separate variation & Joint variation \\\\",
        "\\midrule",
    ]
    for name, label in (
        ("selected", "Original estimator, selected"),
        ("terminal", "Original estimator, terminal"),
        ("pair_selected", "Pair-averaged selected"),
        ("pair_terminal", "Pair-averaged terminal"),
        ("mean_calibration_gaussian", "Direct calibration, Gaussian prior"),
        ("mean_calibration_product", "Direct calibration, product prior"),
    ):
        cells = []
        for protocol in ("axes", "rectangle"):
            estimator = "pair" if name.startswith("pair_") else "raw"
            metric_name = name.removeprefix("pair_")
            vals = [
                r["tvd"][metric_name]
                for r in rows
                if r["protocol"] == protocol and r["estimator"] == estimator
            ]
            if len(vals) != 5:
                raise RuntimeError("expected five fits per table cell")
            pair = [
                r["tvd"][metric_name]
                for r in rows
                if r["protocol"] == protocol and r["estimator"] == "pair"
            ]
            if name.startswith("mean_calibration"):
                assert np.max(np.abs(np.asarray(vals) - np.asarray(pair))) < 1e-14
            cells.append(f"{np.mean(vals):.4f} $\\pm$ {np.std(vals, ddof=1):.4f}")
        lines.append(f"{label} & {cells[0]} & {cells[1]} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/gaussian2d-phase-finite-replay"
    )
    parser.add_argument("--tables", type=Path)
    args = parser.parse_args()
    if args.package:
        package()
    replay(args.output)
    if args.tables:
        write_table(args.tables)
        if args.package:
            path = DATA / "finite_validation_provenance.json"
            provenance = json.loads(path.read_text())
            provenance["table_sha256"] = sha(args.tables)
            provenance["table_path"] = str(args.tables.resolve().relative_to(ROOT))
            path.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()

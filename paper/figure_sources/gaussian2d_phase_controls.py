"""Replay Gaussian phase-input controls and selected/terminal checkpoint errors."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "values": values.tolist(),
    }


def evaluate(path):
    """Evaluate stored parameters; do not fit parameters or select checkpoints."""
    data = np.load(path, allow_pickle=False)
    n = 16
    k = np.arange(n + 1)
    m = n / 2 - k
    x = np.diag(np.sqrt((k[:-1] + 1) * (n - k[:-1])), 1)
    x += x.T
    eig, vec = eigh(x)
    gaussian = np.exp(-((k - 8) ** 2) / 8)
    gaussian /= gaussian.sum()
    product = np.asarray([comb(n, int(j)) for j in k], float) / 2**n
    a = np.sqrt(gaussian)
    ax = float(a @ x @ a / 2)
    theta = float((np.pi - np.arcsin(6 / ax)) / 2)

    def probabilities(conditions, angle, prior):
        phases = np.exp(
            -0.5j * conditions[:, 0, None] * m - 0.025j * conditions[:, 1, None] * m**2
        )
        unitary = (vec * np.exp(1j * angle * eig)) @ vec.T
        return abs((np.sqrt(prior) * phases) @ unitary.T) ** 2

    rows = {}
    for protocol in ("rectangle", "axes"):
        c = data[f"{protocol}_conditions"]
        mask = data[f"{protocol}_mask"]
        target = np.exp(
            -((k - (8 + 3 * c[:, 0, None])) ** 2)
            / (2 * (2 * (1 + 0.2 * c[:, 1, None])) ** 2)
        )
        target /= target.sum(axis=1, keepdims=True)

        def errors(angle, prior=gaussian, c=c, target=target, mask=mask):
            q = probabilities(c, angle, prior)
            assert np.max(abs(q.sum(axis=1) - 1)) < 1e-12
            return float((0.5 * abs(q - target).sum(axis=1))[mask].mean())

        rows[protocol] = {
            stage: stats([errors(t) for t in data[f"{protocol}_{stage}_theta"]])
            for stage in ("initial", "selected", "terminal")
        }
        rows[protocol].update(
            encoder_only=stats([errors(0)]),
            fixed_rotation=stats([errors(3 * np.pi / 8)]),
            analytic_gaussian=stats([errors(theta)]),
            analytic_product=stats([errors(theta, product)]),
            selected_steps=data[f"{protocol}_selected_steps"].tolist(),
            seeds=data[f"{protocol}_seeds"].tolist(),
            n_reporting_conditions=int(mask.sum()),
        )
    return {
        "scope": (
            "Exact replay of frozen sampled parameters and post-hoc target-informed "
            "analytic controls; no new fitting or prospective baseline comparison."
        ),
        "protocol_order": ["rectangle", "axes"],
        "analytic_theta": theta,
        "mean_Jx": ax,
        "gaussian_product_squared_overlap": float((a @ np.sqrt(product)) ** 2),
        "tables": rows,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data/gaussian2d_phase/control_parameters.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root.parent
        / "results/gaussian2d-phase-publication-replay/controls.json",
    )
    parser.add_argument(
        "--tables", type=Path, help="Optional output directory for tables"
    )
    args = parser.parse_args()
    result = evaluate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.tables:
        args.tables.mkdir(parents=True, exist_ok=True)
        write_tables(result, args.tables)


def write_tables(result, output):
    rows = result["tables"]

    def values(key):
        cells = []
        for p in ("axes", "rectangle"):
            s = rows[p][key]
            cell = f"{s['mean']:.4f}"
            if len(s["values"]) == 5:
                cell += rf" $\pm$ {s['sample_sd']:.4f}"
            cells.append(cell)
        return " & ".join(cells)

    start = (
        "\\centering\n"
        "\\begin{tabular}{lcc}\\toprule\n"
        " & Separate variation & Joint variation\\\\\n"
        " & $16$ reporting pairs & $32$ reporting pairs\\\\\\midrule\n"
    )
    main = (
        "\\begin{table*}[!t]\n"
        "\\caption{Full-distribution TVD on each design's reporting set, excluding "
        "fitting and development conditions. For the sampled fits, we report the "
        "mean and sample standard deviation of per-fit mean errors across five "
        "runs. The fixed rotation was chosen after the capacity pilot and frozen "
        "before sampled training. Exact development TVD selects checkpoints. "
        "The designs use different reporting sets.}\n"
        "\\label{tab:gaussian-two-dimensional}\n" + start
    )
    for key, label in (
        ("encoder_only", "Encoder only"),
        ("fixed_rotation", r"Fixed $\theta=3\pi/8$"),
        ("selected", "Sampled calibration, selected checkpoint"),
    ):
        main += label + " & " + values(key) + "\\\\\n"
    main += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    (output / "gaussian2d_phase_tvd.tex").write_text(main)
    appendix = (
        "\\begin{table*}[!t]\n"
        "\\caption{Selection and fixed-control diagnostics on the same reporting "
        "sets as Table~\\ref{tab:gaussian-two-dimensional}. Initial, selected and "
        "terminal rows summarize the same five sampled fits per design. The "
        "analytic controls use target-slope information, were derived after the "
        "campaign and underwent no training. They are descriptive diagnostics. "
        "The product prior retains the quadratic entangling phase.}\n"
        "\\label{tab:gaussian-phase-controls}\n" + start
    )
    for key, label in (
        ("initial", "Initial checkpoint"),
        ("selected", "Selected checkpoint"),
        ("terminal", "Terminal checkpoint"),
        ("fixed_rotation", r"Fixed $\theta=3\pi/8$"),
        ("analytic_gaussian", "Analytic angle, Gaussian prior"),
        ("analytic_product", "Analytic angle, product prior"),
    ):
        appendix += label + " & " + values(key) + "\\\\\n"
    appendix += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    (output / "gaussian2d_phase_controls.tex").write_text(appendix)


if __name__ == "__main__":
    main()

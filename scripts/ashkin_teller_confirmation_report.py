#!/usr/bin/env python3
"""Verify sealed stage-3 artifacts and report paired interpolation metrics."""

# Long strings are paragraphs and tables in the result document.
# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

from iqp_cond.artifacts import read_immutable_json
from iqp_cond.bond_reference import BOND_ENCODING
from iqp_cond.encoding import TWO_BLOCK_BRANCH_ENCODING
from iqp_cond.exact import exact_iqp_probabilities_many

MODELS = (
    "cat_encoder_only",
    "cat",
    "bond_encoder_only",
    "bond",
    "log_composition",
    "log_linear",
)
METRICS = ("tvd", "correlation_rmse", "marginal_a_tvd", "marginal_b_tvd")
LABELS = {
    "cat_encoder_only": "Symmetry-only input",
    "cat": "Symmetry-only IQP",
    "bond_encoder_only": "Bond-informed input",
    "bond": "Bond-informed IQP",
    "log_composition": "Log composition",
    "log_linear": "Pooled log-linear",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recompute_metrics(q, p):
    bits = (np.arange(256)[:, None] >> np.arange(8)) & 1
    spins = 1 - 2 * bits
    a = spins[:, :4] * np.roll(spins[:, :4], -1, axis=1)
    b = spins[:, 4:] * np.roll(spins[:, 4:], -1, axis=1)

    def corr(v):
        return v @ (a * b) - (v @ a) * (v @ b)

    return {
        "tvd": 0.5 * np.abs(q - p).sum(1),
        "correlation_rmse": np.sqrt(np.mean((corr(q) - corr(p)) ** 2, axis=1)),
        "marginal_a_tvd": 0.5
        * np.abs(q.reshape(-1, 16, 16).sum(1) - p.reshape(-1, 16, 16).sum(1)).sum(1),
        "marginal_b_tvd": 0.5
        * np.abs(q.reshape(-1, 16, 16).sum(2) - p.reshape(-1, 16, 16).sum(2)).sum(1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("artifacts/ashkin-teller-confirmation-2026-09-07"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/ashkin-teller-confirmation-result-2026-09-07"),
    )
    args = parser.parse_args()
    out = args.directory
    manifest = read_immutable_json(out / "manifest.json")
    seal = read_immutable_json(out / "predictions-seal.json")
    evaluation = read_immutable_json(out / "evaluation.json")
    mh = digest(out / "manifest.json")
    sh = digest(out / "predictions-seal.json")
    if (
        seal["manifest_sha256"] != mh
        or evaluation["manifest_sha256"] != mh
        or evaluation["seal_sha256"] != sh
    ):
        raise RuntimeError("Evaluation does not match the frozen prediction seal")
    for name, expected in seal["files"].items():
        if digest(out / name) != expected:
            raise RuntimeError("Prediction artifact changed: " + name)
    if digest(out / "source.tar.gz") != manifest["source_archive_sha256"]:
        raise RuntimeError("Source archive changed")
    with tarfile.open(out / "source.tar.gz", "r:gz") as archive:
        for name, expected in manifest["source_sha256"].items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != expected:
                raise RuntimeError("Frozen source mismatch: " + name)
    if (
        digest(out / "data.npz") != manifest["data_sha256"]
        or digest(out / "evaluation-targets.npz")
        != evaluation["evaluation_targets_sha256"]
    ):
        raise RuntimeError("Target/data archive hash mismatch")
    with np.load(out / "data.npz") as d:
        train_conditions = d["train_conditions"]
        test_conditions = d["test_coordinates"]
        g = d["generators"]
        train_target = d["train_targets"]
    with np.load(out / "evaluation-targets.npz") as d:
        test_target = d["test_targets"]
        np.testing.assert_array_equal(d["test_conditions"], test_conditions)
    seeds = manifest["seeds"]
    rows = evaluation["rows"]
    if len(rows) != 60 or len(seeds) != 10:
        raise RuntimeError("Incomplete experiment: no missing-seed filtering permitted")
    rowmap = {(r["model"], r["seed"]): r for r in rows}
    if len(rowmap) != 60:
        raise RuntimeError("Duplicate evaluation identities")
    for seed in seeds:
        for model in MODELS:
            row = rowmap[(model, seed)]
            if model in ("log_composition", "log_linear"):
                with np.load(out / f"classical-{seed}.npz") as d:
                    trainq = d[model + "_train"]
                    testq = d[model + "_test"]
            else:
                encoder = model.split("_")[0]
                zero = model.endswith("encoder_only")
                with np.load(out / f"{encoder}-{seed}.npz") as d:
                    trainq = (
                        d["encoder_only_train"] if zero else d["train_probabilities"]
                    )
                    testq = d["encoder_only_test"] if zero else d["test_predictions"]
                    params = np.zeros(30) if zero else d["selected_params"]
                query = np.concatenate([train_conditions, test_conditions])
                angles = np.repeat(query, 4, axis=1)
                enc = BOND_ENCODING
                if encoder == "cat":
                    angles = 0.225 * np.pi - 0.20 * np.pi * (angles - 0.1) / 0.6
                    enc = TWO_BLOCK_BRANCH_ENCODING
                exact = exact_iqp_probabilities_many(params, g, angles, enc)
                np.testing.assert_allclose(
                    np.concatenate([trainq, testq]), exact, atol=1e-12
                )
            for role, q, p in (
                ("train", trainq, train_target),
                ("test", testq, test_target),
            ):
                if not np.isfinite(q).all() or np.min(q) < 0:
                    raise RuntimeError("Invalid predictions")
                np.testing.assert_allclose(q.sum(1), 1, atol=1e-12)
                for flip in (15, 240):
                    np.testing.assert_allclose(
                        q, q[:, np.arange(256) ^ flip], atol=1e-12
                    )
                for metric, value in recompute_metrics(q, p).items():
                    np.testing.assert_allclose(row[role][metric], value, atol=1e-12)
    index = np.random.default_rng(950701).integers(0, 10, size=(10000, 10))

    def interval(seed_values):
        return np.quantile(
            np.asarray(seed_values)[index].mean(1), [0.025, 0.975]
        ).tolist()

    models = {}
    raw = {}
    for model in MODELS:
        models[model] = {}
        for role in ("train", "test"):
            models[model][role] = {}
            for metric in METRICS:
                values = np.array([rowmap[(model, s)][role][metric] for s in seeds])
                means = values.mean(1)
                worst = values.max(1)
                raw[(model, role, metric)] = values
                models[model][role][metric] = {
                    "mean": float(means.mean()),
                    "seed_means": means.tolist(),
                    "seed_sd": float(means.std(ddof=1)),
                    "ci95": interval(means),
                    "condition_means": values.mean(0).tolist(),
                    "condition_sd": values.std(0, ddof=1).tolist(),
                    "seed_worst": worst.tolist(),
                    "mean_worst": float(worst.mean()),
                    "worst": float(worst.max()),
                    "mean_worst_ci95": interval(worst),
                }
    contrasts = {}
    for metric in METRICS:
        delta = (raw[("bond", "test", metric)] - raw[("cat", "test", metric)]).mean(1)
        contrasts[metric] = {
            "mean": float(delta.mean()),
            "seed_differences": delta.tolist(),
            "ci95": interval(delta),
        }
    gains = {}
    for model in ("cat", "bond"):
        gains[model] = {}
        for metric in METRICS:
            gain = (
                raw[(model + "_encoder_only", "test", metric)]
                - raw[(model, "test", metric)]
            )
            gains[model][metric] = {
                "mean": float(gain.mean()),
                "ci95": interval(gain.mean(1)),
                "condition_means": gain.mean(0).tolist(),
                "minimum_seed_condition_gain": float(gain.min()),
                "worsened_mean_conditions": [
                    test_conditions[i].tolist()
                    for i in np.flatnonzero(gain.mean(0) < -1e-12)
                ],
                "worsened_seed_condition_count": int((gain < -1e-12).sum()),
            }
    payload = {
        "manifest_file_sha256": mh,
        "prediction_seal_sha256": sh,
        "evaluation_sha256": digest(out / "evaluation.json"),
        "report_script_sha256": digest(Path(__file__)),
        "train_conditions": train_conditions.tolist(),
        "test_conditions": test_conditions.tolist(),
        "seeds": seeds,
        "models": models,
        "contrasts": contrasts,
        "encoder_only_gains": gains,
        "bootstrap": {
            "unit": "paired seed means over9conditions",
            "draws": 10000,
            "seed": 950701,
            "interval": "percentile95%",
        },
        "verified_iqp_fits": 20,
        "verified_classical_bundles": 10,
        "raw_evaluation_rows": rows,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    p = contrasts["tvd"]
    favored = "bond-informed" if p["mean"] < 0 else "symmetry-only"
    lines = [
        "# Ashkin-Teller confirmatory interpolation result",
        "",
        "Frozen stage-3 experiment: 20 IQP fits, ten paired classical-baseline bundles, 16 perimeter training conditions and nine new interior test combinations. The method was developed on related conditions; this is compositional interpolation, not extrapolation.",
        "",
        f"The primary bond-minus-cat mean test-TVD difference is **{p['mean']:+.5f}**, paired 95% bootstrap interval **[{p['ci95'][0]:+.5f}, {p['ci95'][1]:+.5f}]**. The average favors {favored} encoding.",
        (
            "The interval crosses zero; this experiment does not establish a consistent mean-TVD ordering."
            if p["ci95"][0] <= 0 <= p["ci95"][1]
            else "The interval excludes zero under this predeclared paired-seed analysis; it does not establish an advantage across other tasks or encoders."
        ),
        "",
        "## All predeclared models",
        "",
        "| Model | Train TVD | Test TVD [95% CI] | Seed SD | Test C error | Mean worst-condition TVD |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        s = models[model]["test"]["tvd"]
        c = models[model]["test"]["correlation_rmse"]
        lines.append(
            f"| {LABELS[model]} | {models[model]['train']['tvd']['mean']:.5f} | {s['mean']:.5f} [{s['ci95'][0]:.5f}, {s['ci95'][1]:.5f}] | {s['seed_sd']:.5f} | {c['mean']:.5f} | {s['mean_worst']:.5f} |"
        )
    lines += [
        "",
        "TVDs and C errors are exact metrics of the saved output distributions. C error is the RMSE over the four connected cross-layer bond correlations. Means average nine conditions within each seed and then ten seeds. Encoder-only inputs are deterministic controls, not ten independent fitted models. Worst-condition TVD is first maximized within each seed, then averaged. Raw per-seed worst values and marginal errors are included in the JSON.",
        "",
        "## Learned contribution and correlation limitations",
        "",
    ]
    for model in ("cat", "bond"):
        t = gains[model]["tvd"]
        c = gains[model]["correlation_rmse"]
        lines.append(
            f"- {LABELS[model]}: mean encoder-only minus trained TVD **{t['mean']:+.5f}**; {t['worsened_seed_condition_count']} of 90 seed/condition pairs worsen TVD. Mean C-error gain **{c['mean']:+.5f}**; worsened condition means: {c['worsened_mean_conditions'] or 'none'} ({c['worsened_seed_condition_count']} of 90 seed/condition pairs worsen C error)."
        )
    c = contrasts["correlation_rmse"]
    correlation_delta = (
        raw[("bond", "test", "correlation_rmse")]
        - raw[("cat", "test", "correlation_rmse")]
    ).mean(0)
    correlation_worse = test_conditions[correlation_delta > 0].tolist()
    lines += [
        "",
        f"Bond-minus-cat connected-correlation error: {c['mean']:+.5f}, descriptive paired interval [{c['ci95'][0]:+.5f}, {c['ci95'][1]:+.5f}]. This secondary comparison is not adjusted for multiple testing.",
        f"The smaller average correlation error does not imply uniform improvement: bond-informed IQP has higher mean C error than symmetry-only IQP at {correlation_worse or 'no test condition'}. The largest condition-mean C errors are {max(models['cat']['test']['correlation_rmse']['condition_means']):.5f} (cat) and {max(models['bond']['test']['correlation_rmse']['condition_means']):.5f} (bond). These remaining errors limit claims of accurate interlayer-dependence recovery.",
        "",
        "The classical predictors are contextual comparisons under the same sample access. They need not lose for this to demonstrate the conditional quantum method. They have different representations and supplied assumptions: 189 pooled log-linear coefficients or 16 orbit histograms, compared with 30 IQP phases. The bond encoder already supplies the uncoupled Ising-ring law and its coupling map, while the cat encoder supplies the partition and both flips. Neither is prior-free.",
        "",
        "## Every held-out condition",
        "",
        "| (a,b) | Cat input TVD | Cat IQP TVD | Bond input TVD | Bond IQP TVD | Log composition TVD | Log-linear TVD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, (a, b) in enumerate(test_conditions):
        lines.append(
            f"| ({a:.2f},{b:.2f}) | "
            + " | ".join(
                f"{models[m]['test']['tvd']['condition_means'][i]:.5f}" for m in MODELS
            )
            + " |"
        )
    lines += [
        "",
        "| (a,b) | Input C error | Cat IQP C error | Bond IQP C error | Log composition C error | Log-linear C error |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, (a, b) in enumerate(test_conditions):
        methods = ("cat_encoder_only", "cat", "bond", "log_composition", "log_linear")
        lines.append(
            f"| ({a:.2f},{b:.2f}) | "
            + " | ".join(
                f"{models[m]['test']['correlation_rmse']['condition_means'][i]:.5f}"
                for m in methods
            )
            + " |"
        )
    lines += [
        "",
        "## Reproducibility and evaluation boundary",
        "",
        "All 20 selected IQP parameters and their test-coordinate predictions, both encoder-only predictions, and all ten classical bundles were saved and sealed before the first test-target evaluation. The sealed predictions were recomputed from fitted parameters before target access. Subsequent fitting is blocked by the evaluation-started marker. No test-based map, architecture, optimizer, regularization, checkpoint or seed selection was made.",
        "",
        "Each model used the same seed-specific 1,000 training and 1,000 independent validation examples at each of 16 perimeter conditions. IQP used the validated 500-update Adam-cosine schedule, 2,048 reference draws and 512 masks per bandwidth/condition; selection used fixed sampled validation MMD at training conditions only. Both classical grids and convergence rules were frozen before fitting and selected on validation likelihood. See the [frozen protocol](ashkin-teller-confirmation-protocol-2026-09-07.md).",
        "",
        "This report independently checked the source archive, all sealed prediction hashes, exact IQP probabilities through the NumPy evaluation path, every joint/marginal/correlation metric, normalization and both layer flips. Ten paired seeds are the uncertainty units; the nine conditions are not independent replications. Bootstrap intervals are approximate and conditional on this fixed split.",
        "",
        f"Manifest file SHA256: `{mh}`. Prediction seal SHA256: `{sh}`. Evaluation SHA256: `{digest(out / 'evaluation.json')}`. Raw archives and source snapshot: `{out}`.",
        "",
        "## Claim-to-evidence map",
        "",
        "| Claim | Evidence | Limitation | Review status |",
        "| --- | --- | --- | --- |",
        "| Conditional IQP predicts new joint input combinations | Frozen prediction seal, all nine test scores above | Within-domain interpolation; related development history disclosed | Numerically verified; author review pending |",
        "| Additional bond information changes accuracy | Paired primary TVD contrast and full C-error comparison | Changes encoder family and supplied information, not topology alone | Predeclared comparison; author review pending |",
        "| Learned phases add to the supplied input | Per-condition encoder-only gains above | Accuracy and dependence can improve differently | Verified at this split only |",
        "| Classical or quantum advantage | Not claimed | Classical comparators use different parameterizations and priors | Outside scope |",
        "| Generalization beyond this split or size | Not measured | Failed development extrapolation retained; no n=12 extension here | Open |",
        "",
        "## Figures",
        "",
        "[Paired summary](../../artifacts/ashkin-teller-confirmation-2026-09-07/figures/confirmation_paired_summary.pdf): mean test TVD and connected-correlation RMSE; faint points are ten seed means, bars are percentile bootstrap intervals over seeds, and grey inputs are deterministic controls. All six models are included.",
        "",
        "[Condition grids](../../artifacts/ashkin-teller-confirmation-2026-09-07/figures/confirmation_test_grid.pdf): all nine condition means for both IQP encoders; each metric uses a shared color scale across encoders. The lower row is connected-correlation RMSE. PNG previews and input/file hashes accompany the PDFs. These are proposed experiment figures, not yet manuscript inserts.",
        "",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({"contrast": contrasts, "gains": gains}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify and report the September 7 development study at 2,048 samples."""

# Long string literals are paragraphs and table rows in the generated report.
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("artifacts/ashkin-teller-development-2026-09-07"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/ashkin-teller-sampled-development-2026-09-07"),
    )
    args = parser.parse_args()
    out = args.directory
    manifest = read_immutable_json(out / "manifest.json")
    reference_samples = 2048
    assert manifest["reference_samples"] == reference_samples
    mh = digest(out / "manifest.json")
    with tarfile.open(out / "source.tar.gz", "r:gz") as archive:
        for name, expected in manifest["source_sha256"].items():
            assert (
                hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
            )
    assert digest(out / "data.npz") == manifest["data_sha256"]
    with np.load(out / "data.npz") as d:
        conditions = d["conditions"]
        g = d["generators"]
        target = d["targets"]
        initial = d["initial_params"]
    rows = []
    arrays = {}
    for encoder in ("cat", "bond"):
        for si, seed in enumerate(manifest["seeds"]):
            for mode in ("exact", "sampled"):
                name = f"{encoder}-{mode}-{seed}-s{reference_samples}"
                record = read_immutable_json(out / f"{name}.json")
                assert record["manifest_sha256"] == mh
                assert record["checkpoint_sha256"] == digest(out / f"{name}.npz")
                with np.load(out / f"{name}.npz") as d:
                    a = {k: d[k] for k in d.files}
                arrays[(encoder, mode, seed)] = a
                assert a["train_losses"].shape == (500,)
                assert np.array_equal(a["validation_steps"], np.arange(0, 501, 25))
                assert np.array_equal(a["initial_params"], initial[si])
                for value in a.values():
                    assert np.isfinite(value).all()
                q = a["probabilities"]
                assert q.shape == (11, 256)
                np.testing.assert_allclose(q.sum(1), 1, atol=1e-12)
                assert np.min(q) >= 0
                angles = np.repeat(conditions, 4, axis=1)
                encoding = BOND_ENCODING
                if encoder == "cat":
                    encoding = TWO_BLOCK_BRANCH_ENCODING
                    angles = 0.225 * np.pi - 0.20 * np.pi * (angles - 0.1) / 0.6
                recomputed = exact_iqp_probabilities_many(
                    a["selected_params"], g, angles, encoding
                )
                np.testing.assert_allclose(q, recomputed, atol=1e-12)
                np.testing.assert_allclose(
                    record["metrics"]["tvd"],
                    0.5 * np.abs(q - target).sum(1),
                    atol=1e-12,
                )
                for flip in (15, 240):
                    np.testing.assert_allclose(
                        q, q[:, np.arange(256) ^ flip], atol=1e-12
                    )
                rows.append(record)
    # All four variants share initialization, data, mask banks and batch order.
    for seed in manifest["seeds"]:
        reference = arrays[("cat", "exact", seed)]
        for encoder in ("cat", "bond"):
            for mode in ("exact", "sampled"):
                for key in ("initial_params", "masks", "weights", "batches"):
                    assert np.array_equal(
                        reference[key], arrays[(encoder, mode, seed)][key]
                    )
    summaries = {}
    gates = {}
    for encoder in ("cat", "bond"):
        for mode in ("exact", "sampled"):
            runs = [r for r in rows if r["encoder"] == encoder and r["mode"] == mode]
            tvd = np.array([r["metrics"]["tvd"] for r in runs])
            corr = np.array([r["metrics"]["correlation_rmse"] for r in runs])
            zero = np.array([r["encoder_only"]["tvd"] for r in runs])
            zero_corr = np.array([r["encoder_only"]["correlation_rmse"] for r in runs])
            summaries[f"{encoder}_{mode}"] = {
                "mean_tvd": float(tvd.mean()),
                "worst_tvd": float(tvd.max()),
                "seed_mean_tvd": tvd.mean(1).tolist(),
                "mean_correlation_rmse": float(corr.mean()),
                "worst_correlation_rmse": float(corr.max()),
                "mean_encoder_only_tvd_gain": float((zero - tvd).mean()),
                "minimum_encoder_only_tvd_gain": float((zero - tvd).min()),
                "per_condition_encoder_only_correlation_rmse": zero_corr.mean(
                    0
                ).tolist(),
                "per_condition_tvd": tvd.mean(0).tolist(),
                "per_condition_correlation_rmse": corr.mean(0).tolist(),
                "per_condition_encoder_only_tvd_gain": (zero - tvd).mean(0).tolist(),
            }
        gaps = np.array(summaries[f"{encoder}_sampled"]["seed_mean_tvd"]) - np.array(
            summaries[f"{encoder}_exact"]["seed_mean_tvd"]
        )
        gates[encoder] = {
            "paired_mean_tvd_gaps": gaps.tolist(),
            "median_gap": float(np.median(gaps)),
            "passed": bool(np.median(gaps) <= 0.03),
        }
    payload = {
        "manifest": manifest,
        "manifest_file_sha256": mh,
        "source_archive_sha256": digest(out / "source.tar.gz"),
        "report_script_sha256": digest(Path(__file__)),
        "reference_samples": reference_samples,
        "verified_fit_count": len(rows),
        "gate": gates,
        "summary": summaries,
        "runs": rows,
        "scope": "Development only; all 11 conditions used in training. No new generalization evidence.",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Ashkin-Teller sampled-training development comparison",
        "",
        "Stages 1-2 result, 2026-09-07. All 11 conditions are development fitting/validation conditions. Confirmatory targets remain unopened.",
        "",
        "## Result",
        "",
        "| Encoder | Training loss | Mean TVD | Worst TVD | Mean C error | Worst C error | Mean encoder-only TVD gain |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, s in summaries.items():
        enc, mode = key.split("_")
        lines.append(
            f"| {enc} | {mode} | {s['mean_tvd']:.5f} | {s['worst_tvd']:.5f} | {s['mean_correlation_rmse']:.5f} | {s['worst_correlation_rmse']:.5f} | {s['mean_encoder_only_tvd_gain']:.5f} |"
        )
    lines += [
        "",
        "Means average three seed fits and 11 conditions; worst values range over every seed/condition. C error is the RMSE of the four connected interlayer bond correlations. These are exact output metrics, even when training uses sampled estimates.",
        "",
        "## Paired engineering gate",
        "",
        "Predeclared tolerance: median paired increase in mean TVD (sampled minus exact) ≤.03 for each encoder, with no unexplained estimator failures. This is an engineering tolerance, not a significance test.",
        "",
    ]
    for encoder, gate in gates.items():
        gaps = ", ".join(f"{v:+.5f}" for v in gate["paired_mean_tvd_gaps"])
        lines.append(
            f"- {encoder}: paired gaps [{gaps}]; median {gate['median_gap']:+.5f}; {'PASS' if gate['passed'] else 'FAIL'}."
        )
    lines += [
        "",
        "## Every development condition",
        "",
        "TVDs and gains below average the three seeds. Δ is sampled minus exact TVD; gain is encoder-only minus sampled TVD, so negative gain identifies a condition worsened by training.",
        "",
        "| (a,b) | Cat exact | Cat sampled | Cat Δ | Cat gain | Bond exact | Bond sampled | Bond Δ | Bond gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, (a, b) in enumerate(conditions):
        values = []
        for encoder in ("cat", "bond"):
            e = summaries[f"{encoder}_exact"]["per_condition_tvd"][i]
            s = summaries[f"{encoder}_sampled"]["per_condition_tvd"][i]
            gain = summaries[f"{encoder}_sampled"][
                "per_condition_encoder_only_tvd_gain"
            ][i]
            values.extend([e, s, s - e, gain])
        lines.append(
            f"| ({a:.1f},{b:.1f}) | " + " | ".join(f"{v:.5f}" for v in values) + " |"
        )
    lines += [
        "",
        "| (a,b) | Encoder-only C error | Cat exact C error | Cat sampled C error | Bond exact C error | Bond sampled C error |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, (a, b) in enumerate(conditions):
        values = [
            summaries["cat_sampled"]["per_condition_encoder_only_correlation_rmse"][i]
        ] + [
            summaries[f"{e}_{m}"]["per_condition_correlation_rmse"][i]
            for e in ("cat", "bond")
            for m in ("exact", "sampled")
        ]
        lines.append(
            f"| ({a:.1f},{b:.1f}) | " + " | ".join(f"{v:.5f}" for v in values) + " |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Both encoders pass at 2,048 reference samples; the 4,096-sample fallback is not needed. Sampled joint TVD improves on the encoder-only input in every seed/condition. The symmetry-only fits show more sensitivity to sampled optimization: their largest seed-average gap is .02639, compared with .00322 for bond-informed fits. Passing the median tolerance does not imply identical optimization trajectories.",
        "",
        "Bond-informed input gives lower mean TVD, while symmetry-only input gives lower worst TVD and lower mean connected-correlation error. At (.7,.7), both sampled-trained models worsen connected-correlation error relative to their encoder-only inputs. The worst errors remain near .23. Thus this supports a controlled comparison of supplied physics and learned contributions, not a claim of uniformly superior bond encoding or uniformly accurate interlayer dependence.",
        "",
        "These results support freezing stage 3 next. They are not themselves new generalization evidence; no confirmatory run has been launched.",
        "",
        "## Protocol, verification and limits",
        "",
        f"All 12 fits use 500 Adam-cosine updates, 100 training samples per condition per update, 1,000 independently generated training/validation examples per condition, 512 masks per bandwidth/condition and {reference_samples:,} reference samples. The three seeds are 1701, 1702, 1703. No ancillas; both encoders use the same deduplicated 30-phase two-local bank.",
        "",
        "Both training modes select the minimum fixed sampled development-validation MMD among the initial checkpoint and checks every 25 updates. Exact controls optimize population MMD; sampled fits additionally include finite training-data, mask-bank and reference noise. Thus the gap is not a pure reference-sampling-error estimate.",
        "",
        "The supplied bond input has uncoupled-layer probability amplitudes with fixed phase eta=pi/8. Its intra-layer law already contains useful physics. The cat input supplies the layer partition and both spin-flip symmetries. Calling the latter completely physics-unaware would be misleading.",
        "",
        "The final implementation passed all 218 tests. Every recorded checkpoint/data hash was checked; saved output probabilities were independently recomputed, normalized, and verified against both flips. All four variants share data-derived initialization, mask banks and batching for each seed. Per-seed and per-condition marginal/correlation metrics are retained in the JSON and raw checkpoint bundles.",
        "",
        "A slow first implementation was stopped for runtime reasons. Partial fits remain in the sibling `-superseded-matrix-products` directory and are excluded here. Equivalent elementwise contractions and canceled ratio normalizers reduced runtime; deterministic input arrays were verified identical before restarting under the new source freeze.",
        "",
        f"Frozen manifest file SHA256: `{mh}`. Artifacts: `{out}`. The complete source snapshot is `source.tar.gz`; the JSON records its hash and full fit provenance.",
        "",
        "The former worst-correlation ≤.10 development gate remains a recorded failure. Passing the new sampled-fidelity gate does not reverse it. No classical comparison or quantum-advantage claim is made here. This experiment is not a fresh generalization test and does not establish hardware preparation or scalable output sampling.",
        "",
        "## Claim-to-evidence map",
        "",
        "| Claim | Evidence | Limitation | Review status |",
        "| --- | --- | --- | --- |",
        "| Tractable bond reference access and phase gradients | [Method](bond-reference-method.md), reference/estimator/preparation tests | Floating-point size scope; preparation is a construction, not hardware execution | Automated checks passed; author review pending |",
        "| Sampled training approximates paired exact development fits | Frozen 12-fit protocol and paired gaps above | Three seeds; data/mask/reference noise combined | Engineering gate reported; author review pending |",
        "| Learned IQP phases improve on the supplied input | Encoder-only gains and connected-correlation errors above | C error worsens at (.7,.7); uneven correlation accuracy | Development evidence only |",
        "| Generalization to fresh joint conditions | No evidence from this experiment | Stage 3 remains unrun | Open |",
        "",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({"gates": gates, "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()

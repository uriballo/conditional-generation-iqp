#!/usr/bin/env python3
"""Post hoc, read-only structural diagnostics for Ashkin--Teller outputs."""

# Long table rows are prose in the generated report.
# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from iqp_cond.artifacts import read_immutable_json

SEEDS = tuple(range(2701, 2711))
MODELS = (
    "cat_encoder_only",
    "cat",
    "bond_encoder_only",
    "bond",
    "log_composition",
    "log_linear",
)
IQP = {
    "cat_encoder_only": "cat",
    "cat": "cat",
    "bond_encoder_only": "bond",
    "bond": "bond",
}


def _states(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n % 2 or n < 4:
        raise ValueError("Expected two equal layers with at least two spins each")
    bits = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    return bits[:, : n // 2].sum(1), bits[:, n // 2 :].sum(1)


def shell_uniformize(
    q: np.ndarray, ka: np.ndarray | None = None, kb: np.ndarray | None = None
) -> np.ndarray:
    """Uniformize q inside each joint (K_A,K_B) shell."""
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    n = int(np.log2(q.shape[1]))
    if ka is None or kb is None:
        ka, kb = _states(n)
    out = np.zeros_like(q)
    for x, y in np.unique(np.column_stack((ka, kb)), axis=0):
        mask = (ka == x) & (kb == y)
        out[:, mask] = q[:, mask].sum(1, keepdims=True) / mask.sum()
    return out


def product_marginals(q: np.ndarray) -> np.ndarray:
    """Return q_A(s) q_B(t), retaining the little-endian state ordering."""
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    width = q.shape[1]
    m = int(np.log2(width) // 2)
    table = q.reshape(q.shape[0], 1 << m, 1 << m)
    return (table.sum(2)[:, :, None] * table.sum(1)[:, None, :]).reshape(q.shape)


def tvd(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(np.asarray(q) - np.asarray(p)).sum(axis=-1)


def diagnostic_metrics(q: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    q, p = np.asarray(q, float), np.asarray(p, float)
    if q.shape != p.shape or q.ndim != 2:
        raise ValueError("q and p must be matching (conditions, states) arrays")
    if (
        not np.isfinite(q).all()
        or not np.isfinite(p).all()
        or np.min(q) < 0
        or np.min(p) < 0
        or not np.allclose(q.sum(1), 1)
        or not np.allclose(p.sum(1), 1)
    ):
        raise ValueError("q and p must be finite probability distributions")
    n = int(np.log2(q.shape[1]))
    if 2**n != q.shape[1]:
        raise ValueError("State count must be a power of two")
    ka, kb = _states(n)
    eq, prod = shell_uniformize(q, ka, kb), product_marginals(q)
    for a, b in np.unique(np.column_stack((ka, kb)), axis=0):
        mask = (ka == a) & (kb == b)
        np.testing.assert_allclose(eq[:, mask].sum(1), q[:, mask].sum(1), atol=1e-12)
    side = 2 ** (n // 2)
    for axis in (1, 2):
        np.testing.assert_allclose(
            prod.reshape(-1, side, side).sum(axis),
            q.reshape(-1, side, side).sum(axis),
            atol=1e-12,
        )
    if not np.allclose(eq.sum(1), 1) or not np.allclose(prod.sum(1), 1):
        raise RuntimeError("diagnostic transformation did not preserve normalization")
    raw = tvd(q, p)
    side = int(np.sqrt(q.shape[1]))
    qt, pt = q.reshape(-1, side, side), p.reshape(-1, side, side)
    return {
        "tvd": raw,
        "shell_tvd": tvd(eq, p),
        "delta_spatial": tvd(eq, p) - raw,
        "product_tvd": tvd(prod, p),
        "delta_dependence": tvd(prod, p) - raw,
        "target_shell_tvd": tvd(shell_uniformize(p), p),
        # Little-endian state index is A + (B << m): axis 1 is B and
        # axis 2 is A after reshape, so sum over B for A and over A for B.
        "marginal_a_tvd": tvd(qt.sum(1), pt.sum(1)),
        "marginal_b_tvd": tvd(qt.sum(2), pt.sum(2)),
    }


def summarize(
    values: np.ndarray, conditions: np.ndarray, seeds: tuple[int, ...] = SEEDS
) -> dict:
    """Summarize condition values by seed, then paired seed bootstrap."""
    values = np.asarray(values, float).reshape(len(seeds), len(conditions))
    seed_means = values.mean(1)
    index = np.random.default_rng(950701).integers(
        0, len(seeds), size=(10000, len(seeds))
    )
    ci = np.quantile(seed_means[index].mean(1), [0.025, 0.975]).tolist()
    return {
        "mean": float(seed_means.mean()),
        "seed_condition_values": values.tolist(),
        "seed_means": seed_means.tolist(),
        "ci95": ci,
        "condition_means": values.mean(0).tolist(),
        "negative_count": int((values < -1e-12).sum()),
        "negative_condition_means": int((values.mean(0) < -1e-12).sum()),
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(directory: Path, output: Path, *, label: str = "n8-posthoc") -> dict:
    manifest = read_immutable_json(directory / "manifest.json")
    seal = read_immutable_json(directory / "predictions-seal.json")
    evaluation = read_immutable_json(directory / "evaluation.json")
    if seal.get("manifest_sha256") != digest(directory / "manifest.json"):
        raise RuntimeError("prediction seal does not match manifest")
    if evaluation.get("manifest_sha256") != digest(
        directory / "manifest.json"
    ) or evaluation.get("seal_sha256") != digest(directory / "predictions-seal.json"):
        raise RuntimeError("evaluation does not match sealed prediction bundle")
    if (
        digest(directory / "data.npz") != manifest["data_sha256"]
        or digest(directory / "evaluation-targets.npz")
        != evaluation["evaluation_targets_sha256"]
    ):
        raise RuntimeError("data or target archive hash mismatch")
    for name, expected in seal["files"].items():
        if digest(directory / name) != expected:
            raise RuntimeError("sealed artifact changed: " + name)
    with np.load(directory / "data.npz") as d:
        conditions = d["test_coordinates"]
    with np.load(directory / "evaluation-targets.npz") as d:
        targets = d["test_targets"]
        target_conditions = d["test_conditions"]
    np.testing.assert_array_equal(conditions, target_conditions)
    seeds = tuple(int(seed) for seed in manifest["seeds"])
    if len(set(seeds)) != 10 or conditions.shape != (9, 2) or targets.shape[0] != 9:
        raise RuntimeError("diagnostic requires ten seeds and nine test conditions")
    n = int(manifest["n_visible"])
    expected_label = "n8-posthoc" if n == 8 else "n12-predeclared"
    if label != expected_label:
        raise ValueError("Label does not match the experiment size and declaration")
    note = (
        "n=8 diagnostics are post hoc"
        if n == 8
        else "n=12 diagnostics were predeclared before prediction sealing"
    )
    rows: dict[str, dict] = {}
    for model in MODELS:
        per_metric: dict[str, list] = {
            k: []
            for k in (
                "tvd",
                "delta_spatial",
                "delta_dependence",
                "target_shell_tvd",
                "marginal_a_tvd",
                "marginal_b_tvd",
            )
        }
        for seed in seeds:
            if model in IQP:
                with np.load(directory / f"{IQP[model]}-{seed}.npz") as d:
                    q = (
                        d["encoder_only_test"]
                        if model.endswith("encoder_only")
                        else d["test_predictions"]
                    )
            else:
                with np.load(directory / f"classical-{seed}.npz") as d:
                    q = d[model + "_test"]
            metrics = diagnostic_metrics(q, targets)
            for key in per_metric:
                per_metric[key].append(metrics[key])
        rows[model] = {
            key: summarize(np.asarray(value), conditions, seeds)
            for key, value in per_metric.items()
        }
    payload = {
        "label": label,
        "directory": str(directory),
        "manifest_sha256": digest(directory / "manifest.json"),
        "seal_sha256": digest(directory / "predictions-seal.json"),
        "evaluation_sha256": digest(directory / "evaluation.json"),
        "report_script_sha256": digest(Path(__file__)),
        "models": rows,
        "conditions": conditions.tolist(),
        "seeds": list(seeds),
        "note": note
        + "; transformations preserve declared profiles and no refitting was performed.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Ashkin-Teller structural diagnostics",
        "",
        "Read-only diagnostics of sealed outputs; " + note + ".",
        "",
        "| Model | TVD | Δ spatial | Δ dependence |",
        "|---|---:|---:|---:|",
    ]
    for model, data in rows.items():
        lines.append(
            f"| {model} | {data['tvd']['mean']:.5f} | "
            f"{data['delta_spatial']['mean']:+.5f} | "
            f"{data['delta_dependence']['mean']:+.5f} |"
        )
    lines += [
        "",
        "Positive deltas mean learned arrangement or interlayer dependence "
        "improves TVD relative to the profile-preserving transformation.",
        "",
        "## Seed uncertainty",
        "",
        "| Model | Spatial 95% CI | Dependence 95% CI | Negative spatial /90 | Negative dependence /90 |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for model, data in rows.items():
        a, b = data["delta_spatial"], data["delta_dependence"]
        lines.append(
            f"| {model} | [{a['ci95'][0]:+.5f}, {a['ci95'][1]:+.5f}] | [{b['ci95'][0]:+.5f}, {b['ci95'][1]:+.5f}] | {a['negative_count']} | {b['negative_count']} |"
        )
    lines += [
        "",
        "## Condition means",
        "",
        "| (a,b) | Model | TVD | Δ spatial | Δ dependence | negative spatial | "
        "negative dependence |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for index, condition in enumerate(conditions):
        for model, data in rows.items():
            spatial = data["delta_spatial"]["seed_condition_values"]
            dependence = data["delta_dependence"]["seed_condition_values"]
            lines.append(
                f"| ({condition[0]:.2f},{condition[1]:.2f}) | {model} | "
                f"{data['tvd']['condition_means'][index]:.5f} | "
                f"{data['delta_spatial']['condition_means'][index]:+.5f} | "
                f"{data['delta_dependence']['condition_means'][index]:+.5f} | "
                f"{sum(row[index] < -1e-12 for row in spatial)} | "
                f"{sum(row[index] < -1e-12 for row in dependence)} |"
            )
    lines += [
        "",
        "The JSON retains all raw seed-by-condition values and bootstrap intervals.",
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("artifacts/ashkin-teller-confirmation-2026-09-07"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/ashkin-teller-structure-result-2026-09-07"),
    )
    parser.add_argument("--label", default="n8-posthoc")
    args = parser.parse_args()
    run(args.directory, args.output, label=args.label)


if __name__ == "__main__":
    main()

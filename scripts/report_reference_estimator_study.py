#!/usr/bin/env python3
"""Validate and summarize the reference-estimator precision study."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

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
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "axes.xmargin": 0.02,
        "axes.ymargin": 0.02,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        **(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
            }
            if shutil.which("latex")
            else {"text.usetex": False, "mathtext.fontset": "cm"}
        ),
    }
)

EXPECTED_BUDGETS = (128, 512, 2048, 8192)
N_BOOT = 2000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _active_masks(ops: np.ndarray, encoder: str) -> dict[str, np.ndarray]:
    nonzero = np.any(ops != 0, axis=1)
    m = ops.shape[1] // 2
    even = (ops[:, :m].sum(axis=1) % 2 == 0) & (ops[:, m:].sum(axis=1) % 2 == 0)
    forbidden = (
        np.zeros(len(ops), dtype=bool) if encoder == "product" else nonzero & ~even
    )
    return {
        "active": nonzero & ~forbidden,
        "identity": ~nonzero,
        "forbidden": forbidden,
    }


def load_records(
    manifest_path: Path, result_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    manifest_sha = hashlib.sha256(raw).hexdigest()
    freeze = json.loads(manifest_path.with_name("freeze.json").read_text())
    if freeze["manifest_sha256"] != manifest_sha:
        raise ValueError("frozen manifest digest mismatch")
    json_paths = sorted(result_dir.rglob("*.json"))
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    source_hashes: dict[str, str] | None = None
    for meta_path in json_paths:
        if meta_path.name == "summary.json":
            continue
        meta = json.loads(meta_path.read_text())
        ident = meta.get("identity", {})
        case_id, budget = ident.get("case_id"), ident.get("sample_budget")
        if case_id is None or budget is None:
            raise ValueError(f"{meta_path}: missing identity")
        key = (str(case_id), int(budget))
        if key in seen:
            raise ValueError(f"duplicate case/budget output: {key}")
        seen.add(key)
        if ident.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"{meta_path}: manifest hash mismatch")
        if meta.get("pilot", False):
            raise ValueError(f"{meta_path}: pilot output is not reportable")
        current_sources = meta.get("source_hashes")
        if not isinstance(current_sources, dict):
            raise ValueError(f"{meta_path}: missing source hashes")
        if any(freeze["source_sha256"].get(p) != h for p, h in current_sources.items()):
            raise ValueError("result source does not match frozen source")
        if source_hashes is None:
            source_hashes = current_sources
        elif current_sources != source_hashes:
            raise ValueError(f"{meta_path}: source hashes differ across records")
        npz_path = meta_path.with_suffix(".npz")
        if not npz_path.exists():
            raise ValueError(f"{meta_path}: missing paired NPZ")
        if meta.get("npz_sha256") != sha256(npz_path):
            raise ValueError(f"{meta_path}: NPZ digest mismatch")
        arrays = dict(np.load(npz_path, allow_pickle=False))
        expected = {
            "sample_values",
            "sample_gradients",
            "exact_values",
            "exact_gradients",
        }
        if set(arrays) != expected:
            raise ValueError(f"{npz_path}: array schema mismatch")
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError(f"{npz_path}: non-finite array")
        reps = int(ident.get("repetitions", arrays["sample_values"].shape[0]))
        if reps != 32 or arrays["sample_values"].shape[0] != 32:
            raise ValueError(f"{meta_path}: expected R=32")
        case = meta.get("case", {})
        manifest_case = next(
            (c for c in manifest["cases"] if str(c["case_id"]) == str(case_id)), None
        )
        if manifest_case is None or case != manifest_case:
            raise ValueError(
                f"{meta_path}: case metadata differs from manifest fixture"
            )
        ops = np.asarray(case.get("ops"), dtype=int)
        if ops.ndim != 2 or arrays["sample_values"].shape[1] != len(ops):
            raise ValueError(f"{meta_path}: operator shape mismatch")
        if arrays["sample_gradients"].shape[1:] != arrays["exact_gradients"].shape:
            raise ValueError(f"{meta_path}: gradient shape mismatch")
        n_ops, n_params = len(ops), len(case["params"])
        for name, shape in {
            "sample_values": (32, n_ops),
            "sample_gradients": (32, n_ops, n_params),
            "exact_values": (n_ops,),
            "exact_gradients": (n_ops, n_params),
        }.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{meta_path}: {name} shape mismatch")
        records.append(
            {
                "meta": meta,
                "arrays": arrays,
                "case": case,
                "ops": ops,
                "masks": _active_masks(ops, str(case["encoder"])),
                "path": str(meta_path),
                "manifest_sha": manifest_sha,
                "source_hashes": current_sources,
            }
        )
    if len(records) != 60 * 4:
        raise ValueError(
            f"expected exactly 60 cases x 4 budgets, found {len(records)} outputs"
        )
    expected_keys = {
        (str(c["case_id"]), b) for c in manifest["cases"] for b in EXPECTED_BUDGETS
    }
    if len(manifest.get("cases", [])) != 60 or seen != expected_keys:
        raise ValueError("manifest/output case set is not exactly 60 cases x 4 budgets")
    if {int(r["meta"]["identity"]["sample_budget"]) for r in records} != set(
        EXPECTED_BUDGETS
    ):
        raise ValueError("unexpected sample budgets")
    return manifest, records


def metric_replicates(
    record: dict[str, Any], kind: str, mask_name: str = "active"
) -> np.ndarray:
    a = record["arrays"]
    mask = record["masks"][mask_name]
    if not mask.any():
        return np.full(a["sample_values"].shape[0], np.nan)
    if kind == "moment":
        errors = a["sample_values"][:, mask] - a["exact_values"][mask]
    else:
        errors = a["sample_gradients"][:, mask, :] - a["exact_gradients"][mask, :]
    return np.mean(errors**2, axis=tuple(range(1, errors.ndim)))


def metric(
    record: dict[str, Any], kind: str, mask_name: str = "active"
) -> float | None:
    values = metric_replicates(record, kind, mask_name)
    return None if np.isnan(values).all() else float(np.sqrt(np.mean(values)))


def bootstrap_case_mse(
    records: list[dict[str, Any]], kind: str, seed: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    case_mse = np.asarray([metric_replicates(r, kind) for r in records])
    draws = np.empty((N_BOOT, len(records)))
    for i in range(N_BOOT):
        for j in range(len(records)):
            draws[i, j] = np.mean(
                case_mse[j, rng.integers(0, case_mse.shape[1], case_mse.shape[1])]
            )
    aggregate = np.sqrt(np.mean(case_mse))
    boot = np.sqrt(np.mean(draws, axis=1))
    return (
        float(aggregate),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def summarize(
    manifest: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    cases = {str(c["case_id"]): c for c in manifest["cases"]}
    rows = []
    for r in records:
        ident = r["meta"]["identity"]
        c = cases[str(ident["case_id"])]
        arrays = r["arrays"]
        moment_errors = arrays["sample_values"] - arrays["exact_values"]
        gradient_errors = arrays["sample_gradients"] - arrays["exact_gradients"]
        active = r["masks"]["active"]
        mci = bootstrap_case_mse([r], "moment", 93000 + len(rows))
        gci = bootstrap_case_mse([r], "gradient", 94000 + len(rows))
        rows.append(
            {
                "case_id": ident["case_id"],
                "encoder": c["encoder"],
                "n_qubits": int(c["n_qubits"]),
                "budget": int(ident["sample_budget"]),
                "phase_regime": c["phase_regime"],
                "trained": c["phase_regime"] == "trained",
                "moment_rmse": metric(r, "moment"),
                "gradient_rmse": metric(r, "gradient"),
                "moment_ci": mci[1:],
                "gradient_ci": gci[1:],
                "moment_bias_rms": float(
                    np.sqrt(np.mean(moment_errors[:, active].mean(axis=0) ** 2))
                )
                if active.any()
                else None,
                "gradient_bias_rms": float(
                    np.sqrt(np.mean(gradient_errors[:, active, :].mean(axis=0) ** 2))
                )
                if active.any()
                else None,
                "identity_moment_rmse": metric(r, "moment", "identity"),
                "forbidden_moment_rmse": metric(r, "moment", "forbidden"),
                "wall_seconds_median": float(
                    np.median(r["meta"]["timing"]["wall_seconds"])
                ),
                "wall_seconds_min": float(np.min(r["meta"]["timing"]["wall_seconds"])),
                "wall_seconds_max": float(np.max(r["meta"]["timing"]["wall_seconds"])),
                "sampled_compile_seconds": float(
                    r["meta"]["timing"]["sampled_compile_seconds"]
                ),
            }
        )
    synthetic = [r for r in records if r["case"]["phase_regime"] != "trained"]
    plotted = []
    for enc in ["product", "cat", "bond"]:
        for n in sorted({int(r["case"]["n_qubits"]) for r in synthetic}):
            for b in EXPECTED_BUDGETS:
                group = [
                    r
                    for r in synthetic
                    if r["case"]["encoder"] == enc
                    and int(r["case"]["n_qubits"]) == n
                    and int(r["meta"]["identity"]["sample_budget"]) == b
                ]
                if len(group) != 6:
                    raise ValueError(
                        f"expected 6 synthetic cases for {enc}, N={n}, S={b}"
                    )
                m = bootstrap_case_mse(group, "moment", 91000 + n + b)
                g = bootstrap_case_mse(group, "gradient", 92000 + n + b)
                plotted.append(
                    {
                        "encoder": enc,
                        "n_qubits": n,
                        "budget": b,
                        "moment_rmse": m[0],
                        "moment_ci": m[1:],
                        "gradient_rmse": g[0],
                        "gradient_ci": g[1:],
                    }
                )
    return {
        "study_id": manifest["study_id"],
        "manifest_cases": len(manifest["cases"]),
        "budgets": list(EXPECTED_BUDGETS),
        "repetitions": 32,
        "n_bootstrap": N_BOOT,
        "rows": rows,
        "synthetic_summary": plotted,
    }


def plot_accuracy(summary: dict[str, Any], out: Path) -> None:
    rows = summary["synthetic_summary"]
    encoders = ["product", "cat", "bond"]
    fig, axes = plt.subplots(
        2, 3, figsize=(170 * MM, 115 * MM), layout="constrained", sharex=True
    )
    for col, enc in enumerate(encoders):
        ax_m, ax_g = axes[:, col]
        for n, color in zip(
            sorted({r["n_qubits"] for r in rows}), OKABE_ITO[:3], strict=True
        ):
            vals = [r for r in rows if r["encoder"] == enc and r["n_qubits"] == n]
            by_b = {b: [r for r in vals if r["budget"] == b] for b in EXPECTED_BUDGETS}
            # Stored intervals resample repetitions within each fixed case.
            for ax, key, label in (
                (ax_m, "moment_rmse", "moment RMSE"),
                (ax_g, "gradient_rmse", "gradient RMSE"),
            ):
                y, lo, hi = [], [], []
                for b in EXPECTED_BUDGETS:
                    row = by_b[b][0]
                    y.append(row[key])
                    lo.append(row[f"{key.removesuffix('_rmse')}_ci"][0])
                    hi.append(row[f"{key.removesuffix('_rmse')}_ci"][1])
                ax.plot(EXPECTED_BUDGETS, y, marker="o", color=color, label=f"$N={n}$")
                ax.fill_between(
                    EXPECTED_BUDGETS, lo, hi, color=color, alpha=0.14, linewidth=0
                )
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                ax.set_xticks(EXPECTED_BUDGETS, [str(b) for b in EXPECTED_BUDGETS])
                ax.set_ylabel(label)
                ax.grid(alpha=0.15, which="both")
        ax_m.set_title(
            {"product": "Product", "cat": "Symmetry-only", "bond": "Bond-informed"}[enc]
        )
        ax_g.set_xlabel("reference samples $S$")
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3)
    fig.savefig(out / "reference_estimator_accuracy.pdf")
    fig.savefig(out / "reference_estimator_accuracy.png", dpi=300)
    plt.close(fig)


def plot_cost(summary: dict[str, Any], out: Path) -> None:
    rows = [r for r in summary["rows"] if r["budget"] == 8192 and not r["trained"]]
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 60 * MM), layout="constrained")
    for enc, color in zip(["product", "cat", "bond"], OKABE_ITO[:3], strict=True):
        vals = [r for r in rows if r["encoder"] == enc]
        by_n = {
            n: [r for r in vals if r["n_qubits"] == n]
            for n in sorted({r["n_qubits"] for r in vals})
        }
        ns = sorted(by_n)
        for ax, key, label in (
            (axes[0], "wall_seconds_median", "median warmed estimator time [s]"),
            (axes[1], "sampled_compile_seconds", "Lowering + compilation (s)"),
        ):
            med = [np.median([r[key] for r in by_n[n]]) for n in ns]
            lo = [np.min([r[key] for r in by_n[n]]) for n in ns]
            hi = [np.max([r[key] for r in by_n[n]]) for n in ns]
            ax.plot(
                ns,
                med,
                marker="o",
                color=color,
                label={
                    "product": "Product",
                    "cat": "Symmetry-only",
                    "bond": "Bond-informed",
                }[enc],
            )
            ax.fill_between(ns, lo, hi, color=color, alpha=0.12, linewidth=0)
            ax.set_xlabel("qubits $N$")
            ax.set_xticks(ns)
            ax.set_ylabel(label)
            ax.grid(alpha=0.15)
    axes[0].legend()
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "reference_estimator_cost.pdf")
    fig.savefig(out / "reference_estimator_cost.png", dpi=300)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, records = load_records(args.manifest, args.results)
    summary = summarize(manifest, records)
    summary["manifest_sha256"] = sha256(args.manifest)
    summary["result_metadata"] = [r["path"] for r in records]
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    plot_accuracy(summary, args.output)
    plot_cost(summary, args.output)
    report = (
        "# Reference-estimator precision study\n\n"
        "This report evaluates reference-sampling error against exact moments and "
        "all phase gradients. Curves aggregate six synthetic cases per encoder "
        "and width with equal case weight; bands are replicate-index bootstrap "
        "intervals (2,000 draws). Identity and forbidden masks are retained as "
        "controls and excluded from the active-operator curves. The results "
        "support a precision and cost characterization, not a claim of "
        "unbiasedness, quantum advantage, or learned-distribution generalization. "
        "Timing includes the complete reference-sampling estimator and is "
        "descriptive; compile times are reported separately.\n"
    )
    (args.output / "report.md").write_text(report)


if __name__ == "__main__":
    main()

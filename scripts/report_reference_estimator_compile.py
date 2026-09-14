#!/usr/bin/env python3
# ruff: noqa: E501
"""Validate and summarize the reference-estimator compile probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

EXPECTED_NS = (8, 12, 16)
EXPECTED_REPS = (0, 1, 2)
EXPECTED_HOSTS = ("192.168.1.58", "192.168.1.67")
EXPECTED_P = {8: 36, 12: 78, 16: 136}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median(values):
    values = sorted(values)
    return (
        values[len(values) // 2]
        if len(values) % 2
        else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
    )


def _range(values):
    return [min(values), max(values)]


def _operation_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if " = " in line)


def _observed_shapes(text: str) -> list[str]:
    return sorted(set(re.findall(r"\[[0-9]+(?:,[0-9]+)+\]", text)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/reference-estimator-study-2026-09-07"),
    )
    args = parser.parse_args()
    root = args.root
    probe_root = root / "compile-probe"
    freeze = json.loads((root / "freeze.json").read_text())
    expected_manifest = freeze["manifest_sha256"]
    frozen_sources = freeze["source_sha256"]
    hosts = sorted(path.name for path in probe_root.iterdir() if path.is_dir())
    if tuple(hosts) != EXPECTED_HOSTS:
        raise RuntimeError(f"expected hosts {EXPECTED_HOSTS}, found {hosts}")
    expected_source_keys = {"scripts/run_reference_estimator_study.py"} | {
        str(path) for path in Path("src/iqp_cond").rglob("*.py")
    }
    expected_paths = {
        probe_root / host / f"n{n}-r{rep}" / "probe.json"
        for host in EXPECTED_HOSTS for n in EXPECTED_NS for rep in EXPECTED_REPS
    }
    if set(probe_root.rglob("probe.json")) != expected_paths:
        raise RuntimeError("unexpected or missing probe paths")
    rows = []
    for host in hosts:
        for n in EXPECTED_NS:
            for repetition in EXPECTED_REPS:
                directory = probe_root / host / f"n{n}-r{repetition}"
                probe_path = directory / "probe.json"
                before_path = directory / "sampled_before.hlo.txt"
                after_path = directory / "sampled_after.hlo.txt"
                if not all(
                    path.exists() for path in (probe_path, before_path, after_path)
                ):
                    raise RuntimeError(f"incomplete probe: {directory}")
                probe = json.loads(probe_path.read_text())
                expected_case = f"n{n}_product_c1_p0"
                if (
                    probe["manifest_sha256"] != expected_manifest
                    or not probe["persistent_compilation_cache_disabled"]
                    or probe.get("case_id") != expected_case
                    or probe.get("n_qubits") != n
                    or probe.get("repetition") != repetition
                    or probe.get("sample_budget") != 8192
                    or probe.get("hostname") != {
                        "192.168.1.58": "strix-halo", "192.168.1.67": "strix-halo2"
                    }[host]
                ):
                    raise RuntimeError(f"invalid manifest/cache gate: {directory}")
                if set(probe["source_hashes"]) != expected_source_keys:
                    raise RuntimeError(f"source key mismatch: {directory}")
                for name, digest in probe["source_hashes"].items():
                    if frozen_sources.get(name) != digest:
                        raise RuntimeError(
                            f"source hash mismatch: {name} ({directory})"
                        )
                artifacts = probe["artifacts"]
                if artifacts["before_hlo_sha256"] != _sha256(before_path) or artifacts[
                    "after_hlo_sha256"
                ] != _sha256(after_path):
                    raise RuntimeError(f"HLO hash mismatch: {directory}")
                before, after = before_path.read_text(), after_path.read_text()
                timings = probe["timing_seconds"]
                timing_values = [
                    timings["lowering"],
                    timings["compilation"],
                    timings["first_execution"],
                    *timings["warm_executions"],
                ]
                if not all(
                    isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                    for value in timing_values
                ):
                    raise RuntimeError(f"invalid timing: {directory}")
                cost = probe.get("cost_analysis") or {}
                shapes = _observed_shapes(after)
                if (
                    "[16,16,1,8192]" not in shapes
                    or f"[256,{EXPECTED_P[n]}]" not in shapes
                ):
                    raise RuntimeError(f"required HLO shapes missing: {directory}")
                dot_lines = [line.strip() for line in after.splitlines() if " dot(" in line]
                if not any(f"f64[256,{EXPECTED_P[n]}]" in line for line in dot_lines):
                    raise RuntimeError(f"expected Jacobian dot missing: {directory}")
                rows.append(
                    {
                        "host": host,
                        "n": n,
                        "repetition": repetition,
                        "lowering": timings["lowering"],
                        "compilation": timings["compilation"],
                        "sum": timings["lowering"] + timings["compilation"],
                        "warm_median": _median(timings["warm_executions"]),
                        "flops": cost.get("flops"),
                        "bytes_accessed": cost.get("bytes accessed"),
                        "before_ops": _operation_count(before),
                        "after_ops": _operation_count(after),
                        "observed_shapes": shapes,
                        "dot_instructions": dot_lines,
                    }
                )
    report = {
        "coverage": {
            "hosts": hosts,
            "n_qubits": list(EXPECTED_NS),
            "repetitions": list(EXPECTED_REPS),
            "count": len(rows),
        },
        "manifest_sha256": expected_manifest,
        "probe_source_sha256": _sha256(
            Path(__file__).with_name("probe_reference_estimator_compile.py")
        ),
        "report_source_sha256": _sha256(Path(__file__)),
        "rows": rows,
        "summary": {},
    }
    for host in hosts:
        report["summary"][host] = {}
        for n in EXPECTED_NS:
            selected = [row for row in rows if row["host"] == host and row["n"] == n]
            report["summary"][host][str(n)] = {
                metric: {
                    "median": _median([row[metric] for row in selected]),
                    "range": _range([row[metric] for row in selected]),
                }
                for metric in (
                    "lowering",
                    "compilation",
                    "sum",
                    "warm_median",
                    "flops",
                    "bytes_accessed",
                )
            }
            report["summary"][host][str(n)]["operation_counts"] = {
                "before": sorted({row["before_ops"] for row in selected}),
                "after": sorted({row["after_ops"] for row in selected}),
            }
            report["summary"][host][str(n)]["observed_shapes"] = sorted(
                {shape for row in selected for shape in row["observed_shapes"]}
            )
    out_json = probe_root / "summary.json"
    out_md = probe_root / "summary.md"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Reference-estimator compile probe",
        "",
        f"Validated {len(rows)} probes across {len(hosts)} hosts; manifest `{expected_manifest}`.",
        "",
        "The cost-analysis values are compiler estimates, not measured instruction counts. This report records observations and does not assign causal explanations to compiler behavior.",
        "",
        "| Host | n | Lowering median (s) | Compilation median (s) | Lowering+compile median (s) | Warm median (s) | FLOPs estimate | Bytes estimate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for host in hosts:
        for n in EXPECTED_NS:
            item = report["summary"][host][str(n)]
            lines.append(
                f"| {host} | {n} | {item['lowering']['median']:.4g} | {item['compilation']['median']:.4g} | {item['sum']['median']:.4g} | {item['warm_median']['median']:.4g} | {item['flops']['median']:.4g} | {item['bytes_accessed']['median']:.4g} |"
            )
    lines += [
        "",
        "Operation counts and observed intermediate shapes are retained in `summary.json`; in particular, the report preserves any `[16,16,1,8192]` full-Jacobian and `[256,P]` dot shapes present in the compiled HLO.",
        "",
        f"Probe source SHA256: `{report['probe_source_sha256']}`",
        f"Report source SHA256: `{report['report_source_sha256']}`",
        "",
    ]
    out_md.write_text("\n".join(lines))
    print(f"validated {len(rows)} probes; wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()

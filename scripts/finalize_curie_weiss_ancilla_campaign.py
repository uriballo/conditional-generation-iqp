#!/usr/bin/env python3
"""Finalize the fixed-m=5 Curie--Weiss report and claim registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json, sha256_file, write_immutable_json
from iqp_cond.curie_weiss_ancilla_campaign import CurieWeissAncillaCampaignStore
from iqp_cond.curie_weiss_ancilla_protocol import CurieWeissAncillaManifest
from iqp_cond.curie_weiss_ancilla_report import build_report, write_report


def _write_text_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"immutable generated evidence differs: {path}")
        return
    path.write_text(content)


def _registry_markdown(registry: dict) -> str:
    lines = [
        "# Ancilla Curie--Weiss claim--evidence registry",
        "",
        f"Generated from all {registry['completed_runs']} immutable fits. Positive",
        "means that the named conditioned or five-ancilla model has lower TVD.",
        "No performance gate",
        "is applied; interval signs determine supported, mixed, or unsupported.",
        "",
        f"- Manifest: `{registry['manifest_digest']}`",
        f"- Report SHA-256: `{registry['report_sha256']}`",
        "",
        "| Claim key | Status | Estimate | 95% paired interval |",
        "|---|---:|---:|---:|",
    ]
    for record in registry["records"]:
        evidence = record["evidence"]
        lines.append(
            f"| `{record['comparison']}` | {record['status']} | "
            f"{evidence['estimate']:+.3f} | "
            f"[{evidence['lower']:+.3f}, {evidence['upper']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Scope restrictions",
            "",
            *[f"- No claim of {claim}." for claim in registry["prohibited_claims"]],
            "",
        ]
    )
    return "\n".join(lines)


def _tex_effect(interval: dict) -> str:
    return (
        f"{interval['estimate']:+.3f} "
        f"[{interval['lower']:+.3f}, {interval['upper']:+.3f}]"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/curie_weiss_m5_foundations_v1.json"),
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("results/curie-weiss-m5-foundations-v1"),
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=Path("paper/generated/curie_weiss_ancilla_claims.tex"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("docs/curie-weiss-ancilla-claim-evidence-registry.md"),
    )
    args = parser.parse_args(argv)
    manifest = CurieWeissAncillaManifest.from_json(args.manifest)
    store = CurieWeissAncillaCampaignStore(args.campaign / "store", manifest)
    report = build_report(store)
    report_path = write_report(
        args.campaign / "report" / "campaign-report.json", report
    )
    report_sha = sha256_file(report_path)
    registry = {
        "schema_version": 1,
        "completed_runs": report["completed_runs"],
        "manifest_digest": manifest.digest,
        "report_path": str(report_path),
        "report_sha256": report_sha,
        "claim_policy": "paired interval sign; no empirical publication gate",
        "records": [
            {
                "comparison": row["comparison"],
                "status": row["interval"]["status"],
                "evidence": row["interval"],
            }
            for row in report["comparisons"]
        ],
        "prohibited_claims": report["prohibited_claims"],
    }
    registry_path = args.campaign / "report" / "claim-registry.json"
    if registry_path.exists():
        if read_immutable_json(registry_path) != registry:
            raise RuntimeError("immutable ancilla claim registry differs")
    else:
        write_immutable_json(registry_path, registry)
    _write_text_immutable(args.registry, _registry_markdown(registry))
    intervals = {
        record["comparison"]: record["interval"] for record in report["comparisons"]
    }
    macros = {
        "CWAncillaProfileControlTraining": "conditional_m5_profile_tvd_training",
        "CWAncillaProfileControlHeldOut": "conditional_m5_profile_tvd_held_out",
        "CWAncillaRawControlTraining": "conditional_m5_bitstring_tvd_training",
        "CWAncillaRawControlHeldOut": "conditional_m5_bitstring_tvd_held_out",
        "CWAncillaProfileMixtureHeldOut": "m5_vs_mixture_profile_tvd_held_out",
        "CWAncillaRawMixtureHeldOut": "m5_vs_mixture_bitstring_tvd_held_out",
        "CWAncillaProfileOracleHeldOut": "m5_vs_profile_oracle_held_out",
        "CWAncillaRawOracleHeldOut": "m5_vs_raw_oracle_held_out",
        "CWAncillaProfileCritical": "critical_no_condition_profile_tvd",
        "CWAncillaRawCritical": "critical_no_condition_bitstring_tvd",
        "CWAncillaProfileCold": "cold_no_condition_profile_tvd",
        "CWAncillaRawCold": "cold_no_condition_bitstring_tvd",
    }
    tex_lines = [
        "% Generated from all 80 immutable fixed-m=5 Curie--Weiss fits.",
        "\\newif\\ifcurieweissancillaresults",
        "\\curieweissancillaresultstrue",
        f"\\newcommand{{\\CWAncillaReportSHA}}{{{report_sha}}}",
        "\\newcommand{\\CWAncillaAtomicFits}{80}",
    ]
    for macro, comparison in macros.items():
        tex_lines.append(
            f"\\newcommand{{\\{macro}}}{{{_tex_effect(intervals[comparison])}}}"
        )
    tex_lines.append("")
    tex = "\n".join(tex_lines)
    _write_text_immutable(args.tex, tex)
    print(
        json.dumps(
            {
                "manifest_digest": manifest.digest,
                "report_sha256": report_sha,
                "comparisons": len(report["comparisons"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

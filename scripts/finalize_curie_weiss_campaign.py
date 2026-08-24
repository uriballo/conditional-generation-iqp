"""Finalize Curie--Weiss paired statistics and evidence-bound paper macros."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqp_cond.artifacts import sha256_file, write_immutable_json
from iqp_cond.curie_weiss_campaign import CurieWeissCampaignStore
from iqp_cond.curie_weiss_protocol import CurieWeissCampaignManifest
from iqp_cond.curie_weiss_report import build_curie_weiss_report, write_report


def _macro_name(n: int, comparison: str, field: str) -> str:
    token = "".join(part.title() for part in comparison.split("_"))
    return f"CW{token}N{n}{field.title()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/curie_weiss_foundations_v1.json")
    )
    parser.add_argument(
        "--campaign", type=Path, default=Path("results/curie-weiss-foundations-v1")
    )
    parser.add_argument(
        "--tex", type=Path, default=Path("paper/generated/curie_weiss_claims.tex")
    )
    args = parser.parse_args(argv)
    manifest = CurieWeissCampaignManifest.from_json(args.manifest)
    store = CurieWeissCampaignStore(args.campaign / "store", manifest)
    report = build_curie_weiss_report(store)
    report_path = write_report(
        args.campaign / "report" / "campaign-report.json", report
    )
    report_sha = sha256_file(report_path)
    registry = {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "report_path": str(report_path),
        "report_sha256": report_sha,
        "claim_policy": (
            "supported/mixed/unsupported from frozen paired intervals; no pass gate"
        ),
        "records": [
            {
                "n_qubits": row["n_qubits"],
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
        from iqp_cond.artifacts import read_immutable_json

        if read_immutable_json(registry_path) != registry:
            raise RuntimeError("immutable Curie--Weiss registry differs")
    else:
        write_immutable_json(registry_path, registry)
    lines = [
        "% Generated from all 270 immutable Curie--Weiss fits. Do not edit.",
        "\\newif\\ifcurieweissresults",
        "\\curieweissresultstrue",
        f"\\newcommand{{\\CWReportSHA}}{{{report_sha}}}",
    ]
    for row in report["comparisons"]:
        for field in ("estimate", "lower", "upper"):
            name = _macro_name(row["n_qubits"], row["comparison"], field)
            lines.append(f"\\newcommand{{\\{name}}}{{{row['interval'][field]:.3f}}}")
    content = "\n".join(lines) + "\n"
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    if args.tex.exists() and args.tex.read_text() != content:
        raise RuntimeError("immutable Curie--Weiss TeX evidence differs")
    if not args.tex.exists():
        args.tex.write_text(content)
    print(
        json.dumps(
            {"report_sha256": report_sha, "comparisons": len(report["comparisons"])},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

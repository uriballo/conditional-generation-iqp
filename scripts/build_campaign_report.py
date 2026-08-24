#!/usr/bin/env python3
"""Regenerate the deterministic Quantum campaign acceptance report."""

from __future__ import annotations

import argparse
from pathlib import Path

from iqp_cond.campaign_report import build_campaign_report, write_campaign_report


def _parser(repository: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository / "configs" / "quantum_campaign_v1.json",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=repository / "results" / "quantum-campaign-v1" / "campaign",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "results" / "quantum-campaign-v1" / "report",
    )
    parser.add_argument(
        "--claim-registry",
        type=Path,
        default=repository / "docs" / "claim-evidence-registry.md",
    )
    return parser


def main() -> int:
    """Build JSON and Markdown artifacts and print their stable digest."""
    repository = Path(__file__).resolve().parents[1]
    arguments = _parser(repository).parse_args()
    report = build_campaign_report(
        arguments.repository,
        arguments.manifest,
        arguments.campaign_root,
        claim_registry_path=arguments.claim_registry,
    )
    json_path, markdown_path = write_campaign_report(report, arguments.output)
    print(f"campaign_report_digest={report.digest}")
    print(f"json={json_path}")
    print(f"markdown={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

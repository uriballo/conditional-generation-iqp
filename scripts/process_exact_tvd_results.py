#!/usr/bin/env python3
"""Process exact-TVD run envelopes into JSON and CSV summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iqp_cond.exact_tvd_campaign import load_manifest  # noqa: E402
from iqp_cond.exact_tvd_results import (  # noqa: E402
    aggregate,
    load_records,
    write_summary_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/exact_tvd_ancilla_v1.json",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="process a manifest-valid subset (default requires all campaign runs)",
    )
    args = parser.parse_args()
    config = load_manifest(args.manifest)
    records = load_records(
        args.campaign / "evaluations",
        config,
        require_complete=not args.allow_partial,
    )
    summary = aggregate(records, config)
    write_summary_tables(
        summary,
        args.json or args.campaign / "summary.json",
        args.csv or args.campaign / "summary.csv",
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "summary": str(args.json or args.campaign / "summary.json"),
            }
        )
    )


if __name__ == "__main__":
    main()

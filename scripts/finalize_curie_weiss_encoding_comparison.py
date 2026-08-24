#!/usr/bin/env python3
"""Write the immutable encoding-comparison report and Markdown summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqp_cond.curie_weiss_ancilla_campaign import CurieWeissAncillaCampaignStore
from iqp_cond.curie_weiss_ancilla_protocol import CurieWeissAncillaManifest
from iqp_cond.curie_weiss_encoding_report import (
    build_encoding_report,
    render_encoding_summary,
    write_encoding_report,
)


def _write_text_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"immutable generated evidence differs: {path}")
    else:
        path.write_text(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/curie_weiss_encoding_comparison_v1.json"),
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("results/curie-weiss-encoding-comparison-v1"),
    )
    args = parser.parse_args(argv)
    manifest = CurieWeissAncillaManifest.from_json(args.manifest)
    store = CurieWeissAncillaCampaignStore(args.campaign / "store", manifest)
    report = build_encoding_report(store)
    path = write_encoding_report(args.campaign / "report" / "report.json", report)
    _write_text_immutable(
        args.campaign / "summary" / "metrics.md",
        render_encoding_summary(report),
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "manifest_digest": manifest.digest,
                "report": str(path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

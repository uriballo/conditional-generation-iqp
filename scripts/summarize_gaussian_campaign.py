"""Build the immutable complete report for the Gaussian campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json, write_immutable_json
from iqp_cond.gaussian_campaign import GaussianCampaignStore
from iqp_cond.gaussian_protocol import GaussianCampaignManifest
from iqp_cond.gaussian_report import build_gaussian_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/gaussian_foundations_v1.json")
    )
    parser.add_argument(
        "--campaign", type=Path, default=Path("results/gaussian-foundations-v1")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gaussian-foundations-v1/report/complete-report.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = Path.cwd().resolve()
    manifest = GaussianCampaignManifest.from_json(args.manifest)
    store = GaussianCampaignStore(args.campaign / "store", manifest)
    payload = build_gaussian_report(repository, manifest, store)
    if args.output.exists():
        if read_immutable_json(args.output) != payload:
            raise RuntimeError("complete Gaussian report differs")
    else:
        write_immutable_json(args.output, payload)
    compact = {
        "manifest_digest": manifest.digest,
        "completed_atomic_models": payload["completed_atomic_models"],
        "factorial_cells": len(payload["factorial_cells"]),
        "condition_count_cells": len(payload["condition_count_cells"]),
        "output": str(args.output),
    }
    print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare immutable exact-draw datasets for ``arxiv_foundations_v1``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqp_cond.artifacts import sha256_file, write_immutable_json
from iqp_cond.foundations_campaign import (
    FoundationsCampaignStore,
    FoundationsGate,
    GateDecision,
)
from iqp_cond.foundations_data import prepare_foundations_datasets
from iqp_cond.foundations_protocol import FoundationsManifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/arxiv_foundations_v1.json"),
    )
    parser.add_argument(
        "--source-targets",
        type=Path,
        default=Path("results/quantum-campaign-v1/targets"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/arxiv-foundations-v1"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = FoundationsManifest.from_json(args.manifest)
    store = FoundationsCampaignStore(args.output / "store", manifest)
    archives = prepare_foundations_datasets(
        manifest,
        args.source_targets,
        args.output / "datasets",
    )
    payload = {
        "schema_version": 1,
        "campaign_version": manifest.campaign_version,
        "manifest_digest": manifest.digest,
        "source_campaign": manifest.source_campaign.to_dict(),
        "held_out_draws_generated": 0,
        "archives": [
            {
                "coupling_seed": archive.coupling_seed,
                "path": archive.path,
                "sha256": archive.sha256,
                "target_sha256": archive.target_sha256,
            }
            for archive in archives
        ],
    }
    evidence = args.output / "evidence" / "data-ready.json"
    write_immutable_json(evidence, payload)
    store.finalize_gate(
        FoundationsGate.DATA_READY,
        GateDecision.PASSED,
        evidence_digest=sha256_file(evidence),
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "manifest_digest": manifest.digest,
                "active_instances": len(archives),
                "held_out_draws_generated": 0,
                "evidence": str(evidence),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

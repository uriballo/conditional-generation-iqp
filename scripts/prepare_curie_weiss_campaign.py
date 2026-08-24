"""Prepare immutable Curie--Weiss training and validation datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json, write_immutable_json
from iqp_cond.curie_weiss_data import prepare_all_datasets
from iqp_cond.curie_weiss_protocol import CurieWeissCampaignManifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/curie_weiss_foundations_v1.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/curie-weiss-foundations-v1/datasets"),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("results/curie-weiss-foundations-v1/evidence/data-ready.json"),
    )
    args = parser.parse_args(argv)
    manifest = CurieWeissCampaignManifest.from_json(args.manifest)
    prepared = prepare_all_datasets(manifest, args.output)
    payload = {
        "schema_version": 1,
        "manifest_digest": manifest.digest,
        "archive_count": len(prepared),
        "held_out_draws": 0,
        "archives": [
            {"path": str(path), "sha256": digest} for path, digest in prepared
        ],
    }
    if args.evidence.exists():
        if read_immutable_json(args.evidence) != payload:
            raise RuntimeError("Curie--Weiss data evidence differs")
    else:
        write_immutable_json(args.evidence, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

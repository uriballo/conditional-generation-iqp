"""Run and archive the binding foundations estimator calibration."""

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
from iqp_cond.foundations_estimator import run_estimator_audit
from iqp_cond.foundations_protocol import FoundationsManifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/arxiv_foundations_v1.json"),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("results/quantum-campaign-v1/targets/coupling_606.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/arxiv-foundations-v1"),
    )
    parser.add_argument("--repetitions", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = FoundationsManifest.from_json(args.manifest)
    store = FoundationsCampaignStore(args.output / "store", manifest)
    audit = run_estimator_audit(
        manifest,
        args.target,
        repetitions=args.repetitions,
    )
    evidence = args.output / "evidence" / "estimator-calibration.json"
    write_immutable_json(evidence, audit.to_dict())
    store.finalize_gate(
        FoundationsGate.ESTIMATOR_CALIBRATION,
        GateDecision.PASSED,
        evidence_digest=sha256_file(evidence),
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "selected_budget": audit.selection.operator_budget,
                "passing_budgets": audit.selection.passing_budgets,
                "manifest_digest": manifest.digest,
                "evidence": str(evidence),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

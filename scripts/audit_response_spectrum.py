"""Archive a synthetic spectrum fixture or a production readiness preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqp_cond.spectrum_audit import (
    archive_spectrum_report,
    production_spectrum_preflight,
    synthetic_spectrum_fixture,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit finite response spectra. Production mode never substitutes "
            "synthetic responses for missing frozen checkpoints."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/quantum_campaign_v1.json"),
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path("results/quantum-campaign-v1/campaign"),
    )
    parser.add_argument("--fixed-bundle", type=Path)
    parser.add_argument("--phase-bundle", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/quantum-campaign-v1/evidence/response-spectrum.json",
        ),
    )
    parser.add_argument(
        "--synthetic-fixture",
        action="store_true",
        help="validate the audit on a known function, not campaign evidence",
    )
    parser.add_argument("--grid-size", type=int, default=193)
    parser.add_argument("--injected-amplitude", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Create an immutable audit and return nonzero for blocked production."""
    args = _parser().parse_args(argv)
    if args.synthetic_fixture:
        report = synthetic_spectrum_fixture(
            grid_size=args.grid_size,
            injected_amplitude=args.injected_amplitude,
        )
    else:
        report = production_spectrum_preflight(
            manifest_path=args.manifest,
            repository_root=args.repository_root,
            campaign_root=args.campaign_root,
            fixed_bundle_path=args.fixed_bundle,
            phase_bundle_path=args.phase_bundle,
            grid_size=args.grid_size,
        )
    artifact_digest = archive_spectrum_report(args.output, report)
    summary = {
        "status": report["status"],
        "output": str(args.output),
        "artifact_digest": artifact_digest,
        "configuration_digest": report["configuration_digest"],
        "full_training_runs_used": 0,
    }
    if report["status"] == "blocked":
        summary["blockers"] = report["blockers"]
    print(json.dumps(summary, sort_keys=True))
    return 2 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())

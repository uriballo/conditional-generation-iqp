#!/usr/bin/env python3
"""Run the confirmatory exact-versus-Monte-Carlo estimator audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from iqp_cond.confirmatory_estimator_audit import (
    DEFAULT_MANIFEST,
    load_manifest,
    run_audit,
    write_immutable_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run all encoder/topology families at the first two sample budgets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_audit(quick=args.quick, manifest=load_manifest(args.manifest))
    write_immutable_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()

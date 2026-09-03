#!/usr/bin/env python3
"""Select the global optimizer from completed pilot JSONL rows."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iqp_cond.artifacts import read_immutable_json  # noqa: E402
from iqp_cond.exact_tvd_campaign import load_manifest  # noqa: E402
from iqp_cond.exact_tvd_pilot import (  # noqa: E402
    select_optimizer,
    write_selection,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--preflight", type=Path, action="append", required=True)
    p.add_argument("--selection-output", type=Path, required=True)
    a = p.parse_args()
    manifest = load_manifest(a.manifest)
    if a.rows.is_dir():
        rows = [read_immutable_json(path) for path in sorted(a.rows.glob("*.json"))]
    else:
        rows = [json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()]
    result = select_optimizer(rows, manifest)
    calibration = read_immutable_json(a.calibration)
    preflights = [read_immutable_json(path) for path in a.preflight]
    write_selection(a.selection_output, manifest, calibration, result, preflights)
    print(result["optimizer"])


if __name__ == "__main__":
    main()

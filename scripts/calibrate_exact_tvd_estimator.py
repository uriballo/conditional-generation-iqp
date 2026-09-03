#!/usr/bin/env python3
"""Select the first passing exact-TVD estimator calibration candidate."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iqp_cond.artifacts import read_immutable_json, write_immutable_json  # noqa: E402
from iqp_cond.exact_tvd_campaign import (  # noqa: E402
    load_manifest,
    manifest_digest,
)
from iqp_cond.exact_tvd_pilot import select_n_samples  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = load_manifest(a.manifest)
    if a.rows.is_dir():
        rows = [read_immutable_json(path) for path in sorted(a.rows.glob("*.json"))]
    else:
        rows = [json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()]
    selected = select_n_samples(rows, manifest)
    write_immutable_json(
        a.output,
        {
            "schema_version": 1,
            "stage": "estimator_calibration",
            "configuration_digest": manifest_digest(manifest),
            "selected_n_samples": selected,
            "cell_count": len(rows),
            "rows_source": str(a.rows),
        },
    )
    print(selected)


if __name__ == "__main__":
    main()

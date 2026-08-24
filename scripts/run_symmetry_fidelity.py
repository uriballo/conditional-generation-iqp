"""Evaluate the frozen symmetry-aligned fidelity protocol."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from iqp_cond.symmetry_fidelity import (
    evaluate_endpoint_seed,
    evaluate_heldout_seed,
    save_study_record,
)


def _parse_seeds(value: str) -> list[int]:
    if "-" in value:
        start, end = (int(part) for part in value.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--endpoint-runs", type=Path, default=Path("results/runs"))
    parser.add_argument(
        "--interpolation-runs",
        type=Path,
        default=Path("results/interpolation/runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/symmetry/runs"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    for seed in _parse_seeds(args.seeds):
        started = time.monotonic()
        print(f"evaluate symmetry fidelity seed={seed}", flush=True)
        records = evaluate_endpoint_seed(seed, args.endpoint_runs)
        records.extend(
            evaluate_heldout_seed(
                seed,
                args.interpolation_runs,
                args.endpoint_runs,
            ),
        )
        for record in records:
            save_study_record(
                record,
                args.output,
                skip_existing=args.skip_existing,
            )
        print(
            f"done symmetry fidelity seed={seed} "
            f"seconds={time.monotonic() - started:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

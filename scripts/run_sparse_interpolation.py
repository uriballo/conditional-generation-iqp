"""Run sparse-grid held-out interpolation with ancilla-assisted models."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from iqp_cond.sparse_interpolation import (
    run_sparse_ancilla,
    save_sparse_result,
)
from iqp_cond.training import TrainingConfig


def _parse_seeds(value: str) -> list[int]:
    if "-" in value:
        start, end = (int(part) for part in value.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def main() -> None:
    """Run selected sparse grids, ancilla counts, and seeds."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        type=int,
        default=3,
        choices=[2, 3],
        help="number of training peaks (2 or 3)",
    )
    parser.add_argument("--ancillas", default="2,3,5,7")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument(
        "--width-scale",
        type=float,
        default=1.0,
        help="target peak width scale (smaller spreads peaks apart)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sparse-interpolation/runs"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    ancilla_counts = [int(x) for x in args.ancillas.split(",")]
    config = TrainingConfig(steps=args.steps)

    for n_ancillas in ancilla_counts:
        for seed in _parse_seeds(args.seeds):
            model_name = (
                f"sparse{args.grid}_ancilla_{n_ancillas}"
                "_broadcast_all_complete"
            )
            if args.width_scale != 1.0:
                model_name += f"_w{args.width_scale:g}"
            destination = args.output / f"{model_name}_seed{seed}.json"
            if args.skip_existing and destination.exists():
                print(f"skip {model_name} seed={seed}")
                continue

            started = time.monotonic()
            print(f"run {model_name} seed={seed}", flush=True)
            result, metrics = run_sparse_ancilla(
                n_peaks=args.grid,
                n_ancillas=n_ancillas,
                seed=seed,
                config=config,
                width_scale=args.width_scale,
            )
            save_sparse_result(result, metrics, config, args.output)
            elapsed = time.monotonic() - started
            print(
                f"done {model_name} seed={seed} seconds={elapsed:.1f} "
                f"heldout_shell={metrics['heldout_shell_tvd']:.6f} "
                f"heldout_bit={metrics['heldout_bitstring_tvd']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

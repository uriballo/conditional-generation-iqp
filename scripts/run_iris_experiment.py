"""Run the Iris three-class ancilla-only conditional IQP experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from iqp_cond.iris import run_iris_experiment
from iqp_cond.training import TrainingConfig


def _parse_seeds(value: str) -> list[int]:
    if "-" in value:
        start, end = (int(part) for part in value.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def _parse_ancillas(value: str) -> list[int]:
    return [int(part) for part in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0-2")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--bits-per-feature", type=int, choices=[1, 2], default=1)
    parser.add_argument("--ancillas", default="2")
    parser.add_argument("--output", type=Path, default=Path("results/iris"))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(steps=args.steps)
    for n_ancillas in _parse_ancillas(args.ancillas):
        for seed in _parse_seeds(args.seeds):
            stem = (
                f"iris_ancilla_only_{args.bits_per_feature}bit_m{n_ancillas}"
                f"_complete_seed{seed}"
            )
            json_path = args.output / f"{stem}.json"
            if args.skip_existing and json_path.exists():
                print(f"skip {stem}")
                continue

            started = time.monotonic()
            print(f"run {stem}", flush=True)
            result, metrics = run_iris_experiment(
                seed,
                config,
                bits_per_feature=args.bits_per_feature,
                n_ancillas=n_ancillas,
            )
            json_path.write_text(json.dumps(metrics, indent=2) + "\n")
            np.savez_compressed(
                args.output / f"{stem}.npz",
                best_params=result.best_params,
                final_params=result.final_params,
                generators=result.generators,
                condition_angles=result.condition_angles,
                train_steps=result.train_steps,
                train_losses=result.train_losses,
                validation_steps=result.validation_steps,
                validation_losses=result.validation_losses,
            )
            elapsed = time.monotonic() - started
            tvds = [item["bitstring_tvd"] for item in metrics["classes"]]
            print(
                f"done {stem} sec={elapsed:.1f} "
                f"median_class_tvd={np.median(tvds):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

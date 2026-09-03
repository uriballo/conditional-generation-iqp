#!/usr/bin/env python3
"""Execute resumable estimator-calibration and optimizer-pilot cells."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iqp_cond.artifacts import write_immutable_json  # noqa: E402
from iqp_cond.exact_tvd_campaign import (  # noqa: E402
    RunIdentity,
    load_manifest,
    manifest_digest,
)
from iqp_cond.exact_tvd_runner import (  # noqa: E402
    calibration_cell,
    evaluate_run,
    train_run,
)


def _cells() -> list[tuple[str, int, int, int]]:
    return [
        (dataset, n_visible, ancillas, seed)
        for dataset, n_visible in (("gaussian", 8), ("curie_weiss", 8), ("mfc", 3))
        for ancillas in (0, 10)
        for seed in (0, 1, 2)
    ]


def _optimizer_candidates(config: dict) -> list[dict]:
    result = [
        {
            "name": "adam_cosine",
            "optimizer": "adam_cosine",
            "learning_rate": 0.01,
            "final_learning_rate": 0.001,
            "b1": 0.9,
        }
    ]
    for value in config["optimizer_pilot"]["schedule_free_adamw"]:
        result.append(
            {
                "name": f"schedule_free_lr{value['learning_rate']}_b1{value['b1']}",
                "optimizer": "schedule_free_adamw",
                **value,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "configs/exact_tvd_ancilla_v1.json"
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=ROOT / "results/exact-tvd-ancilla-v1/data",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "results/exact-tvd-ancilla-v1/development",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("kind", choices=("calibration", "optimizer"))
    calibration = subparsers.add_parser("calibration")
    calibration.add_argument("--cell-index", type=int, required=True)
    calibration.add_argument("--candidate-index", type=int, required=True)
    calibration.add_argument("--repeats", type=int, default=5)
    pilot = subparsers.add_parser("optimizer")
    pilot.add_argument("--cell-index", type=int, required=True)
    pilot.add_argument("--candidate-index", type=int, required=True)
    pilot.add_argument("--n-samples", type=int, required=True)
    args = parser.parse_args()
    config = load_manifest(args.manifest)
    cells = _cells()
    optimizers = _optimizer_candidates(config)
    if args.command == "plan":
        size = (
            len(cells) * len(config["estimator_calibration"]["candidates"])
            if args.kind == "calibration"
            else len(cells) * len(optimizers)
        )
        print(json.dumps({"cells": len(cells), "total": size}, indent=2))
        return
    if not 0 <= args.cell_index < len(cells):
        raise SystemExit("cell index is outside the 18-cell development grid")
    dataset, n_visible, ancillas, seed = cells[args.cell_index]
    if args.command == "calibration":
        candidates = config["estimator_calibration"]["candidates"]
        if not 0 <= args.candidate_index < len(candidates):
            raise SystemExit("candidate index is outside the estimator grid")
        n_samples = int(candidates[args.candidate_index])
        row = calibration_cell(
            config=config,
            dataset=dataset,
            ancillas=ancillas,
            seed=seed,
            n_samples=n_samples,
            data_directory=args.data_directory,
            repeats=args.repeats,
        )
        write_immutable_json(
            args.output_directory
            / "calibration"
            / f"cell{args.cell_index}-candidate{args.candidate_index}.json",
            {
                "schema_version": 1,
                "stage": "estimator_calibration_cell",
                "configuration_digest": manifest_digest(config),
                **row,
            },
        )
        print(json.dumps(row))
        return
    if not 0 <= args.candidate_index < len(optimizers):
        raise SystemExit("candidate index is outside the optimizer grid")
    candidate = optimizers[args.candidate_index]
    selection = {
        "status": "passed",
        "n_samples": args.n_samples,
        "optimizer": candidate["optimizer"],
        "optimizer_settings": {
            **candidate,
            "warmup_steps": config["optimizer_pilot"]["warmup_steps"],
            "b2": config["optimizer_pilot"]["b2"],
        },
    }
    run = RunIdentity(
        manifest_digest(config),
        dataset,
        n_visible,
        ancillas,
        seed,
        "conditioned",
        "shared",
        None,
    )
    candidate_directory = args.output_directory / "optimizer" / candidate["name"]
    train_run(
        repository=ROOT,
        config=config,
        selection=selection,
        run=run,
        data_directory=args.data_directory,
        output_directory=candidate_directory,
    )
    evaluation = evaluate_run(
        repository=ROOT,
        config=config,
        run=run,
        data_directory=args.data_directory,
        output_directory=candidate_directory,
    )
    row = {
        "dataset": dataset,
        "visible_qubits": n_visible,
        "ancillas": ancillas,
        "seed": seed,
        "optimizer": candidate["optimizer"],
        "learning_rate": candidate["learning_rate"],
        "b1": candidate["b1"],
        "validation_tvd": min(
            evaluation["selection"]["candidate_mean_exact_validation_tvd"]
        ),
    }
    write_immutable_json(
        args.output_directory
        / "optimizer-rows"
        / f"cell{args.cell_index}-candidate{args.candidate_index}.json",
        {
            "schema_version": 1,
            "stage": "optimizer_pilot_cell",
            "configuration_digest": manifest_digest(config),
            **row,
        },
    )
    print(json.dumps(row))


if __name__ == "__main__":
    main()

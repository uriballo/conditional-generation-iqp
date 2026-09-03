#!/usr/bin/env python3
"""Plan, prepare, train, and exactly evaluate the exact-TVD campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iqp_cond.exact_tvd_campaign import (  # noqa: E402
    load_manifest,
    production_identities,
)
from iqp_cond.exact_tvd_runner import (  # noqa: E402
    evaluate_run,
    load_protocol_selection,
    prepare_data_archive,
    train_run,
)


def _parser() -> argparse.ArgumentParser:
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
        default=ROOT / "results/exact-tvd-ancilla-v1",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--json", action="store_true")
    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument(
        "--mfc-events",
        type=Path,
        default=ROOT / ".scratch/mfc-confirmatory/events.npz",
    )
    for command in ("train", "evaluate"):
        action = subparsers.add_parser(command)
        action.add_argument("--run-id")
        action.add_argument("--run-index", type=int)
        action.add_argument("--shard-index", type=int, default=0)
        action.add_argument("--shard-count", type=int, default=1)
        action.add_argument("--max-runs", type=int)
        action.add_argument("--resume", action="store_true")
        if command == "train":
            action.add_argument(
                "--selection",
                type=Path,
                default=ROOT / "results/exact-tvd-ancilla-v1/protocol-selection.json",
            )
    return parser


def _selected_runs(args: argparse.Namespace, runs: tuple) -> list:
    if args.run_id is not None:
        matches = [run for run in runs if run.run_id == args.run_id]
        if len(matches) != 1:
            raise SystemExit(f"unknown run id: {args.run_id}")
        return matches
    if args.run_index is not None:
        if not 0 <= args.run_index < len(runs):
            raise SystemExit("run index is outside the plan")
        return [runs[args.run_index]]
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard coordinates")
    selected = list(runs[args.shard_index :: args.shard_count])
    return selected if args.max_runs is None else selected[: args.max_runs]


def main() -> None:
    args = _parser().parse_args()
    config = load_manifest(args.manifest)
    runs = production_identities(config)
    if args.command == "plan":
        payload = [dict(run.to_dict(), run_id=run.run_id) for run in runs]
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            counts: dict[str, int] = {}
            for run in runs:
                counts[run.dataset] = counts.get(run.dataset, 0) + 1
            print(json.dumps({"total": len(runs), "datasets": counts}, indent=2))
        return
    if args.command == "prepare-data":
        prepared = []
        for dataset, spec in config["datasets"].items():
            for n_visible in spec["visible_qubits"]:
                path, digest = prepare_data_archive(
                    config,
                    args.data_directory,
                    dataset=dataset,
                    n_visible=n_visible,
                    mfc_events=args.mfc_events if dataset == "mfc" else None,
                )
                prepared.append({"path": str(path), "sha256": digest})
        print(json.dumps(prepared, indent=2))
        return
    selected = _selected_runs(args, runs)
    selection = (
        load_protocol_selection(args.selection, config)
        if args.command == "train"
        else None
    )
    completed = 0
    for run in selected:
        destination = (
            args.output_directory
            / ("training" if args.command == "train" else "evaluations")
            / f"{run.run_id}.json"
        )
        if args.resume and destination.exists():
            continue
        if args.command == "train":
            assert selection is not None
            train_run(
                repository=ROOT,
                config=config,
                selection=selection,
                run=run,
                data_directory=args.data_directory,
                output_directory=args.output_directory,
            )
        else:
            evaluate_run(
                repository=ROOT,
                config=config,
                run=run,
                data_directory=args.data_directory,
                output_directory=args.output_directory,
            )
        completed += 1
        print(json.dumps({"completed": completed, "run_id": run.run_id}), flush=True)


if __name__ == "__main__":
    main()

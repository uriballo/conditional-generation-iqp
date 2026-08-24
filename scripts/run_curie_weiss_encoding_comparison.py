#!/usr/bin/env python3
"""Prepare, preflight, inspect, or run the frozen encoding comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json, sha256_file
from iqp_cond.curie_weiss_ancilla_campaign import CurieWeissAncillaCampaignStore
from iqp_cond.curie_weiss_ancilla_protocol import (
    CurieWeissAncillaFamily,
    CurieWeissAncillaManifest,
    plan_curie_weiss_ancilla_runs,
)
from iqp_cond.curie_weiss_ancilla_runner import execute_run, runtime_preflight


def _verify_inputs(repository: Path, manifest: CurieWeissAncillaManifest) -> None:
    for source in manifest.dataset_sources:
        if sha256_file(repository / source.path) != source.sha256:
            raise SystemExit(f"dataset digest differs: {source.path}")
    baseline = manifest.raw["baseline_reference"]
    for path_key, digest_key in (
        ("manifest_path", "manifest_sha256"),
        ("report_path", "report_sha256"),
    ):
        if sha256_file(repository / baseline[path_key]) != baseline[digest_key]:
            raise SystemExit(f"baseline digest differs: {baseline[path_key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/curie_weiss_encoding_comparison_v1.json"),
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("results/curie-weiss-encoding-comparison-v1"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("preflight")
    commands.add_parser("status")
    run = commands.add_parser("run")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--seed", type=int)
    run.add_argument(
        "--family",
        choices=(
            CurieWeissAncillaFamily.ROLE_LINEAR.value,
            CurieWeissAncillaFamily.ROLE_QUADRATIC.value,
        ),
    )
    run.add_argument("--limit", type=int)
    run.add_argument("--updates", type=int)
    run.add_argument("--lightweight", action="store_true")
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    manifest = CurieWeissAncillaManifest.from_json(repository / args.manifest)
    store = CurieWeissAncillaCampaignStore(
        repository / args.campaign / "store",
        manifest,
    )
    planned = list(plan_curie_weiss_ancilla_runs(manifest))

    if args.command == "prepare":
        _verify_inputs(repository, manifest)
        print(
            json.dumps(
                {
                    "manifest_digest": manifest.digest,
                    "planned_new_fits": len(planned),
                    "reused_baseline_fits": 10,
                    "baseline_reference": manifest.raw["baseline_reference"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "preflight":
        _verify_inputs(repository, manifest)
        evidence = runtime_preflight(repository, manifest)
        print(json.dumps(evidence, indent=2, sort_keys=True))
        store.record_preflight(evidence)
        return 0 if evidence["passed"] else 2
    if args.command == "status":
        preflight = (
            read_immutable_json(store.preflight_path)
            if store.preflight_path.exists()
            else None
        )
        completed = len(store.completed_runs())
        print(
            json.dumps(
                {
                    "manifest_digest": manifest.digest,
                    "planned": len(planned),
                    "completed": completed,
                    "remaining": len(planned) - completed,
                    "preflight": (
                        "unrun"
                        if preflight is None
                        else "passed"
                        if preflight["passed"]
                        else "failed"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.shard_count < 1 or args.shard_index not in range(args.shard_count):
        raise SystemExit("invalid shard coordinates")
    selected = [
        run
        for index, run in enumerate(planned)
        if index % args.shard_count == args.shard_index
    ]
    if args.seed is not None:
        selected = [run for run in selected if run.optimizer_seed == args.seed]
    if args.family is not None:
        selected = [run for run in selected if run.family.value == args.family]
    selected = [run for run in selected if not store.has_complete(run)]
    if args.limit is not None:
        selected = selected[: args.limit]
    for position, run_identity in enumerate(selected, start=1):
        print(
            json.dumps(
                {
                    "event": "starting",
                    "position": position,
                    "selected": len(selected),
                    "run_id": run_identity.run_id,
                    "family": run_identity.family.value,
                    "seed": run_identity.optimizer_seed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        execute_run(
            repository,
            manifest,
            store,
            run_identity,
            invocation=tuple(sys.argv),
            updates=args.updates,
            require_preflight=not args.lightweight,
            lightweight=args.lightweight,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

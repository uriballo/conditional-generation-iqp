#!/usr/bin/env python3
"""Prepare, preflight, inspect, or execute a frozen ancilla campaign."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/curie_weiss_m5_foundations_v1.json"),
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("results/curie-weiss-m5-foundations-v1"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("preflight")
    subparsers.add_parser("status")
    run = subparsers.add_parser("run")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--ancillas", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument(
        "--family", choices=[family.value for family in CurieWeissAncillaFamily]
    )
    run.add_argument("--reference-beta", type=float)
    run.add_argument("--limit", type=int)
    run.add_argument("--lightweight", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    manifest = CurieWeissAncillaManifest.from_json(repository / args.manifest)
    campaign = repository / args.campaign
    store = CurieWeissAncillaCampaignStore(campaign / "store", manifest)
    runs = list(plan_curie_weiss_ancilla_runs(manifest))

    if args.command == "prepare":
        sources = []
        for source in manifest.dataset_sources:
            path = repository / source.path
            digest = sha256_file(path)
            if digest != source.sha256:
                raise SystemExit(f"dataset digest differs: {path}")
            sources.append({"seed": source.seed, "path": source.path, "sha256": digest})
        selection = manifest.raw.get("development_selection")
        if selection is not None:
            pilot_manifest = repository / selection["campaign_path"] / "manifest.json"
            if sha256_file(pilot_manifest) != selection["manifest_sha256"]:
                raise SystemExit("development pilot manifest digest differs")
        print(
            json.dumps(
                {
                    "manifest_digest": manifest.digest,
                    "planned": len(runs),
                    "datasets": sources,
                    "development_selection": selection,
                },
                indent=2,
            )
        )
        return
    if args.command == "preflight":
        evidence = runtime_preflight(repository, manifest)
        print(json.dumps(evidence, indent=2, sort_keys=True))
        store.record_preflight(evidence)
        if not evidence["passed"]:
            raise SystemExit(2)
        return
    if args.command == "status":
        completed = len(store.completed_runs())
        preflight = (
            read_immutable_json(store.preflight_path)
            if store.preflight_path.exists()
            else None
        )
        payload = {
            "manifest_digest": manifest.digest,
            "planned": len(runs),
            "completed": completed,
            "remaining": len(runs) - completed,
            "preflight": (
                "unrun"
                if preflight is None
                else "passed"
                if preflight["passed"]
                else "failed"
            ),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.shard_count < 1 or args.shard_index not in range(args.shard_count):
        raise SystemExit("invalid shard coordinates")
    selected = [
        run
        for index, run in enumerate(runs)
        if index % args.shard_count == args.shard_index
    ]
    if args.ancillas is not None:
        selected = [run for run in selected if run.ancillas == args.ancillas]
    if args.seed is not None:
        selected = [run for run in selected if run.optimizer_seed == args.seed]
    if args.family is not None:
        selected = [run for run in selected if run.family.value == args.family]
    if args.reference_beta is not None:
        selected = [
            run for run in selected if run.reference_beta == args.reference_beta
        ]
    selected = [run for run in selected if not store.has_complete(run)]
    if args.limit is not None:
        selected = selected[: args.limit]
    for position, declared in enumerate(selected, start=1):
        print(
            json.dumps(
                {
                    "event": "starting",
                    "position": position,
                    "selected": len(selected),
                    "run_id": declared.run_id,
                    "ancillas": declared.ancillas,
                    "family": declared.family.value,
                    "seed": declared.optimizer_seed,
                    "reference_beta": declared.reference_beta,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        execute_run(
            repository,
            manifest,
            store,
            declared,
            invocation=tuple(sys.argv),
            require_preflight=not args.lightweight,
            lightweight=args.lightweight,
        )
        print(json.dumps({"event": "completed", "run_id": declared.run_id}), flush=True)


if __name__ == "__main__":
    main()

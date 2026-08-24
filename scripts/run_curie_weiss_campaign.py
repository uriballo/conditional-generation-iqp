"""Inspect, preflight, or execute the immutable Curie--Weiss campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json, write_immutable_json
from iqp_cond.curie_weiss_campaign import CurieWeissCampaignStore
from iqp_cond.curie_weiss_protocol import (
    CurieWeissCampaignManifest,
    CurieWeissFamily,
    plan_curie_weiss_runs,
)
from iqp_cond.curie_weiss_runner import execute_curie_weiss_run, runtime_preflight


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/curie_weiss_foundations_v1.json")
    )
    parser.add_argument(
        "--campaign", type=Path, default=Path("results/curie-weiss-foundations-v1")
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("status", "preflight"):
        _common(commands.add_parser(name))
    run = commands.add_parser("run")
    _common(run)
    run.add_argument("--run-id")
    run.add_argument("--family", choices=[family.value for family in CurieWeissFamily])
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--limit", type=int)
    run.add_argument("--dry-run", action="store_true")
    return result


def context(args: argparse.Namespace):
    repository = Path.cwd().resolve()
    manifest = CurieWeissCampaignManifest.from_json(args.manifest)
    campaign = args.campaign.resolve()
    return (
        repository,
        manifest,
        campaign,
        CurieWeissCampaignStore(campaign / "store", manifest),
        campaign / "datasets",
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repository, manifest, campaign, store, datasets = context(args)
    runs = list(plan_curie_weiss_runs(manifest))
    if args.command == "status":
        completed = {run.run_id for run in store.completed_runs()}
        print(
            json.dumps(
                {
                    "manifest_digest": manifest.digest,
                    "planned": 270,
                    "completed": len(completed),
                    "remaining": 270 - len(completed),
                    "preflight_passed": store.preflight_path.exists(),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "preflight":
        ready = read_immutable_json(campaign / "evidence" / "data-ready.json")
        if ready.get("manifest_digest") != manifest.digest:
            raise RuntimeError("data-ready evidence is incompatible")
        evidence = runtime_preflight(repository, manifest, datasets)
        path = campaign / "evidence" / "runtime-preflight.json"
        if path.exists() and read_immutable_json(path) != evidence:
            raise RuntimeError("runtime evidence differs")
        if not path.exists():
            write_immutable_json(path, evidence)
        if evidence["passed"]:
            store.record_preflight(evidence)
        print(json.dumps(evidence, sort_keys=True))
        return 0 if evidence["passed"] else 2
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if args.run_id:
        runs = [run for run in runs if run.run_id == args.run_id]
        if not runs:
            raise ValueError("run-id is undeclared")
    if args.family:
        runs = [run for run in runs if run.family is CurieWeissFamily(args.family)]
    runs = [
        run
        for index, run in enumerate(runs)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        runs = runs[: args.limit]
    if args.dry_run:
        print(
            json.dumps(
                {"count": len(runs), "run_ids": [run.run_id for run in runs]},
                sort_keys=True,
            )
        )
        return 0
    store.require_preflight()
    for index, run in enumerate(runs, 1):
        if store.has_complete(run):
            print(f"[{index}/{len(runs)}] skip {run.run_id}", flush=True)
            continue
        print(
            f"[{index}/{len(runs)}] n={run.n_qubits} {run.family.value} "
            f"beta={run.reference_beta} seed={run.optimizer_seed}",
            flush=True,
        )
        execute_curie_weiss_run(
            repository, manifest, store, datasets, run, invocation=tuple(sys.argv)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

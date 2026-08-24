"""Preflight, inspect, or execute the immutable Gaussian campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json, write_immutable_json
from iqp_cond.gaussian_campaign import GaussianCampaignStore
from iqp_cond.gaussian_protocol import (
    GaussianCampaignManifest,
    GaussianFamily,
    GaussianStudy,
    plan_gaussian_runs,
)
from iqp_cond.gaussian_runner import execute_gaussian_run, runtime_preflight


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/gaussian_foundations_v1.json")
    )
    parser.add_argument(
        "--campaign", type=Path, default=Path("results/gaussian-foundations-v1")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "status"):
        child = subparsers.add_parser(name)
        _common(child)
    run = subparsers.add_parser("run")
    _common(run)
    run.add_argument("--run-id")
    run.add_argument("--study", choices=[item.value for item in GaussianStudy])
    run.add_argument("--family", choices=[item.value for item in GaussianFamily])
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--limit", type=int)
    run.add_argument("--dry-run", action="store_true")
    return parser


def _context(args: argparse.Namespace):
    repository = Path.cwd().resolve()
    manifest = GaussianCampaignManifest.from_json(args.manifest)
    campaign = args.campaign.resolve()
    store = GaussianCampaignStore(campaign / "store", manifest)
    datasets = campaign / "datasets"
    return repository, manifest, campaign, store, datasets


def _preflight(args: argparse.Namespace) -> int:
    repository, manifest, campaign, store, datasets = _context(args)
    data_evidence = read_immutable_json(campaign / "evidence" / "data-ready.json")
    if data_evidence.get("manifest_digest") != manifest.digest:
        raise RuntimeError("data-ready evidence belongs to another manifest")
    evidence = runtime_preflight(repository, manifest, datasets)
    path = campaign / "evidence" / "runtime-preflight.json"
    if path.exists():
        if read_immutable_json(path) != evidence:
            raise RuntimeError("runtime-preflight evidence differs")
    else:
        write_immutable_json(path, evidence)
    if evidence["passed"]:
        store.record_preflight(evidence)
    print(json.dumps(evidence, sort_keys=True))
    return 0 if evidence["passed"] else 2


def _status(args: argparse.Namespace) -> int:
    _, manifest, _, store, _ = _context(args)
    runs = plan_gaussian_runs(manifest)
    completed = {run.run_id for run in store.completed_runs()}
    payload = {
        "manifest_digest": manifest.digest,
        "planned": len(runs),
        "completed": len(completed),
        "remaining": len(runs) - len(completed),
        "preflight_passed": store.preflight_path.exists(),
        "by_study": {
            study.value: {
                "planned": sum(run.study is study for run in runs),
                "completed": sum(
                    run.study is study and run.run_id in completed for run in runs
                ),
            }
            for study in GaussianStudy
        },
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    repository, manifest, _, store, datasets = _context(args)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index/count are invalid")
    runs = list(plan_gaussian_runs(manifest))
    if args.run_id:
        runs = [run for run in runs if run.run_id == args.run_id]
        if not runs:
            raise ValueError("run-id is not declared")
    if args.study:
        runs = [run for run in runs if run.study is GaussianStudy(args.study)]
    if args.family:
        runs = [run for run in runs if run.family is GaussianFamily(args.family)]
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
    for number, run in enumerate(runs, start=1):
        if store.has_complete(run):
            print(f"[{number}/{len(runs)}] skip {run.run_id}", flush=True)
            continue
        print(
            f"[{number}/{len(runs)}] {run.study.value} n={run.n_qubits} "
            f"w={run.width_factor:.2f} K={run.peak_count} "
            f"{run.family.value} seed={run.optimizer_seed}",
            flush=True,
        )
        execute_gaussian_run(
            repository,
            manifest,
            store,
            datasets,
            run,
            invocation=tuple(sys.argv),
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        return _preflight(args)
    if args.command == "status":
        return _status(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())

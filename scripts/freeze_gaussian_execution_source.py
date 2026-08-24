"""Seal the complete numerical dependency closure for the Gaussian campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json, sha256_file, write_immutable_json
from iqp_cond.gaussian_protocol import GaussianCampaignManifest
from iqp_cond.gaussian_runner import _source_digest
from iqp_cond.protocol import content_digest

EXECUTION_FILES = (
    "configs/gaussian_foundations_v1.json",
    "pyproject.toml",
    "uv.lock",
    "scripts/prepare_gaussian_campaign.py",
    "scripts/run_gaussian_campaign.py",
    "src/iqp_cond/__init__.py",
    "src/iqp_cond/ansatz.py",
    "src/iqp_cond/artifacts.py",
    "src/iqp_cond/encoding.py",
    "src/iqp_cond/exact.py",
    "src/iqp_cond/gaussian_campaign.py",
    "src/iqp_cond/gaussian_data.py",
    "src/iqp_cond/gaussian_protocol.py",
    "src/iqp_cond/gaussian_runner.py",
    "src/iqp_cond/mmd.py",
    "src/iqp_cond/protocol.py",
    "src/iqp_cond/simulator.py",
    "src/iqp_cond/targets.py",
    "src/iqp_cond/training.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/gaussian_foundations_v1.json")
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=Path("results/gaussian-foundations-v1/store/preflight.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gaussian-foundations-v1/evidence/execution-source.json"),
    )
    return parser


def build_snapshot(
    repository: Path,
    manifest: GaussianCampaignManifest,
    preflight_path: Path,
) -> dict[str, object]:
    files = {name: sha256_file(repository / name) for name in EXECUTION_FILES}
    preflight = read_immutable_json(preflight_path)
    if preflight.get("manifest_digest") != manifest.digest:
        raise RuntimeError("execution-source snapshot and preflight manifests differ")
    if preflight.get("source_tree_digest") != _source_digest(repository):
        raise RuntimeError("preflight numerical-core source digest differs")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema_version": 1,
        "campaign_version": manifest.campaign_version,
        "manifest_digest": manifest.digest,
        "preflight_source_tree_digest": preflight["source_tree_digest"],
        "code_revision": revision,
        "files": files,
        "dependency_closure_digest": content_digest(files),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = Path.cwd().resolve()
    manifest = GaussianCampaignManifest.from_json(args.manifest)
    payload = build_snapshot(repository, manifest, args.preflight)
    if args.output.exists():
        if read_immutable_json(args.output) != payload:
            raise RuntimeError("sealed Gaussian execution source differs")
    else:
        write_immutable_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

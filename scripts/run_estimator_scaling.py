"""Archive a deterministic, zero-training-run estimator-scaling audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from iqp_cond.estimator_scaling import (
    archive_scaling_report,
    build_axis_scaling_cases,
    run_scaling_plan,
    runtime_environment,
)
from iqp_cond.protocol import CampaignManifest


def _positive_integer_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated integer list",
        ) from error
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("all list values must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark IQP expectation and gradient estimators at synthetic "
            "frozen parameters; this command performs no training."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/quantum_campaign_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/quantum-campaign-v1/evidence/estimator-scaling.json",
        ),
    )
    parser.add_argument(
        "--qubits",
        type=_positive_integer_list,
        default=(4, 8),
        help="comma-separated qubit counts",
    )
    parser.add_argument(
        "--conditions",
        type=_positive_integer_list,
        default=(1, 4, 16),
        help="comma-separated condition counts K",
    )
    parser.add_argument(
        "--dimensions",
        type=_positive_integer_list,
        default=(1, 2, 4),
        help="comma-separated condition dimensions d",
    )
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--exact-max-qubits", type=int, default=5)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use a tiny local smoke-test plan",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _provenance(manifest: Path, argv: list[str]) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    sources = (
        Path(__file__).resolve(),
        repository / "src" / "iqp_cond" / "estimator_scaling.py",
        repository / "src" / "iqp_cond" / "simulator.py",
        repository / "src" / "iqp_cond" / "exact.py",
        repository / "src" / "iqp_cond" / "response.py",
        manifest.resolve(),
    )
    return {
        "code_revision": revision,
        "working_tree_dirty": dirty,
        "invocation": [sys.executable, str(Path(__file__).resolve()), *argv],
        "source_digests": {
            str(path.relative_to(repository)): _sha256(path) for path in sources
        },
        "environment": runtime_environment(),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the synthetic scaling plan and create one immutable JSON report."""
    invocation_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(invocation_argv)
    manifest = CampaignManifest.from_json(args.manifest)
    qubits = (3,) if args.quick else args.qubits
    conditions = (1, 2) if args.quick else args.conditions
    dimensions = (1, 2) if args.quick else args.dimensions
    samples = 64 if args.quick else args.samples
    repeats = 2 if args.quick else args.repeats
    exact_max = 3 if args.quick else args.exact_max_qubits
    cases = build_axis_scaling_cases(
        qubit_sizes=qubits,
        condition_counts=conditions,
        condition_dimensions=dimensions,
        n_samples=samples,
        n_repeats=repeats,
        seed=args.seed,
        exact_max_qubits=exact_max,
    )
    report = run_scaling_plan(
        cases,
        campaign_version=manifest.campaign_version,
        manifest_digest=manifest.digest,
        provenance=_provenance(args.manifest, invocation_argv),
    )
    artifact_digest = archive_scaling_report(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "artifact_digest": artifact_digest,
                "configuration_digest": report["configuration_digest"],
                "numerical_results_digest": report["numerical_results_digest"],
                "case_count": len(report["records"]),
                "full_training_runs_used": 0,
            },
            sort_keys=True,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

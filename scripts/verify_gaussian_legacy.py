"""Regenerate and verify every archived Gaussian summary used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json, write_immutable_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/gaussian-foundations-v1/evidence/legacy-regeneration.json"
        ),
    )
    return parser


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(_sha(path)))
    return digest.hexdigest()


def _run(command: list[str], repository: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"legacy regeneration failed: {' '.join(command)}\n"
            f"{(completed.stdout + completed.stderr)[-6000:]}"
        )


def _compare(
    repository: Path, generated: Path, pairs: tuple[tuple[str, str], ...]
) -> dict[str, str]:
    result = {}
    for expected_name, actual_name in pairs:
        expected = repository / expected_name
        actual = generated / actual_name
        if expected.read_bytes() != actual.read_bytes():
            raise RuntimeError(f"legacy Gaussian summary differs: {expected_name}")
        result[expected_name] = _sha(expected)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = Path.cwd().resolve()
    with tempfile.TemporaryDirectory(prefix="gaussian-legacy-") as directory:
        root = Path(directory)
        paths = {
            "summary": root / "summary",
            "interpolation": root / "interpolation",
            "encoding": root / "encoding",
            "ancilla": root / "ancilla",
        }
        _run(
            [
                "uv",
                "run",
                "python",
                "scripts/summarize_results.py",
                "results/runs",
                "--output",
                str(paths["summary"]),
            ],
            repository,
        )
        _run(
            [
                "uv",
                "run",
                "python",
                "scripts/summarize_interpolation.py",
                "--output",
                str(paths["interpolation"]),
            ],
            repository,
        )
        _run(
            [
                "uv",
                "run",
                "python",
                "scripts/summarize_encodings.py",
                "--output",
                str(paths["encoding"]),
            ],
            repository,
        )
        _run(
            [
                "uv",
                "run",
                "python",
                "scripts/summarize_ancilla.py",
                "--output",
                str(paths["ancilla"]),
            ],
            repository,
        )
        verified = {}
        verified.update(
            _compare(
                repository,
                paths["summary"],
                (
                    ("results/summary/metrics.md", "metrics.md"),
                    ("results/summary/results_table.tex", "results_table.tex"),
                    (
                        "results/summary/compact_topology_table.tex",
                        "compact_topology_table.tex",
                    ),
                ),
            )
        )
        verified.update(
            _compare(
                repository,
                paths["interpolation"],
                (
                    ("results/interpolation/summary/audit.json", "audit.json"),
                    ("results/interpolation/summary/metrics.md", "metrics.md"),
                    (
                        "results/interpolation/summary/interpolation_table.tex",
                        "interpolation_table.tex",
                    ),
                ),
            )
        )
        verified.update(
            _compare(
                repository,
                paths["encoding"],
                (("results/encoding/summary/metrics.json", "metrics.json"),),
            )
        )
        verified.update(
            _compare(
                repository,
                paths["ancilla"],
                (
                    ("results/ancilla/summary/metrics.md", "metrics.md"),
                    (
                        "results/ancilla/summary/ancilla_table.tex",
                        "ancilla_table.tex",
                    ),
                ),
            )
        )
    payload = {
        "schema_version": 1,
        "campaign_version": "gaussian_foundations_v1",
        "exact_agreement": True,
        "verified_summary_sha256": verified,
        "input_tree_digests": {
            name: _tree_digest(repository / path)
            for name, path in {
                "main_runs": "results/runs",
                "interpolation_runs": "results/interpolation/runs",
                "encoding_runs": "results/encoding/runs",
                "ancilla_runs": "results/ancilla/runs",
                "ancilla_baseline_runs": "results/ancilla-baseline/runs",
                "ancilla_control_runs": "results/ancilla-control/runs",
            }.items()
        },
        "commands": [
            "uv run python scripts/summarize_results.py results/runs",
            "uv run python scripts/summarize_interpolation.py",
            "uv run python scripts/summarize_encodings.py",
            "uv run python scripts/summarize_ancilla.py",
        ],
    }
    if args.output.exists():
        if read_immutable_json(args.output) != payload:
            raise RuntimeError("immutable legacy-regeneration evidence differs")
    else:
        write_immutable_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

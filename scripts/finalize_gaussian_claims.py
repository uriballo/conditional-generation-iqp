"""Generate evidence-bound Gaussian claims for the manuscript and registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/gaussian-foundations-v1/report/complete-report.json"),
    )
    parser.add_argument(
        "--registry-json",
        type=Path,
        default=Path("results/gaussian-foundations-v1/report/claim-registry.json"),
    )
    parser.add_argument(
        "--registry-md",
        type=Path,
        default=Path("docs/gaussian-claim-evidence-registry.md"),
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=Path("paper/generated/gaussian_claims.tex"),
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _status(interval: dict[str, float]) -> str:
    if interval["lower"] > 0.0:
        return "supported"
    if interval["upper"] < 0.0:
        return "unsupported"
    return "mixed"


def _cell_id(cell: dict[str, object]) -> str:
    return f"n{cell['n_qubits']}-w{float(cell['width_factor']):.2f}"


def _claim(
    cells: list[dict[str, object]], claim_id: str, interval_name: str
) -> dict[str, object]:
    win_name = {
        "training_no_condition_minus_conditioned_profile_tvd": (
            "training_vs_no_condition_profile_tvd"
        ),
        "heldout_lookup_minus_conditioned_profile_tvd": (
            "heldout_vs_lookup_profile_tvd"
        ),
        "heldout_mixture_minus_conditioned_profile_tvd": (
            "heldout_vs_mixture_profile_tvd"
        ),
        "training_no_condition_minus_conditioned_bitstring_tvd": (
            "training_vs_no_condition_bitstring_tvd"
        ),
        "heldout_lookup_minus_conditioned_bitstring_tvd": (
            "heldout_vs_lookup_bitstring_tvd"
        ),
    }[interval_name]
    rows = []
    for cell in cells:
        interval = cell["paired_intervals"][interval_name]
        rows.append(
            {
                "cell": _cell_id(cell),
                "status": _status(interval),
                "estimate": interval["estimate"],
                "lower": interval["lower"],
                "upper": interval["upper"],
                "wins": cell["cell_wins"][win_name],
            }
        )
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("supported", "mixed", "unsupported")
    }
    return {
        "claim_id": claim_id,
        "interval_name": interval_name,
        "interpretation": "positive values favor the conditioned model",
        "counts": counts,
        "cells": rows,
    }


def _interval_text(interval: dict[str, float]) -> str:
    return (
        f"{interval['estimate']:.4f} [{interval['lower']:.4f}, {interval['upper']:.4f}]"
    )


def _tex_command(name: str, value: object) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _write_exact(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode()
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError(f"immutable generated claim artifact differs: {path}")
    if not path.exists():
        path.write_bytes(encoded)


def _registry(report: dict[str, object], report_sha: str) -> dict[str, object]:
    cells = report["factorial_cells"]
    claims = [
        _claim(
            cells,
            "G-CONTROL-PROFILE",
            "training_no_condition_minus_conditioned_profile_tvd",
        ),
        _claim(
            cells,
            "G-INTERPOLATION-LOOKUP-PROFILE",
            "heldout_lookup_minus_conditioned_profile_tvd",
        ),
        _claim(
            cells,
            "G-INTERPOLATION-MIXTURE-PROFILE",
            "heldout_mixture_minus_conditioned_profile_tvd",
        ),
        _claim(
            cells,
            "G-CONTROL-BITSTRING",
            "training_no_condition_minus_conditioned_bitstring_tvd",
        ),
        _claim(
            cells,
            "G-INTERPOLATION-LOOKUP-BITSTRING",
            "heldout_lookup_minus_conditioned_bitstring_tvd",
        ),
    ]
    return {
        "schema_version": 1,
        "campaign_version": report["campaign_version"],
        "manifest_digest": report["manifest_digest"],
        "report_sha256": report_sha,
        "completed_atomic_models": report["completed_atomic_models"],
        "performance_gate": False,
        "claim_policy": (
            "supported iff the paired 95% interval is strictly positive; "
            "unsupported iff strictly negative; mixed otherwise"
        ),
        "claims": claims,
        "prohibited_claims": [
            "extrapolation",
            "universal conditional expressivity",
            "quantum advantage",
            "efficient classical full-distribution sampling",
        ],
    }


def _markdown(registry: dict[str, object]) -> str:
    lines = [
        "# Gaussian claim--evidence registry",
        "",
        (
            "This registry is generated only after all 1,020 declared atomic "
            "models and exact evaluations are complete. Positive values favor "
            "the product-state-conditioned circuit. There is no empirical "
            "pass/fail gate: every size--width cell is reported."
        ),
        "",
        f"- Manifest digest: `{registry['manifest_digest']}`",
        f"- Complete report SHA-256: `{registry['report_sha256']}`",
        "- Paired intervals: 10,000 resamples, seed `20260811`",
        "",
        "| Claim | Supported | Mixed | Unsupported |",
        "|---|---:|---:|---:|",
    ]
    for claim in registry["claims"]:
        counts = claim["counts"]
        lines.append(
            f"| {claim['claim_id']} | {counts['supported']} | "
            f"{counts['mixed']} | {counts['unsupported']} |"
        )
    for claim in registry["claims"]:
        lines.extend(
            [
                "",
                f"## {claim['claim_id']}",
                "",
                "| Cell | Status | Paired effect [95% interval] | Wins |",
                "|---|---|---:|---:|",
            ]
        )
        for row in claim["cells"]:
            lines.append(
                f"| {row['cell']} | {row['status']} | "
                f"{row['estimate']:.4f} [{row['lower']:.4f}, {row['upper']:.4f}] | "
                f"{row['wins']['wins']}/{row['wins']['total']} |"
            )
    lines.extend(
        [
            "",
            "## Prohibited claims",
            "",
            *[f"- {claim}" for claim in registry["prohibited_claims"]],
            "",
        ]
    )
    return "\n".join(lines)


def _tex(registry: dict[str, object], report: dict[str, object]) -> str:
    claims = {claim["claim_id"]: claim for claim in registry["claims"]}
    headline = next(
        cell
        for cell in report["factorial_cells"]
        if cell["n_qubits"] == 19 and cell["width_factor"] == 0.5
    )
    intervals = headline["paired_intervals"]
    lines = [
        "% Generated by scripts/finalize_gaussian_claims.py. Do not edit.",
        _tex_command("GaussianAtomicModels", "1,020"),
        _tex_command("GaussianManifestDigest", registry["manifest_digest"]),
    ]
    names = {
        "G-CONTROL-PROFILE": "ProfileControl",
        "G-INTERPOLATION-LOOKUP-PROFILE": "ProfileLookup",
        "G-INTERPOLATION-MIXTURE-PROFILE": "ProfileMixture",
        "G-CONTROL-BITSTRING": "BitstringControl",
        "G-INTERPOLATION-LOOKUP-BITSTRING": "BitstringLookup",
    }
    for claim_id, prefix in names.items():
        counts = claims[claim_id]["counts"]
        for status in ("supported", "mixed", "unsupported"):
            lines.append(
                _tex_command(
                    prefix + status.title(),
                    counts[status],
                )
            )
    headline_names = {
        "training_no_condition_minus_conditioned_profile_tvd": "HeadlineProfileControl",
        "training_conditioned_minus_single_peak_profile_tvd": "HeadlineProfileBankGap",
        "heldout_lookup_minus_conditioned_profile_tvd": "HeadlineProfileLookup",
        "heldout_mixture_minus_conditioned_profile_tvd": "HeadlineProfileMixture",
        "heldout_lookup_minus_conditioned_bitstring_tvd": "HeadlineBitstringLookup",
    }
    for key, name in headline_names.items():
        lines.append(_tex_command(name, _interval_text(intervals[key])))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = read_immutable_json(args.report)
    if report.get("completed_atomic_models") != 1_020:
        raise RuntimeError("claims remain unavailable until all 1,020 models complete")
    report_sha = _sha256(args.report)
    registry = _registry(report, report_sha)
    registry_text = json.dumps(registry, indent=2, sort_keys=True) + "\n"
    _write_exact(args.registry_json, registry_text)
    _write_exact(args.registry_md, _markdown(registry))
    _write_exact(args.tex, _tex(registry, report))
    print(json.dumps({"report_sha256": report_sha, "claims": len(registry["claims"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

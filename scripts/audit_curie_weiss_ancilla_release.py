#!/usr/bin/env python3
"""Audit the completed ancilla Curie--Weiss evidence and combined release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json, sha256_file
from iqp_cond.curie_weiss_ancilla_campaign import CurieWeissAncillaCampaignStore
from iqp_cond.curie_weiss_ancilla_protocol import (
    CurieWeissAncillaManifest,
    plan_curie_weiss_ancilla_runs,
)


def _check(passed: bool, detail: str) -> dict[str, object]:
    return {"passed": bool(passed), "detail": detail}


def audit(
    repository: Path, campaign_path: Path, manifest_path: Path
) -> dict[str, object]:
    repository = repository.resolve()
    campaign = repository / campaign_path
    manifest = CurieWeissAncillaManifest.from_json(repository / manifest_path)
    store = CurieWeissAncillaCampaignStore(campaign / "store", manifest)
    planned = plan_curie_weiss_ancilla_runs(manifest)
    completed = store.completed_runs()
    planned_ids = {run.run_id for run in planned}
    completed_ids = {run.run_id for run in completed}
    checks = {
        "manifest_cardinality": _check(
            len(planned_ids) == 80, f"{len(planned_ids)}/80 declared"
        ),
        "completed_results": _check(
            completed_ids == planned_ids, f"{len(completed_ids)}/80 completed"
        ),
    }
    try:
        preflight = store.require_preflight()
        preflight_valid = (
            preflight.get("compilation_cache") == "fresh_temporary_directory"
            and preflight.get("projected_seconds_per_model", float("inf")) <= 3_600
            and preflight.get("peak_rss_bytes", 1 << 100) <= 30_064_771_072
        )
        checks["runtime_preflight"] = _check(
            preflight_valid,
            f"projected={preflight.get('projected_seconds_per_model')} s",
        )
    except RuntimeError as error:
        checks["runtime_preflight"] = _check(False, str(error))

    artifacts_valid = completed_ids == planned_ids
    verified = 0
    if artifacts_valid:
        for run in completed:
            try:
                bundle = store.load_result(run)
                for relative, expected in (
                    (bundle.checkpoint_path, bundle.checkpoint_sha256),
                    (bundle.probabilities_path, bundle.probabilities_sha256),
                    (bundle.dataset_path, bundle.dataset_sha256),
                ):
                    path = Path(relative)
                    path = path if path.is_absolute() else repository / path
                    if not path.is_file() or sha256_file(path) != expected:
                        artifacts_valid = False
                        break
                    verified += 1
            except (OSError, RuntimeError, ValueError):
                artifacts_valid = False
            if not artifacts_valid:
                break
    checks["artifact_integrity"] = _check(
        artifacts_valid, f"{verified} bound artifact references verified"
    )

    report_path = campaign / "report" / "campaign-report.json"
    registry_path = campaign / "report" / "claim-registry.json"
    report_valid = registry_valid = False
    if report_path.is_file():
        report = read_immutable_json(report_path)
        report_valid = (
            report.get("manifest_digest") == manifest.digest
            and report.get("completed_runs") == 80
            and report.get("headline_ancillas") == 5
            and report.get("performance_thresholds_applied") is False
            and report.get("co_primary_metrics") == ["profile_tvd", "bitstring_tvd"]
        )
    if report_valid and registry_path.is_file():
        registry = read_immutable_json(registry_path)
        registry_valid = registry.get(
            "manifest_digest"
        ) == manifest.digest and registry.get("report_sha256") == sha256_file(
            report_path
        )
    checks["complete_report"] = _check(report_valid, str(report_path))
    checks["claim_registry"] = _check(registry_valid, str(registry_path))

    paper = repository / "paper"
    main_and_supplement = (paper / "main.tex").read_text() + (
        paper / "supplement.tex"
    ).read_text()
    excluded = [
        phrase
        for phrase in (
            "\\ifcurieweissresults",
            "all 270",
            "symmetry-tied",
            "symmetry-aligned",
            "curie_weiss_figures/",
        )
        if phrase in main_and_supplement
    ]
    checks["active_manuscript_scope"] = _check(
        not excluded, "clean" if not excluded else ", ".join(excluded)
    )
    expected_generated = (
        "curie_weiss_ancilla_claims.tex",
        "curie_weiss_ancilla_effects.tex",
        "curie_weiss_ancilla_response.tex",
        "curie_weiss_ancilla_all_rows.tex",
    )
    expected_figures = (
        "curie_weiss_ancilla_temperature_response.pdf",
        "curie_weiss_m5_fidelity.pdf",
        "matched_capacity_v2.pdf",
    )
    checks["generated_evidence"] = _check(
        all((paper / "generated" / name).is_file() for name in expected_generated)
        and all(
            (paper / "curie_weiss_ancilla_figures" / name).is_file()
            for name in expected_figures
        ),
        "four tables/claim files and three figures",
    )
    capacity = repository / "results/curie-weiss-capacity-control-v2"
    checks["matched_capacity_control"] = _check(
        (capacity / "summary.json").is_file()
        and len(tuple((capacity / "results").glob("seed-*.json"))) == 10,
        str(capacity),
    )
    review = repository / "output" / "author-review" / "conditional-iqp-draft"
    checks["author_review_package"] = _check(
        (review / "SHA256SUMS.json").is_file()
        and (review / "paper/ARXIV_METADATA_DRAFT.md").is_file(),
        str(review),
    )
    passed = all(row["passed"] for row in checks.values())
    return {
        "schema_version": 1,
        "campaign_version": manifest.campaign_version,
        "manifest_digest": manifest.digest,
        "passed": passed,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/curie_weiss_m5_foundations_v2.json"),
    )
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("results/curie-weiss-m5-foundations-v2"),
    )
    args = parser.parse_args()
    payload = audit(Path.cwd(), args.campaign, args.manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

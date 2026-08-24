"""Audit the Gaussian-only campaign, evidence, manuscript, and release package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from iqp_cond.artifacts import read_immutable_json, sha256_file
from iqp_cond.gaussian_campaign import GaussianCampaignStore
from iqp_cond.gaussian_protocol import GaussianCampaignManifest, plan_gaussian_runs

FORBIDDEN = (
    "ising",
    "theorem gate",
    "kill tree",
    "quantum-ready",
    "quantum readiness",
    "failure draft",
)
GENERATED = (
    "gaussian_claims.tex",
    "factorial_cells.tex",
    "factorial_wins.tex",
    "raw_bitstring_cells.tex",
    "secondary_diagnostics.tex",
    "condition_count_cells.tex",
    "condition_count_per_condition.tex",
    "condition_count_intervals.tex",
    "all_intervals.tex",
    "all_intervals_paginated.tex",
)
FIGURES = tuple(
    f"fig{index}_{name}"
    for index, name in enumerate(
        (
            "pipeline_targets",
            "headline_profiles",
            "size_width_surface",
            "condition_count",
            "ablations",
        ),
        start=1,
    )
)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_ids(directory: Path, suffix: str) -> set[str]:
    return {path.stem for path in directory.glob(f"*{suffix}")}


def _check(condition: bool, detail: str) -> dict[str, object]:
    return {"passed": bool(condition), "detail": detail}


def _verify_completed_bundles(
    repository: Path,
    store: GaussianCampaignStore,
    completed_runs: tuple,
    source_digest: str,
) -> tuple[bool, int]:
    verified_files: dict[Path, str] = {}
    try:
        for run in completed_runs:
            bundle = store.load_result(run)
            if (
                bundle.environment.get("source_tree_digest") != source_digest
                or bundle.metrics.get("source_tree_digest") != source_digest
                or bundle.metrics.get("optimizer_updates") != 2_000
                or bundle.metrics.get("performance_thresholds_applied") is not False
            ):
                return False, len(verified_files)
            for raw_path, expected in (
                (bundle.dataset_path, bundle.dataset_sha256),
                (bundle.checkpoint_path, bundle.checkpoint_sha256),
                (bundle.probabilities_path, bundle.probabilities_sha256),
            ):
                path = Path(raw_path)
                if not path.is_absolute():
                    path = repository / path
                previous = verified_files.get(path)
                if previous is not None and previous != expected:
                    return False, len(verified_files)
                if previous is None:
                    if not path.is_file() or sha256_file(path) != expected:
                        return False, len(verified_files)
                    verified_files[path] = expected
    except (OSError, RuntimeError, ValueError):
        return False, len(verified_files)
    return True, len(verified_files)


def audit(repository: Path) -> dict[str, object]:
    repository = repository.resolve()
    campaign = repository / "results" / "gaussian-foundations-v1"
    manifest = GaussianCampaignManifest.from_json(
        repository / "configs" / "gaussian_foundations_v1.json"
    )
    store = GaussianCampaignStore(campaign / "store", manifest)
    planned = plan_gaussian_runs(manifest)
    expected_ids = {run.run_id for run in planned}
    completed_runs = store.completed_runs()
    completed_ids = {run.run_id for run in completed_runs}
    checks: dict[str, dict[str, object]] = {}

    checks["manifest_cardinality"] = _check(
        len(planned) == len(expected_ids) == 1_020,
        f"{len(expected_ids)}/1020 unique declared identities",
    )
    data = read_immutable_json(campaign / "evidence" / "data-ready.json")
    data_valid = (
        data.get("manifest_digest") == manifest.digest
        and data.get("archive_count") == 120
        and data.get("held_out_draws") == 0
        and all(
            (repository / row["path"]).is_file()
            and sha256_file(repository / row["path"]) == row["sha256"]
            for row in data.get("archives", [])
        )
    )
    checks["role_isolated_datasets"] = _check(
        data_valid,
        (
            f"{data.get('archive_count')} archives; "
            f"held_out_draws={data.get('held_out_draws')}"
        ),
    )
    preflight = store.require_preflight()
    checks["runtime_preflight"] = _check(
        preflight.get("passed") is True and preflight.get("performance_gate") is False,
        (
            f"projected={preflight.get('projected_seconds_per_model')} s; "
            f"rss={preflight.get('peak_rss_bytes')} bytes"
        ),
    )
    source = read_immutable_json(campaign / "evidence" / "execution-source.json")
    source_snapshot = campaign / "evidence" / "execution-source-files"
    source_valid = (
        source.get("manifest_digest") == manifest.digest
        and source.get("preflight_source_tree_digest")
        == preflight.get("source_tree_digest")
        and all(
            (source_snapshot / relative).is_file()
            and sha256_file(source_snapshot / relative) == digest
            for relative, digest in source.get("files", {}).items()
        )
    )
    checks["execution_source"] = _check(
        source_valid,
        f"sealed snapshot {source.get('dependency_closure_digest')}",
    )
    checks["completed_results"] = _check(
        completed_ids == expected_ids,
        f"{len(completed_ids)}/1020 completed",
    )
    bundles_valid, verified_file_count = _verify_completed_bundles(
        repository,
        store,
        completed_runs,
        str(preflight["source_tree_digest"]),
    )
    checks["completed_bundle_integrity"] = _check(
        bundles_valid,
        f"{len(completed_ids)} envelopes; {verified_file_count} unique artifacts",
    )
    artifact_sets = {
        "reservations": _artifact_ids(store.ledger_directory, ".json"),
        "results": _artifact_ids(store.results_directory, ".json"),
        "checkpoints": _artifact_ids(store.checkpoints_directory, ".npz"),
        "probabilities": _artifact_ids(store.probabilities_directory, ".npz"),
    }
    exact_artifacts = all(ids == expected_ids for ids in artifact_sets.values())
    checks["exact_artifact_sets"] = _check(
        exact_artifacts,
        ", ".join(f"{name}={len(ids)}" for name, ids in artifact_sets.items()),
    )
    legacy = read_immutable_json(campaign / "evidence" / "legacy-regeneration.json")
    legacy_valid = legacy.get("exact_agreement") is True and all(
        (repository / relative).is_file()
        and sha256_file(repository / relative) == digest
        for relative, digest in legacy.get("verified_summary_sha256", {}).items()
    )
    checks["legacy_regeneration"] = _check(
        legacy_valid,
        f"{len(legacy.get('verified_summary_sha256', {}))} exact summaries",
    )

    report_path = campaign / "report" / "complete-report.json"
    registry_path = campaign / "report" / "claim-registry.json"
    report_valid = False
    registry_valid = False
    if report_path.is_file():
        report = read_immutable_json(report_path)
        report_valid = (
            report.get("manifest_digest") == manifest.digest
            and report.get("completed_atomic_models") == 1_020
            and report.get("performance_thresholds_applied") is False
            and report.get("execution_source") == source
        )
    if report_valid and registry_path.is_file():
        registry = json.loads(registry_path.read_text())
        registry_valid = (
            registry.get("manifest_digest") == manifest.digest
            and registry.get("completed_atomic_models") == 1_020
            and registry.get("performance_gate") is False
            and registry.get("report_sha256") == _file_digest(report_path)
        )
    checks["complete_report"] = _check(report_valid, str(report_path))
    checks["claim_registry"] = _check(registry_valid, str(registry_path))

    paper = repository / "paper"
    generated_valid = all((paper / "generated" / name).is_file() for name in GENERATED)
    generated_count = sum((paper / "generated" / name).is_file() for name in GENERATED)
    figure_valid = all(
        (paper / "gaussian_figures" / f"{stem}.{suffix}").is_file()
        for stem in FIGURES
        for suffix in ("pdf", "png", "pgf")
    )
    figure_count = sum(
        (paper / "gaussian_figures" / f"{stem}.pdf").is_file() for stem in FIGURES
    )
    checks["generated_tables_and_claims"] = _check(
        generated_valid,
        f"{generated_count}/{len(GENERATED)}",
    )
    checks["principal_figures"] = _check(
        figure_valid,
        f"{figure_count}/5 PDFs",
    )
    manuscript_paths = (
        paper / "main.tex",
        paper / "supplement.tex",
        paper / "references.bib",
    )
    excluded = [
        f"{path.name}:{phrase}"
        for path in manuscript_paths
        for phrase in FORBIDDEN
        if phrase in path.read_text().lower()
    ]
    checks["gaussian_only_manuscript"] = _check(
        not excluded,
        "no excluded campaign language" if not excluded else ", ".join(excluded),
    )

    review = repository / "output" / "author-review" / "conditional-iqp-draft"
    sums_path = review / "SHA256SUMS.json"
    review_valid = False
    if sums_path.is_file():
        sums = json.loads(sums_path.read_text())
        review_valid = (
            (review / "README.md").is_file()
            and (review / "output/pdf/conditional_iqp_generation.pdf").is_file()
            and (review / "paper/CLAIM_CHANGES.md").is_file()
            and all(
                (review / relative).is_file()
                and _file_digest(review / relative) == digest
                for relative, digest in sums.items()
            )
            and not any(
                "upload" in path.name.lower() or "submit" in path.name.lower()
                for path in review.rglob("*")
                if path.is_file()
            )
        )
    checks["author_review_package"] = _check(review_valid, str(review))

    passed = all(row["passed"] for row in checks.values())
    return {
        "schema_version": 1,
        "campaign_version": manifest.campaign_version,
        "manifest_digest": manifest.digest,
        "passed": passed,
        "checks": checks,
    }


def main() -> int:
    payload = audit(Path.cwd())
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

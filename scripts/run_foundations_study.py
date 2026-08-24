"""Execute and finalize one stage of the compact foundations Ising study."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from iqp_cond.artifacts import read_immutable_json, sha256_file, write_immutable_json
from iqp_cond.campaign_statistics import (
    CONDITION_MODULATED_PHASE,
    FIXED_INPUT,
    NO_CONDITION,
    ORACLE_ADJACENT,
    PairedEvaluationRecord,
    oracle_adjacent_beta_lookup,
)
from iqp_cond.foundations_campaign import (
    FoundationsCampaignError,
    FoundationsCampaignStore,
    FoundationsGate,
    GateDecision,
)
from iqp_cond.foundations_data import load_foundations_exact_archive
from iqp_cond.foundations_estimator import EstimatorSelection
from iqp_cond.foundations_protocol import (
    FOUNDATIONS_SHARED_FAMILIES,
    FoundationsManifest,
    FoundationsTuningDecision,
    FoundationsTuningSelection,
)
from iqp_cond.foundations_runner import execute_foundations_run
from iqp_cond.foundations_statistics import (
    finalize_conditional_control,
    finalize_interpolation,
)
from iqp_cond.protocol import ModelFamily, Stage, content_digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("runtime", "representability", "tuning", "primary"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/arxiv_foundations_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/arxiv-foundations-v1"),
    )
    parser.add_argument(
        "--source-targets",
        type=Path,
        default=Path("results/quantum-campaign-v1/targets"),
    )
    parser.add_argument("--run-id", help="execute only one declared identity")
    return parser


def _selection(output: Path) -> EstimatorSelection:
    payload = read_immutable_json(output / "evidence" / "estimator-calibration.json")
    return EstimatorSelection.from_dict(payload["selection"])


def _execute(
    store: FoundationsCampaignStore,
    stage: Stage,
    args: argparse.Namespace,
) -> tuple:
    planned = store.planned_runs(stage)
    if args.run_id is not None:
        planned = tuple(run for run in planned if run.run_id == args.run_id)
        if len(planned) != 1:
            raise ValueError("--run-id is not declared for this foundations stage")
    selection = _selection(args.output)
    for index, run in enumerate(planned, start=1):
        if store.artifacts.has_bundle(run.run_id):
            print(f"[{index}/{len(planned)}] complete {run.run_id}")
            continue
        if store.reservation_path(run).exists():
            raise RuntimeError(
                f"partial reservation {run.run_id} has no complete result; "
                "refusing reuse"
            )
        print(
            f"[{index}/{len(planned)}] {run.stage.value} "
            f"instance={run.coupling_seed} family={run.model_family.value} "
            f"seed={run.optimizer_seed}",
            flush=True,
        )
        execute_foundations_run(
            store,
            run,
            datasets_directory=args.output / "datasets",
            targets_directory=args.source_targets,
            selection=selection,
            repository=Path(__file__).resolve().parents[1],
            invocation=(sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]),
        )
    return planned


def _archive_and_gate(
    store: FoundationsCampaignStore,
    gate: FoundationsGate,
    passed: bool,
    path: Path,
    payload: dict[str, object],
) -> None:
    write_immutable_json(path, payload)
    store.finalize_gate(
        gate,
        GateDecision.PASSED if passed else GateDecision.FAILED,
        evidence_digest=sha256_file(path),
    )


def _finalize_runtime(
    store: FoundationsCampaignStore, runs: tuple, output: Path
) -> bool:
    if len(runs) != 1:
        return False
    bundle = store.require_complete(runs[0])
    projected = bundle.runtime_seconds * (
        store.manifest.training.max_updates / store.manifest.runtime.smoke_updates
    )
    passed = (
        projected < store.manifest.runtime.max_projected_seconds_per_run
        and bundle.peak_memory_bytes < store.manifest.runtime.max_peak_rss_bytes
    )
    payload = {
        "schema_version": 1,
        "manifest_digest": store.manifest.digest,
        "run_id": runs[0].run_id,
        "smoke_runtime_seconds": bundle.runtime_seconds,
        "projected_runtime_seconds": projected,
        "peak_rss_bytes": bundle.peak_memory_bytes,
        "thresholds": store.manifest.runtime.to_dict(),
        "passed": passed,
    }
    _archive_and_gate(
        store,
        FoundationsGate.RUNTIME_FEASIBILITY,
        passed,
        output / "evidence" / "runtime-feasibility.json",
        payload,
    )
    return passed


def _finalize_representability(
    store: FoundationsCampaignStore, runs: tuple, output: Path
) -> bool:
    if len(runs) != store.manifest.run_allocation.representability:
        return False
    rows = []
    for run in runs:
        metrics = store.require_complete(run).raw_metrics["exact_tvd"]
        if len(metrics) != 1:
            raise ValueError("representability result lacks its one exact TVD")
        rows.append(
            {
                "run_id": run.run_id,
                "coupling_seed": run.coupling_seed,
                "optimizer_seed": run.optimizer_seed,
                "condition_scope": run.condition_scope,
                "tvd": float(metrics[0]["tvd"]),
            }
        )
    values = np.asarray([row["tvd"] for row in rows])
    median = float(np.median(values))
    maximum = float(np.max(values))
    passed = (
        median <= store.manifest.representability.median_tvd_max
        and maximum <= store.manifest.representability.cell_tvd_max
    )
    payload = {
        "schema_version": 1,
        "manifest_digest": store.manifest.digest,
        "rows": rows,
        "median_tvd": median,
        "maximum_tvd": maximum,
        "thresholds": store.manifest.representability.to_dict(),
        "passed": passed,
    }
    _archive_and_gate(
        store,
        FoundationsGate.REPRESENTABILITY,
        passed,
        output / "evidence" / "representability.json",
        payload,
    )
    return passed


def _finalize_tuning(
    store: FoundationsCampaignStore, runs: tuple, output: Path
) -> bool:
    if len(runs) != store.manifest.run_allocation.tuning:
        return False
    grouped: dict[str, list[float]] = defaultdict(list)
    rows = []
    for run in runs:
        value = float(
            store.require_complete(run).raw_metrics["best_validation_conditional_mmd"]
        )
        grouped[run.tuning_candidate_id].append(value)
        rows.append(
            {
                "run_id": run.run_id,
                "coupling_seed": run.coupling_seed,
                "model_family": run.model_family.value,
                "candidate_id": run.tuning_candidate_id,
                "validation_conditional_mmd": value,
            }
        )
    medians = {
        candidate: float(np.median(values)) for candidate, values in grouped.items()
    }
    selections = []
    for family in FOUNDATIONS_SHARED_FAMILIES:
        candidates = store.manifest.training.candidates_for(family)
        selected = min(
            candidates, key=lambda item: (medians[item.candidate_id], item.candidate_id)
        )
        selections.append(FoundationsTuningSelection(family, selected.candidate_id))
    evidence_payload = {
        "schema_version": 1,
        "manifest_digest": store.manifest.digest,
        "selection_rule": store.manifest.training.tuning_selection_rule,
        "rows": rows,
        "candidate_median_validation_conditional_mmd": medians,
        "selections": [selection.to_dict() for selection in selections],
    }
    evidence_digest = content_digest(evidence_payload)
    decision = FoundationsTuningDecision(
        store.manifest.digest, evidence_digest, tuple(selections)
    )
    evidence_path = output / "evidence" / "tuning.json"
    write_immutable_json(evidence_path, evidence_payload)
    store.finalize_tuning(decision)
    return True


def _interval(value) -> dict[str, float | int]:
    return {
        "point_estimate": value.point_estimate,
        "lower": value.lower,
        "upper": value.upper,
        "confidence_level": value.confidence_level,
        "seed": value.seed,
    }


def _primary_records(
    store: FoundationsCampaignStore,
    targets: Path,
) -> tuple[list[PairedEvaluationRecord], list[PairedEvaluationRecord]]:
    bundles = {
        (
            run.coupling_seed,
            run.optimizer_seed,
            run.model_family,
        ): store.require_complete(run)
        for run in store.planned_runs(Stage.PRIMARY)
    }
    training = []
    held_out = []
    manifest = store.manifest
    for instance in manifest.instances.primary:
        exact = load_foundations_exact_archive(
            targets / f"coupling_{instance}.npz", manifest, instance
        )
        oracle = [
            oracle_adjacent_beta_lookup(
                beta,
                manifest.beta_grid.training,
                exact.training.probabilities,
                exact.held_out.probabilities[index],
            ).tvd
            for index, beta in enumerate(manifest.beta_grid.held_out)
        ]
        for seed in manifest.optimizer_seeds.primary:
            values = {}
            for family, label in (
                (ModelFamily.FIXED_INPUT, FIXED_INPUT),
                (ModelFamily.CONDITION_MODULATED_PHASE, CONDITION_MODULATED_PHASE),
                (ModelFamily.NO_CONDITION, NO_CONDITION),
            ):
                metrics = bundles[(instance, seed, family)].raw_metrics["exact_tvd"]
                values[label] = [float(row["tvd"]) for row in metrics]
            for index, beta in enumerate(manifest.beta_grid.training):
                training.append(
                    PairedEvaluationRecord(
                        str(instance),
                        seed,
                        beta,
                        {label: rows[index] for label, rows in values.items()},
                        split="training",
                    )
                )
            offset = len(manifest.beta_grid.training)
            for index, beta in enumerate(manifest.beta_grid.held_out):
                held_out.append(
                    PairedEvaluationRecord(
                        str(instance),
                        seed,
                        beta,
                        {
                            **{
                                label: rows[offset + index]
                                for label, rows in values.items()
                            },
                            ORACLE_ADJACENT: oracle[index],
                        },
                        split="held_out",
                    )
                )
    return training, held_out


def _finalize_primary(
    store: FoundationsCampaignStore, runs: tuple, output: Path, targets: Path
) -> bool:
    if len(runs) != store.manifest.run_allocation.primary:
        return False
    training, held_out = _primary_records(store, targets)
    control = finalize_conditional_control(store.manifest, training)
    interpolation = finalize_interpolation(store.manifest, held_out)
    payload = {
        "schema_version": 1,
        "manifest_digest": store.manifest.digest,
        "conditional_control": {
            "training_median_fixed_tvd": control.training_median_fixed_tvd,
            "improvement_vs_no_condition": _interval(control.improvement_interval),
            "cell_wins": control.cell_wins,
            "cell_total": control.cell_total,
            "cell_win_fraction": control.cell_win_fraction,
            "passed": control.passed,
        },
        "interpolation": {
            "improvement_vs_no_condition": _interval(
                interpolation.improvement_vs_no_condition
            ),
            "improvement_vs_adjacent_lookup": _interval(
                interpolation.improvement_vs_lookup
            ),
            "cell_wins": interpolation.cell_wins,
            "cell_total": interpolation.cell_total,
            "cell_win_fraction": interpolation.cell_win_fraction,
            "instances_improved_vs_lookup": interpolation.instances_improved_vs_lookup,
            "improved_instances": list(interpolation.improved_instances),
            "passed": interpolation.passed,
        },
        "phase_comparator_is_descriptive": True,
        "passed": control.passed and interpolation.passed,
    }
    evidence = output / "evidence" / "primary-claims.json"
    write_immutable_json(evidence, payload)
    evidence_sha = sha256_file(evidence)
    store.finalize_gate(
        FoundationsGate.CONDITIONAL_CONTROL,
        GateDecision.PASSED if control.passed else GateDecision.FAILED,
        evidence_digest=evidence_sha,
    )
    store.finalize_gate(
        FoundationsGate.INTERPOLATION,
        GateDecision.PASSED if interpolation.passed else GateDecision.FAILED,
        evidence_digest=evidence_sha,
    )
    return control.passed and interpolation.passed


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = FoundationsManifest.from_json(args.manifest)
    store = FoundationsCampaignStore(args.output / "store", manifest)
    stage = {
        "runtime": Stage.RUNTIME_FEASIBILITY,
        "representability": Stage.REPRESENTABILITY_SCREEN,
        "tuning": Stage.TUNING,
        "primary": Stage.PRIMARY,
    }[args.stage]
    try:
        runs = _execute(store, stage, args)
    except FoundationsCampaignError as error:
        print(
            json.dumps(
                {"stage": args.stage, "passed": False, "blocked": str(error)},
                sort_keys=True,
            )
        )
        return 2
    if args.run_id is not None:
        return 0
    if args.stage == "runtime":
        passed = _finalize_runtime(store, runs, args.output)
    elif args.stage == "representability":
        passed = _finalize_representability(store, runs, args.output)
    elif args.stage == "tuning":
        passed = _finalize_tuning(store, runs, args.output)
    else:
        passed = _finalize_primary(store, runs, args.output, args.source_targets)
    print(json.dumps({"stage": args.stage, "passed": passed}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

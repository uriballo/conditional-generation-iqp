"""Deterministic finalization for the confirmatory experimental campaigns."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import read_immutable_json
from .confirmatory_curie import (
    CurieConfirmatoryArm,
    CurieConfirmatoryManifest,
    CurieConfirmatoryRunIdentity,
    plan_curie_confirmatory_runs,
)
from .field_symmetry_runner import (
    FIELD_EVALUATION,
    FieldSymmetryArm,
    FieldSymmetryRunIdentity,
    plan_field_symmetry_runs,
)
from .gaussian_statistics import paired_bootstrap_interval
from .mfc_confirmatory import (
    MFCCampaignManifest,
    MFCCoding,
    MFCModelFamily,
    MFCRunIdentity,
    passes_promotion_gate,
    plan_mfc_conditional_runs,
    plan_mfc_screen_runs,
)

CURIE_METRICS = (
    "profile_tvd",
    "bitstring_tvd",
    "absolute_magnetization",
    "energy_per_spin",
    "susceptibility",
    "binder_cumulant",
    "heat_capacity_per_spin",
    "exact_mmd",
)
FIELD_METRICS = (
    "profile_tvd",
    "bitstring_tvd",
    "magnetization_error",
    "equivariance_defect",
)


class ConfirmatoryFinalizationError(RuntimeError):
    """Raised when a report would summarize incomplete or conflicting evidence."""


def gaussian_interpolation_criteria(
    held_out_tvd: float,
    no_condition_held_out_tvd: float,
    training_interior_tvd: float,
    *,
    maximum_generalization_gap: float = 0.05,
) -> dict[str, float | bool]:
    """Apply the frozen Gaussian interpolation boundary to one TVD metric."""
    values = np.asarray(
        [
            held_out_tvd,
            no_condition_held_out_tvd,
            training_interior_tvd,
            maximum_generalization_gap,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Gaussian interpolation values must be finite and nonnegative")
    gap = held_out_tvd - training_interior_tvd
    beats_no_condition = held_out_tvd < no_condition_held_out_tvd
    gap_within_boundary = gap <= maximum_generalization_gap
    return {
        "held_out_tvd": held_out_tvd,
        "no_condition_held_out_tvd": no_condition_held_out_tvd,
        "training_interior_tvd": training_interior_tvd,
        "generalization_gap": gap,
        "maximum_generalization_gap": maximum_generalization_gap,
        "beats_no_condition": beats_no_condition,
        "gap_within_boundary": gap_within_boundary,
        "supported": beats_no_condition and gap_within_boundary,
        "adjacent_distribution_superiority_required": False,
    }


def median_iqr(values: Sequence[float]) -> dict[str, float | int]:
    """Return the declared descriptive statistics without changing units."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be a non-empty finite vector")
    return {
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "n_replication_units": int(array.size),
        "replication_unit": "seed",
    }


def paired_interval(
    values: Mapping[int, float],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, object]:
    """Bootstrap a paired seed-level contrast with frozen randomness."""
    return paired_bootstrap_interval(
        dict(values),
        replicates=replicates,
        seed=seed,
        confidence_level=confidence_level,
    ).to_dict()


def _read_exact_results(
    directory: Path,
    expected: Mapping[str, Any],
    *,
    identity_key: str,
    parse_identity: Callable[[Mapping[str, Any]], Any],
    manifest_digest: str,
) -> dict[str, dict[str, Any]]:
    if not directory.is_dir():
        raise ConfirmatoryFinalizationError(f"result directory is absent: {directory}")
    actual = {path.stem: path for path in directory.glob("*.json")}
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ConfirmatoryFinalizationError(
            "result identities differ from the manifest; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    records: dict[str, dict[str, Any]] = {}
    for run_id, run in expected.items():
        payload = read_immutable_json(actual[run_id])
        raw_identity = payload.get(identity_key)
        if not isinstance(raw_identity, dict) or parse_identity(raw_identity) != run:
            raise ConfirmatoryFinalizationError(
                f"conflicting run identity in {actual[run_id]}"
            )
        if payload.get("manifest_digest") != manifest_digest:
            raise ConfirmatoryFinalizationError(
                f"manifest digest differs in {actual[run_id]}"
            )
        payload_run_id = payload.get("run_id")
        if payload_run_id is not None and payload_run_id != run_id:
            raise ConfirmatoryFinalizationError(
                f"payload run_id differs in {actual[run_id]}"
            )
        records[run_id] = payload
    return records


def _rows_by_seed(
    runs: Sequence[Any],
    records: Mapping[str, Mapping[str, Any]],
    row_filter: Callable[[Mapping[str, Any]], bool],
    metric: str,
) -> dict[int, float]:
    result: dict[int, float] = {}
    for run in runs:
        rows = records[run.run_id].get("metrics", {}).get("evaluation_rows")
        if not isinstance(rows, list):
            raise ConfirmatoryFinalizationError("evaluation_rows are absent")
        selected = [row for row in rows if isinstance(row, dict) and row_filter(row)]
        if not selected:
            raise ConfirmatoryFinalizationError("a declared condition scope is empty")
        try:
            values = np.asarray([float(row[metric]) for row in selected])
        except (KeyError, TypeError, ValueError) as error:
            raise ConfirmatoryFinalizationError(
                f"metric {metric} is absent or invalid"
            ) from error
        if not np.all(np.isfinite(values)):
            raise ConfirmatoryFinalizationError(f"metric {metric} is non-finite")
        if run.seed in result:
            raise ConfirmatoryFinalizationError("a cell repeats a seed identity")
        result[run.seed] = float(values.mean())
    return result


def _summaries_for_scopes(
    runs: Sequence[Any],
    records: Mapping[str, Mapping[str, Any]],
    scopes: Mapping[str, Callable[[Mapping[str, Any]], bool]],
    metrics: Sequence[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        scope: {
            metric: median_iqr(
                list(_rows_by_seed(runs, records, predicate, metric).values())
            )
            for metric in metrics
        }
        for scope, predicate in scopes.items()
    }


def summarize_curie_confirmatory(
    manifest: CurieConfirmatoryManifest,
    campaign: Path,
) -> dict[str, Any]:
    """Finalize every Curie temperature/ancilla cell and mechanism contrast."""
    planned = plan_curie_confirmatory_runs(manifest)
    records = _read_exact_results(
        campaign / "runs",
        {run.run_id: run for run in planned},
        identity_key="run_identity",
        parse_identity=CurieConfirmatoryRunIdentity.from_dict,
        manifest_digest=manifest.digest,
    )
    scopes = {
        "training": lambda row: row.get("role") == "training",
        "held_out": lambda row: row.get("role") == "held_out",
        "critical_held_out": lambda row: (
            row.get("role") == "held_out" and row.get("critical_region") is True
        ),
    }
    cells: list[dict[str, Any]] = []
    indexed: dict[tuple[int, CurieConfirmatoryArm], tuple[Any, ...]] = {}
    for n_qubits in manifest.system_sizes:
        for arm in CurieConfirmatoryArm:
            runs = tuple(
                run for run in planned if run.n_qubits == n_qubits and run.arm is arm
            )
            if {run.seed for run in runs} != set(manifest.seeds):
                raise ConfirmatoryFinalizationError("Curie cell seed set is incomplete")
            indexed[n_qubits, arm] = runs
            cells.append(
                {
                    "n_qubits": n_qubits,
                    "arm": arm.value,
                    "ancillas": arm.ancillas,
                    "summaries": _summaries_for_scopes(
                        runs, records, scopes, CURIE_METRICS
                    ),
                }
            )

    stats = manifest.raw["statistics"]
    mechanisms: list[dict[str, Any]] = []
    for n_qubits in manifest.system_sizes:
        for ancillas in manifest.ancilla_rungs:
            arms = {
                "full": CurieConfirmatoryArm(f"full_joint_r{ancillas}"),
                "disconnected": CurieConfirmatoryArm(f"disconnected_r{ancillas}"),
                "visible_matched": CurieConfirmatoryArm(f"visible_matched_r{ancillas}"),
                "m0": CurieConfirmatoryArm.M0_SPIN_FLIP,
            }
            for metric in ("bitstring_tvd", "profile_tvd"):
                seed_values = {
                    name: _rows_by_seed(
                        indexed[n_qubits, arm],
                        records,
                        scopes["critical_held_out"],
                        metric,
                    )
                    for name, arm in arms.items()
                }
                for name, comparator, treatment in (
                    ("purified_input_rank", "m0", "disconnected"),
                    ("cross_register_interactions", "disconnected", "full"),
                    ("ancilla_beyond_parameter_count", "visible_matched", "full"),
                ):
                    differences = {
                        seed: seed_values[comparator][seed]
                        - seed_values[treatment][seed]
                        for seed in manifest.seeds
                    }
                    interval = paired_interval(
                        differences,
                        replicates=int(stats["bootstrap_replicates"]),
                        seed=int(stats["bootstrap_seed"]),
                        confidence_level=float(stats["confidence_level"]),
                    )
                    mechanisms.append(
                        {
                            "n_qubits": n_qubits,
                            "ancillas": ancillas,
                            "mechanism": name,
                            "metric": metric,
                            "contrast": f"{comparator}_minus_{treatment}",
                            "positive_favors_mechanism": True,
                            "interval": interval,
                            "evidence": (
                                "supported"
                                if float(interval["lower"]) > 0.0
                                else "not_supported"
                            ),
                        }
                    )
    return {
        "schema_version": 1,
        "campaign": "curie_temperature_ancilla",
        "manifest_digest": manifest.digest,
        "complete_run_count": len(records),
        "cells": cells,
        "mechanism_comparisons": mechanisms,
        "negative_results_suppressed": False,
    }


def summarize_field_confirmatory(
    manifest: CurieConfirmatoryManifest,
    campaign: Path,
) -> dict[str, Any]:
    """Finalize the signed-field campaign and its three frozen criteria."""
    planned = plan_field_symmetry_runs(manifest)
    records = _read_exact_results(
        campaign / "results",
        {run.run_id: run for run in planned},
        identity_key="run",
        parse_identity=FieldSymmetryRunIdentity.from_dict,
        manifest_digest=manifest.digest,
    )
    scopes = {
        "all_fields": lambda row: float(row.get("field", math.nan)) in FIELD_EVALUATION,
        "held_out": lambda row: row.get("role") == "held_out",
        "negative_held_out": lambda row: (
            row.get("role") == "held_out" and float(row.get("field", 0.0)) < 0.0
        ),
    }
    indexed: dict[tuple[int, FieldSymmetryArm], tuple[Any, ...]] = {}
    cells: list[dict[str, Any]] = []
    for n_qubits in manifest.system_sizes:
        for arm in FieldSymmetryArm:
            runs = tuple(
                run for run in planned if run.n_qubits == n_qubits and run.arm is arm
            )
            if {run.seed for run in runs} != set(manifest.seeds):
                raise ConfirmatoryFinalizationError("field cell seed set is incomplete")
            indexed[n_qubits, arm] = runs
            cells.append(
                {
                    "n_qubits": n_qubits,
                    "arm": arm.value,
                    "summaries": _summaries_for_scopes(
                        runs, records, scopes, FIELD_METRICS
                    ),
                }
            )

    settings = manifest.raw["statistics"]
    field_settings = manifest.raw["field_study"]
    comparisons: list[dict[str, Any]] = []
    criteria: list[dict[str, Any]] = []
    for n_qubits in manifest.system_sizes:
        equivariant = indexed[n_qubits, FieldSymmetryArm.EQUIVARIANT_POSITIVE]
        exact = all(
            float(records[run.run_id]["metrics"]["max_equivariance_defect"])
            <= float(field_settings["equivariance_tolerance"])
            for run in equivariant
        )
        criteria.append(
            {
                "n_qubits": n_qubits,
                "criterion": "exact_equivariance",
                "threshold": float(field_settings["equivariance_tolerance"]),
                "satisfied": exact,
            }
        )
        for metric in ("bitstring_tvd", "profile_tvd"):
            by_arm = {
                arm: _rows_by_seed(
                    indexed[n_qubits, arm],
                    records,
                    scopes["negative_held_out"],
                    metric,
                )
                for arm in (
                    FieldSymmetryArm.EQUIVARIANT_POSITIVE,
                    FieldSymmetryArm.PRODUCT_POSITIVE,
                    FieldSymmetryArm.PRODUCT_TWO_SIDED,
                )
            }
            for criterion, comparator, orientation, margin in (
                (
                    "improves_over_positive_only",
                    FieldSymmetryArm.PRODUCT_POSITIVE,
                    "comparator_minus_equivariant",
                    0.0,
                ),
                (
                    "noninferior_to_two_sided",
                    FieldSymmetryArm.PRODUCT_TWO_SIDED,
                    "equivariant_minus_comparator",
                    float(field_settings["two_sided_noninferiority_tvd"]),
                ),
            ):
                differences = {
                    seed: (
                        by_arm[comparator][seed]
                        - by_arm[FieldSymmetryArm.EQUIVARIANT_POSITIVE][seed]
                        if orientation == "comparator_minus_equivariant"
                        else by_arm[FieldSymmetryArm.EQUIVARIANT_POSITIVE][seed]
                        - by_arm[comparator][seed]
                    )
                    for seed in manifest.seeds
                }
                interval = paired_interval(
                    differences,
                    replicates=int(settings["bootstrap_replicates"]),
                    seed=int(settings["bootstrap_seed"]),
                    confidence_level=float(settings["confidence_level"]),
                )
                satisfied = (
                    float(interval["lower"]) > 0.0
                    if criterion == "improves_over_positive_only"
                    else float(interval["upper"]) <= margin
                )
                comparisons.append(
                    {
                        "n_qubits": n_qubits,
                        "metric": metric,
                        "criterion": criterion,
                        "contrast": orientation,
                        "margin": margin,
                        "interval": interval,
                        "satisfied": satisfied,
                    }
                )
    return {
        "schema_version": 1,
        "campaign": "signed_field_symmetry",
        "manifest_digest": manifest.digest,
        "complete_run_count": len(records),
        "cells": cells,
        "comparisons": comparisons,
        "criteria": criteria,
        "negative_results_suppressed": False,
    }


def summarize_mfc_confirmatory(
    manifest: MFCCampaignManifest,
    campaign: Path,
) -> dict[str, Any]:
    """Finalize all MFC screen cells and every promoted confirmatory cell."""
    wasserstein_metric = (
        "wasserstein_eta_bin"
        if manifest.campaign_version == "mfc_pseudorapidity_confirmatory_v1"
        else "wasserstein_scaled_energy"
    )
    mfc_metrics = ("tvd", wasserstein_metric, "mean_error", "variance_error")
    directory = campaign / "runs"
    screen_runs = plan_mfc_screen_runs(manifest)
    screen_ids = {run.run_id for run in screen_runs}
    if not directory.is_dir():
        raise ConfirmatoryFinalizationError(f"result directory is absent: {directory}")
    all_paths = {path.stem: path for path in directory.glob("*.json")}
    screen_paths = {
        run_id: all_paths[run_id] for run_id in screen_ids if run_id in all_paths
    }
    # Screen records share a directory with conditional records, so validate them
    # directly before determining the promotion-dependent expected set.
    if set(screen_paths) != screen_ids:
        raise ConfirmatoryFinalizationError("MFC screening seed set is incomplete")
    screen_records: dict[str, dict[str, Any]] = {}
    for run in screen_runs:
        payload = read_immutable_json(screen_paths[run.run_id])
        if (
            not isinstance(payload.get("run"), dict)
            or MFCRunIdentity.from_dict(payload["run"]) != run
            or payload.get("manifest_digest") != manifest.digest
            or payload.get("run_id") != run.run_id
        ):
            raise ConfirmatoryFinalizationError("conflicting MFC screen identity")
        screen_records[run.run_id] = payload
    promoted: dict[tuple[int, MFCCoding], bool] = {}
    screen_cells: list[dict[str, Any]] = []
    for bin_count in manifest.bin_counts:
        for coding in manifest.codings:
            scores: dict[int, dict[int, float]] = {}
            for anchor in manifest.anchor_energies_gev:
                scores[anchor] = {}
                for seed in manifest.development_seeds:
                    run = next(
                        item
                        for item in screen_runs
                        if item.bin_count == bin_count
                        and item.coding is coding
                        and item.anchor_energy_gev == anchor
                        and item.seed == seed
                    )
                    scores[anchor][seed] = float(
                        screen_records[run.run_id]["metrics"]["tvd"]
                    )
            passed = passes_promotion_gate(scores, manifest)
            promoted[bin_count, coding] = passed
            screen_cells.append(
                {
                    "bin_count": bin_count,
                    "coding": coding.value,
                    "promoted": passed,
                    "anchors": {
                        str(anchor): {
                            "median_validation_tvd": float(
                                np.median(list(scores[anchor].values()))
                            ),
                            "worst_seed_tvd": max(scores[anchor].values()),
                            "seed_values": {
                                str(seed): value
                                for seed, value in scores[anchor].items()
                            },
                        }
                        for anchor in manifest.anchor_energies_gev
                    },
                }
            )

    conditional_plan = plan_mfc_conditional_runs(manifest)
    expected_conditional = {
        run.run_id: run
        for run in conditional_plan
        if promoted[run.bin_count, run.coding]
    }
    expected_ids = screen_ids | set(expected_conditional)
    if set(all_paths) != expected_ids:
        raise ConfirmatoryFinalizationError(
            "MFC results differ from the promotion-dependent manifest"
        )
    conditional_records: dict[str, dict[str, Any]] = {}
    for run_id, run in expected_conditional.items():
        payload = read_immutable_json(all_paths[run_id])
        if (
            not isinstance(payload.get("run"), dict)
            or MFCRunIdentity.from_dict(payload["run"]) != run
            or payload.get("manifest_digest") != manifest.digest
            or payload.get("run_id") != run_id
        ):
            raise ConfirmatoryFinalizationError("conflicting MFC conditional identity")
        conditional_records[run_id] = payload

    conditional_cells: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    bootstrap_seed = 20_260_825
    for bin_count in manifest.bin_counts:
        for coding in manifest.codings:
            if not promoted[bin_count, coding]:
                conditional_cells.append(
                    {
                        "bin_count": bin_count,
                        "coding": coding.value,
                        "promoted": False,
                        "families": None,
                    }
                )
                continue
            families: dict[str, Any] = {}
            seed_averages: dict[MFCModelFamily, dict[int, dict[str, float]]] = {}
            for family in MFCModelFamily:
                family_runs = tuple(
                    run
                    for run in expected_conditional.values()
                    if run.bin_count == bin_count
                    and run.coding is coding
                    and run.family is family
                )
                if family.is_trained:
                    expected_count = len(manifest.held_out_energies_gev) * len(
                        manifest.confirmatory_seeds
                    )
                    if len(family_runs) != expected_count:
                        raise ConfirmatoryFinalizationError(
                            "MFC trained family seed/fold set is incomplete"
                        )
                    seed_averages[family] = {}
                    for seed in manifest.confirmatory_seeds:
                        runs = [run for run in family_runs if run.seed == seed]
                        if {run.held_out_energy_gev for run in runs} != set(
                            manifest.held_out_energies_gev
                        ):
                            raise ConfirmatoryFinalizationError(
                                "MFC seed does not contain every declared fold"
                            )
                        seed_averages[family][seed] = {
                            metric: float(
                                np.mean(
                                    [
                                        conditional_records[run.run_id]["metrics"][
                                            metric
                                        ]
                                        for run in runs
                                    ]
                                )
                            )
                            for metric in mfc_metrics
                        }
                    families[family.value] = {
                        metric: median_iqr(
                            [
                                seed_averages[family][seed][metric]
                                for seed in manifest.confirmatory_seeds
                            ]
                        )
                        for metric in mfc_metrics
                    }
                else:
                    if len(family_runs) != len(manifest.held_out_energies_gev):
                        raise ConfirmatoryFinalizationError(
                            "MFC deterministic baseline fold set is incomplete"
                        )
                    families[family.value] = {
                        metric: {
                            "condition_average": float(
                                np.mean(
                                    [
                                        conditional_records[run.run_id]["metrics"][
                                            metric
                                        ]
                                        for run in family_runs
                                    ]
                                )
                            ),
                            "inferential_interval": None,
                            "reason": "deterministic_contextual_baseline",
                        }
                        for metric in mfc_metrics
                    }
            conditional_cells.append(
                {
                    "bin_count": bin_count,
                    "coding": coding.value,
                    "promoted": True,
                    "families": families,
                }
            )
            for metric in ("tvd", wasserstein_metric):
                differences = {
                    seed: seed_averages[MFCModelFamily.NO_CONDITION_IQP][seed][metric]
                    - seed_averages[MFCModelFamily.CONDITIONED_IQP][seed][metric]
                    for seed in manifest.confirmatory_seeds
                }
                interval = paired_interval(
                    differences,
                    replicates=manifest.bootstrap_replicates,
                    seed=bootstrap_seed,
                    confidence_level=manifest.confidence_level,
                )
                comparisons.append(
                    {
                        "bin_count": bin_count,
                        "coding": coding.value,
                        "metric": metric,
                        "contrast": "no_condition_iqp_minus_conditioned_iqp",
                        "positive_favors_conditioning": True,
                        "claim_comparison": True,
                        "interval": interval,
                        "supported": float(interval["lower"]) > 0.0,
                    }
                )
            comparisons.extend(
                {
                    "bin_count": bin_count,
                    "coding": coding.value,
                    "family": family.value,
                    "claim_comparison": False,
                    "role": "contextual_baseline",
                }
                for family in (
                    MFCModelFamily.SEPARATE_IQP,
                    MFCModelFamily.CATEGORICAL_MLP,
                    MFCModelFamily.HISTOGRAM_INTERPOLATION,
                )
            )
    return {
        "schema_version": 1,
        "campaign": "mfc_applied_benchmark",
        "manifest_digest": manifest.digest,
        "complete_run_count": len(all_paths),
        "screen_cells": screen_cells,
        "conditional_cells": conditional_cells,
        "comparisons": comparisons,
        "claim_scope": "conditioned_iqp_vs_no_condition_iqp_only",
        "negative_results_suppressed": False,
    }

"""Complete evidence synthesis for the 1,020-run Gaussian campaign."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import read_immutable_json, sha256_file
from .exact import (
    enumerate_bitstrings,
    fwht,
    shell_probabilities,
    total_variation_distance,
)
from .gaussian_campaign import GaussianCampaignStore, GaussianResultBundle
from .gaussian_data import gaussian_target
from .gaussian_protocol import (
    DataRegime,
    GaussianCampaignManifest,
    GaussianFamily,
    GaussianRunIdentity,
    GaussianStudy,
    plan_gaussian_runs,
)
from .gaussian_statistics import (
    GaussianMetricRecord,
    barycentric_mixture,
    oracle_adjacent_profile_lookup,
    paired_bootstrap_interval,
)
from .protocol import content_digest


def _probabilities(
    repository: Path, bundle: GaussianResultBundle
) -> tuple[np.ndarray, np.ndarray]:
    for raw_path, expected_digest in (
        (bundle.dataset_path, bundle.dataset_sha256),
        (bundle.checkpoint_path, bundle.checkpoint_sha256),
        (bundle.probabilities_path, bundle.probabilities_sha256),
    ):
        artifact = Path(raw_path)
        if not artifact.is_absolute():
            artifact = repository / artifact
        if not artifact.is_file() or sha256_file(artifact) != expected_digest:
            raise ValueError(f"Gaussian result artifact digest differs: {artifact}")
    path = Path(bundle.probabilities_path)
    if not path.is_absolute():
        path = repository / path
    with np.load(path, allow_pickle=False) as archive:
        if archive["manifest_digest"].item().decode() != bundle.manifest_digest:
            raise ValueError("probability artifact manifest differs")
        return (
            np.asarray(archive["conditions"], dtype=np.float64),
            np.asarray(archive["model_probabilities"], dtype=np.float64),
        )


def _index_runs(
    store: GaussianCampaignStore,
    manifest: GaussianCampaignManifest,
    expected_revision: str,
) -> dict[tuple[object, ...], GaussianRunIdentity]:
    runs = plan_gaussian_runs(manifest)
    if len(store.completed_runs()) != len(runs):
        raise ValueError("Gaussian final report requires all 1,020 completed runs")
    expected_ids = {run.run_id for run in runs}
    actual_artifacts = {
        "reservations": {path.stem for path in store.ledger_directory.glob("*.json")},
        "results": {path.stem for path in store.results_directory.glob("*.json")},
        "checkpoints": {
            path.stem for path in store.checkpoints_directory.glob("*.npz")
        },
        "probabilities": {
            path.stem for path in store.probabilities_directory.glob("*.npz")
        },
    }
    for artifact_type, actual_ids in actual_artifacts.items():
        if actual_ids != expected_ids:
            raise ValueError(
                f"Gaussian {artifact_type} do not exactly match declared identities"
            )
    preflight_source = str(store.require_preflight().get("source_tree_digest"))
    revisions = set()
    result = {}
    for run in runs:
        bundle = store.load_result(run)
        revisions.add(bundle.code_revision)
        if (
            bundle.environment.get("source_tree_digest") != preflight_source
            or bundle.metrics.get("source_tree_digest") != preflight_source
            or bundle.metrics.get("performance_thresholds_applied") is not False
            or bundle.metrics.get("optimizer_updates") != 2_000
        ):
            raise ValueError("Gaussian result provenance differs from the preflight")
        key = (
            run.study,
            run.n_qubits,
            run.width_factor,
            run.peak_count,
            run.data_regime,
            run.family,
            run.bank_peak_index,
            run.optimizer_seed,
        )
        result[key] = run
    if revisions != {expected_revision}:
        raise ValueError("Gaussian campaign source revision differs from its snapshot")
    return result


def _validate_execution_source(
    repository: Path,
    manifest: GaussianCampaignManifest,
    store: GaussianCampaignStore,
) -> dict[str, object]:
    evidence = read_immutable_json(
        store.root.parent / "evidence" / "execution-source.json"
    )
    preflight = store.require_preflight()
    if (
        evidence.get("campaign_version") != manifest.campaign_version
        or evidence.get("manifest_digest") != manifest.digest
        or evidence.get("preflight_source_tree_digest")
        != preflight.get("source_tree_digest")
    ):
        raise ValueError("Gaussian execution-source evidence is incompatible")
    files = evidence.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Gaussian execution-source file inventory is empty")
    for relative, expected_digest in files.items():
        path = repository / str(relative)
        if not path.is_file() or sha256_file(path) != expected_digest:
            raise ValueError(f"Gaussian execution source changed: {relative}")
    if evidence.get("dependency_closure_digest") != content_digest(files):
        raise ValueError("Gaussian execution-source closure digest differs")
    return evidence


def _run(
    index: dict[tuple[object, ...], GaussianRunIdentity],
    *,
    study: GaussianStudy,
    n_qubits: int,
    width_factor: float,
    peak_count: int,
    regime: DataRegime,
    family: GaussianFamily,
    seed: int,
    bank_peak_index: int | None = None,
) -> GaussianRunIdentity:
    return index[
        (
            study,
            n_qubits,
            width_factor,
            peak_count,
            regime,
            family,
            bank_peak_index,
            seed,
        )
    ]


def _bank_run(
    index: dict[tuple[object, ...], GaussianRunIdentity],
    *,
    n_qubits: int,
    width_factor: float,
    peak_index: int,
    seed: int,
) -> GaussianRunIdentity:
    if n_qubits == 19 and width_factor == 0.5 and peak_index % 2:
        return _run(
            index,
            study=GaussianStudy.CONDITION_COUNT,
            n_qubits=19,
            width_factor=0.5,
            peak_count=9,
            regime=DataRegime.FIXED_PER_PEAK,
            family=GaussianFamily.SINGLE_PEAK,
            seed=seed,
            bank_peak_index=peak_index,
        )
    return _run(
        index,
        study=GaussianStudy.FACTORIAL,
        n_qubits=n_qubits,
        width_factor=width_factor,
        peak_count=5,
        regime=DataRegime.FIXED_PER_PEAK,
        family=GaussianFamily.SINGLE_PEAK,
        seed=seed,
        bank_peak_index=peak_index,
    )


def _metric(
    target: np.ndarray,
    model: np.ndarray,
    bitstrings: np.ndarray,
    sigmas: tuple[float, ...],
) -> dict[str, float]:
    weights = bitstrings.sum(axis=1).astype(np.float64)
    target_mean = float(weights @ target)
    model_mean = float(weights @ model)
    target_std = float(
        np.sqrt(max(float(((weights - target_mean) ** 2) @ target), 0.0))
    )
    model_std = float(np.sqrt(max(float(((weights - model_mean) ** 2) @ model), 0.0)))
    target_fourier = np.asarray(fwht(target).real, dtype=np.float64)
    model_fourier = np.asarray(fwht(model).real, dtype=np.float64)
    mask_weights = bitstrings.sum(axis=1)
    mmd_values = []
    for sigma in sigmas:
        probability = (1.0 - np.exp(-1.0 / (2.0 * sigma**2))) / 2.0
        mask_law = probability**mask_weights * (1.0 - probability) ** (
            bitstrings.shape[1] - mask_weights
        )
        mmd_values.append(
            float(np.sum(mask_law * (model_fourier - target_fourier) ** 2))
        )
    return GaussianMetricRecord(
        profile_tvd=total_variation_distance(
            shell_probabilities(target, bitstrings),
            shell_probabilities(model, bitstrings),
        ),
        bitstring_tvd=total_variation_distance(target, model),
        mean_absolute_error=abs(model_mean - target_mean),
        width_absolute_error=abs(model_std - target_std),
        conditional_mmd=float(np.mean(mmd_values)),
    ).to_dict()


def _cell(
    repository: Path,
    manifest: GaussianCampaignManifest,
    store: GaussianCampaignStore,
    index: dict[tuple[object, ...], GaussianRunIdentity],
    *,
    n_qubits: int,
    width_factor: float,
    peak_count: int,
    regime: DataRegime,
    study: GaussianStudy,
) -> dict[str, Any]:
    seed_rows = []
    interval_values: dict[str, dict[int, float]] = defaultdict(dict)
    training_conditions = np.asarray(manifest.training_grid(peak_count))
    selected_bank_indices = np.asarray(
        [round(condition * 8) for condition in training_conditions], dtype=int
    )
    bitstrings = enumerate_bitstrings(n_qubits)
    sigmas = tuple(float(value) for value in manifest.raw["training"]["bandwidths"])
    for seed in manifest.optimizer_seeds:
        conditioned = store.load_result(
            _run(
                index,
                study=study,
                n_qubits=n_qubits,
                width_factor=width_factor,
                peak_count=peak_count,
                regime=regime,
                family=GaussianFamily.CONDITIONED,
                seed=seed,
            )
        )
        no_condition = store.load_result(
            _run(
                index,
                study=study,
                n_qubits=n_qubits,
                width_factor=width_factor,
                peak_count=peak_count,
                regime=regime,
                family=GaussianFamily.NO_CONDITION,
                seed=seed,
            )
        )
        conditions, conditioned_probabilities = _probabilities(repository, conditioned)
        no_conditions, no_probabilities = _probabilities(repository, no_condition)
        if not np.array_equal(conditions, no_conditions):
            raise ValueError("paired shared/no-condition evaluations differ")
        bank_probabilities = []
        for peak_index in selected_bank_indices:
            bank = store.load_result(
                _bank_run(
                    index,
                    n_qubits=n_qubits,
                    width_factor=width_factor,
                    peak_index=int(peak_index),
                    seed=seed,
                )
            )
            bank_conditions, bank_output = _probabilities(repository, bank)
            if not np.isclose(bank_conditions[0], peak_index / 8):
                raise ValueError("bank checkpoint condition differs")
            bank_probabilities.append(bank_output[0])
        bank_probabilities = np.asarray(bank_probabilities)
        training_rows = []
        heldout_rows = []
        for condition_index, condition in enumerate(conditions):
            target = gaussian_target(n_qubits, width_factor, float(condition))
            target_probs = target.bitstring_probabilities(bitstrings)
            conditioned_metrics = _metric(
                target_probs,
                conditioned_probabilities[condition_index],
                bitstrings,
                sigmas,
            )
            no_metrics = _metric(
                target_probs,
                no_probabilities[condition_index],
                bitstrings,
                sigmas,
            )
            if condition_index < peak_count:
                bank_metrics = _metric(
                    target_probs,
                    bank_probabilities[condition_index],
                    bitstrings,
                    sigmas,
                )
                training_rows.append(
                    {
                        "condition": float(condition),
                        "conditioned": conditioned_metrics,
                        "no_condition": no_metrics,
                        "single_peak": bank_metrics,
                    }
                )
                continue
            mixture, mixture_weights = barycentric_mixture(
                training_conditions, bank_probabilities, float(condition)
            )
            lookup, lookup_details = oracle_adjacent_profile_lookup(
                training_conditions,
                bank_probabilities,
                target_probs,
                bitstrings,
                float(condition),
            )
            heldout_rows.append(
                {
                    "condition": float(condition),
                    "conditioned": conditioned_metrics,
                    "no_condition": no_metrics,
                    "adjacent_lookup": _metric(
                        target_probs, lookup, bitstrings, sigmas
                    ),
                    "barycentric_mixture": _metric(
                        target_probs, mixture, bitstrings, sigmas
                    ),
                    "mixture_weights": mixture_weights,
                    "lookup_details": lookup_details,
                }
            )
        comparisons = {}
        for metric in (
            "profile_tvd",
            "bitstring_tvd",
            "mean_absolute_error",
            "width_absolute_error",
            "conditional_mmd",
        ):
            comparisons[f"training_no_condition_minus_conditioned_{metric}"] = np.mean(
                [
                    row["no_condition"][metric] - row["conditioned"][metric]
                    for row in training_rows
                ]
            )
            comparisons[f"training_conditioned_minus_single_peak_{metric}"] = np.mean(
                [
                    row["conditioned"][metric] - row["single_peak"][metric]
                    for row in training_rows
                ]
            )
            for baseline, label in (
                ("no_condition", "no_condition"),
                ("adjacent_lookup", "lookup"),
                ("barycentric_mixture", "mixture"),
            ):
                comparisons[f"heldout_{label}_minus_conditioned_{metric}"] = np.mean(
                    [
                        row[baseline][metric] - row["conditioned"][metric]
                        for row in heldout_rows
                    ]
                )
        for name, value in comparisons.items():
            interval_values[name][seed] = float(value)
        seed_rows.append(
            {
                "optimizer_seed": seed,
                "training": training_rows,
                "held_out": heldout_rows,
                "paired_comparisons": comparisons,
            }
        )
    cell_wins = {}
    for metric in (
        "profile_tvd",
        "bitstring_tvd",
        "mean_absolute_error",
        "width_absolute_error",
        "conditional_mmd",
    ):
        cell_wins[f"training_vs_no_condition_{metric}"] = {
            "wins": sum(
                row["conditioned"][metric] < row["no_condition"][metric]
                for seed_row in seed_rows
                for row in seed_row["training"]
            ),
            "total": len(seed_rows) * peak_count,
        }
        for baseline, label in (
            ("no_condition", "no_condition"),
            ("adjacent_lookup", "lookup"),
            ("barycentric_mixture", "mixture"),
        ):
            cell_wins[f"heldout_vs_{label}_{metric}"] = {
                "wins": sum(
                    row["conditioned"][metric] < row[baseline][metric]
                    for seed_row in seed_rows
                    for row in seed_row["held_out"]
                ),
                "total": len(seed_rows) * len(manifest.held_out_grid),
            }
    return {
        "n_qubits": n_qubits,
        "width_factor": width_factor,
        "peak_count": peak_count,
        "data_regime": regime.value,
        "study": study.value,
        "seed_rows": seed_rows,
        "paired_intervals": {
            name: paired_bootstrap_interval(values).to_dict()
            for name, values in interval_values.items()
        },
        "cell_wins": cell_wins,
        "parameter_accounting": {
            "shared_parameters": n_qubits * (n_qubits + 1) // 2,
            "single_peak_bank_parameters": peak_count * n_qubits * (n_qubits + 1) // 2,
            "shared_optimizer_updates": 2_000,
            "single_peak_bank_optimizer_updates": 2_000 * peak_count,
        },
    }


def build_gaussian_report(
    repository: Path,
    manifest: GaussianCampaignManifest,
    store: GaussianCampaignStore,
) -> dict[str, Any]:
    repository = repository.resolve()
    execution_source = _validate_execution_source(repository, manifest, store)
    index = _index_runs(
        store,
        manifest,
        expected_revision=str(execution_source["code_revision"]),
    )
    factorial = [
        _cell(
            repository,
            manifest,
            store,
            index,
            n_qubits=n_qubits,
            width_factor=width_factor,
            peak_count=5,
            regime=DataRegime.FIXED_PER_PEAK,
            study=GaussianStudy.FACTORIAL,
        )
        for n_qubits in manifest.system_sizes
        for width_factor in manifest.width_factors
    ]
    condition_count = []
    for regime in (DataRegime.FIXED_PER_PEAK, DataRegime.FIXED_TOTAL):
        for peak_count in manifest.condition_counts:
            study = (
                GaussianStudy.FACTORIAL
                if regime is DataRegime.FIXED_PER_PEAK and peak_count == 5
                else GaussianStudy.CONDITION_COUNT
            )
            condition_count.append(
                _cell(
                    repository,
                    manifest,
                    store,
                    index,
                    n_qubits=19,
                    width_factor=0.5,
                    peak_count=peak_count,
                    regime=regime,
                    study=study,
                )
            )
    return {
        "schema_version": 1,
        "campaign_version": manifest.campaign_version,
        "manifest_digest": manifest.digest,
        "execution_source": execution_source,
        "completed_atomic_models": 1_020,
        "performance_thresholds_applied": False,
        "factorial_cells": factorial,
        "condition_count_cells": condition_count,
        "interpretation_policy": (
            "report every confidence interval and cell without an empirical "
            "pass/fail gate"
        ),
    }

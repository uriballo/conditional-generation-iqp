#!/usr/bin/env python3
"""Finalize the Curie--Weiss single-temperature and model-bank references."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from iqp_cond.artifacts import read_immutable_json, sha256_file, write_immutable_json
from iqp_cond.confirmatory_curie import (
    CurieConfirmatoryArm,
    CurieConfirmatoryManifest,
    CurieConfirmatoryRunIdentity,
    plan_curie_confirmatory_runs,
)
from iqp_cond.curie_single_temperature import (
    HELD_OUT_BETAS,
    SingleTemperatureManifest,
    SingleTemperatureRunIdentity,
    plan_single_temperature_runs,
)
from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.exact import (
    enumerate_bitstrings,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.gaussian_statistics import paired_bootstrap_interval

METRICS = ("profile_tvd", "bitstring_tvd")


class FinalizationError(RuntimeError):
    """Raised when a report would summarize incomplete or conflicting evidence."""


def median_iqr(values: Sequence[float]) -> dict[str, float | int | str]:
    """Return the declared descriptive statistics across optimizer seeds."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.all(np.isfinite(array)):
        raise FinalizationError("summary values must be a non-empty finite vector")
    return {
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "n_replication_units": int(array.size),
        "replication_unit": "seed",
    }


def _campaign_files(
    roots: Sequence[Path], subdirectory: str, suffix: str
) -> list[Path]:
    return [
        path
        for root in roots
        for path in sorted((Path(root) / subdirectory).glob(f"*{suffix}"))
    ]


def load_single_results(
    manifest: SingleTemperatureManifest,
    roots: Sequence[Path],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Path],
    dict[str, object],
]:
    """Load and verify the exact 330-run single-temperature campaign."""
    expected_runs = {run.run_id: run for run in plan_single_temperature_runs(manifest)}
    records: dict[str, dict[str, Any]] = {}
    arrays: dict[str, Path] = {}
    source_digests: set[str] = set()
    revisions: set[str] = set()
    for path in _campaign_files(roots, "runs", ".json"):
        payload = read_immutable_json(path)
        run_id = str(payload.get("run_id"))
        if run_id in records:
            raise FinalizationError(f"duplicate single-temperature run: {run_id}")
        if run_id not in expected_runs:
            raise FinalizationError(f"unexpected single-temperature run: {run_id}")
        identity = payload.get("run_identity")
        if not isinstance(identity, dict):
            raise FinalizationError(f"run identity is absent in {path}")
        if SingleTemperatureRunIdentity.from_dict(identity) != expected_runs[run_id]:
            raise FinalizationError(f"conflicting run identity in {path}")
        if payload.get("manifest_digest") != manifest.digest:
            raise FinalizationError(f"manifest digest differs in {path}")
        if payload.get("lightweight") or payload.get("optimizer_updates") != int(
            manifest.raw["training"]["updates"]
        ):
            raise FinalizationError(f"incomplete single-temperature run in {path}")
        arrays_path = path.parent.parent / "arrays" / f"{run_id}.npz"
        if not arrays_path.exists() or sha256_file(arrays_path) != payload.get(
            "arrays_sha256"
        ):
            raise FinalizationError(f"arrays are missing or corrupt for {run_id}")
        records[run_id] = payload
        arrays[run_id] = arrays_path
        source_digests.add(str(payload.get("source_tree_digest")))
        revisions.add(str(payload.get("code_revision")))

    if set(records) != set(expected_runs):
        missing = sorted(set(expected_runs) - set(records))
        raise FinalizationError(
            f"single-temperature campaign is incomplete: missing={missing[:5]}"
        )
    if len(source_digests) != 1 or len(revisions) != 1:
        raise FinalizationError("single-temperature source provenance conflicts")
    return (
        records,
        arrays,
        {
            "source_tree_digest": next(iter(source_digests)),
            "code_revision": next(iter(revisions)),
        },
    )


def load_conditioned_results(
    manifest: CurieConfirmatoryManifest,
    roots: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, object]]:
    """Load and verify all 390 original confirmatory result envelopes."""
    expected_runs = {run.run_id: run for run in plan_curie_confirmatory_runs(manifest)}
    records: dict[str, dict[str, Any]] = {}
    source_digests: set[str] = set()
    revisions: set[str] = set()
    for path in _campaign_files(roots, "runs", ".json"):
        payload = read_immutable_json(path)
        run_id = str(payload.get("run_id"))
        if run_id in records:
            raise FinalizationError(f"duplicate conditioned run: {run_id}")
        if run_id not in expected_runs:
            raise FinalizationError(f"unexpected conditioned run: {run_id}")
        identity = payload.get("run_identity")
        if not isinstance(identity, dict):
            raise FinalizationError(f"conditioned run identity is absent in {path}")
        if CurieConfirmatoryRunIdentity.from_dict(identity) != expected_runs[run_id]:
            raise FinalizationError(f"conflicting conditioned identity in {path}")
        if payload.get("manifest_digest") != manifest.digest:
            raise FinalizationError(f"conditioned manifest digest differs in {path}")
        if payload.get("lightweight") or payload.get("optimizer_updates") != int(
            manifest.raw["training"]["updates"]
        ):
            raise FinalizationError(f"incomplete conditioned run in {path}")
        records[run_id] = payload
        source_digests.add(str(payload.get("source_tree_digest")))
        revisions.add(str(payload.get("code_revision")))

    if set(records) != set(expected_runs):
        missing = sorted(set(expected_runs) - set(records))
        raise FinalizationError(f"conditioned campaign is incomplete: {missing[:5]}")
    if len(source_digests) != 1 or len(revisions) != 1:
        raise FinalizationError("conditioned source provenance conflicts")
    return records, {
        "source_tree_digest": next(iter(source_digests)),
        "code_revision": next(iter(revisions)),
    }


def _single_index(
    manifest: SingleTemperatureManifest,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[
    tuple[int, float, int], tuple[SingleTemperatureRunIdentity, Mapping[str, Any]]
]:
    result = {}
    for run in plan_single_temperature_runs(manifest):
        result[run.n_qubits, run.beta, run.seed] = (run, records[run.run_id])
    return result


def _conditioned_index(
    manifest: CurieConfirmatoryManifest,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[int, float, int], Mapping[str, Any]]:
    result = {}
    for run in plan_curie_confirmatory_runs(manifest):
        if run.arm is not CurieConfirmatoryArm.FULL_JOINT_R5:
            continue
        rows = records[run.run_id]["metrics"]["evaluation_rows"]
        for row in rows:
            result[run.n_qubits, float(row["beta"]), run.seed] = row
    expected = len(manifest.system_sizes) * len(manifest.seeds) * 11
    if len(result) != expected:
        raise FinalizationError("conditioned evaluation rows are incomplete")
    return result


def _load_model_probabilities(path: Path, expected_run_id: str) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if (
                "run_id" not in archive.files
                or "model_probabilities" not in archive.files
            ):
                raise FinalizationError(f"required arrays are absent in {path}")
            stored_run_id = archive["run_id"].item().decode("ascii")
            probabilities = np.asarray(archive["model_probabilities"], dtype=np.float64)
    except (OSError, ValueError) as error:
        raise FinalizationError(f"could not read arrays {path}") from error
    if stored_run_id != expected_run_id:
        raise FinalizationError(f"array run identity differs in {path}")
    if (
        probabilities.ndim != 1
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-10)
    ):
        raise FinalizationError(f"invalid model probabilities in {path}")
    return probabilities


def adjacent_training_mixture(
    beta: float,
    training_betas: Sequence[float],
    stored_probabilities: Mapping[float, np.ndarray],
) -> np.ndarray:
    """Interpolate only adjacent training-temperature model outputs."""
    training = np.asarray(training_betas, dtype=np.float64)
    if beta not in HELD_OUT_BETAS:
        raise ValueError("mixture beta must be a declared held-out temperature")
    right = int(np.searchsorted(training, beta, side="right"))
    left = right - 1
    if left < 0 or right >= training.size:
        raise ValueError("held-out temperature lacks adjacent training models")
    weight_right = float((beta - training[left]) / (training[right] - training[left]))
    mixture = (1.0 - weight_right) * stored_probabilities[float(training[left])]
    mixture = mixture + weight_right * stored_probabilities[float(training[right])]
    return np.asarray(mixture, dtype=np.float64)


def summarize_campaign(
    single_manifest: SingleTemperatureManifest,
    base_manifest: CurieConfirmatoryManifest,
    single_records: Mapping[str, Mapping[str, Any]],
    single_arrays: Mapping[str, Path],
    conditioned_records: Mapping[str, Mapping[str, Any]],
    *,
    single_provenance: Mapping[str, object],
    conditioned_provenance: Mapping[str, object],
) -> dict[str, Any]:
    """Derive direct-oracle cells, training-bank mixtures, and paired contrasts."""
    single = _single_index(single_manifest, single_records)
    conditioned = _conditioned_index(base_manifest, conditioned_records)
    probability_cache: dict[str, np.ndarray] = {}

    def probabilities(run: SingleTemperatureRunIdentity) -> np.ndarray:
        if run.run_id not in probability_cache:
            probability_cache[run.run_id] = _load_model_probabilities(
                single_arrays[run.run_id], run.run_id
            )
        return probability_cache[run.run_id]

    mixture_metrics: dict[tuple[int, float, int], dict[str, float]] = {}
    for n_qubits in single_manifest.system_sizes:
        bitstrings = enumerate_bitstrings(n_qubits)
        for seed in single_manifest.seeds:
            stored = {
                beta: probabilities(single[n_qubits, beta, seed][0])
                for beta in single_manifest.training_betas
            }
            for beta in single_manifest.held_out_betas:
                mixture = adjacent_training_mixture(
                    beta, single_manifest.training_betas, stored
                )
                target = exact_curie_weiss_target(n_qubits, beta)
                mixture_metrics[n_qubits, beta, seed] = {
                    "bitstring_tvd": total_variation_distance(
                        mixture, target.probabilities
                    ),
                    "profile_tvd": total_variation_distance(
                        shell_probabilities(mixture, bitstrings), target.profile
                    ),
                }

    cells = []
    for n_qubits in single_manifest.system_sizes:
        for beta in single_manifest.betas:
            oracle_rows = [
                single[n_qubits, beta, seed][1]["metrics"]["evaluation"]
                for seed in single_manifest.seeds
            ]
            conditioned_rows = [
                conditioned[n_qubits, beta, seed] for seed in single_manifest.seeds
            ]
            cell: dict[str, Any] = {
                "n_qubits": n_qubits,
                "beta": beta,
                "role": "training"
                if beta in single_manifest.training_betas
                else "held_out",
                "direct_single_temperature_is_oracle": beta
                in single_manifest.held_out_betas,
                "summaries": {
                    "conditioned": {
                        metric: median_iqr(
                            [float(row[metric]) for row in conditioned_rows]
                        )
                        for metric in METRICS
                    },
                    "single_temperature": {
                        metric: median_iqr([float(row[metric]) for row in oracle_rows])
                        for metric in METRICS
                    },
                },
            }
            if beta in single_manifest.held_out_betas:
                cell["summaries"]["training_bank_mixture"] = {
                    metric: median_iqr(
                        [
                            mixture_metrics[n_qubits, beta, seed][metric]
                            for seed in single_manifest.seeds
                        ]
                    )
                    for metric in METRICS
                }
            cells.append(cell)

    statistics = single_manifest.raw["statistics"]
    contrasts = []
    for n_qubits in single_manifest.system_sizes:
        for metric in METRICS:
            scopes = {
                "held_out_mixture_minus_conditioned": {
                    seed: float(
                        np.mean(
                            [
                                mixture_metrics[n_qubits, beta, seed][metric]
                                - float(conditioned[n_qubits, beta, seed][metric])
                                for beta in single_manifest.held_out_betas
                            ]
                        )
                    )
                    for seed in single_manifest.seeds
                },
                "all_conditioned_minus_direct_single_temperature": {
                    seed: float(
                        np.mean(
                            [
                                float(conditioned[n_qubits, beta, seed][metric])
                                - float(
                                    single[n_qubits, beta, seed][1]["metrics"][
                                        "evaluation"
                                    ][metric]
                                )
                                for beta in single_manifest.betas
                            ]
                        )
                    )
                    for seed in single_manifest.seeds
                },
                "held_out_conditioned_minus_direct_oracle": {
                    seed: float(
                        np.mean(
                            [
                                float(conditioned[n_qubits, beta, seed][metric])
                                - float(
                                    single[n_qubits, beta, seed][1]["metrics"][
                                        "evaluation"
                                    ][metric]
                                )
                                for beta in single_manifest.held_out_betas
                            ]
                        )
                    )
                    for seed in single_manifest.seeds
                },
            }
            for name, seed_values in scopes.items():
                contrasts.append(
                    {
                        "n_qubits": n_qubits,
                        "metric": metric,
                        "contrast": name,
                        "positive_interpretation": (
                            "conditioned_model_improves_on_training_bank"
                            if name == "held_out_mixture_minus_conditioned"
                            else "accuracy_cost_of_parameter_sharing"
                        ),
                        "interval": paired_bootstrap_interval(
                            seed_values,
                            replicates=int(statistics["bootstrap_replicates"]),
                            seed=int(statistics["bootstrap_seed"]),
                            confidence_level=float(statistics["confidence_level"]),
                        ).to_dict(),
                    }
                )

    resources = [
        {
            "n_qubits": n_qubits,
            "phases_per_model": (n_qubits + 5) * (n_qubits + 6) // 2,
            "conditioned_stored_phases": (n_qubits + 5) * (n_qubits + 6) // 2,
            "training_bank_stored_phases": 6 * (n_qubits + 5) * (n_qubits + 6) // 2,
            "complete_oracle_bank_stored_phases": 11
            * (n_qubits + 5)
            * (n_qubits + 6)
            // 2,
            "conditioned_training_runs_per_seed": 1,
            "training_bank_runs_per_seed": 6,
            "complete_oracle_bank_runs_per_seed": 11,
        }
        for n_qubits in single_manifest.system_sizes
    ]
    return {
        "schema_version": 1,
        "campaign": "curie_weiss_single_temperature_reference",
        "manifest_digest": single_manifest.digest,
        "base_manifest_digest": base_manifest.digest,
        "complete_single_temperature_run_count": len(single_records),
        "complete_conditioned_run_count": len(conditioned_records),
        "single_temperature_provenance": dict(single_provenance),
        "conditioned_provenance": dict(conditioned_provenance),
        "cells": cells,
        "paired_contrasts": contrasts,
        "resource_accounting": resources,
        "direct_held_out_fits_are_oracle_references": True,
        "training_bank_uses_held_out_targets": False,
        "required_win": False,
        "familywise_correction": False,
        "negative_results_suppressed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/curie_weiss_single_temperature_v1.json"),
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path("configs/curie_weiss_confirmatory_v1.json"),
    )
    parser.add_argument(
        "--single-results-dir", action="append", type=Path, required=True
    )
    parser.add_argument(
        "--conditioned-results-dir", action="append", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    single_manifest = SingleTemperatureManifest.from_json(args.manifest)
    base_manifest = CurieConfirmatoryManifest.from_json(args.base_manifest)
    single_records, single_arrays, single_provenance = load_single_results(
        single_manifest, args.single_results_dir
    )
    conditioned_records, conditioned_provenance = load_conditioned_results(
        base_manifest, args.conditioned_results_dir
    )
    report = summarize_campaign(
        single_manifest,
        base_manifest,
        single_records,
        single_arrays,
        conditioned_records,
        single_provenance=single_provenance,
        conditioned_provenance=conditioned_provenance,
    )
    if args.output.exists():
        if read_immutable_json(args.output) != report:
            raise RuntimeError("immutable single-temperature report differs")
    else:
        write_immutable_json(args.output, report)
    print(
        json.dumps(
            {
                "campaign": report["campaign"],
                "single_temperature_runs": report[
                    "complete_single_temperature_run_count"
                ],
                "conditioned_runs": report["complete_conditioned_run_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

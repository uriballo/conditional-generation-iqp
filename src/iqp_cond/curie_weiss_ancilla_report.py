"""Paired evidence report for the eight-visible Curie--Weiss ancilla campaign."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import read_immutable_json, write_immutable_json
from .curie_weiss_ancilla_campaign import CurieWeissAncillaCampaignStore
from .curie_weiss_ancilla_protocol import (
    CurieWeissAncillaFamily,
    plan_curie_weiss_ancilla_runs,
)
from .curie_weiss_data import exact_curie_weiss_target
from .curie_weiss_statistics import (
    adjacent_profile_oracle,
    midpoint_mixture,
    thermodynamic_metrics,
)
from .exact import enumerate_bitstrings, shell_probabilities, total_variation_distance
from .gaussian_runner import exact_mmd_squared
from .gaussian_statistics import paired_bootstrap_interval


def _probabilities(store: CurieWeissAncillaCampaignStore, run) -> np.ndarray:
    bundle = store.load_result(run)
    path = store.root.parents[2] / bundle.probabilities_path
    with np.load(path, allow_pickle=False) as archive:
        if archive["manifest_digest"].item().decode() != store.manifest.digest:
            raise RuntimeError("probability archive belongs to another manifest")
        if archive["run_id"].item().decode() != run.run_id:
            raise RuntimeError("probability archive belongs to another run")
        return np.asarray(archive["model_probabilities"], dtype=np.float64)


def _interval(values: dict[int, float]) -> dict[str, Any]:
    result = paired_bootstrap_interval(values, replicates=10_000, seed=20_260_815)
    status = (
        "supported"
        if result.lower > 0
        else "unsupported"
        if result.upper < 0
        else "mixed"
    )
    return {**result.to_dict(), "status": status}


def _raw_oracle(
    training: np.ndarray, references: np.ndarray, beta: float
) -> np.ndarray:
    right = int(np.searchsorted(training, beta))
    target = exact_curie_weiss_target(8, beta)
    candidates = (references[right - 1], references[right])
    scores = [
        total_variation_distance(target.probabilities, candidate)
        for candidate in candidates
    ]
    return candidates[0 if scores[0] <= scores[1] else 1]


def build_report(store: CurieWeissAncillaCampaignStore) -> dict[str, Any]:
    runs = plan_curie_weiss_ancilla_runs(store.manifest)
    expected_runs = len(runs)
    headline = store.manifest.headline_ancillas
    headline_label = f"m{headline}"
    if len(store.completed_runs()) != expected_runs:
        raise RuntimeError(f"report requires all {expected_runs} immutable results")
    by_coordinate = {
        (run.ancillas, run.optimizer_seed, run.family, run.reference_beta): run
        for run in runs
    }
    response_rows: list[dict[str, Any]] = []
    effects: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    training = np.asarray(store.manifest.training_betas)
    visible_bitstrings = enumerate_bitstrings(8)
    for seed in store.manifest.optimizer_seeds:
        reference_outputs = np.stack(
            [
                _probabilities(
                    store,
                    by_coordinate[
                        (
                            headline,
                            seed,
                            CurieWeissAncillaFamily.TEMPERATURE_REFERENCE,
                            beta,
                        )
                    ],
                )[0]
                for beta in store.manifest.training_betas
            ]
        )
        rung_outputs = {}
        for ancillas in store.manifest.ancilla_rungs:
            conditioned = _probabilities(
                store,
                by_coordinate[
                    (ancillas, seed, CurieWeissAncillaFamily.CONDITIONED, None)
                ],
            )
            no_condition = _probabilities(
                store,
                by_coordinate[
                    (ancillas, seed, CurieWeissAncillaFamily.NO_CONDITION, None)
                ],
            )
            rung_outputs[(ancillas, "conditioned")] = conditioned
            rung_outputs[(ancillas, "no_condition")] = no_condition
            for index, beta in enumerate(store.manifest.evaluation_betas):
                target = exact_curie_weiss_target(8, beta)
                conditioned_metrics = thermodynamic_metrics(target, conditioned[index])
                no_metrics = thermodynamic_metrics(target, no_condition[index])
                role = (
                    "training" if beta in store.manifest.training_betas else "held_out"
                )
                for metric in ("profile_tvd", "bitstring_tvd"):
                    effects[f"conditional_m{ancillas}_{metric}_{role}"][seed].append(
                        getattr(no_metrics, metric)
                        - getattr(conditioned_metrics, metric)
                    )
                row: dict[str, Any] = {
                    "seed": seed,
                    "ancillas": ancillas,
                    "beta": beta,
                    "role": role,
                    "conditioned": conditioned_metrics.to_dict(),
                    "no_condition": no_metrics.to_dict(),
                    "conditioned_exact_mmd_by_bandwidth": {
                        str(sigma): exact_mmd_squared(
                            conditioned[index], target.probabilities, (sigma,)
                        )
                        for sigma in (1.0, 2.0, 4.0)
                    },
                    "no_condition_exact_mmd_by_bandwidth": {
                        str(sigma): exact_mmd_squared(
                            no_condition[index], target.probabilities, (sigma,)
                        )
                        for sigma in (1.0, 2.0, 4.0)
                    },
                    "conditioned_profile": shell_probabilities(
                        conditioned[index], visible_bitstrings
                    ).tolist(),
                    "no_condition_profile": shell_probabilities(
                        no_condition[index], visible_bitstrings
                    ).tolist(),
                }
                if ancillas == headline and role == "training":
                    reference_index = store.manifest.training_betas.index(beta)
                    reference = reference_outputs[reference_index]
                    reference_metrics = thermodynamic_metrics(target, reference)
                    row["temperature_reference"] = reference_metrics.to_dict()
                    row["temperature_reference_profile"] = shell_probabilities(
                        reference, visible_bitstrings
                    ).tolist()
                    for metric in ("profile_tvd", "bitstring_tvd"):
                        effects[f"{headline_label}_vs_reference_{metric}_training"][
                            seed
                        ].append(
                            getattr(reference_metrics, metric)
                            - getattr(conditioned_metrics, metric)
                        )
                if ancillas == headline and role == "held_out":
                    mixture = midpoint_mixture(training, reference_outputs, beta)
                    profile_oracle, _ = adjacent_profile_oracle(
                        training, reference_outputs, target
                    )
                    raw_oracle = _raw_oracle(training, reference_outputs, beta)
                    baseline_metrics = {
                        "adjacent_mixture": thermodynamic_metrics(target, mixture),
                        "profile_oracle": thermodynamic_metrics(target, profile_oracle),
                        "raw_oracle": thermodynamic_metrics(target, raw_oracle),
                    }
                    row.update(
                        {
                            name: metrics.to_dict()
                            for name, metrics in baseline_metrics.items()
                        }
                    )
                    row["adjacent_mixture_profile"] = shell_probabilities(
                        mixture, visible_bitstrings
                    ).tolist()
                    row["profile_oracle_profile"] = shell_probabilities(
                        profile_oracle, visible_bitstrings
                    ).tolist()
                    row["raw_oracle_profile"] = shell_probabilities(
                        raw_oracle, visible_bitstrings
                    ).tolist()
                    effects[f"{headline_label}_vs_mixture_profile_tvd_held_out"][
                        seed
                    ].append(
                        baseline_metrics["adjacent_mixture"].profile_tvd
                        - conditioned_metrics.profile_tvd
                    )
                    effects[f"{headline_label}_vs_mixture_bitstring_tvd_held_out"][
                        seed
                    ].append(
                        baseline_metrics["adjacent_mixture"].bitstring_tvd
                        - conditioned_metrics.bitstring_tvd
                    )
                    effects[f"{headline_label}_vs_profile_oracle_held_out"][
                        seed
                    ].append(
                        baseline_metrics["profile_oracle"].profile_tvd
                        - conditioned_metrics.profile_tvd
                    )
                    effects[f"{headline_label}_vs_raw_oracle_held_out"][seed].append(
                        baseline_metrics["raw_oracle"].bitstring_tvd
                        - conditioned_metrics.bitstring_tvd
                    )
                if ancillas == headline and beta == 1.0:
                    for metric in ("profile_tvd", "bitstring_tvd"):
                        effects[f"critical_no_condition_{metric}"][seed].append(
                            getattr(no_metrics, metric)
                            - getattr(conditioned_metrics, metric)
                        )
                if ancillas == headline and beta == 1.5:
                    for metric in ("profile_tvd", "bitstring_tvd"):
                        effects[f"cold_no_condition_{metric}"][seed].append(
                            getattr(no_metrics, metric)
                            - getattr(conditioned_metrics, metric)
                        )
                response_rows.append(row)
        for family in ("conditioned", "no_condition"):
            if 0 not in store.manifest.ancilla_rungs:
                continue
            zero = rung_outputs[(0, family)]
            headline_output = rung_outputs[(headline, family)]
            for role, betas in (
                ("training", store.manifest.training_betas),
                ("held_out", store.manifest.held_out_betas),
            ):
                indices = [
                    store.manifest.evaluation_betas.index(beta) for beta in betas
                ]
                for metric in ("profile_tvd", "bitstring_tvd"):
                    differences = []
                    for index in indices:
                        target = exact_curie_weiss_target(
                            8, store.manifest.evaluation_betas[index]
                        )
                        zero_metric = getattr(
                            thermodynamic_metrics(target, zero[index]), metric
                        )
                        headline_metric = getattr(
                            thermodynamic_metrics(target, headline_output[index]),
                            metric,
                        )
                        differences.append(zero_metric - headline_metric)
                    effects[f"{headline_label}_vs_m0_{family}_{metric}_{role}"][
                        seed
                    ].append(float(np.mean(differences)))
    comparisons = []
    for name, seed_values in sorted(effects.items()):
        paired = {seed: float(np.mean(values)) for seed, values in seed_values.items()}
        comparisons.append({"comparison": name, "interval": _interval(paired)})
    statuses: dict[str, int] = defaultdict(int)
    for comparison in comparisons:
        statuses[comparison["interval"]["status"]] += 1
    return {
        "schema_version": 1,
        "campaign_version": store.manifest.campaign_version,
        "manifest_digest": store.manifest.digest,
        "completed_runs": expected_runs,
        "headline_ancillas": headline,
        "co_primary_metrics": ["profile_tvd", "bitstring_tvd"],
        "performance_thresholds_applied": False,
        "comparison_direction": (
            "positive means lower TVD for the named headline/conditioned model"
        ),
        "comparisons": comparisons,
        "response_rows": response_rows,
        "status_counts": dict(sorted(statuses.items())),
        "limitations": [
            "eight visible qubits only",
            "ancilla count and parameter count are confounded",
            "within-range interpolation only",
            "no matched-capacity higher-local visible control",
        ],
        "prohibited_claims": [
            "state-of-the-art thermal-state preparation",
            "quantum advantage",
            "extrapolation",
            "exact critical scaling",
            "general many-body thermal generation",
            "ancilla marginalization alone causes the observed effect",
        ],
    }


def write_report(path: Path, report: dict[str, Any]) -> Path:
    if path.exists():
        if read_immutable_json(path) != report:
            raise RuntimeError("immutable ancilla report differs")
    else:
        write_immutable_json(path, report)
    return path

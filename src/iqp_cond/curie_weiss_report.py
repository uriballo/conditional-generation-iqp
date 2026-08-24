"""Content-verified statistics and evidence registry for Curie--Weiss."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import read_immutable_json, write_immutable_json
from .curie_weiss_campaign import CurieWeissCampaignStore
from .curie_weiss_data import exact_curie_weiss_target
from .curie_weiss_protocol import CurieWeissFamily, plan_curie_weiss_runs
from .curie_weiss_statistics import (
    adjacent_profile_oracle,
    evidence_status,
    midpoint_mixture,
    paired_interval,
    thermodynamic_metrics,
)
from .exact import enumerate_bitstrings, shell_probabilities


def _probabilities(store: CurieWeissCampaignStore, run) -> np.ndarray:
    bundle = store.load_result(run)
    path = Path(bundle.probabilities_path)
    if not path.is_absolute():
        path = store.root.parents[2] / path
    with np.load(path, allow_pickle=False) as archive:
        if archive["manifest_digest"].item().decode() != store.manifest.digest:
            raise RuntimeError("probability archive belongs to another manifest")
        if archive["run_id"].item().decode() != run.run_id:
            raise RuntimeError("probability archive belongs to another run")
        return np.asarray(archive["model_probabilities"], dtype=np.float64)


def _interval(values: dict[int, float]) -> dict[str, Any]:
    result = paired_interval(values)
    return {**result.to_dict(), "status": evidence_status(result)}


def build_curie_weiss_report(store: CurieWeissCampaignStore) -> dict[str, Any]:
    runs = plan_curie_weiss_runs(store.manifest)
    if len(store.completed_runs()) != len(runs):
        raise RuntimeError("report requires all 270 immutable results")
    by_coordinate = {
        (run.n_qubits, run.optimizer_seed, run.family, run.reference_beta): run
        for run in runs
    }
    comparisons: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    training = np.asarray(store.manifest.training_betas)
    for n in store.manifest.system_sizes:
        bitstrings = enumerate_bitstrings(n)
        seed_effects: dict[str, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for seed in store.manifest.optimizer_seeds:
            conditioned_run = by_coordinate[
                (n, seed, CurieWeissFamily.CONDITIONED, None)
            ]
            no_run = by_coordinate[(n, seed, CurieWeissFamily.NO_CONDITION, None)]
            unrestricted_run = by_coordinate[
                (n, seed, CurieWeissFamily.UNRESTRICTED_CONDITIONED, None)
            ]
            conditioned = _probabilities(store, conditioned_run)
            no_condition = _probabilities(store, no_run)
            unrestricted = _probabilities(store, unrestricted_run)
            references = np.stack(
                [
                    _probabilities(
                        store,
                        by_coordinate[
                            (n, seed, CurieWeissFamily.TEMPERATURE_REFERENCE, beta)
                        ],
                    )[0]
                    for beta in store.manifest.training_betas
                ]
            )
            for index, beta in enumerate(store.manifest.evaluation_betas):
                target = exact_curie_weiss_target(n, beta)
                fixed_metrics = thermodynamic_metrics(target, conditioned[index])
                no_metrics = thermodynamic_metrics(target, no_condition[index])
                unrestricted_metrics = thermodynamic_metrics(
                    target, unrestricted[index]
                )
                role = (
                    "training" if beta in store.manifest.training_betas else "held_out"
                )
                mixture_profile = None
                mixture_metrics = None
                oracle_metrics = None
                if role == "held_out":
                    mixture = midpoint_mixture(training, references, beta)
                    oracle, _ = adjacent_profile_oracle(training, references, target)
                    mixture_profile = shell_probabilities(mixture, bitstrings).tolist()
                    mixture_metrics = thermodynamic_metrics(target, mixture).to_dict()
                    oracle_metrics = thermodynamic_metrics(target, oracle).to_dict()
                response_rows.append(
                    {
                        "n_qubits": n,
                        "optimizer_seed": seed,
                        "beta": beta,
                        "role": role,
                        "conditioned": fixed_metrics.to_dict(),
                        "no_condition": no_metrics.to_dict(),
                        "unrestricted_conditioned": unrestricted_metrics.to_dict(),
                        "conditioned_profile": shell_probabilities(
                            conditioned[index], bitstrings
                        ).tolist(),
                        "no_condition_profile": shell_probabilities(
                            no_condition[index], bitstrings
                        ).tolist(),
                        "adjacent_mixture_profile": mixture_profile,
                        "adjacent_mixture": mixture_metrics,
                        "adjacent_oracle": oracle_metrics,
                    }
                )
                seed_effects[f"training_no_condition_{role}"][seed].append(
                    no_metrics.profile_tvd - fixed_metrics.profile_tvd
                )
                seed_effects[f"unrestricted_minus_aligned_{role}"][seed].append(
                    unrestricted_metrics.profile_tvd - fixed_metrics.profile_tvd
                )
                if role == "training":
                    ref_index = store.manifest.training_betas.index(beta)
                    ref_metric = thermodynamic_metrics(target, references[ref_index])
                    seed_effects["independent_reference_training"][seed].append(
                        ref_metric.profile_tvd - fixed_metrics.profile_tvd
                    )
                else:
                    mixture = midpoint_mixture(training, references, beta)
                    oracle, _ = adjacent_profile_oracle(training, references, target)
                    mix_metric = thermodynamic_metrics(target, mixture)
                    oracle_metric = thermodynamic_metrics(target, oracle)
                    seed_effects["adjacent_mixture_held_out"][seed].append(
                        mix_metric.profile_tvd - fixed_metrics.profile_tvd
                    )
                    seed_effects["adjacent_oracle_held_out"][seed].append(
                        oracle_metric.profile_tvd - fixed_metrics.profile_tvd
                    )
                    if beta == 1.0:
                        seed_effects["critical_no_condition"][seed].append(
                            no_metrics.profile_tvd - fixed_metrics.profile_tvd
                        )
                        seed_effects["critical_adjacent_mixture"][seed].append(
                            mix_metric.profile_tvd - fixed_metrics.profile_tvd
                        )
                        seed_effects["critical_adjacent_oracle"][seed].append(
                            oracle_metric.profile_tvd - fixed_metrics.profile_tvd
                        )
        for comparison, values in sorted(seed_effects.items()):
            paired = {seed: float(np.mean(items)) for seed, items in values.items()}
            comparisons.append(
                {"n_qubits": n, "comparison": comparison, "interval": _interval(paired)}
            )
    statuses = defaultdict(int)
    for row in comparisons:
        statuses[row["interval"]["status"]] += 1
    return {
        "schema_version": 1,
        "campaign_version": store.manifest.campaign_version,
        "manifest_digest": store.manifest.digest,
        "completed_runs": 270,
        "performance_thresholds_applied": False,
        "comparison_direction": (
            "positive means lower profile TVD for symmetry-aligned conditioned IQP"
        ),
        "comparisons": comparisons,
        "response_rows": response_rows,
        "status_counts": dict(sorted(statuses.items())),
        "prohibited_claims": [
            "state-of-the-art thermal-state preparation",
            "quantum advantage",
            "extrapolation",
            "exact critical scaling",
            "general many-body thermal generation",
        ],
    }


def write_report(path: Path, report: dict[str, Any]) -> Path:
    if path.exists():
        if read_immutable_json(path) != report:
            raise RuntimeError("immutable Curie--Weiss report differs")
    else:
        write_immutable_json(path, report)
    return path

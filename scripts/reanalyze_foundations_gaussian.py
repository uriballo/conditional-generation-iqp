"""Regenerate legacy Gaussian metrics and add paired seed-level intervals."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from summarize_interpolation import (
    _audit,
    _comparison_rows,
    _load_dense,
)

from iqp_cond.artifacts import write_immutable_json
from iqp_cond.interpolation import HELDOUT_ANGLE_FRACTIONS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    parser.add_argument(
        "--interpolation-runs",
        type=Path,
        default=Path("results/interpolation/runs"),
    )
    parser.add_argument(
        "--legacy-audit",
        type=Path,
        default=Path("results/interpolation/summary/audit.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/arxiv-foundations-v1/gaussian/paired-evidence.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_811)
    return parser


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _interval(
    values: dict[int, float], replicates: int, seed: int
) -> dict[str, object]:
    ordered_seeds = tuple(sorted(values))
    observations = np.asarray([values[item] for item in ordered_seeds])
    rng = np.random.default_rng(seed)
    draws = observations[
        rng.integers(0, observations.size, size=(replicates, observations.size))
    ].mean(axis=1)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return {
        "estimate": float(np.mean(observations)),
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": 0.95,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "replication_unit": "optimizer_seed",
        "paired_seed_values": {
            str(item): float(values[item]) for item in ordered_seeds
        },
    }


def _training_control(runs: Path, replicates: int, seed: int) -> dict[str, object]:
    improvements: dict[str, dict[int, float]] = {
        "shell_tvd": {},
        "bitstring_tvd": {},
    }
    for optimizer_seed in range(10):
        fixed = json.loads(
            (runs / f"angle_complete_seed{optimizer_seed}.json").read_text()
        )
        no_condition = json.loads(
            (runs / f"no_condition_complete_seed{optimizer_seed}.json").read_text()
        )
        for metric in improvements:
            fixed_mean = float(
                np.mean([condition[metric] for condition in fixed["conditions"]])
            )
            baseline_mean = float(
                np.mean([condition[metric] for condition in no_condition["conditions"]])
            )
            improvements[metric][optimizer_seed] = baseline_mean - fixed_mean
    intervals = {
        metric: _interval(values, replicates, seed)
        for metric, values in improvements.items()
    }
    return {
        "comparison": "no_condition_minus_product_state_conditioned",
        "conditions": ["A", "B"],
        "intervals": intervals,
        "profile_conditional_control_claim": intervals["shell_tvd"]["lower"] > 0.0,
        "bitstring_result_is_secondary": True,
    }


def _seed_method_means(
    rows: list[dict[str, float | int | str]], metric: str
) -> dict[tuple[str, int], float]:
    result = {}
    for method in ("dense", "nearest", "endpoint"):
        for optimizer_seed in range(10):
            values = [
                float(row[metric])
                for row in rows
                if row["method"] == method and row["seed"] == optimizer_seed
            ]
            if len(values) != 4:
                raise ValueError("Gaussian held-out rows do not form 10x4 blocks")
            result[(method, optimizer_seed)] = float(np.mean(values))
    return result


def _interpolation(
    rows: list[dict[str, float | int | str]],
    audit: dict[str, object],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    comparisons = {}
    for metric in ("shell_tvd", "bitstring_tvd"):
        means = _seed_method_means(rows, metric)
        for baseline in ("endpoint", "nearest"):
            values = {
                optimizer_seed: means[(baseline, optimizer_seed)]
                - means[("dense", optimizer_seed)]
                for optimizer_seed in range(10)
            }
            comparisons[f"{metric}_vs_{baseline}"] = _interval(values, replicates, seed)
    lookup_wins = [
        float(item["angle_fraction_pi"])
        for item in audit["comparisons"]
        if item["test"] == "continuous_conditioning"
        and item["metric"] == "shell_tvd"
        and not item["passed"]
    ]
    shell_endpoint = comparisons["shell_tvd_vs_endpoint"]
    shell_lookup = comparisons["shell_tvd_vs_nearest"]
    bitstring_lookup = comparisons["bitstring_tvd_vs_nearest"]
    return {
        "comparison_interpretation": "positive values favor dense conditioning",
        "paired_intervals": comparisons,
        "aggregate_profile_interpolation_claim": (
            shell_endpoint["lower"] > 0.0 and shell_lookup["lower"] > 0.0
        ),
        "lookup_winning_profile_conditions_pi": lookup_wins,
        "bitstring_interpolation_advantage": bitstring_lookup["lower"] > 0.0,
        "held_out_angle_fractions_pi": [
            float(value) for value in HELDOUT_ANGLE_FRACTIONS
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dense = _load_dense(args.interpolation_runs)
    rows = _comparison_rows(dense, args.interpolation_runs, args.runs)
    regenerated_audit, _ = _audit(rows)
    legacy_audit = json.loads(args.legacy_audit.read_text())
    if regenerated_audit != legacy_audit:
        raise RuntimeError("regenerated Gaussian metrics differ from current summaries")
    payload = {
        "schema_version": 1,
        "study": "archived_gaussian_hamming_profile",
        "legacy_summary_exact_agreement": True,
        "source_hashes": {
            "legacy_audit": _sha(args.legacy_audit),
            **{
                f"dense_seed_{seed}": _sha(
                    args.interpolation_runs / f"interpolation_complete_seed{seed}.json"
                )
                for seed in range(10)
            },
        },
        "conditional_control": _training_control(
            args.runs, args.bootstrap_replicates, args.bootstrap_seed
        ),
        "held_out_interpolation": _interpolation(
            rows,
            regenerated_audit,
            args.bootstrap_replicates,
            args.bootstrap_seed,
        ),
    }
    write_immutable_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict paired aggregation for the exact-TVD ancilla campaign."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import read_immutable_json, sha256_file
from .exact_tvd_campaign import RunIdentity, production_identities


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("probability vectors must be aligned and one-dimensional")
    return float(0.5 * np.abs(p - q).sum())


def interpolate_probabilities(
    left: np.ndarray, right: np.ndarray, weight: float
) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("interpolation weight must lie in [0, 1]")
    result = (1.0 - weight) * np.asarray(left) + weight * np.asarray(right)
    if np.any(result < -1e-12) or not np.isclose(result.sum(), 1.0):
        raise ValueError("interpolation inputs must be normalized probabilities")
    return np.maximum(result, 0.0) / result.sum()


def bootstrap_ci(
    values: Iterable[float],
    *,
    seed: int,
    draws: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(tuple(values), dtype=np.float64)
    if values.size < 1:
        raise ValueError("bootstrap requires at least one replication unit")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, values.size), replace=True)
    statistics = np.median(samples, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(v) for v in np.quantile(statistics, (alpha, 1.0 - alpha)))


def summarize_seed_values(values: Iterable[float], *, seed: int) -> dict[str, Any]:
    values = np.asarray(tuple(values), dtype=np.float64)
    if values.size < 1:
        raise ValueError("summary requires at least one seed")
    return {
        "n_seeds": int(values.size),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "bootstrap_ci": list(bootstrap_ci(values, seed=seed)),
    }


@dataclass(frozen=True, slots=True)
class RunRecord:
    identity: dict[str, Any]
    conditions: np.ndarray
    probabilities: np.ndarray
    targets: np.ndarray
    metrics: tuple[dict[str, Any], ...]
    envelope_path: Path

    @property
    def key(self) -> tuple[Any, ...]:
        value = self.identity
        return (
            value["dataset"],
            int(value["visible_qubits"]),
            int(value["ancillas"]),
            int(value["seed"]),
            value["family"],
            value["condition_role"],
        )

    def at(self, condition: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        matches = np.flatnonzero(np.isclose(self.conditions, condition))
        if matches.size != 1:
            raise ValueError(f"condition {condition} is absent or duplicated")
        index = int(matches[0])
        return self.probabilities[index], self.targets[index], self.metrics[index]


def _identity_tuple(identity: RunIdentity) -> tuple[Any, ...]:
    return (
        identity.dataset,
        identity.visible_qubits,
        identity.ancillas,
        identity.seed,
        identity.family,
        identity.condition_role,
    )


def _resolve_probability_path(directory: Path, raw_path: str | Path) -> Path:
    """Resolve a probability artifact after a cross-host artifact sync."""
    path = Path(raw_path)
    if not path.is_absolute():
        path = directory.parent / path
    if path.exists():
        return path
    if Path(raw_path).is_absolute():
        fallback = directory.parent / "probabilities" / path.name
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"probability artifact is missing: {path}")


def load_records(
    directory: Path, config: dict[str, Any], *, require_complete: bool = True
) -> list[RunRecord]:
    """Verify evaluation envelopes and probability artifacts.

    By default the full manifest is required.  ``require_complete=False`` is
    an explicit exploratory mode for aggregating a manifest-valid subset while
    it is still running; it does not relax artifact, identity, or duplicate
    validation.
    """
    records: list[RunRecord] = []
    for envelope_path in sorted(directory.glob("*.json")):
        payload = read_immutable_json(envelope_path)
        if payload.get("stage") != "evaluation":
            raise ValueError(f"not an evaluation envelope: {envelope_path}")
        identity = payload.get("run_identity")
        if not isinstance(identity, dict):
            raise ValueError(f"missing run identity: {envelope_path}")
        probability_path = _resolve_probability_path(
            directory, payload["probabilities_path"]
        )
        if sha256_file(probability_path) != payload["probabilities_sha256"]:
            raise ValueError(f"probability digest mismatch: {probability_path}")
        with np.load(probability_path, allow_pickle=False) as archive:
            conditions = np.asarray(archive["conditions"], dtype=np.float64)
            probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
            targets = np.asarray(archive["target_probabilities"], dtype=np.float64)
        if probabilities.shape != targets.shape or probabilities.ndim != 2:
            raise ValueError(f"probability shapes differ: {probability_path}")
        if conditions.shape != (probabilities.shape[0],):
            raise ValueError(f"condition shape differs: {probability_path}")
        if not np.allclose(probabilities.sum(axis=1), 1.0):
            raise ValueError(
                f"model probabilities are not normalized: {probability_path}"
            )
        if not np.allclose(targets.sum(axis=1), 1.0):
            raise ValueError(
                f"target probabilities are not normalized: {probability_path}"
            )
        metrics = tuple(payload["metrics"])
        if len(metrics) != len(conditions):
            raise ValueError(f"metric rows differ: {envelope_path}")
        records.append(
            RunRecord(
                identity,
                conditions,
                probabilities,
                targets,
                metrics,
                envelope_path,
            )
        )
    actual = {record.key for record in records}
    expected = {_identity_tuple(run) for run in production_identities(config)}
    if len(actual) != len(records):
        raise ValueError("duplicate run identity")
    unexpected = actual - expected
    if unexpected:
        raise ValueError(
            f"campaign contains identities outside manifest: extra={len(unexpected)}"
        )
    if require_complete and actual != expected:
        raise ValueError(f"campaign incomplete: missing={len(expected - actual)}")
    if not records:
        raise ValueError("campaign contains no evaluation envelopes")
    return records


def _condition_role(spec: dict[str, Any], condition: float) -> str:
    fields = (("train", "training_conditions"), ("held_out", "held_out_conditions"))
    for prefix, field in fields:
        for index, value in enumerate(spec[field]):
            if np.isclose(value, condition):
                return f"{prefix}_{index}"
    raise ValueError("condition is outside the manifest")


def _adjacent(spec: dict[str, Any], condition: float) -> tuple[float, float, float]:
    training = np.asarray(spec["training_conditions"], dtype=np.float64)
    right = int(np.searchsorted(training, condition))
    if right == 0 or right == len(training):
        raise ValueError("held-out condition lacks adjacent training conditions")
    left_value, right_value = training[right - 1], training[right]
    weight = float((condition - left_value) / (right_value - left_value))
    return float(left_value), float(right_value), weight


def comparison_rows(
    records: list[RunRecord], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return per-condition exact comparisons before seed averaging."""
    by = {record.key: record for record in records}
    rows: list[dict[str, Any]] = []
    for shared in records:
        identity = shared.identity
        if identity["family"] != "conditioned":
            continue
        dataset = identity["dataset"]
        n_visible = int(identity["visible_qubits"])
        ancillas = int(identity["ancillas"])
        seed = int(identity["seed"])
        spec = config["datasets"][dataset]
        for condition in shared.conditions:
            prediction, target, metric = shared.at(float(condition))
            is_training = any(
                np.isclose(condition, value) for value in spec["training_conditions"]
            )
            split = "training" if is_training else "held_out"
            role = _condition_role(spec, float(condition))
            oracle = by[(dataset, n_visible, ancillas, seed, "independent", role)]
            oracle_prediction, oracle_target, _ = oracle.at(float(condition))
            if not np.allclose(target, oracle_target):
                raise ValueError("paired runs use different test targets")
            conditioned_tvd = tvd(target, prediction)
            oracle_tvd = tvd(target, oracle_prediction)
            row = {
                "dataset": dataset,
                "visible_qubits": n_visible,
                "ancillas": ancillas,
                "seed": seed,
                "condition": float(condition),
                "split": split,
                "conditioned_tvd": conditioned_tvd,
                "oracle_tvd": oracle_tvd,
                "sharing_cost": conditioned_tvd - oracle_tvd,
                "profile_tvd": float(metric.get("profile_tvd", np.nan)),
                "interpolation_tvd": float("nan"),
                "generalization_advantage": float("nan"),
            }
            if split == "held_out":
                left, right, weight = _adjacent(spec, float(condition))
                left_run = by[
                    (
                        dataset,
                        n_visible,
                        ancillas,
                        seed,
                        "independent",
                        _condition_role(spec, left),
                    )
                ]
                right_run = by[
                    (
                        dataset,
                        n_visible,
                        ancillas,
                        seed,
                        "independent",
                        _condition_role(spec, right),
                    )
                ]
                interpolated = interpolate_probabilities(
                    left_run.at(left)[0], right_run.at(right)[0], weight
                )
                interpolation_tvd = tvd(target, interpolated)
                row["interpolation_tvd"] = interpolation_tvd
                row["generalization_advantage"] = interpolation_tvd - conditioned_tvd
            rows.append(row)
    return rows


def _subgroups(dataset: str, condition: float, split: str) -> tuple[str, ...]:
    groups = [split]
    if dataset == "curie_weiss" and split == "held_out":
        groups.append("curie_full_held_out")
        if np.isclose(condition, 1.0):
            groups.append("curie_beta_1")
        if any(np.isclose(condition, value) for value in (0.8, 1.0, 1.2)):
            groups.append("curie_critical_window")
    return tuple(groups)


def aggregate(records: list[RunRecord], config: dict[str, Any]) -> dict[str, Any]:
    """Average conditions within seed, then summarize paired seed statistics."""
    rows = comparison_rows(records, config)
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for group in _subgroups(row["dataset"], row["condition"], row["split"]):
            buckets[
                (
                    row["dataset"],
                    row["visible_qubits"],
                    row["ancillas"],
                    row["seed"],
                    group,
                )
            ].append(row)
    metrics = (
        "conditioned_tvd",
        "oracle_tvd",
        "sharing_cost",
        "interpolation_tvd",
        "generalization_advantage",
        "profile_tvd",
        "heldout_minus_training_tvd",
    )
    seed_rows = []
    for key, group_rows in sorted(buckets.items()):
        entry = dict(
            zip(
                ("dataset", "visible_qubits", "ancillas", "seed", "group"),
                key,
                strict=True,
            )
        )
        for metric in metrics:
            if metric == "heldout_minus_training_tvd":
                continue
            values = np.asarray([row[metric] for row in group_rows], dtype=np.float64)
            entry[metric] = (
                float(np.nanmean(values))
                if not np.all(np.isnan(values))
                else float("nan")
            )
        seed_rows.append(entry)
    base_seed_index = {
        (
            row["dataset"],
            row["visible_qubits"],
            row["ancillas"],
            row["seed"],
            row["group"],
        ): row
        for row in seed_rows
    }
    for row in seed_rows:
        row["heldout_minus_training_tvd"] = float("nan")
        if row["group"] == "held_out":
            training = base_seed_index[
                (
                    row["dataset"],
                    row["visible_qubits"],
                    row["ancillas"],
                    row["seed"],
                    "training",
                )
            ]
            row["heldout_minus_training_tvd"] = (
                row["conditioned_tvd"] - training["conditioned_tvd"]
            )
    summary_buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        summary_buckets[
            (row["dataset"], row["visible_qubits"], row["ancillas"], row["group"])
        ].append(row)
    summaries: dict[str, Any] = {}
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    for offset, (key, group_rows) in enumerate(sorted(summary_buckets.items())):
        name = "/".join(map(str, key))
        summaries[name] = {
            metric: summarize_seed_values(
                [row[metric] for row in group_rows if np.isfinite(row[metric])],
                seed=bootstrap_seed + offset * len(metrics) + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
            if any(np.isfinite(row[metric]) for row in group_rows)
        }
    seed_index = {
        (
            row["dataset"],
            row["visible_qubits"],
            row["ancillas"],
            row["seed"],
            row["group"],
        ): row
        for row in seed_rows
    }
    ancilla_effects = []
    for row in seed_rows:
        if row["ancillas"] == 0:
            continue
        base = seed_index[
            (row["dataset"], row["visible_qubits"], 0, row["seed"], row["group"])
        ]
        ancilla_effects.append(
            {
                "dataset": row["dataset"],
                "visible_qubits": row["visible_qubits"],
                "ancillas": row["ancillas"],
                "seed": row["seed"],
                "group": row["group"],
                "conditioned_tvd_change_vs_r0": row["conditioned_tvd"]
                - base["conditioned_tvd"],
                "profile_tvd_change_vs_r0": row["profile_tvd"] - base["profile_tvd"],
            }
        )
    return {
        "schema_version": 1,
        "campaign": "exact_tvd_ancilla_v1",
        "condition_rows": rows,
        "seed_rows": seed_rows,
        "ancilla_effect_rows": ancilla_effects,
        "summaries": summaries,
    }


def write_summary_tables(
    summary: dict[str, Any], json_path: Path, csv_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    rows = summary["seed_rows"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

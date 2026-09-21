"""Deterministic sensitivity summaries for scalar Gaussian audit rows."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import analysis, campaign, protocol

MAIN_ENCODERS = (
    "broadcast_affine",
    "qubit_affine",
    "anti_block",
    "condition_blind",
)
CONFIGURATION_FIELDS = (
    "protocol",
    "anchor_count",
    "placement",
    "layout_seed",
    "budget_mode",
    "optimizer_seed",
    "hyperparameter_id",
    "model_kind",
)
METRIC = "bitstring_tvd"


def _read_csv(path: Path) -> list[dict[str, str | None]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return [
            {
                key: value if value is not None and value.strip() else None
                for key, value in row.items()
            }
            for row in csv.DictReader(stream)
        ]


def _number(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _summary(values: list[float]) -> dict[str, int | float]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _sorted_protocols(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["protocol"]) for row in rows if row.get("protocol")})


def _sorted_categories(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["category"]) for row in rows if row.get("category")})


def equal_configuration_sensitivity(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if row.get("protocol") not in campaign.MIRROR_PROTOCOLS
        and row.get("encoder") in MAIN_ENCODERS
        and _number(row.get(METRIC)) is not None
    ]
    result: dict[str, Any] = {}
    for encoder in MAIN_ENCODERS:
        encoder_rows = [row for row in usable if row.get("encoder") == encoder]
        categories: dict[str, Any] = {}
        for category in sorted(
            {str(row["category"]) for row in encoder_rows if row.get("category")}
        ):
            subset = [row for row in encoder_rows if row.get("category") == category]
            configurations: dict[tuple[Any, ...], list[float]] = defaultdict(list)
            for row in subset:
                identity = tuple(row.get(field) for field in CONFIGURATION_FIELDS)
                value = _number(row.get(METRIC))
                if value is not None:
                    configurations[identity].append(value)
            configuration_means = [
                float(np.mean(values))
                for _, values in sorted(
                    configurations.items(),
                    key=lambda item: tuple(str(v) for v in item[0]),
                )
            ]
            raw = [_number(row.get(METRIC)) for row in subset]
            raw_values = [float(value) for value in raw if value is not None]
            categories[category] = {
                "n_rows": len(raw_values),
                "n_configurations": len(configuration_means),
                "raw_row_mean": float(np.mean(raw_values)),
                "raw_row_median": float(np.median(raw_values)),
                "equal_configuration_mean": float(np.mean(configuration_means)),
                "equal_configuration_median": float(np.median(configuration_means)),
                "equal_configuration_max": float(np.max(configuration_means)),
            }
        result[encoder] = categories
    return {
        "estimand": (
            "Raw-row summaries weight condition rows equally within each group. "
            "Equal-configuration summaries average conditions within configurations, "
            "then weight configurations equally. A configuration is identified by "
            "protocol, anchor_count, placement, layout_seed, budget_mode, "
            "optimizer_seed, hyperparameter_id, and model_kind. "
            "Mirrored protocols are excluded."
        ),
        "encoders": result,
    }


def complement_covariance_summaries(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if row.get("encoder") in MAIN_ENCODERS
        and _number(row.get("complement_covariance_error")) is not None
    ]

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        values = [float(_number(row["complement_covariance_error"])) for row in subset]
        pairs = sorted(
            {
                (float(row["condition"]), 1.0 - float(row["condition"]))
                for row in subset
                if _number(row.get("condition")) is not None
            }
        )
        return {
            **_summary(values),
            "protocols": _sorted_protocols(subset),
            "categories": _sorted_categories(subset),
            "condition_pairs": [
                {"c": float(c), "complement": float(complement)}
                for c, complement in pairs
            ],
        }

    overall: dict[str, Any] = {}
    stratified: dict[str, Any] = {}
    for encoder in MAIN_ENCODERS:
        subset = [row for row in usable if row.get("encoder") == encoder]
        if subset:
            overall[encoder] = summarize(subset)
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            groups[
                (str(row.get("protocol") or ""), str(row.get("category") or ""))
            ].append(row)
        stratified[encoder] = [
            {"protocol": protocol_name, "category": category, **summarize(group)}
            for (protocol_name, category), group in sorted(groups.items())
        ]
    return {"overall": overall, "by_protocol_category": stratified}


def condition_blind_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("encoder") == "condition_blind"
        and row.get("protocol") not in campaign.MIRROR_PROTOCOLS
        and _number(row.get(METRIC)) is not None
    ]
    groups: dict[tuple[str, str, str], dict[tuple[Any, ...], list[float]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for row in selected:
        group = (
            str(row.get("protocol") or ""),
            str(row.get("anchor_count") or ""),
            str(row.get("category") or ""),
        )
        identity = tuple(row.get(field) for field in CONFIGURATION_FIELDS)
        value = _number(row.get(METRIC))
        if value is not None:
            groups[group][identity].append(value)
    result = []
    for (protocol_name, anchor_count, category), configurations in sorted(
        groups.items()
    ):
        means = [float(np.mean(values)) for values in configurations.values()]
        result.append(
            {
                "protocol": protocol_name,
                "anchor_count": int(anchor_count)
                if anchor_count.isdigit()
                else anchor_count,
                "category": category,
                "n_configurations": len(means),
                "equal_configuration_mean": float(np.mean(means)),
                "equal_configuration_median": float(np.median(means)),
                "equal_configuration_max": float(np.max(means)),
            }
        )
    return result


def direct_reference_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values_by_condition: dict[float, list[float]] = defaultdict(list)
    all_values: list[float] = []
    for row in rows:
        value = _number(row.get(METRIC))
        condition = _number(row.get("condition"))
        if value is None:
            continue
        all_values.append(value)
        if condition is not None:
            values_by_condition[condition].append(value)
    return {
        "global": _summary(all_values)
        if all_values
        else {"n": 0, "mean": None, "median": None, "max": None},
        "by_condition": [
            {"condition": float(condition), **_summary(values)}
            for condition, values in sorted(values_by_condition.items())
        ],
    }


def classical_interpolation_summaries() -> dict[str, Any]:
    grid = protocol.evaluation_grid("audit")
    rows: list[dict[str, Any]] = []
    for protocol_name in campaign.ORIGINAL_PROTOCOLS:
        for anchor_count in protocol.ANCHOR_COUNTS:
            anchors = protocol.anchor_conditions(protocol_name, anchor_count, "even")
            anchor_distributions = protocol.target_probabilities(anchors)
            rows.extend(
                analysis.interpolation_rows(
                    anchors,
                    anchor_distributions,
                    grid,
                    protocol_name,
                    "audit",
                )
            )
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    condition_grouped: dict[tuple[str, int, str, float], list[float]] = defaultdict(
        list
    )
    for row in rows:
        key = (row["protocol"], int(row["anchor_count"]), row["category"])
        grouped[key].append(row)
        condition_grouped[(*key, float(row["condition"]))].append(float(row[METRIC]))
    aggregates = [
        {
            "protocol": protocol_name,
            "anchor_count": int(anchor_count),
            "placement": "even",
            "category": category,
            **_summary([float(row[METRIC]) for row in group]),
        }
        for (protocol_name, anchor_count, category), group in sorted(grouped.items())
    ]
    by_condition = [
        {
            "protocol": protocol_name,
            "anchor_count": int(anchor_count),
            "placement": "even",
            "category": category,
            "condition": float(condition),
            **_summary(values),
        }
        for (protocol_name, anchor_count, category, condition), values in sorted(
            condition_grouped.items()
        )
    ]
    return {
        "split": "audit",
        "grid_size": int(grid.size),
        "protocols": list(campaign.ORIGINAL_PROTOCOLS),
        "anchor_counts": list(protocol.ANCHOR_COUNTS),
        "placement": "even",
        "aggregates": aggregates,
        "by_condition": by_condition,
    }


def summarize(
    audit_rows: list[dict[str, Any]], direct_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "equal_configuration_sensitivity": equal_configuration_sensitivity(audit_rows),
        "complement_covariance": complement_covariance_summaries(audit_rows),
        "condition_blind": condition_blind_summaries(audit_rows),
        "direct_reference": direct_reference_summaries(direct_rows),
        "classical_anchor_interpolation": classical_interpolation_summaries(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-rows", type=Path, required=True)
    parser.add_argument("--direct-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(_read_csv(args.audit_rows), _read_csv(args.direct_rows))
    analysis._atomic_json(args.output, result)


if __name__ == "__main__":
    main()

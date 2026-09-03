"""Calibration and optimizer-pilot selection for exact-TVD campaign.

The selectors are intentionally pure: workers may run on separate machines and
append JSON rows, while this module provides one deterministic decision point.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_immutable_json
from .exact_tvd_campaign import manifest_digest


def calibration_passes(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    """Return whether one estimator cell satisfies every preregistered gate."""
    return (
        float(row["pauli_expectation_rmse"]) <= float(spec["max_pauli_rmse"])
        and float(row["mmd_cv"]) <= float(spec["max_mmd_cv"])
        and float(row["gradient_cosine"]) >= float(spec["min_gradient_cosine"])
        and float(row["relative_gradient_error"])
        <= float(spec["max_relative_gradient_error"])
    )


def select_n_samples(rows: Iterable[dict[str, Any]], manifest: dict[str, Any]) -> int:
    """Select the smallest candidate passing all calibration cells.

    A candidate must pass every dataset/ancilla/seed cell; missing cells are a
    hard failure, preventing accidental production from a partial calibration.
    """
    rows = list(rows)
    spec = manifest["estimator_calibration"]
    candidates = [int(x) for x in spec["candidates"]]
    required = {
        (d, int(a), int(s), int(n))
        for d in ("gaussian", "curie_weiss", "mfc")
        for a in spec["ancillas"]
        for s in spec["development_seeds"]
        for n in candidates
    }
    present = {
        (str(r["dataset"]), int(r["ancillas"]), int(r["seed"]), int(r["n_samples"]))
        for r in rows
    }
    missing = required - present
    if missing:
        raise ValueError(
            f"incomplete estimator calibration: {len(missing)} cells missing"
        )
    for n in candidates:
        cells = [r for r in rows if int(r["n_samples"]) == n]
        if all(calibration_passes(r, spec) for r in cells):
            return n
    raise RuntimeError(
        "no estimator n_samples candidate satisfies all calibration gates"
    )


def matched_cell_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["dataset"]),
        int(row["visible_qubits"]),
        int(row["ancillas"]),
        int(row["seed"]),
    )


def select_optimizer(
    rows: Iterable[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Select Adam or one global schedule-free candidate under preregistered gates."""
    rows = list(rows)
    spec = manifest["optimizer_pilot"]
    adam = [r for r in rows if str(r["optimizer"]) == "adam_cosine"]
    sf = [r for r in rows if str(r["optimizer"]) == "schedule_free_adamw"]
    configs = {
        (float(c["learning_rate"]), float(c["b1"])) for c in spec["schedule_free_adamw"]
    }
    keys = {matched_cell_key(r) for r in adam}
    if len(keys) != int(spec["matched_cells"]):
        raise ValueError("optimizer pilot must contain exactly 18 Adam matched cells")
    for cfg in configs:
        got = {
            matched_cell_key(r)
            for r in sf
            if (float(r["learning_rate"]), float(r["b1"])) == cfg
        }
        if got != keys:
            raise ValueError(f"incomplete schedule-free pilot for {cfg}")
    # A candidate comparison is paired by cell; median improvement is relative.
    adam_by = {matched_cell_key(r): float(r["validation_tvd"]) for r in adam}
    scores = []
    for cfg in sorted(configs):
        sf_by = {
            matched_cell_key(r): float(r["validation_tvd"])
            for r in sf
            if (float(r["learning_rate"]), float(r["b1"])) == cfg
        }
        improvements = [(adam_by[k] - sf_by[k]) / max(adam_by[k], 1e-15) for k in keys]
        scores.append(
            (float(np.median(improvements)), sum(x > 0 for x in improvements), cfg)
        )
    scores.sort(reverse=True)
    best = scores[0]
    adopt = best[0] >= float(spec["minimum_median_improvement"]) and best[1] >= int(
        spec["minimum_matched_cell_wins"]
    )
    return {
        "optimizer": "schedule_free_adamw" if adopt else "adam_cosine",
        "learning_rate": best[2][0] if adopt else 0.01,
        "b1": best[2][1] if adopt else 0.9,
        "median_improvement": best[0],
        "wins": best[1],
        "candidates": [
            {
                "learning_rate": x[2][0],
                "b1": x[2][1],
                "median_improvement": x[0],
                "wins": x[1],
            }
            for x in scores
        ],
    }


def write_selection(
    path: Path,
    manifest: dict[str, Any],
    calibration: dict[str, Any],
    optimizer: dict[str, Any],
    preflights: Sequence[dict[str, Any]],
) -> None:
    """Write a content-addressed, create-once protocol-selection artifact."""
    selected_n_samples = int(calibration["selected_n_samples"])
    optimizer_name = str(optimizer["optimizer"])
    settings = {
        "learning_rate": float(optimizer["learning_rate"]),
        "b1": float(optimizer["b1"]),
        "b2": float(manifest["optimizer_pilot"]["b2"]),
        "warmup_steps": int(manifest["optimizer_pilot"]["warmup_steps"]),
        "final_learning_rate": 0.001,
    }
    valid_preflights = [
        value
        for value in preflights
        if value.get("passed") is True
        and int(value.get("total_qubits", 0)) == 26
        and int(value.get("peak_rss_bytes", 2**63))
        < int(manifest["evaluation"]["maximum_exact_peak_bytes"])
        and value.get("configuration_digest") == manifest_digest(manifest)
    ]
    if not valid_preflights:
        raise ValueError("production selection requires a passing 26-qubit preflight")
    payload = {
        "schema_version": 1,
        "campaign": "exact_tvd_ancilla_v1",
        "status": "passed",
        "configuration_digest": manifest_digest(manifest),
        "n_samples": selected_n_samples,
        "optimizer": optimizer_name,
        "optimizer_settings": settings,
        "exact_preflights": valid_preflights,
        "calibration": calibration,
        "optimizer_pilot": optimizer,
    }
    if path.exists():
        raise FileExistsError(f"immutable selection already exists: {path}")
    write_immutable_json(path, payload)

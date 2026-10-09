"""Pure selection and objective helpers for the CW follow-up."""

from __future__ import annotations

import numpy as np

from .cw_followup_config import C_STAR


def _trap(y, x):
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def robust_score(errors, betas):
    e = np.asarray(errors, float)
    b = np.asarray(betas, float)
    if e.ndim != 1 or b.ndim != 1 or not e.size or e.size != b.size:
        raise ValueError("errors and betas must be nonempty matched vectors")
    if (
        not np.all(np.isfinite(e))
        or not np.all(np.isfinite(b))
        or np.any(e < 0)
        or np.any(np.diff(b) <= 0)
    ):
        raise ValueError(
            "errors must be finite nonnegative and betas strictly increasing"
        )
    if e.size == 1:
        return float(e[0])
    if b[-1] == b[0]:
        return float(np.max(e))
    return float((2 * np.max(e) + _trap(e, b) / (b[-1] - b[0])) / 3)


def curve_scores(tvd, abs_m_error, c_error, betas):
    t = np.asarray(tvd, float)
    m = np.asarray(abs_m_error, float)
    c = np.asarray(c_error, float) / C_STAR
    return {
        "J": robust_score(t, betas),
        "P": (robust_score(m, betas) + robust_score(c, betas)) / 2,
        "max_tvd": float(np.max(t)),
        "mean_tvd": float(
            t[0] if t.size == 1 else _trap(t, betas) / (betas[-1] - betas[0])
        ),
    }


def select_checkpoint(records, tolerance=0.01):
    valid = sorted(
        [
            r
            for r in records
            if r.get("valid", True) and np.isfinite(r["J"]) and np.isfinite(r["P"])
        ],
        key=lambda r: (r["J"], r["P"], r.get("trial_id", 0)),
    )
    best = min(valid, key=lambda r: (r["J"], r["step"], r["start"]))
    eligible = [r for r in valid if r["J"] <= best["J"] + tolerance]
    return min(eligible, key=lambda r: (r["P"], r["J"], r["step"], r["start"]))


def shortlist(records):
    import json

    valid = sorted(
        [
            r
            for r in records
            if r.get("valid", True) and np.isfinite(r["J"]) and np.isfinite(r["P"])
        ],
        key=lambda r: (r["J"], r["P"], r.get("trial_id", 0)),
    )
    unique = {}
    for r in valid:
        unique.setdefault(
            json.dumps(r.get("config"), sort_keys=True, separators=(",", ":")), r
        )
    valid = list(unique.values())
    if len(valid) < 3:
        raise ValueError(
            "shortlist requires at least three valid distinct configurations"
        )
    primary = sorted(valid, key=lambda r: (r["J"], r["P"], r.get("trial_id", 0)))
    chosen = [primary[0]]
    eligible = [r for r in primary if r["J"] <= primary[0]["J"] + 0.01]
    eligible_choice = min(
        eligible, key=lambda r: (r["P"], r["J"], r.get("trial_id", 0))
    )
    if eligible_choice not in chosen:
        chosen.append(eligible_choice)
    for r in primary:
        if r not in chosen and len(chosen) < 3:
            chosen.append(r)
    best_p = min(valid, key=lambda r: (r["P"], r["J"], r.get("trial_id", 0)))
    if best_p not in chosen and len(chosen) < 4:
        chosen.append(best_p)
    return chosen


def choose_confirmed(records, tolerance=0.01):
    valid = [
        r
        for r in records
        if r.get("valid", True) and np.isfinite(r["J"]) and np.isfinite(r["P"])
    ]
    if not valid:
        raise ValueError("no valid confirmation records")
    best = min(valid, key=lambda r: (r["J"], r.get("trial_id", 0)))
    return min(
        (r for r in valid if r["J"] <= best["J"] + tolerance),
        key=lambda r: (r["P"], r["J"], r.get("trial_id", 0)),
    )

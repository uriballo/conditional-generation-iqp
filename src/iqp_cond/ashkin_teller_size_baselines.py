"""Size-aware classical baselines for periodic Ashkin--Teller rings.

This module mirrors :mod:`ashkin_teller_baselines` while deriving the number
of spins and flip-orbit count from the supplied samples.  It is intended for
the n=12 follow-up; n=8 is supported to make compatibility checks explicit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize


def _orbit_table(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n % 2 or n < 4:
        raise ValueError("n must be an even number of at least 4")
    m = n // 2
    flips = np.asarray([0, (1 << m) - 1, ((1 << m) - 1) << m, (1 << n) - 1])
    states = 1 << n
    mapping = np.empty(states, dtype=np.int64)
    reps: list[int] = []
    seen: set[int] = set()
    for x in range(states):
        if x in seen:
            continue
        orbit = np.unique(x ^ flips)
        reps.append(int(orbit.min()))
        mapping[orbit] = len(reps) - 1
        seen.update(int(v) for v in orbit)
    return mapping, np.asarray(reps, dtype=np.int64)


def _infer_n(samples: np.ndarray) -> int:
    x = np.asarray(samples)
    if x.ndim < 2 or x.shape[-1] % 2 or x.shape[-1] < 4:
        raise ValueError("samples must have an even final dimension of at least 4")
    return int(x.shape[-1])


def orbit_counts(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples)
    n = _infer_n(x)
    if x.ndim not in (2, 3) or not np.all((x == 0) | (x == 1)):
        raise ValueError("samples must be binary with shape (N,n) or (C,N,n)")
    mapping, _ = _orbit_table(n)
    states = x.astype(np.int64) @ (1 << np.arange(n, dtype=np.int64))
    if x.ndim == 2:
        states = states[None, :]
    out = np.zeros((states.shape[0], len(mapping) // 4), dtype=np.int64)
    for i, row in enumerate(states):
        np.add.at(out[i], mapping[row], 1)
    return out[0] if x.ndim == 2 else out


def expand_orbits(q: np.ndarray, n: int | None = None) -> np.ndarray:
    values = np.asarray(q, dtype=float)
    k = values.shape[-1]
    if n is None:
        states = 4 * k
        n = int(np.log2(states))
    mapping, _ = _orbit_table(n)
    if k != len(mapping) // 4:
        raise ValueError("orbit dimension is inconsistent with n")
    return values[..., mapping] / 4.0


def _features(conditions: np.ndarray) -> np.ndarray:
    c = np.asarray(conditions, dtype=float)
    return np.column_stack(
        (np.ones(len(c)), (c[:, 0] - 0.4) / 0.3, (c[:, 1] - 0.4) / 0.3)
    )


def composition_predict(
    conditions: np.ndarray, laws: np.ndarray, query: np.ndarray
) -> np.ndarray:
    train, values, points = map(np.asarray, (conditions, laws, query))
    if train.ndim != 2 or train.shape[1] != 2 or values.shape[0] != len(train):
        raise ValueError("conditions and laws have incompatible shapes")
    lo, hi = train[:, 0].min(), train[:, 0].max()
    blo, bhi = train[:, 1].min(), train[:, 1].max()
    edge = {(float(a), float(b)): i for i, (a, b) in enumerate(train)}
    corners = [(lo, blo), (lo, bhi), (hi, blo), (hi, bhi)]
    try:
        corner_logs = np.log(
            np.clip(values[[edge[c] for c in corners]], np.finfo(float).tiny, None)
        )
    except KeyError as exc:
        raise ValueError(
            "training conditions must contain all perimeter corners"
        ) from exc
    out = np.empty((len(points), values.shape[1]))
    for j, (a, b) in enumerate(points):
        try:
            rows = [
                edge[(float(a), blo)],
                edge[(float(a), bhi)],
                edge[(lo, float(b))],
                edge[(hi, float(b))],
            ]
        except KeyError as exc:
            raise ValueError(
                "query coordinates must have supplied perimeter edge laws"
            ) from exc
        logs = np.log(np.clip(values[rows], np.finfo(float).tiny, None))
        z = 0.5 * logs.sum(axis=0) - 0.25 * corner_logs.sum(axis=0)
        z -= z.max()
        out[j] = np.exp(z) / np.exp(z).sum()
    return out


def loglinear_objective(
    flatcoef: np.ndarray, features: np.ndarray, counts: np.ndarray, l2: float = 0.0
) -> tuple[float, np.ndarray]:
    f, c = np.asarray(features, float), np.asarray(counts, float)
    k = c.shape[1]
    coef = np.asarray(flatcoef).reshape(3, k - 1)
    empirical = c / c.sum(axis=1, keepdims=True)
    logits = np.column_stack((f @ coef, np.zeros(len(f))))
    logits -= logits.max(axis=1, keepdims=True)
    model = np.exp(logits)
    model /= model.sum(axis=1, keepdims=True)
    loss = -np.sum(
        empirical * np.log(np.clip(model, 1e-300, None)), axis=1
    ).mean() + 0.5 * l2 * np.sum(coef * coef)
    grad = (f.T @ (model[:, :-1] - empirical[:, :-1])) / len(f) + l2 * coef
    return float(loss), grad.ravel()


def fit_baselines(
    train_conditions: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    predict_conditions: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    tc, tr, va, pc = map(
        np.asarray, (train_conditions, training, validation, predict_conditions)
    )
    if (
        tr.ndim != 3
        or va.ndim != 3
        or tr.shape[0] != len(tc)
        or va.shape[0] != tr.shape[0]
        or va.shape[2] != tr.shape[2]
        or tr.shape[1] < 1
        or va.shape[1] < 1
    ):
        raise ValueError(
            "training and validation must have matching conditions and qubits, "
            "with nonempty example axes"
        )
    counts, val_counts = orbit_counts(tr), orbit_counts(va)
    k = counts.shape[1]
    masses = (6.4, 64.0, 640.0)
    lambdas = tuple(m / k for m in masses)
    comp_candidates = []
    for lam in lambdas:
        laws = (counts + lam) / (counts.sum(1, keepdims=True) + k * lam)
        pred = composition_predict(tc, laws, tc)
        nll = float(
            np.mean(
                -np.sum(
                    val_counts
                    / val_counts.sum(1, keepdims=True)
                    * np.log(np.clip(pred, 1e-300, None)),
                    axis=1,
                )
            )
        )
        comp_candidates.append((nll, lam, laws))
    _, best_lam, laws = min(comp_candidates, key=lambda z: (z[0], lambdas.index(z[1])))
    comp_pred = composition_predict(tc, laws, pc)
    feats = _features(tc)
    grid = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    candidates = []
    for l2 in grid:

        def fun(z, penalty=l2):
            return loglinear_objective(z, feats, counts, penalty)

        res = minimize(
            fun,
            np.zeros(3 * (k - 1)),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-8},
        )
        if (
            not np.isfinite(res.fun)
            or not res.success
            or np.max(np.abs(res.jac)) > 1e-6
        ):
            res = minimize(
                fun,
                res.x,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 20000, "ftol": 1e-14, "gtol": 1e-8},
            )
        if (
            not np.isfinite(res.fun)
            or not res.success
            or np.max(np.abs(res.jac)) > 1e-6
        ):
            raise RuntimeError(
                f"log-linear optimization failed for l2={l2}: {res.message}"
            )
        coef = res.x.reshape(3, k - 1)
        logits = np.column_stack((feats @ coef, np.zeros(len(tc))))
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        nll = float(
            np.mean(
                -np.sum(
                    val_counts
                    / val_counts.sum(1, keepdims=True)
                    * np.log(np.clip(p, 1e-300, None)),
                    axis=1,
                )
            )
        )
        candidates.append((nll, l2, coef, res))
    _, best_l2, coef, _ = min(candidates, key=lambda z: (z[0], grid.index(z[1])))
    logits = np.column_stack((_features(pc) @ coef, np.zeros(len(pc))))
    logits -= logits.max(1, keepdims=True)
    log_pred = np.exp(logits)
    log_pred /= log_pred.sum(1, keepdims=True)
    metadata = {
        "orbit_count": k,
        "pseudocount_masses": list(masses),
        "lambda_grid": list(lambdas),
        "loglinear_lambda_grid": list(grid),
        "selected_hyperparams": {
            "composition_lambda": best_lam,
            "loglinear_l2": best_l2,
        },
        "all_validation_NLL": {
            "composition": [x[0] for x in comp_candidates],
            "log_linear": [x[0] for x in candidates],
        },
        "convergence": [
            {
                "lambda": float(x[1]),
                "objective": float(x[3].fun),
                "success": bool(x[3].success),
                "grad_inf": float(np.max(np.abs(x[3].jac))),
                "nit": int(x[3].nit),
                "message": str(x[3].message),
            }
            for x in candidates
        ],
        "fitted_coef": coef.tolist(),
        "counts": counts.tolist(),
    }
    return {
        "log_composition": expand_orbits(comp_pred),
        "log_linear": expand_orbits(log_pred),
    }, metadata

"""Classical orbit-symmetric baselines for the Ashkin--Teller study."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

_FLIPS = np.asarray([0, 15, 240, 255], dtype=np.int64)
_LO, _HI = 0.1, 0.7


def _orbit_map() -> np.ndarray:
    mapping = np.empty(256, dtype=np.int64)
    seen: set[int] = set()
    k = 0
    for x in range(256):
        if x in seen:
            continue
        orbit = np.unique(x ^ _FLIPS)
        mapping[orbit] = k
        seen.update(int(v) for v in orbit)
        k += 1
    return mapping


_STATE_TO_ORBIT = _orbit_map()


def orbit_counts(samples: np.ndarray) -> np.ndarray:
    """Count flip-orbits, aggregating the sample axis."""
    x = np.asarray(samples)
    if x.ndim not in (2, 3) or x.shape[-1] != 8:
        raise ValueError("samples must have final dimension 8")
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("samples must be binary")
    states = np.asarray(x, dtype=np.int64) @ (1 << np.arange(8, dtype=np.int64))
    if x.ndim == 2:
        states = states[None, :]
    out_shape = (states.shape[0], 64)
    out = np.zeros(out_shape, dtype=np.int64)
    for i, row in enumerate(states):
        np.add.at(out[i], _STATE_TO_ORBIT[row], 1)
    return out[0] if x.ndim == 2 else out


def expand_orbits(q64: np.ndarray) -> np.ndarray:
    """Expand orbit probabilities to 256 state probabilities."""
    q = np.asarray(q64, dtype=float)
    if q.shape[-1] != 64:
        raise ValueError("q64 must have final dimension 64")
    return q[..., _STATE_TO_ORBIT] / 4.0


def _features(conditions: np.ndarray) -> np.ndarray:
    c = np.asarray(conditions, dtype=float)
    return np.column_stack(
        (np.ones(len(c)), (c[:, 0] - 0.4) / 0.3, (c[:, 1] - 0.4) / 0.3)
    )


def composition_predict(
    conditions: np.ndarray, smoothed_orbit_laws: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """Log-linear four-corner composition of orbit laws, normalized on 64 orbits."""
    train = np.asarray(conditions, dtype=float)
    laws = np.asarray(smoothed_orbit_laws, dtype=float)
    q = np.asarray(query, dtype=float)
    if train.shape != (16, 2) or laws.shape != (16, 64):
        raise ValueError("composition inputs must have shapes (16,2) and (16,64)")
    edge = {(float(a), float(b)): i for i, (a, b) in enumerate(train)}
    ids = [edge[(_LO, _LO)], edge[(_LO, _HI)], edge[(_HI, _LO)], edge[(_HI, _HI)]]
    logq = np.log(np.clip(laws, np.finfo(float).tiny, None))
    basis = np.asarray(q, dtype=float)
    # L(a,b)=.5 L(a,lo)+.5 L(a,hi)+.5 L(lo,b)+.5 L(hi,b)-.25 sum corners.
    out = np.empty((len(q), 64))
    for j, (a, b) in enumerate(basis):
        try:
            qa_lo, qa_hi, qlo_b, qhi_b = (
                logq[edge[key]]
                for key in (
                    (float(a), _LO),
                    (float(a), _HI),
                    (_LO, float(b)),
                    (_HI, float(b)),
                )
            )
        except KeyError as exc:
            raise ValueError(
                "query coordinates must have supplied perimeter edge laws"
            ) from exc
        out[j] = 0.5 * (qa_lo + qa_hi + qlo_b + qhi_b) - 0.25 * logq[ids].sum(axis=0)
    out -= out.max(axis=1, keepdims=True)
    return np.exp(out) / np.exp(out).sum(axis=1, keepdims=True)


def loglinear_objective(
    flatcoef: np.ndarray,
    features: np.ndarray,
    empirical_counts: np.ndarray,
    l2: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Return mean orbit NLL and analytic gradient for pooled softmax logits."""
    coef = np.asarray(flatcoef).reshape(3, 63)
    f = np.asarray(features, dtype=float)
    counts = np.asarray(empirical_counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    probs = counts / totals
    logits = np.column_stack((f @ coef, np.zeros(len(f))))
    logits -= logits.max(axis=1, keepdims=True)
    model = np.exp(logits)
    model /= model.sum(axis=1, keepdims=True)
    loss = -np.sum(probs * np.log(np.clip(model, 1e-300, None)), axis=1).mean()
    grad = (f.T @ (model[:, :63] - probs[:, :63])) / len(f)
    loss += 0.5 * l2 * np.sum(coef * coef)
    grad += l2 * coef
    return float(loss), grad.ravel()


def fit_baselines(
    train_conditions: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    predict_conditions: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit composition and pooled log-linear orbit baselines."""
    tc, tr, va, pc = map(
        np.asarray, (train_conditions, training, validation, predict_conditions)
    )
    if tr.shape[:2] != (16, 1000) or va.shape[:2] != (16, 1000):
        raise ValueError("training and validation must have shape (16, 1000, 8)")
    trc, vac = orbit_counts(tr), orbit_counts(va)
    lambdas = (0.1, 1.0, 10.0)
    comp_candidates = []
    for lam in lambdas:
        laws = (trc + lam) / (trc.sum(axis=1, keepdims=True) + 64 * lam)
        pred = composition_predict(tc, laws, tc)
        nll = float(
            np.mean(
                -np.sum(
                    vac
                    / vac.sum(1, keepdims=True)
                    * np.log(np.clip(pred, 1e-300, None)),
                    axis=1,
                )
            )
        )
        comp_candidates.append((nll, lam, laws))
    _, best_lam, laws = min(comp_candidates, key=lambda z: (z[0], lambdas.index(z[1])))
    comp_pred = composition_predict(tc, laws, pc)

    feats = _features(tc)
    log_candidates = []
    for l2 in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):

        def fun(z, penalty=l2):
            return loglinear_objective(z, feats, trc, penalty)

        res = minimize(
            fun,
            np.zeros(3 * 63),
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
        coef = res.x.reshape(3, 63)
        logits = _features(tc) @ coef
        logits = np.column_stack((logits, np.zeros(16)))
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        nll = float(
            np.mean(
                -np.sum(
                    vac / vac.sum(1, keepdims=True) * np.log(np.clip(p, 1e-300, None)),
                    axis=1,
                )
            )
        )
        log_candidates.append((nll, l2, coef, res))
    _, best_l2, coef, _ = min(
        log_candidates,
        key=lambda z: (z[0], (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1).index(z[1])),
    )
    logits = _features(pc) @ coef
    logits = np.column_stack((logits, np.zeros(len(pc))))
    logits -= logits.max(1, keepdims=True)
    log_pred = np.exp(logits)
    log_pred /= log_pred.sum(1, keepdims=True)
    metadata = {
        "selected_hyperparams": {
            "composition_lambda": best_lam,
            "loglinear_l2": best_l2,
        },
        "all_validation_NLL": {
            "composition": [x[0] for x in comp_candidates],
            "log_linear": [x[0] for x in log_candidates],
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
            for x in log_candidates
        ],
        "fitted_coef": coef.tolist(),
        "counts": trc.tolist(),
    }
    return {
        "log_composition": expand_orbits(comp_pred),
        "log_linear": expand_orbits(log_pred),
    }, metadata

"""Confirmatory audit of the Monte Carlo IQP estimator against exact values."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .encoding import (
    ANGLE_ENCODING,
    FIELD_EQUIVARIANT_ENCODING,
    SPIN_FLIP_BRANCH_ENCODING,
)
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many_jax,
    generator_characters,
)
from .mmd import (
    exact_conditional_mmd,
    exact_target_fouriers,
    mmd_mask_weights,
    mmd_operator_probability,
)
from .simulator import estimate_pauli_z

CAMPAIGN_VERSION = "estimator_audit_confirmatory_v1"
SAMPLE_BUDGETS = (256, 1024, 4096, 16384)
ENCODER_FAMILIES = ("product", "purified", "field_equivariant")
TOPOLOGIES = ("sparse", "complete")
SIGMA = 1.5
DEFAULT_MANIFEST = Path("configs/estimator_audit_confirmatory_v1.json")
EXPECTED_MANIFEST = {
    "campaign_version": CAMPAIGN_VERSION,
    "encoders": list(ENCODER_FAMILIES),
    "gradient_reference": "exact_autodiff",
    "mmd_bandwidth": SIGMA,
    "repetitions": 20,
    "root_seed": 20260825,
    "sample_budgets": list(SAMPLE_BUDGETS),
    "topologies": list(TOPOLOGIES),
    "visible_qubits": 4,
}


@dataclass(frozen=True)
class AuditCase:
    """One exact small-system model used by the estimator audit."""

    encoder: str
    topology: str
    n_visible: int
    n_ancillas: int
    encoding: str
    angles: np.ndarray
    generators: np.ndarray
    params: np.ndarray
    target_fouriers: np.ndarray

    @property
    def n_model_qubits(self) -> int:
        return self.n_visible + self.n_ancillas


def canonical_json(value: Any) -> str:
    """Serialize a result or protocol payload canonically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    """Return the digest of a canonical JSON payload."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load and strictly validate the frozen estimator-audit manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != EXPECTED_MANIFEST:
        raise ValueError("estimator-audit manifest differs from the frozen protocol")
    return payload


def _generators(n_qubits: int, topology: str) -> np.ndarray:
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology: {topology!r}")
    rows = [row for row in np.eye(n_qubits, dtype=np.float64)]
    if topology == "sparse":
        edges: Iterable[tuple[int, int]] = zip(
            range(n_qubits - 1), range(1, n_qubits), strict=True
        )
    else:
        edges = (
            (left, right)
            for left in range(n_qubits)
            for right in range(left + 1, n_qubits)
        )
    for left, right in edges:
        row = np.zeros(n_qubits, dtype=np.float64)
        row[[left, right]] = 1.0
        rows.append(row)
    return np.asarray(rows)


def _visible_marginal(probabilities: jnp.ndarray, n_visible: int) -> jnp.ndarray:
    n_states = probabilities.shape[-1]
    n_model = round(math.log2(n_states))
    n_ancillas = n_model - n_visible
    if n_ancillas < 0 or 2**n_model != n_states:
        raise ValueError("probability width is incompatible with n_visible")
    if n_ancillas == 0:
        return probabilities
    shape = (probabilities.shape[0], 2**n_ancillas, 2**n_visible)
    return probabilities.reshape(shape).sum(axis=1)


def build_case(encoder: str, topology: str) -> AuditCase:
    """Build a deterministic product, purified, or field-equivariant fixture."""
    if encoder not in ENCODER_FAMILIES:
        raise ValueError(f"unknown encoder: {encoder!r}")
    n_visible = 4
    n_ancillas = 2 if encoder == "purified" else 0
    n_model = n_visible + n_ancillas
    encoding = {
        "product": ANGLE_ENCODING,
        "purified": SPIN_FLIP_BRANCH_ENCODING,
        "field_equivariant": FIELD_EQUIVARIANT_ENCODING,
    }[encoder]
    if encoder == "field_equivariant":
        angle = -7.0 * np.pi * 0.225 / 6.0
        angles = np.full((1, n_model), angle, dtype=np.float64)
    elif encoder == "purified":
        angles = np.full((1, n_model), 0.31, dtype=np.float64)
    else:
        base = np.asarray([0.20, -0.30, 0.50, -0.40, 0.70, -0.60])
        angles = base[:n_model][None, :]
    generators = _generators(n_model, topology)
    index = np.arange(generators.shape[0], dtype=np.float64)
    params = 0.23 * np.sin(0.61 * (index + 1.0)) / np.sqrt(generators.shape[0])
    target_params = params + 0.11 * np.cos(0.43 * (index + 1.0))
    target_joint = exact_iqp_probabilities_many_jax(
        jnp.asarray(target_params),
        jnp.asarray(generators),
        jnp.asarray(angles),
        encoding,
    )
    target_visible = _visible_marginal(target_joint, n_visible)
    target_fouriers = np.asarray(exact_target_fouriers(target_visible))
    return AuditCase(
        encoder=encoder,
        topology=topology,
        n_visible=n_visible,
        n_ancillas=n_ancillas,
        encoding=encoding,
        angles=angles,
        generators=generators,
        params=params,
        target_fouriers=target_fouriers,
    )


def _observable_masks(case: AuditCase) -> tuple[jnp.ndarray, jnp.ndarray]:
    visible = jnp.asarray(enumerate_bitstrings(case.n_visible), dtype=jnp.float64)
    if case.n_ancillas:
        zeros = jnp.zeros((visible.shape[0], case.n_ancillas), dtype=visible.dtype)
        model = jnp.concatenate([visible, zeros], axis=1)
    else:
        model = visible
    return visible, model


def _mask_weights(n_visible: int, sigma: float = SIGMA) -> jnp.ndarray:
    p = mmd_operator_probability(sigma)
    weights = mmd_mask_weights(n_visible)
    return (p**weights) * ((1.0 - p) ** (n_visible - weights))


def _estimated_mmd(
    params: jnp.ndarray,
    case: AuditCase,
    key: jax.Array,
    n_samples: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _, model_masks = _observable_masks(case)
    estimate = estimate_pauli_z(
        params,
        jnp.asarray(case.generators),
        model_masks,
        jnp.asarray(case.angles),
        key,
        n_samples,
        case.encoding,
    )
    values = estimate.sample_values[0]
    sums = jnp.sum(values, axis=0)
    square_sums = jnp.sum(values**2, axis=0)
    unbiased_squares = (sums**2 - square_sums) / (n_samples * (n_samples - 1))
    target = jnp.asarray(case.target_fouriers[0])
    per_mask = unbiased_squares - 2.0 * estimate.values[0] * target + target**2
    return jnp.sum(_mask_weights(case.n_visible) * per_mask), estimate.values[0]


def _exact_mmd(params: jnp.ndarray, case: AuditCase) -> jnp.ndarray:
    result = exact_conditional_mmd(
        params,
        jnp.asarray(case.generators),
        jnp.asarray(case.angles),
        jnp.asarray(case.target_fouriers),
        (SIGMA,),
        case.encoding,
        jnp.asarray(generator_characters(case.generators)),
    )
    return result.loss


def exact_model_fouriers(case: AuditCase) -> np.ndarray:
    """Return exact visible Pauli moments for the audit model."""
    joint = exact_iqp_probabilities_many_jax(
        jnp.asarray(case.params),
        jnp.asarray(case.generators),
        jnp.asarray(case.angles),
        case.encoding,
    )
    visible = np.asarray(_visible_marginal(joint, case.n_visible))[0]
    return np.asarray(exact_target_fouriers(visible[None, :]))[0]


def audit_cell(
    case: AuditCase,
    n_samples: int,
    repetition: int,
    *,
    root_seed: int = 20260825,
) -> dict[str, Any]:
    """Compare moments, MMD, and gradients for one deterministic cell."""
    if n_samples < 2 or repetition < 0:
        raise ValueError("n_samples must be at least two and repetition nonnegative")
    encoder_index = ENCODER_FAMILIES.index(case.encoder)
    topology_index = TOPOLOGIES.index(case.topology)
    key = jax.random.fold_in(jax.random.key(root_seed), encoder_index)
    key = jax.random.fold_in(key, topology_index)
    key = jax.random.fold_in(key, n_samples)
    key = jax.random.fold_in(key, repetition)
    params = jnp.asarray(case.params)

    exact_loss = _exact_mmd(params, case)
    exact_gradient = jax.grad(_exact_mmd)(params, case)
    exact_moments = jnp.asarray(exact_model_fouriers(case))

    started = time.perf_counter()
    estimated_loss, estimated_moments = _estimated_mmd(params, case, key, n_samples)
    def estimated_objective(value: jnp.ndarray) -> jnp.ndarray:
        return _estimated_mmd(value, case, key, n_samples)[0]

    estimated_gradient = jax.grad(estimated_objective)(params)
    estimated_gradient.block_until_ready()
    runtime_seconds = time.perf_counter() - started

    moment_rmse = jnp.sqrt(jnp.mean((estimated_moments - exact_moments) ** 2))
    gradient_rmse = jnp.sqrt(jnp.mean((estimated_gradient - exact_gradient) ** 2))
    return {
        "run_identity": (
            f"{CAMPAIGN_VERSION}:{case.encoder}:{case.topology}:"
            f"S{n_samples}:r{repetition}"
        ),
        "encoder": case.encoder,
        "topology": case.topology,
        "n_visible": case.n_visible,
        "n_ancillas": case.n_ancillas,
        "n_parameters": int(case.generators.shape[0]),
        "sample_budget": n_samples,
        "repetition": repetition,
        "moment_rmse": float(moment_rmse),
        "mmd_exact": float(exact_loss),
        "mmd_estimate": float(estimated_loss),
        "mmd_absolute_error": float(jnp.abs(estimated_loss - exact_loss)),
        "gradient_rmse": float(gradient_rmse),
        "runtime_seconds": runtime_seconds,
    }


def convergence_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repetitions and fit empirical log-error/log-budget slopes."""
    summaries = []
    for encoder in ENCODER_FAMILIES:
        for topology in TOPOLOGIES:
            selected = [
                record
                for record in records
                if record["encoder"] == encoder and record["topology"] == topology
            ]
            if not selected:
                continue
            budgets = sorted({int(record["sample_budget"]) for record in selected})
            metrics: dict[str, Any] = {}
            for metric in ("moment_rmse", "mmd_absolute_error", "gradient_rmse"):
                rmse_by_budget = []
                for budget in budgets:
                    values = np.asarray(
                        [
                            float(record[metric])
                            for record in selected
                            if int(record["sample_budget"]) == budget
                        ]
                    )
                    rmse_by_budget.append(float(np.sqrt(np.mean(values**2))))
                safe = np.maximum(rmse_by_budget, np.finfo(float).tiny)
                slope = (
                    float(np.polyfit(np.log(budgets), np.log(safe), 1)[0])
                    if len(budgets) > 1
                    else None
                )
                metrics[metric] = {
                    "rmse_by_budget": dict(
                        zip(map(str, budgets), rmse_by_budget, strict=True)
                    ),
                    "log_log_slope": slope,
                    "reference_slope": -0.5,
                }
            runtime_by_budget = {
                str(budget): float(
                    np.median(
                        [
                            record["runtime_seconds"]
                            for record in selected
                            if record["sample_budget"] == budget
                        ]
                    )
                )
                for budget in budgets
            }
            summaries.append(
                {
                    "encoder": encoder,
                    "topology": topology,
                    "metrics": metrics,
                    "median_runtime_seconds_by_budget": runtime_by_budget,
                }
            )
    return summaries


def run_audit(
    *,
    quick: bool = False,
    encoders: tuple[str, ...] = ENCODER_FAMILIES,
    topologies: tuple[str, ...] = TOPOLOGIES,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the frozen audit matrix and return its immutable envelope."""
    if manifest is None:
        manifest = EXPECTED_MANIFEST
    if manifest != EXPECTED_MANIFEST:
        raise ValueError("estimator-audit manifest differs from the frozen protocol")
    budgets = SAMPLE_BUDGETS[:2] if quick else SAMPLE_BUDGETS
    repetitions = 2 if quick else 20
    records = [
        audit_cell(build_case(encoder, topology), budget, repetition)
        for encoder in encoders
        for topology in topologies
        for budget in budgets
        for repetition in range(repetitions)
    ]
    protocol = {
        "campaign_version": CAMPAIGN_VERSION,
        "sample_budgets": list(budgets),
        "repetitions": repetitions,
        "encoders": list(encoders),
        "topologies": list(topologies),
        "sigma": SIGMA,
        "root_seed": 20260825,
        "mode": "quick" if quick else "confirmatory",
    }
    return {
        "schema_version": 1,
        "protocol": protocol,
        "protocol_sha256": sha256_json(protocol),
        "manifest_sha256": sha256_json(manifest),
        "records": records,
        "convergence": convergence_summary(records),
    }


def write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a canonical result once; existing run envelopes are never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(payload))
        stream.write("\n")

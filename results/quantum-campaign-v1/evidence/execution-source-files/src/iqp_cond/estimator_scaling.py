"""Synthetic, zero-training-run evidence for IQP estimator scaling.

This module benchmarks estimation, not optimization.  In particular, repeated
gradient estimates are compared for bias or consistency; their raw magnitude
is never interpreted as evidence for a favorable training landscape.
"""

from __future__ import annotations

import platform
import resource
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_immutable_json
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many_jax,
    generator_characters,
)
from .protocol import content_digest
from .response import active_generator_indices
from .simulator import estimate_pauli_z


class ScalingTrack(StrEnum):
    """Generator-growth tracks required by the estimator audit."""

    BOUNDED_DEGREE = "bounded_degree"
    COMPLETE_TWO_LOCAL = "complete_two_local"


def runtime_environment() -> dict[str, object]:
    """Describe the host-dependent runtime attached to resource measurements."""
    devices = [
        {
            "platform": device.platform,
            "device_kind": device.device_kind,
        }
        for device in jax.devices()
    ]
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "jax": version("jax"),
        "jaxlib": version("jaxlib"),
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "devices": devices,
    }


@dataclass(frozen=True, slots=True)
class ScalingCase:
    """One deterministic synthetic estimator-scaling coordinate."""

    track: ScalingTrack
    axis: str
    n_qubits: int
    n_generators: int
    n_conditions: int
    condition_dimension: int
    n_samples: int
    n_repeats: int
    seed: int
    exact_reference: bool

    def __post_init__(self) -> None:
        if not isinstance(self.track, ScalingTrack):
            raise ValueError("track must use ScalingTrack vocabulary")
        if self.axis not in {
            "qubit_count",
            "generator_count",
            "condition_count",
            "condition_dimension",
            "fixture",
        }:
            raise ValueError("axis uses unknown scaling vocabulary")
        integer_values = (
            self.n_qubits,
            self.n_generators,
            self.n_conditions,
            self.condition_dimension,
            self.n_samples,
            self.n_repeats,
            self.seed,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_values
        ):
            raise ValueError("scaling coordinates must be integers")
        if (
            self.n_qubits < 1
            or self.n_generators < 1
            or self.n_conditions < 1
            or self.condition_dimension < 1
            or self.n_samples < 2
            or self.n_repeats < 2
            or self.seed < 0
        ):
            raise ValueError("scaling coordinates are out of range")
        maximum = generator_bank(self.track, self.n_qubits).shape[0]
        if self.n_generators > maximum:
            raise ValueError(
                f"n_generators exceeds {self.track.value} maximum {maximum}",
            )

    @property
    def digest(self) -> str:
        """Stable digest of this coordinate and its deterministic seed."""
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "track": self.track.value,
            "axis": self.axis,
            "n_qubits": self.n_qubits,
            "n_generators": self.n_generators,
            "n_conditions": self.n_conditions,
            "condition_dimension": self.condition_dimension,
            "n_samples": self.n_samples,
            "n_repeats": self.n_repeats,
            "seed": self.seed,
            "exact_reference": self.exact_reference,
        }


def generator_bank(track: ScalingTrack, n_qubits: int) -> np.ndarray:
    """Return a deterministic one-body-plus-edge generator bank."""
    if not isinstance(track, ScalingTrack):
        raise ValueError("track must use ScalingTrack vocabulary")
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer")

    rows = [row for row in np.eye(n_qubits, dtype=np.int8)]
    if track is ScalingTrack.BOUNDED_DEGREE:
        edges = {
            tuple(sorted((left, (left + 1) % n_qubits)))
            for left in range(n_qubits)
            if n_qubits > 1
        }
    else:
        edges = {
            (left, right)
            for left in range(n_qubits)
            for right in range(left + 1, n_qubits)
        }
    for left, right in sorted(edges):
        row = np.zeros(n_qubits, dtype=np.int8)
        row[left] = 1
        row[right] = 1
        rows.append(row)
    return np.stack(rows)


def build_axis_scaling_cases(
    *,
    qubit_sizes: tuple[int, ...],
    condition_counts: tuple[int, ...],
    condition_dimensions: tuple[int, ...],
    n_samples: int,
    n_repeats: int,
    seed: int,
    exact_max_qubits: int,
) -> tuple[ScalingCase, ...]:
    """Build a deterministic one-axis-at-a-time plan covering all resources."""
    for name, values in (
        ("qubit_sizes", qubit_sizes),
        ("condition_counts", condition_counts),
        ("condition_dimensions", condition_dimensions),
    ):
        if not values or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values
        ):
            raise ValueError(f"{name} must contain positive integers")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(exact_max_qubits, bool)
        or not isinstance(exact_max_qubits, int)
        or exact_max_qubits < 1
    ):
        raise ValueError("seed and exact_max_qubits are out of range")

    qubits = tuple(sorted(set(qubit_sizes)))
    conditions = tuple(sorted(set(condition_counts)))
    dimensions = tuple(sorted(set(condition_dimensions)))
    reference_n = max(qubits)
    inexpensive_n = min(qubits)
    cases: list[ScalingCase] = []

    def append(
        track: ScalingTrack,
        axis: str,
        n_qubits: int,
        n_generators: int,
        n_conditions: int,
        condition_dimension: int,
    ) -> None:
        cases.append(
            ScalingCase(
                track=track,
                axis=axis,
                n_qubits=n_qubits,
                n_generators=n_generators,
                n_conditions=n_conditions,
                condition_dimension=condition_dimension,
                n_samples=n_samples,
                n_repeats=n_repeats,
                seed=seed + len(cases),
                exact_reference=n_qubits <= exact_max_qubits,
            ),
        )

    for track in ScalingTrack:
        for n_qubits in qubits:
            append(
                track,
                "qubit_count",
                n_qubits,
                generator_bank(track, n_qubits).shape[0],
                conditions[0],
                dimensions[0],
            )

        maximum = generator_bank(track, reference_n).shape[0]
        generator_counts = sorted(
            {reference_n, max(reference_n, maximum // 2), maximum},
        )
        for n_generators in generator_counts:
            append(
                track,
                "generator_count",
                reference_n,
                n_generators,
                conditions[0],
                dimensions[0],
            )

        inexpensive_maximum = generator_bank(track, inexpensive_n).shape[0]
        for n_conditions in conditions:
            append(
                track,
                "condition_count",
                inexpensive_n,
                inexpensive_maximum,
                n_conditions,
                dimensions[0],
            )
        for condition_dimension in dimensions:
            append(
                track,
                "condition_dimension",
                inexpensive_n,
                inexpensive_maximum,
                conditions[0],
                condition_dimension,
            )
    return tuple(cases)


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _synthetic_fixture(case: ScalingCase) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(case.seed)
    generators = generator_bank(case.track, case.n_qubits)[: case.n_generators]
    weights = rng.normal(
        0.0,
        0.18 / np.sqrt(case.condition_dimension),
        size=(case.n_qubits, case.condition_dimension),
    )
    biases = rng.uniform(-0.35, 0.35, size=case.n_qubits)
    conditions = rng.uniform(
        -1.0,
        1.0,
        size=(case.n_conditions, case.condition_dimension),
    )
    angles = biases[None, :] + conditions @ weights.T
    parameters = rng.uniform(-0.3, 0.3, size=case.n_generators)
    observable = (np.arange(case.n_qubits) % 2 == 0).astype(np.int8)
    return {
        "generators": generators,
        "weights": weights,
        "biases": biases,
        "conditions": conditions,
        "angles": angles,
        "parameters": parameters,
        "observable": observable,
    }


def _array_digest(values: dict[str, np.ndarray]) -> str:
    return content_digest(
        {
            name: {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "values": array.tolist(),
            }
            for name, array in sorted(values.items())
        },
    )


def _exact_values_and_gradient(
    parameters: np.ndarray,
    generators: np.ndarray,
    angles: np.ndarray,
    observable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    bitstrings = enumerate_bitstrings(generators.shape[1])
    observable_characters = 1.0 - 2.0 * ((bitstrings @ observable) % 2)
    phase_characters = generator_characters(generators)
    generator_values = jnp.asarray(generators, dtype=jnp.int8)
    angle_values = jnp.asarray(angles, dtype=jnp.float64)
    character_values = jnp.asarray(phase_characters, dtype=jnp.float64)
    observable_values = jnp.asarray(observable_characters, dtype=jnp.float64)

    def response(theta: jnp.ndarray) -> jnp.ndarray:
        probabilities = exact_iqp_probabilities_many_jax(
            theta,
            generator_values,
            angle_values,
            characters=character_values,
        )
        return probabilities @ observable_values

    theta = jnp.asarray(parameters, dtype=jnp.float64)
    values = response(theta)
    gradient = jax.jacrev(response)(theta)
    jax.block_until_ready((values, gradient))
    return (
        np.asarray(values, dtype=np.float64),
        np.asarray(gradient, dtype=np.float64),
        time.perf_counter() - started,
    )


def _error_summary(samples: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    errors = np.asarray(samples, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    bias = np.mean(errors, axis=0)
    return {
        "mean_signed_bias": float(np.mean(bias)),
        "max_absolute_bias": float(np.max(np.abs(bias))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean_empirical_variance": float(np.mean(np.var(samples, axis=0, ddof=1))),
    }


def _consistency_summary(samples: np.ndarray) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    repeat_sd = np.std(values, axis=0, ddof=1)
    consecutive = values[1:] - values[:-1]
    return {
        "mean_repeat_standard_deviation": float(np.mean(repeat_sd)),
        "max_repeat_standard_deviation": float(np.max(repeat_sd)),
        "consecutive_repeat_rmse": float(np.sqrt(np.mean(consecutive**2))),
    }


def _numerical_results_digest(record: dict[str, Any]) -> str:
    deterministic = {
        name: value
        for name, value in record.items()
        if name
        not in {
            "runtime_seconds",
            "peak_rss_bytes",
            "peak_rss_increment_bytes",
            "numerical_results_digest",
        }
    }
    reference = deterministic.get("reference")
    if isinstance(reference, dict):
        deterministic["reference"] = {
            name: value
            for name, value in reference.items()
            if name != "runtime_seconds"
        }
    return content_digest(deterministic)


def run_scaling_case(case: ScalingCase) -> dict[str, Any]:
    """Run one estimator/gradient case and return a JSON-compatible record."""
    case_started = time.perf_counter()
    peak_before = _peak_rss_bytes()
    fixture_started = time.perf_counter()
    fixture = _synthetic_fixture(case)
    fixture_runtime = time.perf_counter() - fixture_started
    generators = fixture["generators"]
    parameters = fixture["parameters"]
    angles = fixture["angles"]
    observable = fixture["observable"]
    active = active_generator_indices(generators, observable)

    exact_values = None
    exact_gradient = None
    exact_runtime = None
    if case.exact_reference:
        exact_values, exact_gradient, exact_runtime = _exact_values_and_gradient(
            parameters,
            generators,
            angles,
            observable,
        )

    theta = jnp.asarray(parameters, dtype=jnp.float64)
    generator_values = jnp.asarray(generators, dtype=jnp.float64)
    observable_values = jnp.asarray(observable, dtype=jnp.float64)
    angle_values = jnp.asarray(angles, dtype=jnp.float64)
    estimates = []
    gradients = []
    standard_errors = []
    runtimes = []
    for repeat in range(case.n_repeats):
        key = jax.random.PRNGKey(case.seed + 10_000 + repeat)

        def sampled_response(
            values: jnp.ndarray,
            *,
            frozen_key: jax.Array = key,
        ) -> jnp.ndarray:
            return estimate_pauli_z(
                values,
                generator_values,
                observable_values,
                angle_values,
                frozen_key,
                case.n_samples,
            ).values[:, 0]

        started = time.perf_counter()
        estimate = estimate_pauli_z(
            theta,
            generator_values,
            observable_values,
            angle_values,
            key,
            case.n_samples,
        )
        gradient = jax.jacrev(sampled_response)(theta)
        jax.block_until_ready((estimate.values, estimate.standard_errors, gradient))
        runtimes.append(time.perf_counter() - started)
        estimates.append(np.asarray(estimate.values[:, 0], dtype=np.float64))
        standard_errors.append(
            np.asarray(estimate.standard_errors[:, 0], dtype=np.float64),
        )
        gradients.append(np.asarray(gradient, dtype=np.float64))
    peak_after = _peak_rss_bytes()

    estimate_array = np.stack(estimates)
    gradient_array = np.stack(gradients)
    standard_error_array = np.stack(standard_errors)
    active_gradients = gradient_array[:, :, active]
    record: dict[str, Any] = {
        "case": case.to_dict(),
        "case_digest": case.digest,
        "synthetic_inputs_digest": _array_digest(fixture),
        "estimator_repeat_seeds": [
            case.seed + 10_000 + repeat for repeat in range(case.n_repeats)
        ],
        "generator_growth": (
            "O(n) one-body plus cycle edges"
            if case.track is ScalingTrack.BOUNDED_DEGREE
            else "O(n^2) one-body plus complete two-local edges"
        ),
        "active_generator_count": int(active.size),
        "full_training_runs_used": 0,
        "budget_pool": "none",
        "runtime_seconds": {
            "fixture_preparation": fixture_runtime,
            "per_repeat": [float(value) for value in runtimes],
            "median": float(np.median(runtimes)),
            "mean": float(np.mean(runtimes)),
            "case_total": time.perf_counter() - case_started,
        },
        "peak_rss_bytes": peak_after,
        "peak_rss_increment_bytes": max(0, peak_after - peak_before),
        "synthetic_input_bytes": int(
            sum(array.nbytes for array in fixture.values()),
        ),
        "reported_expectation_standard_error": {
            "mean": float(np.mean(standard_error_array)),
            "maximum": float(np.max(standard_error_array)),
        },
        "interpretation": (
            "estimator accuracy/consistency only; gradient statistics are not "
            "landscape-trainability evidence"
        ),
    }
    if case.exact_reference:
        record["reference"] = {
            "kind": "exact_statevector",
            "runtime_seconds": float(exact_runtime),
        }
        record["expectation_error"] = _error_summary(estimate_array, exact_values)
        record["gradient_error"] = _error_summary(
            gradient_array,
            exact_gradient,
        )
        record["active_gradient_error"] = _error_summary(
            active_gradients,
            exact_gradient[:, active],
        )
    else:
        record["reference"] = {"kind": "repeated_estimator_consistency"}
        record["expectation_consistency"] = _consistency_summary(estimate_array)
        record["gradient_consistency"] = _consistency_summary(gradient_array)
        record["active_gradient_consistency"] = _consistency_summary(active_gradients)
    record["numerical_results_digest"] = _numerical_results_digest(record)
    return record


def run_scaling_plan(
    cases: tuple[ScalingCase, ...],
    *,
    campaign_version: str,
    manifest_digest: str,
    provenance: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Run a complete synthetic plan with zero campaign-training accounting."""
    if not cases:
        raise ValueError("at least one scaling case is required")
    configuration = {
        "schema_version": 1,
        "campaign_version": campaign_version,
        "manifest_digest": manifest_digest,
        "cases": [case.to_dict() for case in cases],
    }
    records = [run_scaling_case(case) for case in cases]
    provenance_payload = provenance or {
        "code_revision": "unrecorded_library_call",
        "working_tree_dirty": None,
        "invocation": ["iqp_cond.estimator_scaling.run_scaling_plan"],
        "source_digests": {},
        "environment": runtime_environment(),
    }
    required_provenance = {
        "code_revision",
        "working_tree_dirty",
        "invocation",
        "source_digests",
        "environment",
    }
    if set(provenance_payload) != required_provenance:
        raise ValueError("scaling provenance must use the exact required schema")
    return {
        "schema_version": 1,
        "evidence_type": "estimator_scaling",
        "status": "completed",
        "campaign_version": campaign_version,
        "manifest_digest": manifest_digest,
        "configuration_digest": content_digest(configuration),
        "numerical_results_digest": content_digest(
            [record["numerical_results_digest"] for record in records],
        ),
        "data_source": "deterministic_synthetic_frozen_parameters",
        "full_training_runs_used": 0,
        "budget_accounting": "excluded_from_all_full-training-run_pools",
        "track_vocabulary": [track.value for track in ScalingTrack],
        "interpretation": (
            "classical estimator and gradient-estimator evidence only; no claim "
            "about raw gradient magnitude, optimizer convergence, or barren plateaus"
        ),
        "determinism_scope": (
            "configuration, synthetic inputs, estimator draws, and numerical "
            "summaries are content-digested; wall time and RSS are host-dependent"
        ),
        "provenance": provenance_payload,
        "records": records,
    }


def archive_scaling_report(path: Path, report: dict[str, Any]) -> str:
    """Write one immutable, content-verified scaling JSON artifact."""
    return write_immutable_json(path, report)

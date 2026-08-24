"""Finite-support response-spectrum audits with strict production preflight."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    CompletionStatus,
    ResultBundle,
    read_immutable_json,
    write_immutable_json,
)
from .campaign import Gate, GateDecision, GateRecord
from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities,
    exact_iqp_probabilities_many,
)
from .ising import (
    BetaGrid as IsingBetaGrid,
)
from .ising import (
    beta_to_broadcast_angles,
    ising_energies,
    periodic_lattice_couplings,
)
from .phase_conditioning import exact_effective_phases, normalize_conditions
from .protocol import CampaignManifest, ModelFamily, Stage, content_digest
from .response import full_frequency_support, observable_lightcone


class SpectrumAuditError(ValueError):
    """Raised when spectral inputs or frozen provenance are incompatible."""


@dataclass(frozen=True, slots=True)
class FrozenCheckpoint:
    """Verified production checkpoint paired with its immutable result bundle."""

    family: ModelFamily
    coupling_seed: int
    optimizer_seed: int
    manifest_digest: str
    source_run_id: str
    checkpoint_sha256: str
    parameters: np.ndarray
    generators: np.ndarray
    phase_feature: str | None


def _as_frequency_matrix(
    frequencies: np.ndarray,
    dimension: int,
    *,
    integer: bool,
) -> np.ndarray:
    values = np.asarray(frequencies, dtype=np.float64)
    if values.ndim == 1 and dimension == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] != dimension or values.shape[0] < 1:
        raise SpectrumAuditError(
            "frequencies must have shape (n_frequencies, dimension)",
        )
    if not np.all(np.isfinite(values)):
        raise SpectrumAuditError("frequencies must be finite")
    if integer:
        rounded = np.rint(values)
        if not np.allclose(values, rounded, atol=1e-12, rtol=0.0):
            raise SpectrumAuditError("periodic-grid frequencies must be integers")
        return rounded.astype(np.int64)
    return values


def _periodic_coefficients(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(values)
    if samples.ndim < 1 or any(size < 3 for size in samples.shape):
        raise SpectrumAuditError("periodic values need at least three points per axis")
    if not np.all(np.isfinite(samples)):
        raise SpectrumAuditError("periodic values must be finite")
    coefficients = np.fft.fftn(samples) / samples.size
    axes = [
        np.rint(np.fft.fftfreq(size) * size).astype(np.int64) for size in samples.shape
    ]
    grid = np.meshgrid(*axes, indexing="ij")
    frequencies = np.stack([axis.ravel() for axis in grid], axis=1)
    return frequencies, coefficients.ravel()


def _mode_records(
    frequencies: np.ndarray,
    coefficients: np.ndarray,
    selected: np.ndarray,
) -> list[dict[str, object]]:
    records = [
        {
            "frequency": [int(value) for value in frequency],
            "coefficient_real": float(np.real(coefficient)),
            "coefficient_imag": float(np.imag(coefficient)),
            "power": float(abs(coefficient) ** 2),
        }
        for frequency, coefficient in zip(
            frequencies[selected],
            coefficients[selected],
            strict=True,
        )
    ]
    return sorted(
        records,
        key=lambda record: (
            -float(record["power"]),
            tuple(record["frequency"]),
        ),
    )


def recover_periodic_spectrum(
    values: np.ndarray,
    *,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Recover significant integer Fourier modes from a dense periodic grid."""
    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise SpectrumAuditError("spectrum tolerances must be non-negative")
    frequencies, coefficients = _periodic_coefficients(values)
    maximum = float(np.max(np.abs(coefficients)))
    threshold = max(absolute_tolerance, relative_tolerance * maximum)
    selected = np.abs(coefficients) > threshold
    return {
        "grid_shape": list(np.asarray(values).shape),
        "threshold": threshold,
        "parseval_power": float(np.sum(np.abs(coefficients) ** 2)),
        "significant_mode_count": int(np.count_nonzero(selected)),
        "modes": _mode_records(frequencies, coefficients, selected),
    }


def periodic_spectral_mass(
    values: np.ndarray,
    allowed_frequencies: np.ndarray,
    *,
    mode_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Measure exact-grid Fourier power inside and outside an allowed set."""
    frequencies, coefficients = _periodic_coefficients(values)
    allowed = _as_frequency_matrix(
        allowed_frequencies,
        frequencies.shape[1],
        integer=True,
    )
    nyquist_bounds = np.asarray(
        [(size - 1) // 2 for size in np.asarray(values).shape],
        dtype=np.int64,
    )
    if np.any(np.abs(allowed) > nyquist_bounds[None, :]):
        raise SpectrumAuditError(
            "periodic grid is too small for alias-free allowed frequencies",
        )
    allowed_set = {tuple(row) for row in allowed.tolist()}
    allowed_mask = np.asarray(
        [tuple(row) in allowed_set for row in frequencies.tolist()],
        dtype=bool,
    )
    power = np.abs(coefficients) ** 2
    total = float(np.sum(power))
    outside = float(np.sum(power[~allowed_mask]))
    selected_outside = (~allowed_mask) & (np.abs(coefficients) > mode_tolerance)
    return {
        "allowed_frequency_count": len(allowed_set),
        "alias_free_allowed_support": True,
        "nyquist_bounds": nyquist_bounds.tolist(),
        "total_power": total,
        "allowed_power": total - outside,
        "outside_power": outside,
        "outside_power_fraction": 0.0 if total == 0.0 else outside / total,
        "out_of_support_modes": _mode_records(
            frequencies,
            coefficients,
            selected_outside,
        ),
    }


def project_onto_frequency_support(
    points: np.ndarray,
    values: np.ndarray,
    allowed_frequencies: np.ndarray,
) -> dict[str, Any]:
    """Project arbitrary dense-grid values onto a finite frequency dictionary.

    Unlike the FFT audit, this accepts noninteger frequencies and nonperiodic
    sampling windows.  The residual is a finite-grid unresolved-power
    diagnostic, not a claim about a continuum Fourier transform.
    """
    point_values = np.asarray(points, dtype=np.float64)
    sample_values = np.asarray(values, dtype=np.complex128)
    if point_values.ndim == 1:
        point_values = point_values[:, None]
    if point_values.ndim != 2 or point_values.shape[0] < 2:
        raise SpectrumAuditError("points must have shape (n_points, dimension)")
    if sample_values.shape != (point_values.shape[0],):
        raise SpectrumAuditError("values must align with flattened points")
    if not np.all(np.isfinite(point_values)) or not np.all(np.isfinite(sample_values)):
        raise SpectrumAuditError("projection inputs must be finite")
    frequencies = _as_frequency_matrix(
        allowed_frequencies,
        point_values.shape[1],
        integer=False,
    )
    design = np.exp(1j * (point_values @ frequencies.T))
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design,
        sample_values,
        rcond=None,
    )
    reconstruction = design @ coefficients
    residual_power = float(np.mean(np.abs(sample_values - reconstruction) ** 2))
    total_power = float(np.mean(np.abs(sample_values) ** 2))
    condition_number = None
    if singular_values.size and singular_values[-1] > 0.0:
        condition_number = float(singular_values[0] / singular_values[-1])
    return {
        "n_points": point_values.shape[0],
        "frequency_count": frequencies.shape[0],
        "design_rank": int(rank),
        "condition_number": condition_number,
        "total_power": total_power,
        "unresolved_power": residual_power,
        "unresolved_power_fraction": (
            0.0 if total_power == 0.0 else residual_power / total_power
        ),
        "reconstruction_rmse": float(np.sqrt(residual_power)),
    }


def describe_unresolved_mass_relation(
    unresolved_mass: np.ndarray,
    interpolation_error: np.ndarray,
) -> dict[str, Any]:
    """Describe, without causal language, their finite-sample association."""
    mass = np.asarray(unresolved_mass, dtype=np.float64)
    error = np.asarray(interpolation_error, dtype=np.float64)
    if mass.ndim != 1 or error.shape != mass.shape or mass.size < 2:
        return {
            "status": "insufficient_points",
            "n_points": int(mass.size),
            "interpretation": "no association estimated",
        }
    if not np.all(np.isfinite(mass)) or not np.all(np.isfinite(error)):
        raise SpectrumAuditError("relation inputs must be finite")
    centered_mass = mass - np.mean(mass)
    centered_error = error - np.mean(error)
    denominator = float(
        np.sqrt(np.sum(centered_mass**2) * np.sum(centered_error**2)),
    )
    correlation = None
    if denominator > 0.0:
        correlation = float(np.sum(centered_mass * centered_error) / denominator)
    design = np.column_stack((np.ones(mass.size), mass))
    intercept, slope = np.linalg.lstsq(design, error, rcond=None)[0]
    return {
        "status": "descriptive",
        "n_points": int(mass.size),
        "pearson_correlation": correlation,
        "least_squares_intercept": float(intercept),
        "least_squares_slope": float(slope),
        "interpretation": (
            "descriptive finite-fixture association only; not causal and not a "
            "generalization or trainability claim"
        ),
    }


def compare_response_spectra(
    *,
    target_values: np.ndarray,
    fixed_values: np.ndarray,
    phase_values: np.ndarray,
    fixed_allowed_frequencies: np.ndarray,
    phase_allowed_frequencies: np.ndarray,
    interpolation_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare fixed-input and reprogrammable phase responses on one grid."""
    target = np.asarray(target_values, dtype=np.float64)
    fixed = np.asarray(fixed_values, dtype=np.float64)
    phase = np.asarray(phase_values, dtype=np.float64)
    if target.shape != fixed.shape or target.shape != phase.shape:
        raise SpectrumAuditError("target and response grids must align")
    if interpolation_mask is None:
        selected = np.ones(target.shape, dtype=bool)
    else:
        selected = np.asarray(interpolation_mask, dtype=bool)
        if selected.shape != target.shape or not np.any(selected):
            raise SpectrumAuditError("interpolation_mask must select aligned points")

    fixed_mass = periodic_spectral_mass(fixed, fixed_allowed_frequencies)
    phase_mass = periodic_spectral_mass(phase, phase_allowed_frequencies)
    target_fixed = periodic_spectral_mass(target, fixed_allowed_frequencies)
    target_phase = periodic_spectral_mass(target, phase_allowed_frequencies)
    return {
        "fixed_input": {
            "response_spectral_leakage": fixed_mass,
            "target_mass_outside_allowed_support": target_fixed,
            "interpolation_rmse": float(
                np.sqrt(np.mean((fixed[selected] - target[selected]) ** 2)),
            ),
        },
        "condition_modulated_phase": {
            "response_spectral_leakage": phase_mass,
            "target_mass_outside_allowed_support": target_phase,
            "interpolation_rmse": float(
                np.sqrt(np.mean((phase[selected] - target[selected]) ** 2)),
            ),
        },
        "interpretation": (
            "response-spectrum and interpolation-error comparison only; phase "
            "reprogramming is outside fixed-circuit deployment"
        ),
    }


def synthetic_spectrum_fixture(
    *,
    grid_size: int = 32,
    injected_amplitude: float = 0.25,
) -> dict[str, Any]:
    """Return a deterministic multivariate audit with known injected leakage."""
    if grid_size < 8 or injected_amplitude <= 0.0:
        raise SpectrumAuditError("synthetic fixture settings are out of range")
    axis = 2.0 * np.pi * np.arange(grid_size) / grid_size
    first, second = np.meshgrid(axis, axis, indexing="ij")
    shared = 0.5 + 0.2 * np.cos(first) + 0.15 * np.sin(second)
    injected = injected_amplitude * np.cos(2.0 * first)
    target = shared + injected
    fixed = shared
    phase = target.copy()
    first_indices, second_indices = np.meshgrid(
        np.arange(grid_size),
        np.arange(grid_size),
        indexing="ij",
    )
    interpolation_mask = (first_indices + second_indices) % 4 == 1
    fixed_allowed = full_frequency_support(np.eye(2, dtype=np.float64))
    phase_allowed = np.unique(
        np.concatenate(
            (fixed_allowed, np.array([[-2.0, 0.0], [2.0, 0.0]])),
        ),
        axis=0,
    )
    comparison = compare_response_spectra(
        target_values=target,
        fixed_values=fixed,
        phase_values=phase,
        fixed_allowed_frequencies=fixed_allowed,
        phase_allowed_frequencies=phase_allowed,
        interpolation_mask=interpolation_mask,
    )

    masses = []
    errors = []
    for amplitude in (0.05, 0.15, injected_amplitude):
        fixture_target = shared + amplitude * np.cos(2.0 * first)
        mass = periodic_spectral_mass(fixture_target, fixed_allowed)
        masses.append(mass["outside_power_fraction"])
        errors.append(
            float(
                np.sqrt(
                    np.mean(
                        (fixture_target[interpolation_mask] - fixed[interpolation_mask])
                        ** 2,
                    ),
                ),
            ),
        )

    configuration = {
        "grid_size": grid_size,
        "injected_amplitude": injected_amplitude,
        "fixed_allowed_frequencies": fixed_allowed.tolist(),
        "phase_allowed_frequencies": phase_allowed.tolist(),
        "interpolation_mask_rule": "(axis_0_index + axis_1_index) mod 4 equals 1",
    }
    detected_injected_power = comparison["fixed_input"][
        "target_mass_outside_allowed_support"
    ]["outside_power"]
    expected_injected_power = injected_amplitude**2 / 2.0
    return {
        "schema_version": 1,
        "evidence_type": "response_spectrum",
        "status": "synthetic_fixture_completed",
        "data_source": "known_multivariate_periodic_function",
        "configuration_digest": content_digest(configuration),
        "full_training_runs_used": 0,
        "budget_pool": "none",
        "budget_accounting": "excluded_from_all_full-training-run_pools",
        "fixture_truth": {
            "theorem_support_source": "affine_RY_full_probability_support",
            "injected_frequencies": [[-2, 0], [2, 0]],
            "expected_injected_power": expected_injected_power,
            "detected_injected_power": detected_injected_power,
            "absolute_power_error": abs(
                detected_injected_power - expected_injected_power,
            ),
            "interpolation_point_count": int(np.count_nonzero(interpolation_mask)),
        },
        "fixed_response_spectrum": recover_periodic_spectrum(fixed),
        "target_spectrum": recover_periodic_spectrum(target),
        "comparison": comparison,
        "unresolved_mass_error_relation": describe_unresolved_mass_relation(
            np.asarray(masses),
            np.asarray(errors),
        ),
        "interpretation": (
            "the injected out-of-support component is known by construction; "
            "this fixture validates the audit and is not campaign evidence"
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_path(
    repository_root: Path,
    bundle_path: Path,
    artifact_path: str,
) -> Path:
    root = repository_root.resolve()
    campaign_root = bundle_path.resolve().parents[2]
    candidates = (
        (campaign_root / artifact_path).resolve(),
        (root / artifact_path).resolve(),
    )
    for candidate in candidates:
        if candidate.is_relative_to(root) and candidate.is_file():
            return candidate
    if any(not candidate.is_relative_to(root) for candidate in candidates):
        raise SpectrumAuditError("checkpoint artifact escapes repository root")
    return candidates[0]


def load_frozen_checkpoint(
    bundle_path: Path,
    *,
    repository_root: Path,
    manifest: CampaignManifest,
    expected_family: ModelFamily,
) -> FrozenCheckpoint:
    """Load and cross-check a completed primary bundle and immutable NPZ."""
    try:
        bundle = ResultBundle.from_dict(read_immutable_json(bundle_path))
    except Exception as error:
        raise SpectrumAuditError(f"invalid result bundle: {bundle_path}") from error
    if bundle.completion_status is not CompletionStatus.COMPLETED:
        raise SpectrumAuditError("spectrum audit requires a completed result bundle")
    if bundle.manifest_digest != manifest.digest:
        raise SpectrumAuditError("checkpoint bundle belongs to another manifest")
    if bundle.run.stage is not Stage.PRIMARY:
        raise SpectrumAuditError("checkpoint must come from the frozen primary stage")
    if bundle.run.model_family is not expected_family:
        raise SpectrumAuditError(
            f"expected {expected_family.value} checkpoint bundle",
        )
    if bundle.checkpoint is None:
        raise SpectrumAuditError("completed bundle has no checkpoint provenance")
    checkpoint_path = _checkpoint_path(
        repository_root,
        bundle_path,
        bundle.checkpoint.artifact_path,
    )
    if not checkpoint_path.is_file():
        raise SpectrumAuditError("checkpoint artifact is missing")
    if _sha256_file(checkpoint_path) != bundle.checkpoint.artifact_sha256:
        raise SpectrumAuditError("checkpoint artifact digest does not match bundle")
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        required = {
            "best_parameters",
            "final_parameters",
            "generators",
            "best_step",
            "best_validation_loss",
            "train_steps",
            "train_losses",
            "validation_steps",
            "validation_losses",
        }
        if set(archive.files) != required:
            raise SpectrumAuditError("checkpoint NPZ keys do not match schema")
        parameters = np.asarray(archive["best_parameters"], dtype=np.float64)
        generators = np.asarray(archive["generators"], dtype=np.int8)
        best_step = int(archive["best_step"])
        best_loss = float(archive["best_validation_loss"])
    if (
        generators.ndim != 2
        or generators.shape[0] < 1
        or generators.shape[1] < 1
        or parameters.shape != (generators.shape[0],)
    ):
        raise SpectrumAuditError("checkpoint parameters and generators do not align")
    if not np.all(np.isfinite(parameters)) or not np.all(
        (generators == 0) | (generators == 1),
    ):
        raise SpectrumAuditError("checkpoint arrays must be finite and binary")
    if best_step != bundle.checkpoint.selected_step or not np.isclose(
        best_loss,
        bundle.checkpoint.validation_value,
        atol=0.0,
        rtol=0.0,
    ):
        raise SpectrumAuditError("checkpoint values differ from bundle provenance")
    parameters.setflags(write=False)
    generators.setflags(write=False)
    phase_feature = bundle.run.phase_feature_assignment
    return FrozenCheckpoint(
        family=expected_family,
        coupling_seed=bundle.run.coupling_seed,
        optimizer_seed=bundle.run.optimizer_seed,
        manifest_digest=bundle.manifest_digest,
        source_run_id=bundle.run.run_id,
        checkpoint_sha256=bundle.checkpoint.artifact_sha256,
        parameters=parameters,
        generators=generators,
        phase_feature=phase_feature,
    )


def read_theorem_novelty_gate(
    campaign_root: Path,
    manifest: CampaignManifest,
) -> GateRecord | None:
    """Read the immutable campaign gate; never substitute a source-code verdict."""
    path = campaign_root / "gates" / f"{Gate.THEOREM_NOVELTY.value}.json"
    if not path.is_file():
        return None
    try:
        record = GateRecord.from_dict(read_immutable_json(path))
    except Exception as error:
        raise SpectrumAuditError("theorem novelty gate artifact is invalid") from error
    if record.gate is not Gate.THEOREM_NOVELTY:
        raise SpectrumAuditError("theorem novelty gate artifact has the wrong gate")
    if record.manifest_digest != manifest.digest:
        raise SpectrumAuditError("theorem novelty gate belongs to another manifest")
    return record


def _checkpoint_probability_grid(
    checkpoint: FrozenCheckpoint,
    betas: np.ndarray,
    grid: IsingBetaGrid,
) -> np.ndarray:
    n_qubits = checkpoint.generators.shape[1]
    if checkpoint.family is ModelFamily.FIXED_INPUT:
        angles = beta_to_broadcast_angles(betas, grid, n_qubits)
        return exact_iqp_probabilities_many(
            checkpoint.parameters,
            checkpoint.generators,
            angles,
        )
    if checkpoint.family is not ModelFamily.CONDITION_MODULATED_PHASE:
        raise SpectrumAuditError("unsupported checkpoint family")
    if checkpoint.phase_feature is None:
        raise SpectrumAuditError("phase checkpoint has no frozen feature assignment")
    normalized = normalize_conditions(betas, grid.training[0], grid.training[-1])
    effective = exact_effective_phases(
        checkpoint.parameters,
        normalized,
        checkpoint.phase_feature,
    )
    fixed_input = beta_to_broadcast_angles(grid.anchor, grid, n_qubits)
    return np.stack(
        [
            exact_iqp_probabilities(
                parameters,
                checkpoint.generators,
                fixed_input,
            )
            for parameters in effective
        ],
    )


def _dense_target_probabilities(
    coupling_seed: int,
    betas: np.ndarray,
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    side = round(np.sqrt(n_qubits))
    if side * side != n_qubits:
        raise SpectrumAuditError("production Ising target needs a square lattice")
    bitstrings = enumerate_bitstrings(n_qubits)
    couplings = periodic_lattice_couplings(coupling_seed, (side, side))
    energies = ising_energies(bitstrings, couplings)
    log_weights = -betas[:, None] * energies[None, :]
    log_weights -= np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights)
    return bitstrings, weights / np.sum(weights, axis=1, keepdims=True)


def _default_output_indices(n_states: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys((0, 1, 3, n_states - 1)))


def _default_pauli_masks(n_qubits: int) -> np.ndarray:
    masks = np.zeros((4, n_qubits), dtype=np.int8)
    masks[0, 0] = 1
    masks[1, min(1, n_qubits - 1)] = 1
    masks[2, : min(2, n_qubits)] = 1
    masks[3, :] = 1
    return np.unique(masks, axis=0)


def _series_audit(
    *,
    label: str,
    selection: object,
    betas: np.ndarray,
    heldout_mask: np.ndarray,
    target: np.ndarray,
    fixed: np.ndarray,
    phase: np.ndarray,
    allowed_frequencies: np.ndarray,
) -> dict[str, Any]:
    fixed_projection = project_onto_frequency_support(
        betas,
        fixed,
        allowed_frequencies,
    )
    phase_projection = project_onto_frequency_support(
        betas,
        phase,
        allowed_frequencies,
    )
    target_projection = project_onto_frequency_support(
        betas,
        target,
        allowed_frequencies,
    )
    return {
        "response_type": label,
        "selection": selection,
        "predicted_fixed_input_frequencies": allowed_frequencies[:, 0].tolist(),
        "fixed_response_support_audit": fixed_projection,
        "phase_response_on_fixed_support": phase_projection,
        "target_unresolved_mass": target_projection,
        "heldout_interpolation_rmse": {
            "fixed_input": float(
                np.sqrt(np.mean((fixed[heldout_mask] - target[heldout_mask]) ** 2)),
            ),
            "condition_modulated_phase": float(
                np.sqrt(np.mean((phase[heldout_mask] - target[heldout_mask]) ** 2)),
            ),
        },
        "windowed_dense_grid_spectrum": {
            "basis": "DFT index modes over the finite beta window; diagnostic only",
            "fixed_input": recover_periodic_spectrum(fixed),
            "condition_modulated_phase": recover_periodic_spectrum(phase),
            "target": recover_periodic_spectrum(target),
        },
        "dense_values": {
            "target": target.tolist(),
            "fixed_input": fixed.tolist(),
            "condition_modulated_phase": phase.tolist(),
        },
    }


def evaluate_production_spectrum(
    manifest: CampaignManifest,
    fixed: FrozenCheckpoint,
    phase: FrozenCheckpoint,
    *,
    grid_size: int = 193,
) -> dict[str, Any]:
    """Evaluate paired frozen checkpoints on the dense physical beta grid."""
    if grid_size < 65 or grid_size % 2 == 0:
        raise SpectrumAuditError(
            "production beta grid size must be odd and at least 65"
        )
    n_qubits = fixed.generators.shape[1]
    if n_qubits != 16 or phase.generators.shape[1] != n_qubits:
        raise SpectrumAuditError("production spectra require paired 16-qubit models")
    if phase.phase_feature is None:
        raise SpectrumAuditError("phase checkpoint has no frozen feature assignment")
    grid = IsingBetaGrid(
        training=manifest.beta_grid.training,
        heldout=manifest.beta_grid.held_out,
        provenance=manifest.calibration.method,
    )
    betas = np.linspace(grid.training[0], grid.training[-1], grid_size)
    heldout_mask = np.any(
        np.isclose(
            betas[:, None],
            np.asarray(grid.heldout)[None, :],
            atol=1e-14,
            rtol=0.0,
        ),
        axis=1,
    )
    if int(np.count_nonzero(heldout_mask)) != len(grid.heldout):
        raise SpectrumAuditError(
            "dense beta grid must contain all four frozen Held-out Conditions",
        )
    fixed_probabilities = _checkpoint_probability_grid(fixed, betas, grid)
    phase_probabilities = _checkpoint_probability_grid(phase, betas, grid)
    bitstrings, target_probabilities = _dense_target_probabilities(
        fixed.coupling_seed,
        betas,
        n_qubits,
    )
    n_states = target_probabilities.shape[1]
    output_indices = _default_output_indices(n_states)
    pauli_masks = _default_pauli_masks(n_qubits)
    pauli_characters = 1.0 - 2.0 * (
        (bitstrings.astype(np.int64) @ pauli_masks.T.astype(np.int64)) % 2
    )
    target_paulis = target_probabilities @ pauli_characters
    fixed_paulis = fixed_probabilities @ pauli_characters
    phase_paulis = phase_probabilities @ pauli_characters

    angle_slope = (np.pi / 4.0) / (grid.training[-1] - grid.training[0])
    output_support = (
        np.arange(-n_qubits, n_qubits + 1, dtype=np.float64) * angle_slope
    )[:, None]
    records = []
    for output_index in output_indices:
        records.append(
            _series_audit(
                label="output_probability",
                selection={"little_endian_state_index": output_index},
                betas=betas,
                heldout_mask=heldout_mask,
                target=target_probabilities[:, output_index],
                fixed=fixed_probabilities[:, output_index],
                phase=phase_probabilities[:, output_index],
                allowed_frequencies=output_support,
            ),
        )
    for index, mask in enumerate(pauli_masks):
        lightcone_size = int(
            np.count_nonzero(observable_lightcone(fixed.generators, mask)),
        )
        observable_support = (
            np.arange(-lightcone_size, lightcone_size + 1, dtype=np.float64)
            * angle_slope
        )[:, None]
        records.append(
            _series_audit(
                label="pauli_z_expectation",
                selection={"mask": mask.tolist()},
                betas=betas,
                heldout_mask=heldout_mask,
                target=target_paulis[:, index],
                fixed=fixed_paulis[:, index],
                phase=phase_paulis[:, index],
                allowed_frequencies=observable_support,
            ),
        )

    unresolved = np.asarray(
        [
            record["target_unresolved_mass"]["unresolved_power_fraction"]
            for record in records
        ],
    )
    interpolation = np.asarray(
        [record["heldout_interpolation_rmse"]["fixed_input"] for record in records],
    )
    normalized_endpoints = np.asarray([0.0, 1.0])
    phase_endpoints = exact_effective_phases(
        phase.parameters,
        normalized_endpoints,
        phase.phase_feature,
    )
    phase_slopes = (phase_endpoints[1] - phase_endpoints[0]) / (
        grid.training[-1] - grid.training[0]
    )
    phase_bandwidth_bound = float(2.0 * np.sum(np.abs(phase_slopes)))
    configuration = {
        "manifest_digest": manifest.digest,
        "coupling_seed": fixed.coupling_seed,
        "optimizer_seed": fixed.optimizer_seed,
        "grid_size": grid_size,
        "beta_min": grid.training[0],
        "beta_max": grid.training[-1],
        "output_indices": list(output_indices),
        "pauli_masks": pauli_masks.tolist(),
        "fixed_checkpoint_sha256": fixed.checkpoint_sha256,
        "phase_checkpoint_sha256": phase.checkpoint_sha256,
    }
    return {
        "schema_version": 1,
        "evidence_type": "response_spectrum_production",
        "status": "completed",
        "campaign_version": manifest.campaign_version,
        "manifest_digest": manifest.digest,
        "configuration_digest": content_digest(configuration),
        "full_training_runs_used": 0,
        "budget_pool": "none",
        "budget_accounting": "checkpoint-only analysis; no new training run",
        "dense_beta_grid": betas.tolist(),
        "heldout_beta_indices": np.flatnonzero(heldout_mask).tolist(),
        "fixed_input_predicted_output_bandwidth": float(n_qubits * angle_slope),
        "phase_predicted_output_bandwidth_upper_bound": phase_bandwidth_bound,
        "response_records": records,
        "support_leakage_summary": {
            "maximum_fixed_response_unresolved_fraction": float(
                max(
                    record["fixed_response_support_audit"]["unresolved_power_fraction"]
                    for record in records
                ),
            ),
            "maximum_phase_mass_outside_fixed_support_fraction": float(
                max(
                    record["phase_response_on_fixed_support"][
                        "unresolved_power_fraction"
                    ]
                    for record in records
                ),
            ),
        },
        "target_unresolved_mass_vs_interpolation_error": (
            describe_unresolved_mass_relation(unresolved, interpolation)
        ),
        "interpretation": (
            "selected probabilities and Pauli expectations were evaluated on the "
            "dense physical beta grid; fixed-support residuals are numerical "
            "projection audits, and the phase comparison is not a fixed-circuit claim"
        ),
    }


def production_spectrum_preflight(
    *,
    manifest_path: Path,
    repository_root: Path,
    fixed_bundle_path: Path | None,
    phase_bundle_path: Path | None,
    campaign_root: Path | None = None,
    grid_size: int = 193,
) -> dict[str, Any]:
    """Read the actual gate and either block or execute paired checkpoint audits."""
    manifest = CampaignManifest.from_json(manifest_path)
    if campaign_root is None:
        campaign_root = repository_root / "results" / "quantum-campaign-v1" / "campaign"
    blockers: list[dict[str, str]] = []
    gate_record = read_theorem_novelty_gate(campaign_root, manifest)
    if gate_record is None:
        blockers.append(
            {
                "code": "theorem_novelty_missing",
                "detail": "the frozen theorem novelty gate has no decision",
            },
        )
        theorem_verdict = "missing"
        theorem_evidence_digest = None
    else:
        theorem_verdict = gate_record.decision.value
        theorem_evidence_digest = gate_record.evidence_digest
        if gate_record.decision is GateDecision.FAILED:
            blockers.append(
                {
                    "code": "theorem_novelty_failed",
                    "detail": (
                        "the frozen theorem novelty gate failed; production checkpoint "
                        "generation and downstream spectrum claims are not released"
                    ),
                },
            )

    loaded: dict[str, FrozenCheckpoint] = {}
    for family, path in (
        (ModelFamily.FIXED_INPUT, fixed_bundle_path),
        (ModelFamily.CONDITION_MODULATED_PHASE, phase_bundle_path),
    ):
        if path is None:
            blockers.append(
                {
                    "code": f"{family.value}_checkpoint_missing",
                    "detail": (
                        f"a frozen {family.value} primary result bundle is required"
                    ),
                },
            )
            continue
        try:
            loaded[family.value] = load_frozen_checkpoint(
                path,
                repository_root=repository_root,
                manifest=manifest,
                expected_family=family,
            )
        except SpectrumAuditError as error:
            blockers.append(
                {
                    "code": f"{family.value}_checkpoint_incompatible",
                    "detail": str(error),
                },
            )

    if len(loaded) == 2:
        fixed = loaded[ModelFamily.FIXED_INPUT.value]
        phase = loaded[ModelFamily.CONDITION_MODULATED_PHASE.value]
        if (
            fixed.coupling_seed != phase.coupling_seed
            or fixed.optimizer_seed != phase.optimizer_seed
            or not np.array_equal(fixed.generators, phase.generators)
        ):
            blockers.append(
                {
                    "code": "checkpoint_pair_mismatch",
                    "detail": (
                        "fixed and phase checkpoints must share coupling instance, "
                        "optimizer seed, and compiled generators"
                    ),
                },
            )

    configuration = {
        "manifest_digest": manifest.digest,
        "fixed_bundle_path": (
            None if fixed_bundle_path is None else str(fixed_bundle_path)
        ),
        "phase_bundle_path": (
            None if phase_bundle_path is None else str(phase_bundle_path)
        ),
        "theorem_verdict": theorem_verdict,
        "theorem_evidence_digest": theorem_evidence_digest,
        "campaign_root": str(campaign_root),
        "grid_size": grid_size,
        "compatible_checkpoint_sha256": {
            name: checkpoint.checkpoint_sha256
            for name, checkpoint in sorted(loaded.items())
        },
    }
    if not blockers:
        completed = evaluate_production_spectrum(
            manifest,
            loaded[ModelFamily.FIXED_INPUT.value],
            loaded[ModelFamily.CONDITION_MODULATED_PHASE.value],
            grid_size=grid_size,
        )
        completed["configuration_digest"] = content_digest(configuration)
        completed["theorem_gate"] = gate_record.to_dict()
        completed["compatible_checkpoints"] = {
            name: {
                "source_run_id": checkpoint.source_run_id,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            }
            for name, checkpoint in sorted(loaded.items())
        }
        return completed
    return {
        "schema_version": 1,
        "evidence_type": "response_spectrum_production_preflight",
        "status": "blocked" if blockers else "ready",
        "campaign_version": manifest.campaign_version,
        "manifest_digest": manifest.digest,
        "configuration_digest": content_digest(configuration),
        "full_training_runs_used": 0,
        "budget_pool": "none",
        "budget_accounting": "excluded_from_all_full-training-run_pools",
        "blockers": blockers,
        "compatible_checkpoints": {
            name: {
                "source_run_id": checkpoint.source_run_id,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            }
            for name, checkpoint in sorted(loaded.items())
        },
        "interpretation": (
            "blocked status is an evidence result; no synthetic response is "
            "substituted for absent production checkpoints"
        ),
    }


def archive_spectrum_report(path: Path, report: dict[str, Any]) -> str:
    """Write one immutable, content-verified spectrum-audit JSON artifact."""
    return write_immutable_json(path, report)

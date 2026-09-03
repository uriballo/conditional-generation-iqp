"""Execution layer for the versioned exact-TVD ancilla campaign.

Training and exact evaluation are deliberately separate operations.  This
keeps 26-qubit statevectors away from concurrent training workers and lets a
failed evaluator be retried without changing a fitted checkpoint.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import resource
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .ansatz import complete_ansatz
from .artifacts import (
    read_immutable_json,
    sha256_file,
    write_immutable_bytes,
    write_immutable_json,
)
from .curie_weiss_data import exact_curie_weiss_target
from .curie_weiss_statistics import thermodynamic_metrics
from .encoding import ANGLE_ENCODING, SPIN_FLIP_BRANCH_ENCODING
from .exact import (
    enumerate_bitstrings,
    exact_iqp_visible_probabilities,
    fwht,
    shell_probabilities,
    total_variation_distance,
)
from .exact_tvd_campaign import (
    RunIdentity,
    data_initialize,
    expected_bodyness_sigmas,
    manifest_digest,
)
from .exact_tvd_training import (
    CampaignTrainingConfig,
    sample_nonidentity_masks,
    train_campaign,
)
from .gaussian_data import condition_angle_matrix, gaussian_target
from .mfc_reference_iqp import (
    bit_codes,
    load_events,
    physical_bins,
)
from .mfc_reference_iqp import (
    condition_angle as mfc_condition_angle,
)
from .mfc_reference_iqp import (
    metrics as mfc_metrics,
)
from .mmd import _target_terms
from .simulator import _estimate_pauli_z_unchecked


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _derived_seed(*coordinates: Any) -> int:
    digest = hashlib.sha256(_canonical(coordinates)).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _write_immutable_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            equal = set(stored.files) == set(arrays) and all(
                np.array_equal(stored[name], np.asarray(value))
                for name, value in arrays.items()
            )
        if not equal:
            raise RuntimeError(f"refusing to replace incompatible artifact: {path}")
        return sha256_file(path)
    return write_immutable_bytes(path, _npz_bytes(arrays))


def _condition_values(config: Mapping[str, Any], dataset: str) -> np.ndarray:
    spec = config["datasets"][dataset]
    return np.asarray(
        sorted((*spec["training_conditions"], *spec["held_out_conditions"])),
        dtype=np.float64,
    )


def _exact_target(
    dataset: str, n_visible: int, condition: float, width: float
) -> np.ndarray:
    if dataset == "gaussian":
        target = gaussian_target(n_visible, width, condition)
        return target.bitstring_probabilities(enumerate_bitstrings(n_visible))
    if dataset == "curie_weiss":
        return exact_curie_weiss_target(n_visible, condition).probabilities
    raise ValueError("MFC targets are defined by fixed test histograms")


def _draw_synthetic(
    dataset: str,
    n_visible: int,
    condition: float,
    count: int,
    seed: int,
    width: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dataset == "gaussian":
        return gaussian_target(n_visible, width, condition).sample(rng, count)
    target = exact_curie_weiss_target(n_visible, condition).probabilities
    states = enumerate_bitstrings(n_visible)
    return states[rng.choice(states.shape[0], count, p=target)].astype(np.int8)


def data_archive_path(directory: Path, dataset: str, n_visible: int) -> Path:
    return directory / f"{dataset}-n{n_visible}.npz"


def prepare_data_archive(
    config: dict[str, Any],
    directory: Path,
    *,
    dataset: str,
    n_visible: int,
    mfc_events: Path | None = None,
) -> tuple[Path, str]:
    """Create one fixed, role-separated 1000/1000/1000 archive per cell."""
    if n_visible not in config["datasets"][dataset]["visible_qubits"]:
        raise ValueError("dataset size is outside the campaign")
    conditions = _condition_values(config, dataset)
    count = int(config["data"]["train_samples"])
    root = int(config["data"]["root_seed"])
    digest = manifest_digest(config)
    roles: dict[str, list[np.ndarray]] = {
        name: [] for name in ("train", "validation", "test")
    }
    role_seeds: dict[str, list[int]] = {name: [] for name in roles}
    source_digest = "analytic"
    events = None
    if dataset == "mfc":
        if mfc_events is None:
            raise ValueError("MFC preparation requires an event archive")
        events = load_events(mfc_events)
        source_digest = sha256_file(mfc_events)
    for condition in conditions:
        for role in roles:
            seed = _derived_seed(
                digest, root, dataset, n_visible, float(condition), role
            )
            role_seeds[role].append(seed)
            if dataset == "mfc":
                assert events is not None
                rng = np.random.default_rng(seed)
                indices = rng.choice(len(events[int(condition)]), count, replace=False)
                # Derive all three seeds independently, then reject overlap below.
                labels, _ = physical_bins(events[int(condition)], int(condition), 8)
                roles[role].append(bit_codes(labels[indices]).astype(np.int8))
            else:
                roles[role].append(
                    _draw_synthetic(
                        dataset,
                        n_visible,
                        float(condition),
                        count,
                        seed,
                        float(config["datasets"]["gaussian"]["width"]),
                    )
                )
    # MFC uses a single permutation so its role splits are provably disjoint.
    if dataset == "mfc":
        assert events is not None
        roles = {name: [] for name in roles}
        for condition in conditions:
            energy = int(condition)
            seed = _derived_seed(digest, root, dataset, n_visible, energy, "all_roles")
            order = np.random.default_rng(seed).permutation(len(events[energy]))[
                : 3 * count
            ]
            labels, _ = physical_bins(events[energy], energy, 8)
            for offset, role in enumerate(("train", "validation", "test")):
                roles[role].append(
                    bit_codes(
                        labels[order[offset * count : (offset + 1) * count]]
                    ).astype(np.int8)
                )
    arrays: dict[str, np.ndarray] = {
        "configuration_digest": np.asarray(digest, dtype="S64"),
        "dataset": np.asarray(dataset, dtype=f"S{len(dataset)}"),
        "visible_qubits": np.asarray(n_visible, dtype=np.int64),
        "conditions": conditions,
        "source_digest": np.asarray(source_digest, dtype="S64"),
    }
    for role, values in roles.items():
        arrays[role] = np.asarray(values, dtype=np.int8)
        arrays[f"{role}_seeds"] = np.asarray(role_seeds[role], dtype=np.uint64)
    path = data_archive_path(directory, dataset, n_visible)
    return path, _write_immutable_npz(path, arrays)


def load_data_archive(
    config: dict[str, Any], directory: Path, dataset: str, n_visible: int
) -> dict[str, np.ndarray]:
    path = data_archive_path(directory, dataset, n_visible)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if arrays["configuration_digest"].item().decode() != manifest_digest(config):
        raise ValueError("data archive belongs to another manifest")
    conditions = arrays["conditions"]
    if not np.array_equal(conditions, _condition_values(config, dataset)):
        raise ValueError("data conditions differ from the manifest")
    for role in ("train", "validation", "test"):
        expected = (len(conditions), int(config["data"][f"{role}_samples"]), n_visible)
        if arrays[role].shape != expected:
            raise ValueError(f"{role} data shape differs")
    return arrays


def _curie_angle(beta: float) -> float:
    pieces = (
        (0.5, 0.9, 0.175, -0.14),
        (0.9, 1.0, 0.7, -0.07),
        (1.0, 1.2, 0.4, 0.0),
        (1.2, 1.5, 0.2, 0.08),
    )
    for start, end, slope, anchor in pieces:
        if start <= beta <= end:
            return float(np.pi * (anchor + slope * (beta - start)))
    raise ValueError("Curie--Weiss condition lies outside the frozen v3 map")


def condition_angles(
    dataset: str, conditions: Sequence[float], total_qubits: int
) -> np.ndarray:
    values = np.asarray(conditions, dtype=np.float64)
    if dataset == "gaussian":
        return condition_angle_matrix(values, total_qubits)
    if dataset == "curie_weiss":
        return np.repeat(
            np.asarray([_curie_angle(float(v)) for v in values])[:, None],
            total_qubits,
            axis=1,
        )
    if dataset == "mfc":
        return np.repeat(
            np.asarray([mfc_condition_angle(float(v)) for v in values])[:, None],
            total_qubits,
            axis=1,
        )
    raise ValueError(f"unknown dataset: {dataset}")


def _role_indices(
    config: dict[str, Any], run: RunIdentity
) -> tuple[np.ndarray, np.ndarray]:
    all_conditions = _condition_values(config, run.dataset)
    spec = config["datasets"][run.dataset]
    if run.family == "conditioned":
        selected = np.asarray(spec["training_conditions"], dtype=np.float64)
    else:
        if run.condition is None:
            raise ValueError("independent run lacks its condition")
        selected = np.asarray([run.condition], dtype=np.float64)
    indices = np.asarray(
        [
            int(np.flatnonzero(np.isclose(all_conditions, value))[0])
            for value in selected
        ]
    )
    return selected, indices


def _ancilla_support(generators: np.ndarray, n_visible: int) -> np.ndarray:
    return np.any(np.asarray(generators)[:, n_visible:] != 0, axis=1)


def _weighted_loss_functions(
    *,
    generators: np.ndarray,
    angles: np.ndarray,
    validation_targets: np.ndarray,
    sigmas: tuple[float, ...],
    n_ops: int,
    n_samples: int,
    seed: int,
    encoding: str,
) -> tuple[Any, Any, list[tuple[np.ndarray, np.ndarray]]]:
    """Build fixed-operator train and validation sampled-MMD functions."""
    n_conditions = angles.shape[0]
    n_visible = validation_targets.shape[-1]
    n_ancillas = generators.shape[1] - n_visible
    banks: list[list[tuple[jnp.ndarray, jnp.ndarray]]] = []
    serializable: list[tuple[np.ndarray, np.ndarray]] = []
    for bandwidth, sigma in enumerate(sigmas):
        per_condition = []
        for condition in range(n_conditions):
            key = jax.random.PRNGKey(
                _derived_seed(seed, bandwidth, condition) & (2**32 - 1)
            )
            masks, weights = sample_nonidentity_masks(key, n_ops, n_visible, sigma)
            per_condition.append((masks, weights))
            serializable.append((np.asarray(masks), np.asarray(weights)))
        banks.append(per_condition)

    generators_jax = jnp.asarray(generators)
    angles_jax = jnp.asarray(angles)

    def estimate(
        params: jnp.ndarray, targets: jnp.ndarray, key: jax.Array
    ) -> jnp.ndarray:
        values = []
        keys = jax.random.split(key, len(sigmas) * n_conditions)
        cursor = 0
        for per_condition in banks:
            condition_values = []
            for condition, (visible_masks, weights) in enumerate(per_condition):
                if n_ancillas:
                    model_masks = jnp.pad(visible_masks, ((0, 0), (0, n_ancillas)))
                else:
                    model_masks = visible_masks
                model = _estimate_pauli_z_unchecked(
                    params,
                    generators_jax,
                    model_masks,
                    angles_jax[condition : condition + 1],
                    keys[cursor],
                    n_samples,
                    encoding,
                )
                cursor += 1
                target_means, target_squares = _target_terms(
                    targets[condition : condition + 1], visible_masks
                )
                sums = jnp.sum(model.sample_values, axis=1)
                squares = (sums**2 - jnp.sum(model.sample_values**2, axis=1)) / (
                    n_samples * (n_samples - 1)
                )
                per_mask = squares - 2.0 * model.values * target_means + target_squares
                condition_values.append(jnp.sum(per_mask[0] * weights))
            values.append(jnp.stack(condition_values))
        return jnp.mean(jnp.stack(values))

    fixed_validation = jnp.asarray(validation_targets)
    validation_key = jax.random.PRNGKey(_derived_seed(seed, "validation") & (2**32 - 1))

    def training_loss(
        params: jnp.ndarray, batch: jnp.ndarray, key: jax.Array
    ) -> jnp.ndarray:
        return estimate(params, batch, key)

    def validation_mmd(params: jnp.ndarray) -> jnp.ndarray:
        return estimate(params, fixed_validation, validation_key)

    return training_loss, validation_mmd, serializable


def load_protocol_selection(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Load the immutable gate that freezes estimator and optimizer choices."""
    payload = read_immutable_json(path)
    if payload.get("configuration_digest") != manifest_digest(config):
        raise ValueError("protocol selection belongs to another manifest")
    if payload.get("status") != "passed":
        raise RuntimeError("production is blocked: protocol selection has not passed")
    n_samples = payload.get("n_samples")
    if n_samples not in config["estimator_calibration"]["candidates"]:
        raise ValueError("selected n_samples is outside the calibration candidates")
    if payload.get("optimizer") not in {"adam_cosine", "schedule_free_adamw"}:
        raise ValueError("unknown selected optimizer")
    preflights = payload.get("exact_preflights", [])
    if not any(
        value.get("passed") is True
        and int(value.get("total_qubits", 0)) == 26
        and int(value.get("peak_rss_bytes", 2**63))
        < int(config["evaluation"]["maximum_exact_peak_bytes"])
        and value.get("configuration_digest") == manifest_digest(config)
        for value in preflights
    ):
        raise RuntimeError("production is blocked: exact preflight evidence is absent")
    return payload


def _source_tree_digest(repository: Path) -> str:
    names = (
        "configs/exact_tvd_ancilla_v1.json",
        "src/iqp_cond/exact.py",
        "src/iqp_cond/exact_tvd_campaign.py",
        "src/iqp_cond/exact_tvd_runner.py",
        "src/iqp_cond/exact_tvd_training.py",
        "src/iqp_cond/mmd.py",
        "src/iqp_cond/simulator.py",
    )
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((repository / name).read_bytes())
    return digest.hexdigest()


def _git_revision(repository: Path) -> str | None:
    """Return the revision when Git metadata accompanies the source bundle."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    return None


def _runtime_metadata(start: float) -> dict[str, Any]:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if platform.system() == "Darwin" else peak * 1024)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "jax_backend": jax.default_backend(),
        "logical_cpus": os.cpu_count(),
        "runtime_seconds": time.monotonic() - start,
        "peak_rss_bytes": peak_bytes,
    }


def calibration_cell(
    *,
    config: dict[str, Any],
    dataset: str,
    ancillas: int,
    seed: int,
    n_samples: int,
    data_directory: Path,
    repeats: int = 5,
) -> dict[str, Any]:
    """Measure all four estimator gates in one representative pilot cell."""
    if dataset not in {"gaussian", "curie_weiss", "mfc"}:
        raise ValueError("unknown calibration dataset")
    n_visible = 3 if dataset == "mfc" else 8
    archive = load_data_archive(config, data_directory, dataset, n_visible)
    conditions = np.asarray(
        config["datasets"][dataset]["training_conditions"], dtype=np.float64
    )
    all_conditions = _condition_values(config, dataset)
    indices = np.asarray(
        [
            int(np.flatnonzero(np.isclose(all_conditions, value))[0])
            for value in conditions
        ]
    )
    targets = archive["train"][indices, :100]
    total = n_visible + ancillas
    generators = np.asarray(complete_ansatz(total).generators)
    initial = data_initialize(
        archive["train"][indices].reshape(-1, n_visible),
        generators,
        _ancilla_support(generators, n_visible),
        seed,
    )
    angles = condition_angles(dataset, conditions, total)
    sigmas = expected_bodyness_sigmas(n_visible)
    encoding = SPIN_FLIP_BRANCH_ENCODING if dataset == "curie_weiss" else ANGLE_ENCODING

    def make(sample_count: int) -> tuple[Any, Any, list[tuple[np.ndarray, np.ndarray]]]:
        return _weighted_loss_functions(
            generators=generators,
            angles=angles,
            validation_targets=targets,
            sigmas=sigmas,
            n_ops=int(config["training"]["n_ops_per_condition_per_bandwidth"]),
            n_samples=sample_count,
            seed=seed,
            encoding=encoding,
        )

    candidate_loss, _, operator_bank = make(n_samples)
    reference_loss, _, _ = make(max(config["estimator_calibration"]["candidates"]))
    params = jnp.asarray(initial)
    candidate_value_grad = jax.jit(jax.value_and_grad(candidate_loss))
    reference_value_grad = jax.jit(jax.value_and_grad(reference_loss))
    candidate_losses = []
    candidate_gradients = []
    reference_gradients = []
    for repeat in range(repeats):
        key = jax.random.PRNGKey(_derived_seed(seed, n_samples, repeat) & (2**32 - 1))
        value, gradient = candidate_value_grad(params, jnp.asarray(targets), key)
        candidate_losses.append(float(value))
        candidate_gradients.append(np.asarray(gradient))
        reference_key = jax.random.PRNGKey(
            _derived_seed(seed, "reference", repeat) & (2**32 - 1)
        )
        _, reference_gradient = reference_value_grad(
            params, jnp.asarray(targets), reference_key
        )
        reference_gradients.append(np.asarray(reference_gradient))
    candidate_gradient = np.mean(candidate_gradients, axis=0)
    reference_gradient = np.mean(reference_gradients, axis=0)
    denominator = max(
        np.linalg.norm(candidate_gradient) * np.linalg.norm(reference_gradient), 1e-15
    )
    cosine = float(candidate_gradient @ reference_gradient / denominator)
    relative_error = float(
        np.linalg.norm(candidate_gradient - reference_gradient)
        / max(np.linalg.norm(reference_gradient), 1e-15)
    )

    # Compare sampled Pauli expectations with exact visible Fourier coefficients.
    exact_visible = exact_iqp_visible_probabilities(
        initial, generators, angles, n_visible, ancillas, encoding
    )
    exact_fouriers = np.stack([fwht(row).real for row in exact_visible])
    squared_errors = []
    bank_index = 0
    generators_jax = jnp.asarray(generators)
    angles_jax = jnp.asarray(angles)
    for bandwidth in range(len(sigmas)):
        for condition in range(len(conditions)):
            masks, weights = operator_bank[bank_index]
            bank_index += 1
            model_masks = np.pad(masks, ((0, 0), (0, ancillas)))
            estimate = _estimate_pauli_z_unchecked(
                params,
                generators_jax,
                jnp.asarray(model_masks),
                angles_jax[condition : condition + 1],
                jax.random.PRNGKey(
                    _derived_seed(seed, n_samples, "rmse", bandwidth, condition)
                    & (2**32 - 1)
                ),
                n_samples,
                encoding,
            )
            mask_indices = masks.astype(np.int64) @ (1 << np.arange(n_visible))
            normalized_weights = weights / weights.sum()
            squared_errors.append(
                float(
                    normalized_weights
                    @ (
                        np.asarray(estimate.values[0])
                        - exact_fouriers[condition, mask_indices]
                    )
                    ** 2
                )
            )
    loss_values = np.asarray(candidate_losses)
    return {
        "dataset": dataset,
        "visible_qubits": n_visible,
        "ancillas": ancillas,
        "seed": seed,
        "n_samples": n_samples,
        "pauli_expectation_rmse": float(np.sqrt(np.mean(squared_errors))),
        "mmd_cv": float(
            np.std(loss_values, ddof=1) / max(abs(loss_values.mean()), 1e-15)
        ),
        "gradient_cosine": cosine,
        "relative_gradient_error": relative_error,
        "repeats": repeats,
    }


def train_run(
    *,
    repository: Path,
    config: dict[str, Any],
    selection: dict[str, Any],
    run: RunIdentity,
    data_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Fit one identity and persist immutable candidate checkpoints."""
    if run.configuration_digest != manifest_digest(config):
        raise ValueError("run belongs to another campaign")
    training_path = output_directory / "training" / f"{run.run_id}.json"
    if training_path.exists():
        payload = read_immutable_json(training_path)
        if payload.get("run_id") != run.run_id:
            raise ValueError("training artifact belongs to another run")
        return payload
    start = time.monotonic()
    archive = load_data_archive(config, data_directory, run.dataset, run.visible_qubits)
    selected_conditions, selected_indices = _role_indices(config, run)
    training = archive["train"][selected_indices]
    validation = archive["validation"][selected_indices]
    total = run.visible_qubits + run.ancillas
    ansatz = complete_ansatz(total)
    generators = np.asarray(ansatz.generators)
    pooled = training.reshape(-1, run.visible_qubits)
    initial = data_initialize(
        pooled,
        generators,
        _ancilla_support(generators, run.visible_qubits),
        run.seed,
    )
    angles = condition_angles(run.dataset, selected_conditions, total)
    sigmas = expected_bodyness_sigmas(run.visible_qubits)
    encoding = (
        SPIN_FLIP_BRANCH_ENCODING if run.dataset == "curie_weiss" else ANGLE_ENCODING
    )
    training_loss, validation_mmd, masks = _weighted_loss_functions(
        generators=generators,
        angles=angles,
        validation_targets=validation,
        sigmas=sigmas,
        n_ops=int(config["training"]["n_ops_per_condition_per_bandwidth"]),
        n_samples=int(selection["n_samples"]),
        seed=run.seed,
        encoding=encoding,
    )
    optimizer = str(selection["optimizer"])
    optimizer_settings = selection.get("optimizer_settings", {})
    training_config = CampaignTrainingConfig(
        steps=int(config["training"]["updates"]),
        batch_size=int(config["training"]["samples_per_condition_per_update"]),
        n_samples=int(selection["n_samples"]),
        validation_interval=int(config["training"]["validation_interval"]),
        optimizer=optimizer,
        learning_rate=float(optimizer_settings.get("learning_rate", 0.01)),
        final_learning_rate=float(optimizer_settings.get("final_learning_rate", 0.001)),
        warmup_steps=int(optimizer_settings.get("warmup_steps", 100)),
        b1=float(optimizer_settings.get("b1", 0.9)),
        b2=float(optimizer_settings.get("b2", 0.999)),
    )
    mask_digest = hashlib.sha256()
    for mask, weights in masks:
        mask_digest.update(mask.tobytes())
        mask_digest.update(weights.tobytes())
    checkpoint_path = output_directory / "checkpoints" / f"{run.run_id}.npz"
    recovered_checkpoint = checkpoint_path.exists()
    if recovered_checkpoint:
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            checkpoint_digest = checkpoint["configuration_digest"].item().decode()
            if checkpoint_digest != run.configuration_digest:
                raise ValueError("checkpoint belongs to another campaign")
            if checkpoint["run_id"].item().decode() != run.run_id:
                raise ValueError("checkpoint belongs to another run")
        checkpoint_digest = sha256_file(checkpoint_path)
    else:
        result = train_campaign(
            initial,
            training,
            validation_mmd,
            training_loss,
            run.seed,
            training_config,
        )
        checkpoints = np.stack(
            [checkpoint.params for checkpoint in result.checkpoints]
        )
        arrays = {
            "configuration_digest": np.asarray(
                run.configuration_digest, dtype="S64"
            ),
            "run_id": np.asarray(run.run_id, dtype="S64"),
            "initial_params": initial,
            "checkpoint_params": checkpoints,
            "checkpoint_steps": np.asarray(
                [value.step for value in result.checkpoints]
            ),
            "checkpoint_validation_mmd": np.asarray(
                [value.validation_mmd for value in result.checkpoints]
            ),
            "checkpoint_reasons": np.asarray(
                [value.reason for value in result.checkpoints], dtype="S32"
            ),
            "train_steps": result.train_steps,
            "train_losses": result.train_losses,
            "validation_steps": result.validation_steps,
            "validation_losses": result.validation_losses,
            "conditions": selected_conditions,
        }
        checkpoint_digest = _write_immutable_npz(checkpoint_path, arrays)
    payload = {
        "schema_version": 1,
        "campaign": "exact_tvd_ancilla_v1",
        "stage": "training",
        "run_id": run.run_id,
        "run_identity": run.to_dict(),
        "protocol_selection_digest": hashlib.sha256(_canonical(selection)).hexdigest(),
        "data": {
            "path": str(
                data_archive_path(data_directory, run.dataset, run.visible_qubits)
            ),
            "sha256": sha256_file(
                data_archive_path(data_directory, run.dataset, run.visible_qubits)
            ),
            "split_hashes": {
                role: hashlib.sha256(archive[role].tobytes()).hexdigest()
                for role in ("train", "validation", "test")
            },
        },
        "checkpoint_path": str(checkpoint_path.relative_to(output_directory)),
        "checkpoint_sha256": checkpoint_digest,
        "operator_bank_sha256": mask_digest.hexdigest(),
        "parameter_count": ansatz.n_parameters,
        "source_tree_sha256": _source_tree_digest(repository),
        "git_revision": _git_revision(repository),
        "recovered_existing_checkpoint": recovered_checkpoint,
        "runtime": _runtime_metadata(start),
    }
    write_immutable_json(training_path, payload)
    return payload


def _empirical_probabilities(samples: np.ndarray) -> np.ndarray:
    powers = 1 << np.arange(samples.shape[1], dtype=np.int64)
    labels = np.asarray(samples, dtype=np.int64) @ powers
    return np.bincount(labels, minlength=2 ** samples.shape[1]).astype(float) / len(
        labels
    )


def _diagnostic_mmd(
    model: np.ndarray, target: np.ndarray, sigmas: Sequence[float]
) -> float:
    n = int(np.log2(model.size))
    weights = enumerate_bitstrings(n).sum(axis=1)
    model_fourier = fwht(model).real
    target_fourier = fwht(target).real
    values = []
    for sigma in sigmas:
        p = (1.0 - np.exp(-1.0 / (2.0 * sigma**2))) / 2.0
        q = p**weights * (1.0 - p) ** (n - weights)
        values.append(float(q @ ((model_fourier - target_fourier) ** 2)))
    return float(np.mean(values))


def _metrics(
    dataset: str,
    n_visible: int,
    condition: float,
    target: np.ndarray,
    model: np.ndarray,
    sigmas: Sequence[float],
) -> dict[str, float]:
    if dataset == "curie_weiss":
        result = thermodynamic_metrics(
            exact_curie_weiss_target(n_visible, condition), model
        ).to_dict()
        result["test_mmd"] = _diagnostic_mmd(model, target, sigmas)
        return result
    if dataset == "mfc":
        result = mfc_metrics(target, model)
        result["bitstring_tvd"] = result.pop("tvd")
        result["test_mmd"] = _diagnostic_mmd(model, target, sigmas)
        return result
    bits = enumerate_bitstrings(n_visible)
    target_profile = shell_probabilities(target, bits)
    model_profile = shell_probabilities(model, bits)
    weights = bits.sum(axis=1)
    target_mean = float(target @ weights)
    model_mean = float(model @ weights)
    return {
        "bitstring_tvd": total_variation_distance(target, model),
        "profile_tvd": total_variation_distance(target_profile, model_profile),
        "mean_error": model_mean - target_mean,
        "variance_error": float(
            model @ ((weights - model_mean) ** 2)
            - target @ ((weights - target_mean) ** 2)
        ),
        "test_mmd": _diagnostic_mmd(model, target, sigmas),
    }


def evaluate_run(
    *,
    repository: Path,
    config: dict[str, Any],
    run: RunIdentity,
    data_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Select by exact validation TVD and evaluate one fit on fixed test data."""
    start = time.monotonic()
    training_payload = read_immutable_json(
        output_directory / "training" / f"{run.run_id}.json"
    )
    checkpoint_path = Path(training_payload["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = output_directory / checkpoint_path
    if sha256_file(checkpoint_path) != training_payload["checkpoint_sha256"]:
        raise ValueError("checkpoint digest mismatch")
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        params = np.asarray(checkpoint["checkpoint_params"])
        steps = np.asarray(checkpoint["checkpoint_steps"])
        reasons = np.asarray(checkpoint["checkpoint_reasons"])
        training_mmd = np.asarray(checkpoint["checkpoint_validation_mmd"])
    archive = load_data_archive(config, data_directory, run.dataset, run.visible_qubits)
    selected_conditions, selected_indices = _role_indices(config, run)
    total = run.visible_qubits + run.ancillas
    generators = np.asarray(complete_ansatz(total).generators)
    encoding = (
        SPIN_FLIP_BRANCH_ENCODING if run.dataset == "curie_weiss" else ANGLE_ENCODING
    )
    validation_angles = condition_angles(run.dataset, selected_conditions, total)
    validation_targets = [
        _empirical_probabilities(archive["validation"][index])
        for index in selected_indices
    ]
    candidate_tvds = []
    for candidate in params:
        visible = exact_iqp_visible_probabilities(
            candidate,
            generators,
            validation_angles,
            run.visible_qubits,
            run.ancillas,
            encoding,
        )
        candidate_tvds.append(
            float(
                np.mean(
                    [
                        total_variation_distance(target, prediction)
                        for target, prediction in zip(
                            validation_targets, visible, strict=True
                        )
                    ]
                )
            )
        )
    selected_index = int(np.argmin(candidate_tvds))
    selected_params = params[selected_index]
    if run.family == "conditioned":
        evaluation_conditions = _condition_values(config, run.dataset)
        evaluation_indices = np.arange(len(evaluation_conditions))
    else:
        evaluation_conditions = selected_conditions
        evaluation_indices = selected_indices
    predictions = exact_iqp_visible_probabilities(
        selected_params,
        generators,
        condition_angles(run.dataset, evaluation_conditions, total),
        run.visible_qubits,
        run.ancillas,
        encoding,
    )
    sigmas = expected_bodyness_sigmas(run.visible_qubits)
    rows = []
    evaluation_targets = []
    for condition, data_index, prediction in zip(
        evaluation_conditions, evaluation_indices, predictions, strict=True
    ):
        if run.dataset == "mfc":
            target = _empirical_probabilities(archive["test"][data_index])
        else:
            target = _exact_target(
                run.dataset,
                run.visible_qubits,
                float(condition),
                float(config["datasets"]["gaussian"]["width"]),
            )
        evaluation_targets.append(target)
        split = (
            "training"
            if np.any(
                np.isclose(
                    np.asarray(config["datasets"][run.dataset]["training_conditions"]),
                    condition,
                )
            )
            else "held_out"
        )
        row = {
            "condition": float(condition),
            "split": split,
            **_metrics(
                run.dataset,
                run.visible_qubits,
                float(condition),
                target,
                prediction,
                sigmas,
            ),
        }
        if split == "training" or run.family == "independent":
            training_target = _empirical_probabilities(archive["train"][data_index])
            row["training_mmd"] = _diagnostic_mmd(prediction, training_target, sigmas)
        rows.append(row)
    probability_path = output_directory / "probabilities" / f"{run.run_id}.npz"
    probability_digest = _write_immutable_npz(
        probability_path,
        {
            "configuration_digest": np.asarray(run.configuration_digest, dtype="S64"),
            "run_id": np.asarray(run.run_id, dtype="S64"),
            "conditions": evaluation_conditions,
            "probabilities": predictions,
            "target_probabilities": np.asarray(evaluation_targets),
            "selected_params": selected_params,
            "selected_checkpoint_index": np.asarray(selected_index, dtype=np.int64),
        },
    )
    payload = {
        "schema_version": 1,
        "campaign": "exact_tvd_ancilla_v1",
        "stage": "evaluation",
        "run_id": run.run_id,
        "run_identity": run.to_dict(),
        "probabilities_path": str(probability_path.relative_to(output_directory)),
        "probabilities_sha256": probability_digest,
        "probabilities_key": "probabilities",
        "selection": {
            "candidate_steps": steps.tolist(),
            "candidate_reasons": [value.decode() for value in reasons],
            "candidate_validation_mmd": training_mmd.tolist(),
            "candidate_mean_exact_validation_tvd": candidate_tvds,
            "selected_index": selected_index,
            "selected_step": int(steps[selected_index]),
        },
        "metrics": rows,
        "source_tree_sha256": _source_tree_digest(repository),
        "runtime": _runtime_metadata(start),
    }
    write_immutable_json(
        output_directory / "evaluations" / f"{run.run_id}.json", payload
    )
    return payload

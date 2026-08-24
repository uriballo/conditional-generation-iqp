"""Deterministic finite-sample data for the Gaussian foundations campaign."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import StrEnum
from math import pi, sqrt
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .artifacts import sha256_file
from .gaussian_protocol import (
    GaussianCampaignManifest,
    GaussianFamily,
    GaussianRunIdentity,
)
from .targets import GaussianHammingTarget


class GaussianDatasetRole(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class GaussianDatasetBatch:
    training_conditions: np.ndarray
    condition_angles: np.ndarray
    training: np.ndarray
    validation: np.ndarray
    source_path: Path
    source_sha256: str
    training_count_per_peak: int
    validation_count_per_peak: int


def target_mean(n_qubits: int, condition: float) -> float:
    if n_qubits < 2 or not 0.0 <= condition <= 1.0:
        raise ValueError("target coordinates are out of range")
    return n_qubits * (0.15 + 0.70 * condition)


def target_width(n_qubits: int, width_factor: float) -> float:
    if n_qubits < 2 or width_factor <= 0.0:
        raise ValueError("target width coordinates are out of range")
    return width_factor * sqrt(n_qubits)


def gaussian_target(
    n_qubits: int,
    width_factor: float,
    condition: float,
) -> GaussianHammingTarget:
    return GaussianHammingTarget.create(
        n_qubits,
        target_mean(n_qubits, condition),
        target_width(n_qubits, width_factor),
    )


def condition_angle(condition: float) -> float:
    if not 0.0 <= condition <= 1.0:
        raise ValueError("condition must lie in [0, 1]")
    return pi * (0.35 - 0.70 * condition)


def condition_angle_matrix(conditions: np.ndarray, n_qubits: int) -> np.ndarray:
    conditions = np.asarray(conditions, dtype=np.float64)
    if conditions.ndim != 1 or not np.all((0.0 <= conditions) & (conditions <= 1.0)):
        raise ValueError("conditions must be a one-dimensional array in [0, 1]")
    return np.repeat(
        np.asarray([condition_angle(float(value)) for value in conditions])[:, None],
        n_qubits,
        axis=1,
    )


def master_grid() -> np.ndarray:
    return np.linspace(0.0, 1.0, 9, dtype=np.float64)


def grid_indices(peak_count: int) -> tuple[int, ...]:
    if peak_count not in {2, 3, 5, 9}:
        raise ValueError("peak_count must be 2, 3, 5, or 9")
    step = 8 // (peak_count - 1)
    return tuple(range(0, 9, step))


def derive_split_seed(
    manifest: GaussianCampaignManifest,
    *,
    n_qubits: int,
    width_factor: float,
    optimizer_seed: int,
    peak_index: int,
    role: GaussianDatasetRole,
) -> int:
    if (
        n_qubits not in manifest.system_sizes
        or width_factor not in manifest.width_factors
    ):
        raise ValueError("dataset cell is outside the manifest")
    if optimizer_seed not in manifest.optimizer_seeds or peak_index not in range(9):
        raise ValueError("dataset seed coordinates are outside the manifest")
    payload = (
        f"{manifest.digest}|{manifest.raw['data']['root_seed']}|{n_qubits}|"
        f"{width_factor:.2f}|{optimizer_seed}|{peak_index}|{role.value}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def dataset_archive_path(
    directory: Path,
    n_qubits: int,
    width_factor: float,
    optimizer_seed: int,
) -> Path:
    width_code = str(round(width_factor * 100)).zfill(2)
    return directory / f"n{n_qubits}_w{width_code}_seed{optimizer_seed}.npz"


def _arrays_equal(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    try:
        with np.load(path, allow_pickle=False) as stored:
            return set(stored.files) == set(arrays) and all(
                stored[name].dtype == np.asarray(value).dtype
                and stored[name].shape == np.asarray(value).shape
                and np.array_equal(stored[name], value)
                for name, value in arrays.items()
            )
    except (OSError, ValueError):
        return False


def write_immutable_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not _arrays_equal(path, arrays):
            raise RuntimeError(
                f"refusing to rewrite incompatible Gaussian data: {path}"
            )
        return sha256_file(path)
    with NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        return write_immutable_npz(path, arrays)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def prepare_dataset_archive(
    manifest: GaussianCampaignManifest,
    directory: Path,
    *,
    n_qubits: int,
    width_factor: float,
    optimizer_seed: int,
) -> tuple[Path, str]:
    conditions = master_grid()
    training = []
    validation = []
    training_seeds = []
    validation_seeds = []
    actual_means = []
    actual_stds = []
    for peak_index, condition in enumerate(conditions):
        target = gaussian_target(n_qubits, width_factor, float(condition))
        train_seed = derive_split_seed(
            manifest,
            n_qubits=n_qubits,
            width_factor=width_factor,
            optimizer_seed=optimizer_seed,
            peak_index=peak_index,
            role=GaussianDatasetRole.TRAINING,
        )
        validation_seed = derive_split_seed(
            manifest,
            n_qubits=n_qubits,
            width_factor=width_factor,
            optimizer_seed=optimizer_seed,
            peak_index=peak_index,
            role=GaussianDatasetRole.VALIDATION,
        )
        training.append(
            target.sample(np.random.default_rng(train_seed), 10_000).astype(np.int8)
        )
        validation.append(
            target.sample(np.random.default_rng(validation_seed), 2_000).astype(np.int8)
        )
        training_seeds.append(train_seed)
        validation_seeds.append(validation_seed)
        actual_means.append(target.actual_mean)
        actual_stds.append(target.actual_std)
    path = dataset_archive_path(directory, n_qubits, width_factor, optimizer_seed)
    arrays = {
        "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
        "n_qubits": np.asarray(n_qubits, dtype=np.int64),
        "width_factor": np.asarray(width_factor, dtype=np.float64),
        "optimizer_seed": np.asarray(optimizer_seed, dtype=np.int64),
        "conditions": conditions,
        "actual_means": np.asarray(actual_means, dtype=np.float64),
        "actual_stds": np.asarray(actual_stds, dtype=np.float64),
        "training_seeds": np.asarray(training_seeds, dtype=np.int64),
        "validation_seeds": np.asarray(validation_seeds, dtype=np.int64),
        "training": np.asarray(training, dtype=np.int8),
        "validation": np.asarray(validation, dtype=np.int8),
    }
    return path, write_immutable_npz(path, arrays)


def prepare_all_datasets(
    manifest: GaussianCampaignManifest, directory: Path
) -> tuple[tuple[Path, str], ...]:
    return tuple(
        prepare_dataset_archive(
            manifest,
            directory,
            n_qubits=n_qubits,
            width_factor=width_factor,
            optimizer_seed=seed,
        )
        for n_qubits in manifest.system_sizes
        for width_factor in manifest.width_factors
        for seed in manifest.optimizer_seeds
    )


def load_dataset_for_run(
    manifest: GaussianCampaignManifest,
    directory: Path,
    run: GaussianRunIdentity,
) -> GaussianDatasetBatch:
    if run.configuration_digest != manifest.digest:
        raise ValueError("run belongs to a different Gaussian manifest")
    path = dataset_archive_path(
        directory, run.n_qubits, run.width_factor, run.optimizer_seed
    )
    archive_sha = sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        if archive["manifest_digest"].item().decode() != manifest.digest:
            raise ValueError("dataset manifest digest differs")
        all_conditions = np.asarray(archive["conditions"], dtype=np.float64)
        all_training = np.asarray(archive["training"], dtype=np.int8)
        all_validation = np.asarray(archive["validation"], dtype=np.int8)
    if run.family is GaussianFamily.SINGLE_PEAK:
        indices = (int(run.bank_peak_index),)
        train_count, validation_count = 10_000, 2_000
    else:
        indices = grid_indices(run.peak_count)
        train_count, validation_count = manifest.samples_per_peak(
            run.peak_count, run.data_regime
        )
    conditions = all_conditions[list(indices)]
    if run.family is GaussianFamily.NO_CONDITION:
        angles = np.zeros((len(indices), run.n_qubits), dtype=np.float64)
    else:
        angles = condition_angle_matrix(conditions, run.n_qubits)
    return GaussianDatasetBatch(
        training_conditions=conditions,
        condition_angles=angles,
        training=all_training[list(indices), :train_count],
        validation=all_validation[list(indices), :validation_count],
        source_path=path,
        source_sha256=archive_sha,
        training_count_per_peak=train_count,
        validation_count_per_peak=validation_count,
    )

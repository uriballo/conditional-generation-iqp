"""Exact finite-sample preparation for the arXiv foundations campaign."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .foundations_protocol import FoundationsManifest
from .ising_training import (
    DatasetRole,
    ExactTargetSet,
    RoleTaggedDataset,
    TrainingValidationData,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_exact_split_seed(
    manifest: FoundationsManifest,
    coupling_seed: int,
    beta_index: int,
    role: DatasetRole,
) -> int:
    """Derive one explicit, stable, role-separated categorical-draw seed."""
    if role not in {DatasetRole.TRAINING, DatasetRole.VALIDATION}:
        raise ValueError("only Training and Validation split seeds are defined")
    if coupling_seed not in {
        *manifest.instances.development,
        *manifest.instances.primary,
    }:
        raise ValueError("coupling seed is not active in the foundations study")
    if beta_index not in range(len(manifest.beta_grid.training)):
        raise ValueError("beta index is outside the Training grid")
    payload = (
        f"{manifest.campaign_version}|{manifest.data.root_seed}|{coupling_seed}|"
        f"{beta_index}|{role.value}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def draw_exact_dataset(
    *,
    role: DatasetRole,
    coupling_seed: int,
    beta: float,
    data_seed: int,
    bitstrings: np.ndarray,
    probabilities: np.ndarray,
    n_samples: int,
) -> RoleTaggedDataset:
    """Draw one finite split without exposing an exact law to model training."""
    if role is DatasetRole.HELD_OUT:
        raise ValueError("Held-out draws are forbidden in foundations training data")
    if role not in {DatasetRole.TRAINING, DatasetRole.VALIDATION}:
        raise ValueError("exact draws require a Training or Validation role")
    bitstrings = np.asarray(bitstrings, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        bitstrings.ndim != 2
        or probabilities.shape != (bitstrings.shape[0],)
        or n_samples < 2
    ):
        raise ValueError("bitstrings, probabilities, and sample count do not align")
    if not np.all((bitstrings == 0) | (bitstrings == 1)):
        raise ValueError("exact target bitstrings must be binary")
    if (
        np.any(probabilities < 0.0)
        or not np.all(np.isfinite(probabilities))
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0.0)
    ):
        raise ValueError("exact target probabilities must form a normalized law")
    indices = np.random.default_rng(data_seed).choice(
        bitstrings.shape[0],
        size=n_samples,
        replace=True,
        p=probabilities,
    )
    return RoleTaggedDataset(
        role=role,
        coupling_seed=coupling_seed,
        beta=beta,
        data_seed=data_seed,
        samples=bitstrings[indices],
    )


def write_immutable_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Create one typed NPZ or verify byte-independent array identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as existing:
                matches = set(existing.files) == set(arrays) and all(
                    existing[name].dtype == np.asarray(value).dtype
                    and existing[name].shape == np.asarray(value).shape
                    and np.array_equal(existing[name], value)
                    for name, value in arrays.items()
                )
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"existing foundations dataset is invalid: {path}"
            ) from error
        if not matches:
            raise RuntimeError(f"refusing to rewrite foundations dataset: {path}")
        return _sha256_file(path)
    with NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
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
    return _sha256_file(path)


@dataclass(frozen=True, slots=True)
class PreparedDatasetArchive:
    coupling_seed: int
    path: str
    sha256: str
    target_sha256: str


def _target_arrays(
    manifest: FoundationsManifest,
    target_path: Path,
    coupling_seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    target_sha256 = _sha256_file(target_path)
    try:
        with np.load(target_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "coupling_seed",
                "betas",
                "couplings",
                "bitstrings",
                "energies",
                "probabilities",
            }:
                raise ValueError("target archive schema differs from the exact source")
            if int(archive["coupling_seed"]) != coupling_seed:
                raise ValueError("target archive coupling seed differs")
            expected_betas = manifest.beta_grid.training + manifest.beta_grid.held_out
            if not np.allclose(
                archive["betas"],
                expected_betas,
                atol=1e-14,
                rtol=0.0,
            ):
                raise ValueError("target archive beta grid differs")
            bitstrings = np.asarray(archive["bitstrings"], dtype=np.int8)
            probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid exact target archive: {target_path}") from error
    if bitstrings.shape != (65_536, 16) or probabilities.shape != (9, 65_536):
        raise RuntimeError("foundations exact target archive has an invalid shape")
    return bitstrings, probabilities, target_sha256


def prepare_foundations_datasets(
    manifest: FoundationsManifest,
    target_directory: Path,
    output_directory: Path,
    *,
    coupling_seeds: tuple[int, ...] | None = None,
) -> tuple[PreparedDatasetArchive, ...]:
    """Create immutable exact-draw archives for every active instance."""
    active = manifest.instances.development + manifest.instances.primary
    selected = active if coupling_seeds is None else coupling_seeds
    if (
        not selected
        or len(selected) != len(set(selected))
        or not set(selected).issubset(active)
    ):
        raise ValueError("dataset preparation requires unique active coupling seeds")
    prepared = []
    for coupling_seed in selected:
        target_path = target_directory / f"coupling_{coupling_seed}.npz"
        bitstrings, probabilities, target_sha256 = _target_arrays(
            manifest,
            target_path,
            coupling_seed,
        )
        training = []
        validation = []
        training_seeds = []
        validation_seeds = []
        for beta_index, beta in enumerate(manifest.beta_grid.training):
            training_seed = derive_exact_split_seed(
                manifest,
                coupling_seed,
                beta_index,
                DatasetRole.TRAINING,
            )
            validation_seed = derive_exact_split_seed(
                manifest,
                coupling_seed,
                beta_index,
                DatasetRole.VALIDATION,
            )
            training.append(
                draw_exact_dataset(
                    role=DatasetRole.TRAINING,
                    coupling_seed=coupling_seed,
                    beta=beta,
                    data_seed=training_seed,
                    bitstrings=bitstrings,
                    probabilities=probabilities[beta_index],
                    n_samples=manifest.data.training_samples,
                ).samples,
            )
            validation.append(
                draw_exact_dataset(
                    role=DatasetRole.VALIDATION,
                    coupling_seed=coupling_seed,
                    beta=beta,
                    data_seed=validation_seed,
                    bitstrings=bitstrings,
                    probabilities=probabilities[beta_index],
                    n_samples=manifest.data.validation_samples,
                ).samples,
            )
            training_seeds.append(training_seed)
            validation_seeds.append(validation_seed)
        destination = output_directory / f"coupling_{coupling_seed}.npz"
        arrays = {
            "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
            "coupling_seed": np.asarray(coupling_seed, dtype=np.int64),
            "betas": np.asarray(manifest.beta_grid.training, dtype=np.float64),
            "training": np.asarray(training, dtype=np.int8),
            "validation": np.asarray(validation, dtype=np.int8),
            "training_seeds": np.asarray(training_seeds, dtype=np.int64),
            "validation_seeds": np.asarray(validation_seeds, dtype=np.int64),
            "target_sha256": np.asarray(target_sha256, dtype="S64"),
        }
        archive_sha256 = write_immutable_npz(destination, arrays)
        prepared.append(
            PreparedDatasetArchive(
                coupling_seed=coupling_seed,
                path=str(destination),
                sha256=archive_sha256,
                target_sha256=target_sha256,
            ),
        )
    return tuple(prepared)


def load_foundations_training_validation(
    directory: Path,
    manifest: FoundationsManifest,
    coupling_seed: int,
    betas: tuple[float, ...],
) -> TrainingValidationData:
    """Load only declared Training/Validation conditions from one archive."""
    if not betas or not set(betas).issubset(manifest.beta_grid.training):
        raise ValueError("requested datasets must be non-empty Training betas")
    path = directory / f"coupling_{coupling_seed}.npz"
    source_sha256 = _sha256_file(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if bytes(archive["manifest_digest"]).decode("ascii") != manifest.digest:
                raise ValueError("dataset archive belongs to another manifest")
            if int(archive["coupling_seed"]) != coupling_seed:
                raise ValueError("dataset archive coupling seed differs")
            archived_betas = tuple(float(beta) for beta in archive["betas"])
            if archived_betas != manifest.beta_grid.training:
                raise ValueError("dataset archive Training grid differs")
            training = np.asarray(archive["training"], dtype=np.int8)
            validation = np.asarray(archive["validation"], dtype=np.int8)
            training_seeds = np.asarray(archive["training_seeds"], dtype=np.int64)
            validation_seeds = np.asarray(archive["validation_seeds"], dtype=np.int64)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid foundations dataset archive: {path}") from error
    expected_shapes = (
        (5, manifest.data.training_samples, 16),
        (5, manifest.data.validation_samples, 16),
    )
    if (training.shape, validation.shape) != expected_shapes:
        raise RuntimeError("foundations dataset counts or qubit width differ")
    indices = tuple(archived_betas.index(beta) for beta in betas)
    return TrainingValidationData(
        training=tuple(
            RoleTaggedDataset(
                DatasetRole.TRAINING,
                coupling_seed,
                archived_betas[index],
                int(training_seeds[index]),
                training[index],
                source_sha256,
            )
            for index in indices
        ),
        validation=tuple(
            RoleTaggedDataset(
                DatasetRole.VALIDATION,
                coupling_seed,
                archived_betas[index],
                int(validation_seeds[index]),
                validation[index],
                source_sha256,
            )
            for index in indices
        ),
    )


@dataclass(frozen=True, slots=True)
class FoundationsExactArchive:
    """Post-selection exact laws and physical arrays for all nine Conditions."""

    training: ExactTargetSet
    held_out: ExactTargetSet
    bitstrings: np.ndarray
    energies: np.ndarray
    couplings: np.ndarray
    source_sha256: str


def load_foundations_exact_archive(
    path: Path,
    manifest: FoundationsManifest,
    coupling_seed: int,
) -> FoundationsExactArchive:
    """Load exact laws only through the evaluation-facing API."""
    source_sha256 = _sha256_file(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "coupling_seed",
                "betas",
                "couplings",
                "bitstrings",
                "energies",
                "probabilities",
            }
            if set(archive.files) != required:
                raise ValueError("exact target archive schema differs")
            if int(archive["coupling_seed"]) != coupling_seed:
                raise ValueError("exact target archive coupling seed differs")
            betas = tuple(float(value) for value in archive["betas"])
            expected = manifest.beta_grid.training + manifest.beta_grid.held_out
            if not np.allclose(betas, expected, atol=1e-14, rtol=0.0):
                raise ValueError("exact target archive beta grid differs")
            probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
            bitstrings = np.asarray(archive["bitstrings"], dtype=np.int8)
            energies = np.asarray(archive["energies"], dtype=np.float64)
            couplings = np.asarray(archive["couplings"], dtype=np.float64)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid foundations exact archive: {path}") from error
    if (
        probabilities.shape != (9, 65_536)
        or bitstrings.shape != (65_536, 16)
        or energies.shape != (65_536,)
        or couplings.shape != (16, 16)
    ):
        raise RuntimeError("foundations exact archive shapes differ")
    split = len(manifest.beta_grid.training)
    return FoundationsExactArchive(
        training=ExactTargetSet(
            DatasetRole.TRAINING,
            coupling_seed,
            manifest.beta_grid.training,
            probabilities[:split],
            source_sha256,
        ),
        held_out=ExactTargetSet(
            DatasetRole.HELD_OUT,
            coupling_seed,
            manifest.beta_grid.held_out,
            probabilities[split:],
            source_sha256,
        ),
        bitstrings=bitstrings,
        energies=energies,
        couplings=couplings,
        source_sha256=source_sha256,
    )

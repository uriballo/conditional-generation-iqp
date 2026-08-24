"""Exact targets and role-separated finite data for Curie--Weiss."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from .curie_weiss_protocol import (
    CurieWeissCampaignManifest,
    CurieWeissFamily,
    CurieWeissRunIdentity,
    conditioned_angle,
)
from .exact import enumerate_bitstrings
from .gaussian_data import write_immutable_npz


class CurieWeissDatasetRole(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class CurieWeissTarget:
    n_qubits: int
    beta: float
    probabilities: np.ndarray
    magnetization: np.ndarray
    energy: np.ndarray

    @property
    def profile(self) -> np.ndarray:
        bits = enumerate_bitstrings(self.n_qubits)
        return np.bincount(
            bits.sum(axis=1),
            weights=self.probabilities,
            minlength=self.n_qubits + 1,
        )


@dataclass(frozen=True, slots=True)
class CurieWeissDatasetBatch:
    training_betas: np.ndarray
    condition_angles: np.ndarray
    training: np.ndarray
    validation: np.ndarray
    source_path: Path
    source_sha256: str


def exact_curie_weiss_target(n_qubits: int, beta: float) -> CurieWeissTarget:
    if n_qubits not in {8, 12, 16} or not np.isfinite(beta) or beta <= 0:
        raise ValueError("invalid Curie--Weiss target coordinates")
    bits = enumerate_bitstrings(n_qubits)
    magnetization_sum = (2.0 * bits - 1.0).sum(axis=1)
    energy = -(magnetization_sum**2) / (2.0 * n_qubits)
    log_weights = -beta * energy
    probabilities = np.exp(log_weights - np.max(log_weights))
    probabilities /= probabilities.sum()
    return CurieWeissTarget(
        n_qubits,
        float(beta),
        probabilities,
        magnetization_sum / n_qubits,
        energy,
    )


def exact_two_block_curie_weiss_target(
    m_a: int, m_b: int, beta: float
) -> np.ndarray:
    """Exact Z2 x Z2 symmetric two-block Curie--Weiss target probabilities.

    ``p(s) ~ exp(beta (M_A^2/(2 m_a) + M_B^2/(2 m_b)))`` with block
    magnetizations ``M_A`` and ``M_B``.  Invariant under independently
    flipping each block and under permuting spins within each block.
    """
    n = m_a + m_b
    if n < 2 or not np.isfinite(beta) or beta <= 0:
        raise ValueError("invalid two-block target coordinates")
    bits = enumerate_bitstrings(n)
    spins = 2.0 * bits - 1.0
    m_a_val = spins[:, :m_a].sum(axis=1)
    m_b_val = spins[:, m_a:].sum(axis=1)
    log_weights = beta * (
        m_a_val**2 / (2.0 * m_a) + m_b_val**2 / (2.0 * m_b)
    )
    probabilities = np.exp(log_weights - np.max(log_weights))
    probabilities /= probabilities.sum()
    return probabilities


def derive_split_seed(
    manifest: CurieWeissCampaignManifest,
    *,
    n_qubits: int,
    optimizer_seed: int,
    beta_index: int,
    role: CurieWeissDatasetRole,
) -> int:
    if (
        n_qubits not in manifest.system_sizes
        or optimizer_seed not in manifest.optimizer_seeds
    ):
        raise ValueError("dataset cell is outside the manifest")
    if beta_index not in range(len(manifest.training_betas)):
        raise ValueError("beta index is outside the training grid")
    raw = (
        f"{manifest.digest}|{manifest.raw['data']['root_seed']}|{n_qubits}|"
        f"{optimizer_seed}|{beta_index}|{role.value}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & (2**63 - 1)


def dataset_archive_path(directory: Path, n_qubits: int, seed: int) -> Path:
    return directory / f"n{n_qubits}_seed{seed}.npz"


def prepare_dataset_archive(
    manifest: CurieWeissCampaignManifest,
    directory: Path,
    *,
    n_qubits: int,
    optimizer_seed: int,
) -> tuple[Path, str]:
    training, validation, training_seeds, validation_seeds = [], [], [], []
    for index, beta in enumerate(manifest.training_betas):
        target = exact_curie_weiss_target(n_qubits, beta)
        train_seed = derive_split_seed(
            manifest,
            n_qubits=n_qubits,
            optimizer_seed=optimizer_seed,
            beta_index=index,
            role=CurieWeissDatasetRole.TRAINING,
        )
        valid_seed = derive_split_seed(
            manifest,
            n_qubits=n_qubits,
            optimizer_seed=optimizer_seed,
            beta_index=index,
            role=CurieWeissDatasetRole.VALIDATION,
        )
        states = enumerate_bitstrings(n_qubits)
        training.append(
            states[
                np.random.default_rng(train_seed).choice(
                    len(states),
                    10_000,
                    p=target.probabilities,
                )
            ]
        )
        validation.append(
            states[
                np.random.default_rng(valid_seed).choice(
                    len(states),
                    2_000,
                    p=target.probabilities,
                )
            ]
        )
        training_seeds.append(train_seed)
        validation_seeds.append(valid_seed)
    path = dataset_archive_path(directory, n_qubits, optimizer_seed)
    arrays = {
        "manifest_digest": np.asarray(manifest.digest, dtype="S64"),
        "n_qubits": np.asarray(n_qubits, dtype=np.int64),
        "optimizer_seed": np.asarray(optimizer_seed, dtype=np.int64),
        "training_betas": np.asarray(manifest.training_betas, dtype=np.float64),
        "training_seeds": np.asarray(training_seeds, dtype=np.int64),
        "validation_seeds": np.asarray(validation_seeds, dtype=np.int64),
        "training": np.asarray(training, dtype=np.int8),
        "validation": np.asarray(validation, dtype=np.int8),
    }
    return path, write_immutable_npz(path, arrays)


def prepare_all_datasets(
    manifest: CurieWeissCampaignManifest,
    directory: Path,
) -> tuple[tuple[Path, str], ...]:
    return tuple(
        prepare_dataset_archive(
            manifest,
            directory,
            n_qubits=n,
            optimizer_seed=seed,
        )
        for n in manifest.system_sizes
        for seed in manifest.optimizer_seeds
    )


def load_dataset_for_run(
    manifest: CurieWeissCampaignManifest,
    directory: Path,
    run: CurieWeissRunIdentity,
) -> CurieWeissDatasetBatch:
    from .artifacts import sha256_file

    if run.configuration_digest != manifest.digest:
        raise ValueError("run belongs to another manifest")
    path = dataset_archive_path(directory, run.n_qubits, run.optimizer_seed)
    digest = sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        if archive["manifest_digest"].item().decode() != manifest.digest:
            raise ValueError("dataset manifest digest differs")
        betas = np.asarray(archive["training_betas"], dtype=np.float64)
        training = np.asarray(archive["training"], dtype=np.int8)
        validation = np.asarray(archive["validation"], dtype=np.int8)
        if set(archive.files) != {
            "manifest_digest",
            "n_qubits",
            "optimizer_seed",
            "training_betas",
            "training_seeds",
            "validation_seeds",
            "training",
            "validation",
        }:
            raise ValueError("dataset archive contains undeclared roles")
    if run.family is CurieWeissFamily.TEMPERATURE_REFERENCE:
        index = manifest.training_betas.index(float(run.reference_beta))
        betas, training, validation = (
            betas[[index]],
            training[[index]],
            validation[[index]],
        )
    if run.family is CurieWeissFamily.NO_CONDITION:
        angles = np.asarray([conditioned_angle(1.0, run.n_qubits)] * len(betas))
    else:
        angles = np.asarray(
            [conditioned_angle(float(beta), run.n_qubits) for beta in betas]
        )
    return CurieWeissDatasetBatch(betas, angles, training, validation, path, digest)

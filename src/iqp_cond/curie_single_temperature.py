"""Frozen independent single-temperature Curie--Weiss reference campaign."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .confirmatory_curie import CurieConfirmatoryManifest, curie_weiss_probabilities
from .protocol import ProtocolValidationError, content_digest

CAMPAIGN_VERSION = "curie_weiss_single_temperature_v1"
BASE_MANIFEST_DIGEST = (
    "ab2e22e823e08a946d79cadbe375460b6122a748f386c7b9c6d1898c94801977"
)
SYSTEM_SIZES = (8, 12, 16)
SEEDS = tuple(range(100, 110))
TRAINING_BETAS = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
HELD_OUT_BETAS = (0.6, 0.8, 1.0, 1.2, 1.4)
BETAS = tuple(sorted((*TRAINING_BETAS, *HELD_OUT_BETAS)))
ANCILLAS = 5


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ProtocolValidationError(
            f"{context} keys differ; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


@dataclass(frozen=True, slots=True)
class SingleTemperatureManifest:
    """Immutable protocol for the 330 independent reference fits."""

    raw: Mapping[str, Any]
    source_path: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _exact_keys(
            self.raw,
            {
                "schema_version",
                "campaign_version",
                "base_manifest",
                "base_manifest_digest",
                "system_sizes",
                "seeds",
                "training_betas",
                "held_out_betas",
                "model",
                "ansatz",
                "ancillas",
                "encoder",
                "training",
                "statistics",
            },
            "single-temperature manifest",
        )
        if self.raw["schema_version"] != 1:
            raise ProtocolValidationError("manifest schema_version must be 1")
        if self.raw["campaign_version"] != CAMPAIGN_VERSION:
            raise ProtocolValidationError("campaign version differs")
        if tuple(self.raw["system_sizes"]) != SYSTEM_SIZES:
            raise ProtocolValidationError("system sizes differ")
        if tuple(self.raw["seeds"]) != SEEDS:
            raise ProtocolValidationError("optimizer seeds differ")
        if (
            tuple(float(value) for value in self.raw["training_betas"])
            != TRAINING_BETAS
        ):
            raise ProtocolValidationError("training temperatures differ")
        if (
            tuple(float(value) for value in self.raw["held_out_betas"])
            != HELD_OUT_BETAS
        ):
            raise ProtocolValidationError("held-out temperatures differ")
        if self.raw["base_manifest_digest"] != BASE_MANIFEST_DIGEST:
            raise ProtocolValidationError("base manifest digest differs")
        if not isinstance(self.raw["base_manifest"], str):
            raise ProtocolValidationError("base manifest path must be a string")
        if (
            self.raw["model"] != "single_temperature_r5"
            or self.raw["ansatz"] != "full_joint_r5"
            or self.raw["ancillas"] != ANCILLAS
            or self.raw["encoder"] != "spin_flip_branch_neutral"
        ):
            raise ProtocolValidationError("single-temperature model differs")

        expected_training = {
            "optimizer": "adam_cosine_decay",
            "updates": 2000,
            "batch_size": 75,
            "initial_learning_rate": 0.01,
            "final_learning_rate": 0.001,
            "validation_interval": 25,
            "bandwidths": [1.0, 2.0, 4.0],
            "visible_character_masks": 256,
            "training_model_samples": 8192,
            "validation_model_samples": 16384,
            "reference_minibatch": 450,
            "checkpoint_rule": "lowest_validation_conditional_mmd",
        }
        if self.raw["training"] != expected_training:
            raise ProtocolValidationError("single-temperature training differs")
        expected_statistics = {
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260825,
            "confidence_level": 0.95,
            "replication_unit": "seed",
            "summary": "median_and_interquartile_range",
            "interval": "paired_percentile_bootstrap",
            "familywise_correction": False,
            "required_win": False,
        }
        if self.raw["statistics"] != expected_statistics:
            raise ProtocolValidationError("single-temperature statistics differ")

        if self.source_path is not None:
            base_path = self.resolve_base_manifest(self.source_path.parent.parent)
            try:
                base = CurieConfirmatoryManifest.from_json(base_path)
            except (OSError, json.JSONDecodeError) as error:
                raise ProtocolValidationError("base manifest is unreadable") from error
            if base.digest != BASE_MANIFEST_DIGEST:
                raise ProtocolValidationError("base manifest content differs")

    @property
    def digest(self) -> str:
        return content_digest(dict(self.raw))

    @property
    def system_sizes(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["system_sizes"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["seeds"])

    @property
    def training_betas(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.raw["training_betas"])

    @property
    def held_out_betas(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.raw["held_out_betas"])

    @property
    def betas(self) -> tuple[float, ...]:
        return tuple(sorted((*self.training_betas, *self.held_out_betas)))

    def resolve_base_manifest(self, repository: Path) -> Path:
        return (repository / str(self.raw["base_manifest"])).resolve()

    @classmethod
    def from_json(cls, path: Path) -> SingleTemperatureManifest:
        path = Path(path).resolve()
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(f"invalid manifest: {path}") from error
        if not isinstance(value, dict):
            raise ProtocolValidationError("manifest must be a JSON object")
        return cls(value, path)


@dataclass(frozen=True, slots=True)
class SingleTemperatureRunIdentity:
    """One size, temperature, and optimizer-seed reference fit."""

    campaign_version: str
    n_qubits: int
    beta: float
    seed: int
    ancillas: int
    configuration_digest: str

    def __post_init__(self) -> None:
        if self.campaign_version != CAMPAIGN_VERSION:
            raise ProtocolValidationError("run campaign version differs")
        if self.n_qubits not in SYSTEM_SIZES:
            raise ProtocolValidationError("run system size is outside the manifest")
        if not math.isfinite(self.beta) or self.beta not in BETAS:
            raise ProtocolValidationError("run temperature is outside the manifest")
        if self.seed not in SEEDS:
            raise ProtocolValidationError("run seed is outside the manifest")
        if self.ancillas != ANCILLAS:
            raise ProtocolValidationError("run ancilla count differs")
        if len(self.configuration_digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.configuration_digest
        ):
            raise ProtocolValidationError("configuration_digest must be SHA-256")

    @property
    def run_id(self) -> str:
        return content_digest(self.to_dict())

    @property
    def parameter_count(self) -> int:
        total = self.n_qubits + self.ancillas
        return total * (total + 1) // 2

    @property
    def role(self) -> str:
        return "training" if self.beta in TRAINING_BETAS else "held_out_oracle"

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_version": self.campaign_version,
            "n_qubits": self.n_qubits,
            "beta": self.beta,
            "seed": self.seed,
            "ancillas": self.ancillas,
            "configuration_digest": self.configuration_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SingleTemperatureRunIdentity:
        _exact_keys(
            value,
            {
                "campaign_version",
                "n_qubits",
                "beta",
                "seed",
                "ancillas",
                "configuration_digest",
            },
            "single-temperature run identity",
        )
        return cls(
            campaign_version=str(value["campaign_version"]),
            n_qubits=int(value["n_qubits"]),
            beta=float(value["beta"]),
            seed=int(value["seed"]),
            ancillas=int(value["ancillas"]),
            configuration_digest=str(value["configuration_digest"]),
        )


def plan_single_temperature_runs(
    manifest: SingleTemperatureManifest,
) -> tuple[SingleTemperatureRunIdentity, ...]:
    """Enumerate the frozen 3 x 11 x 10 run grid."""
    runs = tuple(
        SingleTemperatureRunIdentity(
            CAMPAIGN_VERSION,
            n_qubits,
            beta,
            seed,
            ANCILLAS,
            manifest.digest,
        )
        for n_qubits in manifest.system_sizes
        for beta in manifest.betas
        for seed in manifest.seeds
    )
    if len(runs) != 330 or len({run.run_id for run in runs}) != 330:
        raise RuntimeError("planner must produce 330 unique identities")
    return runs


def neutral_angles(n_qubits: int, ancillas: int = ANCILLAS) -> np.ndarray:
    """Return the fixed neutral encoder angles for one model input."""
    if n_qubits not in SYSTEM_SIZES or ancillas != ANCILLAS:
        raise ValueError("neutral-angle coordinates differ from the manifest")
    return np.zeros(n_qubits + ancillas, dtype=np.float64)


def target_probabilities(n_qubits: int, beta: float) -> np.ndarray:
    """Return the exact target law at one declared temperature."""
    if n_qubits not in SYSTEM_SIZES or beta not in BETAS:
        raise ValueError("target coordinates differ from the manifest")
    return curie_weiss_probabilities(n_qubits, beta)


def split_seed(run: SingleTemperatureRunIdentity, role: str) -> int:
    """Derive deterministic, role-separated finite-target randomness."""
    if role not in {"training", "validation"}:
        raise ValueError("invalid target-sample role")
    raw = f"{run.run_id}|{run.beta:.17g}|{role}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & (2**63 - 1)

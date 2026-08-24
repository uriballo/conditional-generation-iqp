"""Frozen protocol and run planner for the Curie--Weiss extension."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .protocol import ProtocolValidationError, content_digest


class CurieWeissFamily(StrEnum):
    CONDITIONED = "conditioned"
    NO_CONDITION = "no_condition"
    TEMPERATURE_REFERENCE = "temperature_reference"
    UNRESTRICTED_CONDITIONED = "unrestricted_conditioned"


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ProtocolValidationError(
            f"{context} keys differ; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


@dataclass(frozen=True, slots=True)
class CurieWeissRunIdentity:
    campaign_version: str
    n_qubits: int
    family: CurieWeissFamily
    reference_beta: float | None
    optimizer_seed: int
    data_seed: int
    configuration_digest: str

    def __post_init__(self) -> None:
        if self.campaign_version != "curie_weiss_foundations_v1":
            raise ProtocolValidationError("unknown Curie--Weiss campaign version")
        if self.n_qubits not in {8, 12, 16}:
            raise ProtocolValidationError("unsupported Curie--Weiss system size")
        if self.optimizer_seed not in range(10) or self.data_seed < 0:
            raise ProtocolValidationError("invalid Curie--Weiss run seed")
        if len(self.configuration_digest) != 64:
            raise ProtocolValidationError("configuration_digest must be SHA-256")
        if self.family is CurieWeissFamily.TEMPERATURE_REFERENCE:
            if self.reference_beta not in {0.5, 0.7, 0.9, 1.1, 1.3, 1.5}:
                raise ProtocolValidationError("reference run needs a training beta")
        elif self.reference_beta is not None:
            raise ProtocolValidationError("only reference runs carry reference_beta")

    @property
    def run_id(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_version": self.campaign_version,
            "n_qubits": self.n_qubits,
            "family": self.family.value,
            "reference_beta": self.reference_beta,
            "optimizer_seed": self.optimizer_seed,
            "data_seed": self.data_seed,
            "configuration_digest": self.configuration_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CurieWeissRunIdentity:
        expected = {
            "campaign_version",
            "n_qubits",
            "family",
            "reference_beta",
            "optimizer_seed",
            "data_seed",
            "configuration_digest",
        }
        _exact_keys(value, expected, "Curie--Weiss run identity")
        reference = value["reference_beta"]
        return cls(
            campaign_version=str(value["campaign_version"]),
            n_qubits=int(value["n_qubits"]),
            family=CurieWeissFamily(str(value["family"])),
            reference_beta=None if reference is None else float(reference),
            optimizer_seed=int(value["optimizer_seed"]),
            data_seed=int(value["data_seed"]),
            configuration_digest=str(value["configuration_digest"]),
        )


@dataclass(frozen=True, slots=True)
class CurieWeissCampaignManifest:
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = {
            "schema_version",
            "campaign_version",
            "system_sizes",
            "optimizer_seeds",
            "target",
            "temperatures",
            "condition_map",
            "data",
            "training",
            "runtime_preflight",
            "statistics",
        }
        _exact_keys(self.raw, expected, "Curie--Weiss manifest")
        if self.raw["schema_version"] != 1:
            raise ProtocolValidationError("manifest schema_version must be 1")
        if self.campaign_version != "curie_weiss_foundations_v1":
            raise ProtocolValidationError("campaign version is not frozen")
        if self.system_sizes != (8, 12, 16):
            raise ProtocolValidationError("system sizes must be 8/12/16")
        if self.optimizer_seeds != tuple(range(10)):
            raise ProtocolValidationError("optimizer seeds must be 0--9")
        if self.training_betas != (0.5, 0.7, 0.9, 1.1, 1.3, 1.5):
            raise ProtocolValidationError("training beta grid differs")
        if self.held_out_betas != (0.6, 0.8, 1.0, 1.2, 1.4):
            raise ProtocolValidationError("held-out beta grid differs")
        if set(self.training_betas) & set(self.held_out_betas):
            raise ProtocolValidationError("training and held-out betas overlap")
        self._validate_frozen_sections()

    @property
    def campaign_version(self) -> str:
        return str(self.raw["campaign_version"])

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    @property
    def system_sizes(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.raw["system_sizes"])

    @property
    def optimizer_seeds(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.raw["optimizer_seeds"])

    @property
    def training_betas(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.raw["temperatures"]["training"])

    @property
    def held_out_betas(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.raw["temperatures"]["held_out"])

    @property
    def evaluation_betas(self) -> tuple[float, ...]:
        return tuple(sorted((*self.training_betas, *self.held_out_betas)))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.raw))

    @classmethod
    def from_json(cls, path: Path) -> CurieWeissCampaignManifest:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(f"invalid manifest: {path}") from error
        if not isinstance(value, dict):
            raise ProtocolValidationError("manifest must be an object")
        return cls(value)

    def _validate_frozen_sections(self) -> None:
        if self.raw["target"] != {
            "coupling": 1.0,
            "field": 0.0,
            "critical_inverse_temperature": 1.0,
            "hamiltonian": "-M^2/(2n)",
        }:
            raise ProtocolValidationError("target definition differs")
        if self.raw["condition_map"] != {
            "normalized_coordinate": "beta-0.5",
            "angle_intercept_pi": 0.35,
            "angle_slope_pi": -0.7,
            "outcome_tuned": False,
        }:
            raise ProtocolValidationError("condition map differs")
        if self.raw["data"] != {
            "root_seed": 20260812,
            "training_per_temperature": 10_000,
            "validation_per_temperature": 2_000,
            "held_out_draws": 0,
        }:
            raise ProtocolValidationError("data protocol differs")
        expected_training = {
            "optimizer": "adam_cosine_decay",
            "updates": 2_000,
            "initial_learning_rate": 0.01,
            "final_learning_rate": 0.001,
            "total_target_minibatch": 450,
            "reference_minibatch": 450,
            "validation_interval": 25,
            "checkpoint_rule": "lowest_validation_conditional_mmd",
            "bandwidths": [1.0, 2.0, 4.0],
            "training_operators": 64,
            "training_model_samples": 8_192,
            "validation_operators": 256,
            "validation_model_samples": 16_384,
        }
        if self.raw["training"] != expected_training:
            raise ProtocolValidationError("training protocol differs")
        if self.raw["runtime_preflight"] != {
            "updates": 100,
            "max_projected_seconds_per_model": 3_600.0,
            "max_peak_rss_bytes": 30_064_771_072,
        }:
            raise ProtocolValidationError("runtime protocol differs")
        if self.raw["statistics"] != {
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 20_260_812,
            "confidence_level": 0.95,
            "replication_unit": "optimizer_seed",
            "performance_thresholds": None,
        }:
            raise ProtocolValidationError("statistics protocol differs")


def derive_data_seed(manifest: CurieWeissCampaignManifest, n: int, seed: int) -> int:
    return int(
        content_digest(
            {
                "manifest_digest": manifest.digest,
                "n_qubits": n,
                "optimizer_seed": seed,
                "role": "paired_master_data",
            }
        )[:15],
        16,
    )


def plan_curie_weiss_runs(
    manifest: CurieWeissCampaignManifest,
) -> tuple[CurieWeissRunIdentity, ...]:
    runs: list[CurieWeissRunIdentity] = []
    for n in manifest.system_sizes:
        for seed in manifest.optimizer_seeds:
            data_seed = derive_data_seed(manifest, n, seed)
            for family in (
                CurieWeissFamily.CONDITIONED,
                CurieWeissFamily.NO_CONDITION,
                CurieWeissFamily.UNRESTRICTED_CONDITIONED,
            ):
                runs.append(
                    CurieWeissRunIdentity(
                        manifest.campaign_version,
                        n,
                        family,
                        None,
                        seed,
                        data_seed,
                        manifest.digest,
                    )
                )
            for beta in manifest.training_betas:
                runs.append(
                    CurieWeissRunIdentity(
                        manifest.campaign_version,
                        n,
                        CurieWeissFamily.TEMPERATURE_REFERENCE,
                        beta,
                        seed,
                        data_seed,
                        manifest.digest,
                    )
                )
    if len(runs) != 270 or len({run.run_id for run in runs}) != 270:
        raise RuntimeError("Curie--Weiss planner must produce 270 unique fits")
    return tuple(runs)


def conditioned_angle(beta: float, n_qubits: int) -> tuple[float, ...]:
    if not math.isfinite(beta) or not 0.5 <= beta <= 1.5:
        raise ValueError("beta must lie in [0.5, 1.5]")
    angle = math.pi * (0.35 - 0.70 * (beta - 0.5))
    return (angle,) * n_qubits

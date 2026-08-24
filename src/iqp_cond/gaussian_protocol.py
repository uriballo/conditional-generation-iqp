"""Frozen protocol and deterministic run plan for the Gaussian-only paper."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .protocol import ProtocolValidationError, content_digest


class GaussianStudy(StrEnum):
    FACTORIAL = "factorial"
    CONDITION_COUNT = "condition_count"


class GaussianFamily(StrEnum):
    CONDITIONED = "conditioned"
    NO_CONDITION = "no_condition"
    SINGLE_PEAK = "single_peak"


class DataRegime(StrEnum):
    FIXED_PER_PEAK = "fixed_per_peak"
    FIXED_TOTAL = "fixed_total"


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ProtocolValidationError(
            f"{context} keys differ; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _numbers(value: Any, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ProtocolValidationError(f"{context} must be a non-empty list")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ProtocolValidationError(f"{context} must contain finite numbers")
    return result


def _integers(value: Any, context: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ProtocolValidationError(f"{context} must be a non-empty integer list")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class GaussianRunIdentity:
    campaign_version: str
    study: GaussianStudy
    n_qubits: int
    width_factor: float
    peak_count: int
    data_regime: DataRegime
    family: GaussianFamily
    bank_peak_index: int | None
    optimizer_seed: int
    data_seed: int
    configuration_digest: str

    def __post_init__(self) -> None:
        if self.campaign_version != "gaussian_foundations_v1":
            raise ProtocolValidationError("unknown Gaussian campaign version")
        if self.n_qubits < 2 or self.peak_count not in {2, 3, 5, 9}:
            raise ProtocolValidationError("invalid Gaussian run dimensions")
        if not math.isfinite(self.width_factor) or self.width_factor <= 0.0:
            raise ProtocolValidationError("width_factor must be positive")
        if self.optimizer_seed < 0 or self.data_seed < 0:
            raise ProtocolValidationError("run seeds must be non-negative")
        if len(self.configuration_digest) != 64:
            raise ProtocolValidationError("configuration_digest must be SHA-256")
        if self.family is GaussianFamily.SINGLE_PEAK:
            if self.bank_peak_index is None or not 0 <= self.bank_peak_index < 9:
                raise ProtocolValidationError("single-peak runs require a bank index")
        elif self.bank_peak_index is not None:
            raise ProtocolValidationError("only single-peak runs carry a bank index")

    @property
    def run_id(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_version": self.campaign_version,
            "study": self.study.value,
            "n_qubits": self.n_qubits,
            "width_factor": self.width_factor,
            "peak_count": self.peak_count,
            "data_regime": self.data_regime.value,
            "family": self.family.value,
            "bank_peak_index": self.bank_peak_index,
            "optimizer_seed": self.optimizer_seed,
            "data_seed": self.data_seed,
            "configuration_digest": self.configuration_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GaussianRunIdentity:
        expected = {
            "campaign_version",
            "study",
            "n_qubits",
            "width_factor",
            "peak_count",
            "data_regime",
            "family",
            "bank_peak_index",
            "optimizer_seed",
            "data_seed",
            "configuration_digest",
        }
        _exact_keys(value, expected, "Gaussian run identity")
        bank_index = value["bank_peak_index"]
        if bank_index is not None and (
            isinstance(bank_index, bool) or not isinstance(bank_index, int)
        ):
            raise ProtocolValidationError("bank_peak_index must be integer or null")
        return cls(
            campaign_version=str(value["campaign_version"]),
            study=GaussianStudy(str(value["study"])),
            n_qubits=int(value["n_qubits"]),
            width_factor=float(value["width_factor"]),
            peak_count=int(value["peak_count"]),
            data_regime=DataRegime(str(value["data_regime"])),
            family=GaussianFamily(str(value["family"])),
            bank_peak_index=bank_index,
            optimizer_seed=int(value["optimizer_seed"]),
            data_seed=int(value["data_seed"]),
            configuration_digest=str(value["configuration_digest"]),
        )


@dataclass(frozen=True, slots=True)
class GaussianCampaignManifest:
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = {
            "schema_version",
            "campaign_version",
            "system_sizes",
            "width_factors",
            "optimizer_seeds",
            "target",
            "grids",
            "data",
            "training",
            "runtime_preflight",
            "statistics",
        }
        _exact_keys(self.raw, expected, "Gaussian manifest")
        if self.raw["schema_version"] != 1:
            raise ProtocolValidationError("Gaussian manifest schema_version must be 1")
        if self.raw["campaign_version"] != "gaussian_foundations_v1":
            raise ProtocolValidationError("Gaussian campaign version is not frozen")
        if self.system_sizes != (8, 12, 16, 19):
            raise ProtocolValidationError("system sizes must be 8/12/16/19")
        if self.width_factors != (0.35, 0.5, 0.75):
            raise ProtocolValidationError("width factors must be 0.35/0.50/0.75")
        if self.optimizer_seeds != tuple(range(10)):
            raise ProtocolValidationError("optimizer seeds must be 0--9")
        target = self.raw["target"]
        _exact_keys(
            target,
            {
                "mean_intercept",
                "mean_slope",
                "angle_intercept_pi",
                "angle_slope_pi",
                "uniform_within_shell",
            },
            "target",
        )
        if target != {
            "mean_intercept": 0.15,
            "mean_slope": 0.7,
            "angle_intercept_pi": 0.35,
            "angle_slope_pi": -0.7,
            "uniform_within_shell": True,
        }:
            raise ProtocolValidationError(
                "target path differs from the frozen protocol"
            )
        grids = self.raw["grids"]
        _exact_keys(
            grids, {"factorial_training", "condition_counts", "held_out"}, "grids"
        )
        if self.factorial_grid != (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ProtocolValidationError("factorial training grid differs")
        if self.condition_counts != (2, 3, 5, 9):
            raise ProtocolValidationError("condition-count grid differs")
        if self.held_out_grid != tuple((2 * index + 1) / 16 for index in range(8)):
            raise ProtocolValidationError("held-out grid differs")
        data = self.raw["data"]
        _exact_keys(
            data,
            {
                "root_seed",
                "fixed_per_peak_training",
                "fixed_per_peak_validation",
                "fixed_total_training",
                "fixed_total_validation",
                "held_out_draws",
            },
            "data",
        )
        if data != {
            "root_seed": 20260811,
            "fixed_per_peak_training": 10_000,
            "fixed_per_peak_validation": 2_000,
            "fixed_total_training": 18_000,
            "fixed_total_validation": 3_600,
            "held_out_draws": 0,
        }:
            raise ProtocolValidationError("data protocol differs")
        self._validate_training()
        self._validate_runtime_and_statistics()

    @property
    def campaign_version(self) -> str:
        return str(self.raw["campaign_version"])

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    @property
    def system_sizes(self) -> tuple[int, ...]:
        return _integers(self.raw["system_sizes"], "system_sizes")

    @property
    def width_factors(self) -> tuple[float, ...]:
        return _numbers(self.raw["width_factors"], "width_factors")

    @property
    def optimizer_seeds(self) -> tuple[int, ...]:
        return _integers(self.raw["optimizer_seeds"], "optimizer_seeds")

    @property
    def factorial_grid(self) -> tuple[float, ...]:
        return _numbers(self.raw["grids"]["factorial_training"], "factorial grid")

    @property
    def condition_counts(self) -> tuple[int, ...]:
        return _integers(self.raw["grids"]["condition_counts"], "condition counts")

    @property
    def held_out_grid(self) -> tuple[float, ...]:
        return _numbers(self.raw["grids"]["held_out"], "held-out grid")

    def training_grid(self, peak_count: int) -> tuple[float, ...]:
        if peak_count not in self.condition_counts:
            raise ProtocolValidationError("unsupported peak count")
        return tuple(index / (peak_count - 1) for index in range(peak_count))

    def samples_per_peak(self, peak_count: int, regime: DataRegime) -> tuple[int, int]:
        if regime is DataRegime.FIXED_PER_PEAK:
            return 10_000, 2_000
        if peak_count not in self.condition_counts:
            raise ProtocolValidationError("unsupported peak count")
        return 18_000 // peak_count, 3_600 // peak_count

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.raw))

    @classmethod
    def from_json(cls, path: Path) -> GaussianCampaignManifest:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(
                f"invalid Gaussian manifest: {path}"
            ) from error
        if not isinstance(value, dict):
            raise ProtocolValidationError("Gaussian manifest must be an object")
        return cls(value)

    def _validate_training(self) -> None:
        expected = {
            "optimizer": "adam_cosine_decay",
            "updates": 2_000,
            "initial_learning_rate": 0.01,
            "final_learning_rate": 0.001,
            "total_target_minibatch": 450,
            "validation_interval": 25,
            "checkpoint_rule": "lowest_validation_conditional_mmd",
            "bandwidths": [1.0, 2.0, 4.0],
            "training_operators": 64,
            "training_model_samples": 8_192,
            "validation_operators": 256,
            "validation_model_samples": 16_384,
        }
        _exact_keys(self.raw["training"], set(expected), "training")
        if self.raw["training"] != expected:
            raise ProtocolValidationError("training protocol differs")

    def _validate_runtime_and_statistics(self) -> None:
        runtime = self.raw["runtime_preflight"]
        _exact_keys(
            runtime,
            {"updates", "max_projected_seconds_per_model", "max_peak_rss_bytes"},
            "runtime_preflight",
        )
        if runtime != {
            "updates": 100,
            "max_projected_seconds_per_model": 3_600.0,
            "max_peak_rss_bytes": 30_064_771_072,
        }:
            raise ProtocolValidationError("runtime preflight differs")
        statistics = self.raw["statistics"]
        _exact_keys(
            statistics,
            {
                "bootstrap_replicates",
                "bootstrap_seed",
                "confidence_level",
                "replication_unit",
                "performance_thresholds",
            },
            "statistics",
        )
        if statistics != {
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 20_260_811,
            "confidence_level": 0.95,
            "replication_unit": "optimizer_seed",
            "performance_thresholds": None,
        }:
            raise ProtocolValidationError("statistics protocol differs")


def derive_run_seed(
    manifest: GaussianCampaignManifest,
    *,
    study: GaussianStudy,
    n_qubits: int,
    width_factor: float,
    peak_count: int,
    data_regime: DataRegime,
    family: GaussianFamily,
    bank_peak_index: int | None,
    optimizer_seed: int,
) -> int:
    # Data are paired across every family, grid, and data regime.  The method
    # coordinates remain in the run identity but deliberately do not affect
    # the master-pool seed.
    payload = {
        "manifest_digest": manifest.digest,
        "n_qubits": n_qubits,
        "width_factor": width_factor,
        "optimizer_seed": optimizer_seed,
    }
    return int(content_digest(payload)[:15], 16)


def _identity(
    manifest: GaussianCampaignManifest,
    *,
    study: GaussianStudy,
    n_qubits: int,
    width_factor: float,
    peak_count: int,
    data_regime: DataRegime,
    family: GaussianFamily,
    bank_peak_index: int | None,
    optimizer_seed: int,
) -> GaussianRunIdentity:
    return GaussianRunIdentity(
        campaign_version=manifest.campaign_version,
        study=study,
        n_qubits=n_qubits,
        width_factor=width_factor,
        peak_count=peak_count,
        data_regime=data_regime,
        family=family,
        bank_peak_index=bank_peak_index,
        optimizer_seed=optimizer_seed,
        data_seed=derive_run_seed(
            manifest,
            study=study,
            n_qubits=n_qubits,
            width_factor=width_factor,
            peak_count=peak_count,
            data_regime=data_regime,
            family=family,
            bank_peak_index=bank_peak_index,
            optimizer_seed=optimizer_seed,
        ),
        configuration_digest=manifest.digest,
    )


def plan_factorial_runs(
    manifest: GaussianCampaignManifest,
) -> tuple[GaussianRunIdentity, ...]:
    runs = []
    for n_qubits in manifest.system_sizes:
        for width_factor in manifest.width_factors:
            for seed in manifest.optimizer_seeds:
                for family in (GaussianFamily.CONDITIONED, GaussianFamily.NO_CONDITION):
                    runs.append(
                        _identity(
                            manifest,
                            study=GaussianStudy.FACTORIAL,
                            n_qubits=n_qubits,
                            width_factor=width_factor,
                            peak_count=5,
                            data_regime=DataRegime.FIXED_PER_PEAK,
                            family=family,
                            bank_peak_index=None,
                            optimizer_seed=seed,
                        )
                    )
                for bank_peak_index in (0, 2, 4, 6, 8):
                    runs.append(
                        _identity(
                            manifest,
                            study=GaussianStudy.FACTORIAL,
                            n_qubits=n_qubits,
                            width_factor=width_factor,
                            peak_count=5,
                            data_regime=DataRegime.FIXED_PER_PEAK,
                            family=GaussianFamily.SINGLE_PEAK,
                            bank_peak_index=bank_peak_index,
                            optimizer_seed=seed,
                        )
                    )
    if len(runs) != 840 or len({run.run_id for run in runs}) != 840:
        raise RuntimeError("factorial planner must produce 840 unique identities")
    return tuple(runs)


def plan_condition_count_additions(
    manifest: GaussianCampaignManifest,
) -> tuple[GaussianRunIdentity, ...]:
    runs = []
    for seed in manifest.optimizer_seeds:
        for bank_peak_index in (1, 3, 5, 7):
            runs.append(
                _identity(
                    manifest,
                    study=GaussianStudy.CONDITION_COUNT,
                    n_qubits=19,
                    width_factor=0.5,
                    peak_count=9,
                    data_regime=DataRegime.FIXED_PER_PEAK,
                    family=GaussianFamily.SINGLE_PEAK,
                    bank_peak_index=bank_peak_index,
                    optimizer_seed=seed,
                )
            )
        for regime in (DataRegime.FIXED_PER_PEAK, DataRegime.FIXED_TOTAL):
            for peak_count in manifest.condition_counts:
                if regime is DataRegime.FIXED_PER_PEAK and peak_count == 5:
                    continue
                for family in (
                    GaussianFamily.CONDITIONED,
                    GaussianFamily.NO_CONDITION,
                ):
                    runs.append(
                        _identity(
                            manifest,
                            study=GaussianStudy.CONDITION_COUNT,
                            n_qubits=19,
                            width_factor=0.5,
                            peak_count=peak_count,
                            data_regime=regime,
                            family=family,
                            bank_peak_index=None,
                            optimizer_seed=seed,
                        )
                    )
    if len(runs) != 180 or len({run.run_id for run in runs}) != 180:
        raise RuntimeError("condition-count additions must contain 180 identities")
    return tuple(runs)


def plan_gaussian_runs(
    manifest: GaussianCampaignManifest,
) -> tuple[GaussianRunIdentity, ...]:
    runs = plan_factorial_runs(manifest) + plan_condition_count_additions(manifest)
    if len(runs) != 1_020 or len({run.run_id for run in runs}) != 1_020:
        raise RuntimeError("Gaussian campaign must contain 1,020 unique identities")
    return runs

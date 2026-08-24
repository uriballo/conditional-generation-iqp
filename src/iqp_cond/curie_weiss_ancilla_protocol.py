"""Frozen protocol for the eight-visible Curie--Weiss ancilla campaign."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .protocol import ProtocolValidationError, content_digest

CAMPAIGN_VERSION = "curie_weiss_ancilla_foundations_v1"
M5_CAMPAIGN_VERSION = "curie_weiss_m5_foundations_v1"
V2_CAMPAIGN_VERSION = "curie_weiss_m5_foundations_v2"
V3_CAMPAIGN_VERSION = "curie_weiss_m5_foundations_v3"
ENCODING_COMPARISON_VERSION = "curie_weiss_encoding_comparison_v1"
VISIBLE_QUBITS = 8
ANCILLA_RUNGS = (0, 1, 3, 5, 7, 9)
M5_ANCILLA_RUNGS = (5,)
TRAINING_BETAS = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
HELD_OUT_BETAS = (0.6, 0.8, 1.0, 1.2, 1.4)

# Frozen v2 broadcast map: alpha(beta) in units of pi, piecewise linear in beta.
# alpha(0.5)=-0.14, alpha(0.9)=-0.07, alpha(1.0)=0, alpha(1.2)=0.08, alpha(1.5)=0.14.
V2_CONDITION_MAP = {
    "normalized_coordinate": "beta",
    "map_kind": "broadcast_piecewise_linear",
    "anchor_units": "pi",
    "pieces": [
        {"start": 0.5, "end": 0.9, "slope": 0.175, "anchor": -0.14},
        {"start": 0.9, "end": 1.0, "slope": 0.70, "anchor": -0.07},
        {"start": 1.0, "end": 1.2, "slope": 0.40, "anchor": 0.0},
        {"start": 1.2, "end": 1.5, "slope": 0.20, "anchor": 0.08},
    ],
    "ancilla_scheme": "broadcast_all",
    "outcome_tuned": False,
}

V2_DEVELOPMENT_SELECTION = {
    "campaign_path": "results/curie-weiss-m5-foundations-v1/store",
    "manifest_sha256": (
        "4b097a6dc6424c2f7a4c1f2d9263d1e768691b3abb689378ee0e3161b12ceb9d"
    ),
    "design_basis": (
        "cold-end representability cap analysis and broadcast-map screens; "
        "asymmetric piecewise map caps alpha(1.5)=0.14pi while preserving the "
        "frozen mid-range slope inside [0.9,1.0]"
    ),
    "selected_map": "broadcast_piecewise_linear_v1",
    "optimization_claim": False,
}

V3_DEVELOPMENT_SELECTION = {
    "campaign_path": "results/curie-weiss-m5-foundations-v2/store",
    "manifest_sha256": (
        "611d624f4e480595116a0990c749925e745629f2a577d8f6789aff5daf739550"
    ),
    "design_basis": (
        "matched encoding benchmark and full production screen: spin_flip_branch "
        "halves held-out TVD (0.0476 vs 0.0998) and restores a positive "
        "critical-region conditioning effect; reupload depths L=2,3 did not improve"
    ),
    "selected_map": "broadcast_piecewise_linear_v1",
    "selected_encoding": "spin_flip_branch",
    "optimization_claim": False,
}


class CurieWeissAncillaFamily(StrEnum):
    CONDITIONED = "conditioned"
    NO_CONDITION = "no_condition"
    TEMPERATURE_REFERENCE = "temperature_reference"
    ROLE_LINEAR = "role_linear"
    ROLE_QUADRATIC = "role_quadratic"


@dataclass(frozen=True, slots=True)
class DatasetSource:
    seed: int
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.seed not in range(10) or len(self.sha256) != 64:
            raise ProtocolValidationError("invalid dataset source")
        if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ProtocolValidationError("dataset paths must be repository-relative")


@dataclass(frozen=True, slots=True)
class CurieWeissAncillaRunIdentity:
    campaign_version: str
    visible_qubits: int
    ancillas: int
    family: CurieWeissAncillaFamily
    reference_beta: float | None
    optimizer_seed: int
    configuration_digest: str

    def __post_init__(self) -> None:
        if (
            self.campaign_version
            not in {
                CAMPAIGN_VERSION,
                M5_CAMPAIGN_VERSION,
                V2_CAMPAIGN_VERSION,
                V3_CAMPAIGN_VERSION,
                ENCODING_COMPARISON_VERSION,
            }
            or self.visible_qubits != 8
        ):
            raise ProtocolValidationError("unknown ancilla Curie--Weiss campaign")
        allowed_rungs = (
            M5_ANCILLA_RUNGS
            if self.campaign_version
            in {
                M5_CAMPAIGN_VERSION,
                V2_CAMPAIGN_VERSION,
                V3_CAMPAIGN_VERSION,
                ENCODING_COMPARISON_VERSION,
            }
            else ANCILLA_RUNGS
        )
        if self.ancillas not in allowed_rungs or self.optimizer_seed not in range(10):
            raise ProtocolValidationError("run coordinates are outside the manifest")
        if len(self.configuration_digest) != 64:
            raise ProtocolValidationError("configuration digest must be SHA-256")
        if self.family is CurieWeissAncillaFamily.TEMPERATURE_REFERENCE:
            headline = (
                5
                if self.campaign_version
                in {M5_CAMPAIGN_VERSION, V2_CAMPAIGN_VERSION, V3_CAMPAIGN_VERSION}
                else 9
            )
            if self.ancillas != headline or self.reference_beta not in TRAINING_BETAS:
                raise ProtocolValidationError(
                    f"references require m={headline} and a training beta"
                )
        elif self.reference_beta is not None:
            raise ProtocolValidationError("only references carry reference_beta")
        if self.campaign_version == ENCODING_COMPARISON_VERSION and self.family not in {
            CurieWeissAncillaFamily.ROLE_LINEAR,
            CurieWeissAncillaFamily.ROLE_QUADRATIC,
        }:
            raise ProtocolValidationError("encoding comparison contains only two arms")

    @property
    def total_qubits(self) -> int:
        return self.visible_qubits + self.ancillas

    @property
    def parameter_count(self) -> int:
        return self.total_qubits * (self.total_qubits + 1) // 2

    @property
    def run_id(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_version": self.campaign_version,
            "visible_qubits": self.visible_qubits,
            "ancillas": self.ancillas,
            "family": self.family.value,
            "reference_beta": self.reference_beta,
            "optimizer_seed": self.optimizer_seed,
            "configuration_digest": self.configuration_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CurieWeissAncillaRunIdentity:
        expected = {
            "campaign_version",
            "visible_qubits",
            "ancillas",
            "family",
            "reference_beta",
            "optimizer_seed",
            "configuration_digest",
        }
        if set(value) != expected:
            raise ProtocolValidationError("run identity schema differs")
        beta = value["reference_beta"]
        return cls(
            str(value["campaign_version"]),
            int(value["visible_qubits"]),
            int(value["ancillas"]),
            CurieWeissAncillaFamily(str(value["family"])),
            None if beta is None else float(beta),
            int(value["optimizer_seed"]),
            str(value["configuration_digest"]),
        )


@dataclass(frozen=True, slots=True)
class CurieWeissAncillaManifest:
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = {
            "schema_version",
            "campaign_version",
            "visible_qubits",
            "ancilla_rungs",
            "headline_ancillas",
            "optimizer_seeds",
            "target",
            "temperatures",
            "condition_map",
            "datasets",
            "training",
            "runtime_preflight",
            "statistics",
        }
        if self.raw.get("campaign_version") in {
            M5_CAMPAIGN_VERSION,
            V2_CAMPAIGN_VERSION,
            V3_CAMPAIGN_VERSION,
        }:
            expected.add("development_selection")
        if self.raw.get("campaign_version") == V3_CAMPAIGN_VERSION:
            expected.add("encoding")
        if self.raw.get("campaign_version") == ENCODING_COMPARISON_VERSION:
            expected.add("baseline_reference")
        if set(self.raw) != expected or self.raw["schema_version"] != 1:
            raise ProtocolValidationError("ancilla manifest schema differs")
        if self.campaign_version not in {
            CAMPAIGN_VERSION,
            M5_CAMPAIGN_VERSION,
            V2_CAMPAIGN_VERSION,
            V3_CAMPAIGN_VERSION,
            ENCODING_COMPARISON_VERSION,
        }:
            raise ProtocolValidationError("campaign version differs")
        expected_rungs = (
            M5_ANCILLA_RUNGS
            if self.campaign_version
            in {
                M5_CAMPAIGN_VERSION,
                V2_CAMPAIGN_VERSION,
                V3_CAMPAIGN_VERSION,
                ENCODING_COMPARISON_VERSION,
            }
            else ANCILLA_RUNGS
        )
        if self.visible_qubits != 8 or self.ancilla_rungs != expected_rungs:
            raise ProtocolValidationError("visible/ancilla coordinates differ")
        expected_headline = (
            5
            if self.campaign_version
            in {
                M5_CAMPAIGN_VERSION,
                V2_CAMPAIGN_VERSION,
                V3_CAMPAIGN_VERSION,
                ENCODING_COMPARISON_VERSION,
            }
            else 9
        )
        if int(self.raw["headline_ancillas"]) != expected_headline:
            raise ProtocolValidationError("headline ancilla count differs")
        if self.optimizer_seeds != tuple(range(10)):
            raise ProtocolValidationError("optimizer seeds must be 0--9")
        if (
            self.training_betas != TRAINING_BETAS
            or self.held_out_betas != HELD_OUT_BETAS
        ):
            raise ProtocolValidationError("temperature grid differs")
        if len(self.dataset_sources) != 10 or {
            s.seed for s in self.dataset_sources
        } != set(range(10)):
            raise ProtocolValidationError("exactly ten dataset sources are required")
        self._validate_frozen_sections()

    @property
    def campaign_version(self) -> str:
        return str(self.raw["campaign_version"])

    @property
    def encoding(self) -> str:
        return str(self.raw.get("encoding", "angle"))

    @property
    def visible_qubits(self) -> int:
        return int(self.raw["visible_qubits"])

    @property
    def ancilla_rungs(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.raw["ancilla_rungs"])

    @property
    def optimizer_seeds(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.raw["optimizer_seeds"])

    @property
    def headline_ancillas(self) -> int:
        return int(self.raw["headline_ancillas"])

    @property
    def training_betas(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.raw["temperatures"]["training"])

    @property
    def held_out_betas(self) -> tuple[float, ...]:
        return tuple(float(v) for v in self.raw["temperatures"]["held_out"])

    @property
    def evaluation_betas(self) -> tuple[float, ...]:
        return tuple(sorted((*self.training_betas, *self.held_out_betas)))

    @property
    def dataset_sources(self) -> tuple[DatasetSource, ...]:
        return tuple(
            DatasetSource(**value) for value in self.raw["datasets"]["sources"]
        )

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.raw))

    @classmethod
    def from_json(cls, path: Path) -> CurieWeissAncillaManifest:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(f"invalid manifest: {path}") from error
        if not isinstance(value, dict):
            raise ProtocolValidationError("manifest must be an object")
        return cls(value)

    def source_for_seed(self, seed: int) -> DatasetSource:
        return next(source for source in self.dataset_sources if source.seed == seed)

    def _validate_frozen_sections(self) -> None:
        if self.campaign_version == ENCODING_COMPARISON_VERSION:
            self._validate_encoding_comparison_sections()
            return
        if self.campaign_version == V2_CAMPAIGN_VERSION:
            self._validate_v2_frozen_sections()
            return
        if self.campaign_version == V3_CAMPAIGN_VERSION:
            self._validate_v3_frozen_sections()
            return
        if self.raw["condition_map"] != {
            "normalized_coordinate": "beta-0.5",
            "angle_intercept_pi": 0.35,
            "angle_slope_pi": -0.7,
            "ancilla_scheme": "broadcast_all",
            "outcome_tuned": False,
        }:
            raise ProtocolValidationError("condition map differs")
        datasets = dict(self.raw["datasets"])
        datasets.pop("sources")
        if datasets != {
            "training_per_temperature": 10_000,
            "validation_per_temperature": 2_000,
            "held_out_draws": 0,
        }:
            raise ProtocolValidationError("dataset protocol differs")
        if self.raw["training"] != {
            "optimizer": "adam_cosine_decay",
            "updates": 2_000,
            "initial_learning_rate": 0.01,
            "final_learning_rate": 0.001,
            "total_target_minibatch": 450,
            "reference_minibatch": 450,
            "validation_interval": 25,
            "checkpoint_rule": "lowest_validation_conditional_mmd",
            "bandwidths": [1.0, 2.0, 4.0],
            "visible_character_masks": 256,
            "training_model_samples": 8_192,
            "validation_model_samples": 16_384,
        }:
            raise ProtocolValidationError("training protocol differs")
        if self.raw["statistics"] != {
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 20_260_815,
            "confidence_level": 0.95,
            "replication_unit": "optimizer_seed",
            "performance_thresholds": None,
        }:
            raise ProtocolValidationError("statistics protocol differs")
        if self.raw["runtime_preflight"] != {
            "updates": 100,
            "compilation_cache": "fresh_temporary_directory",
            "max_projected_seconds_per_model": 3_600.0,
            "max_peak_rss_bytes": 30_064_771_072,
        }:
            raise ProtocolValidationError("runtime preflight protocol differs")
        if self.campaign_version == M5_CAMPAIGN_VERSION:
            selection = self.raw["development_selection"]
            if selection != {
                "campaign_path": "results/curie-weiss-ancilla-pilot-v1/store",
                "manifest_sha256": (
                    "739ad7b92101a1ae30c086da3f9f02a8f0841fde03e3783cc986720f18109327"
                ),
                "tested_ancillas": [0, 1, 3],
                "tested_seeds": [0, 1, 2],
                "selected_ancillas": 5,
                "selection_basis": "prospective_compute_capacity_compromise",
                "optimization_claim": False,
            }:
                raise ProtocolValidationError("development-selection record differs")

    def _validate_v2_frozen_sections(self) -> None:
        if self.raw["condition_map"] != V2_CONDITION_MAP:
            raise ProtocolValidationError("v2 condition map differs")
        datasets = dict(self.raw["datasets"])
        datasets.pop("sources")
        if datasets != {
            "training_per_temperature": 10_000,
            "validation_per_temperature": 2_000,
            "held_out_draws": 0,
        }:
            raise ProtocolValidationError("dataset protocol differs")
        if self.raw["training"] != {
            "optimizer": "adam_cosine_decay",
            "updates": 2_000,
            "initial_learning_rate": 0.01,
            "final_learning_rate": 0.001,
            "total_target_minibatch": 450,
            "reference_minibatch": 450,
            "validation_interval": 25,
            "checkpoint_rule": "lowest_validation_conditional_mmd",
            "bandwidths": [1.0, 2.0, 4.0],
            "visible_character_masks": 256,
            "training_model_samples": 8_192,
            "validation_model_samples": 16_384,
        }:
            raise ProtocolValidationError("training protocol differs")
        if self.raw["statistics"] != {
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 20_260_815,
            "confidence_level": 0.95,
            "replication_unit": "optimizer_seed",
            "performance_thresholds": None,
        }:
            raise ProtocolValidationError("statistics protocol differs")
        if self.raw["runtime_preflight"] != {
            "updates": 100,
            "compilation_cache": "fresh_temporary_directory",
            "max_projected_seconds_per_model": 3_600.0,
            "max_peak_rss_bytes": 30_064_771_072,
        }:
            raise ProtocolValidationError("runtime preflight protocol differs")
        if self.raw["development_selection"] != V2_DEVELOPMENT_SELECTION:
            raise ProtocolValidationError("v2 development-selection record differs")

    def _validate_v3_frozen_sections(self) -> None:
        if self.raw["condition_map"] != V2_CONDITION_MAP:
            raise ProtocolValidationError("v3 condition map differs")
        if self.raw.get("encoding") != "spin_flip_branch":
            raise ProtocolValidationError("v3 encoding differs")
        datasets = dict(self.raw["datasets"])
        datasets.pop("sources")
        if datasets != {
            "training_per_temperature": 10_000,
            "validation_per_temperature": 2_000,
            "held_out_draws": 0,
        }:
            raise ProtocolValidationError("dataset protocol differs")
        if self.raw["training"] != {
            "optimizer": "adam_cosine_decay",
            "updates": 2_000,
            "initial_learning_rate": 0.01,
            "final_learning_rate": 0.001,
            "total_target_minibatch": 450,
            "reference_minibatch": 450,
            "validation_interval": 25,
            "checkpoint_rule": "lowest_validation_conditional_mmd",
            "bandwidths": [1.0, 2.0, 4.0],
            "visible_character_masks": 256,
            "training_model_samples": 8_192,
            "validation_model_samples": 16_384,
        }:
            raise ProtocolValidationError("training protocol differs")
        if self.raw["statistics"] != {
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 20_260_815,
            "confidence_level": 0.95,
            "replication_unit": "optimizer_seed",
            "performance_thresholds": None,
        }:
            raise ProtocolValidationError("statistics protocol differs")
        if self.raw["runtime_preflight"] != {
            "updates": 100,
            "compilation_cache": "fresh_temporary_directory",
            "max_projected_seconds_per_model": 3_600.0,
            "max_peak_rss_bytes": 30_064_771_072,
        }:
            raise ProtocolValidationError("runtime preflight protocol differs")
        if self.raw["development_selection"] != V3_DEVELOPMENT_SELECTION:
            raise ProtocolValidationError("v3 development-selection record differs")

    def _validate_encoding_comparison_sections(self) -> None:
        expected_map = {
            "normalized_coordinate": "x=2*(beta-1)",
            "visible_map": V2_CONDITION_MAP,
            "ancilla_amplitude_pi": 0.14,
            "role_linear": "0.14*pi*x",
            "role_quadratic": "0.14*pi*(3*x^2-1)/2",
            "ancilla_scheme": "symmetric_within_register",
            "outcome_tuned": False,
        }
        if self.raw["condition_map"] != expected_map:
            raise ProtocolValidationError("encoding-comparison map differs")
        datasets = dict(self.raw["datasets"])
        datasets.pop("sources")
        if datasets != {
            "training_per_temperature": 10_000,
            "validation_per_temperature": 2_000,
            "held_out_draws": 0,
        }:
            raise ProtocolValidationError("dataset protocol differs")
        if self.raw["training"] != {
            "optimizer": "adam_cosine_decay",
            "updates": 2_000,
            "initial_learning_rate": 0.01,
            "final_learning_rate": 0.001,
            "total_target_minibatch": 450,
            "reference_minibatch": 450,
            "validation_interval": 25,
            "checkpoint_rule": "lowest_validation_conditional_mmd",
            "bandwidths": [1.0, 2.0, 4.0],
            "visible_character_masks": 256,
            "training_model_samples": 8_192,
            "validation_model_samples": 16_384,
        }:
            raise ProtocolValidationError("training protocol differs")
        if self.raw["runtime_preflight"] != {
            "updates": 100,
            "compilation_cache": "fresh_temporary_directory",
            "max_projected_seconds_per_model": 3_600.0,
            "max_peak_rss_bytes": 30_064_771_072,
        }:
            raise ProtocolValidationError("runtime preflight protocol differs")
        if self.raw["statistics"] != {
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 20_260_820,
            "confidence_level": 0.95,
            "replication_unit": "optimizer_seed",
            "primary_metric": "held_out_profile_tvd",
            "bitstring_noninferiority_margin": 0.01,
        }:
            raise ProtocolValidationError("statistics protocol differs")
        baseline = self.raw["baseline_reference"]
        if (
            set(baseline)
            != {
                "campaign_version",
                "manifest_path",
                "manifest_sha256",
                "report_path",
                "report_sha256",
            }
            or baseline["campaign_version"] != V2_CAMPAIGN_VERSION
        ):
            raise ProtocolValidationError("baseline reference differs")
        for key in ("manifest_sha256", "report_sha256"):
            value = baseline[key]
            if not isinstance(value, str) or len(value) != 64:
                raise ProtocolValidationError("baseline digests must be SHA-256")


def conditioned_angle(beta: float, total_qubits: int) -> tuple[float, ...]:
    if beta not in (*TRAINING_BETAS, *HELD_OUT_BETAS) or total_qubits < 8:
        raise ProtocolValidationError("condition coordinate is outside the campaign")
    from math import pi

    angle = pi * (0.35 - 0.70 * (beta - 0.5))
    return (angle,) * total_qubits


def broadcast_angle(
    manifest: CurieWeissAncillaManifest, beta: float, total_qubits: int
) -> tuple[float, ...]:
    """Manifest-driven broadcast angle in units of pi.

    v1 manifests keep the frozen linear map exactly; v2 uses the frozen
    asymmetric piecewise-linear map.
    """
    if total_qubits < 8:
        raise ProtocolValidationError("condition coordinate is outside the campaign")
    mapping = manifest.raw["condition_map"]
    if "visible_map" in mapping:
        mapping = mapping["visible_map"]
    if mapping.get("map_kind") == "broadcast_piecewise_linear":
        for piece in mapping["pieces"]:
            if piece["start"] <= beta <= piece["end"]:
                angle = math.pi * (
                    piece["anchor"] + piece["slope"] * (beta - piece["start"])
                )
                return (angle,) * total_qubits
        raise ProtocolValidationError("condition coordinate is outside the map")
    from math import pi

    angle = pi * (
        mapping["angle_intercept_pi"] + mapping["angle_slope_pi"] * (beta - 0.5)
    )
    return (angle,) * total_qubits


def role_separated_angle(
    manifest: CurieWeissAncillaManifest,
    beta: float,
    family: CurieWeissAncillaFamily,
) -> tuple[float, ...]:
    """Return the frozen 8-visible, 5-ancilla encoding-comparison angles."""
    if manifest.campaign_version != ENCODING_COMPARISON_VERSION:
        raise ProtocolValidationError("role-separated map requires comparison manifest")
    if beta not in (*TRAINING_BETAS, *HELD_OUT_BETAS):
        raise ProtocolValidationError("condition coordinate is outside the campaign")
    visible = broadcast_angle(manifest, beta, VISIBLE_QUBITS)
    coordinate = 2.0 * (beta - 1.0)
    amplitude = math.pi * float(manifest.raw["condition_map"]["ancilla_amplitude_pi"])
    if family is CurieWeissAncillaFamily.ROLE_LINEAR:
        ancilla = amplitude * coordinate
    elif family is CurieWeissAncillaFamily.ROLE_QUADRATIC:
        ancilla = amplitude * (3.0 * coordinate**2 - 1.0) / 2.0
    else:
        raise ProtocolValidationError("unknown role-separated encoding arm")
    return (*visible, *((ancilla,) * 5))


def plan_curie_weiss_ancilla_runs(
    manifest: CurieWeissAncillaManifest,
) -> tuple[CurieWeissAncillaRunIdentity, ...]:
    if manifest.campaign_version == ENCODING_COMPARISON_VERSION:
        runs = tuple(
            CurieWeissAncillaRunIdentity(
                manifest.campaign_version,
                manifest.visible_qubits,
                manifest.headline_ancillas,
                family,
                None,
                seed,
                manifest.digest,
            )
            for seed in manifest.optimizer_seeds
            for family in (
                CurieWeissAncillaFamily.ROLE_LINEAR,
                CurieWeissAncillaFamily.ROLE_QUADRATIC,
            )
        )
        if len(runs) != 20 or len({run.run_id for run in runs}) != 20:
            raise ProtocolValidationError("encoding comparison must contain 20 fits")
        return runs
    runs: list[CurieWeissAncillaRunIdentity] = []
    for ancillas in manifest.ancilla_rungs:
        for seed in manifest.optimizer_seeds:
            for family in (
                CurieWeissAncillaFamily.CONDITIONED,
                CurieWeissAncillaFamily.NO_CONDITION,
            ):
                runs.append(
                    CurieWeissAncillaRunIdentity(
                        manifest.campaign_version,
                        manifest.visible_qubits,
                        ancillas,
                        family,
                        None,
                        seed,
                        manifest.digest,
                    )
                )
    for seed in manifest.optimizer_seeds:
        for beta in manifest.training_betas:
            runs.append(
                CurieWeissAncillaRunIdentity(
                    manifest.campaign_version,
                    manifest.visible_qubits,
                    manifest.headline_ancillas,
                    CurieWeissAncillaFamily.TEMPERATURE_REFERENCE,
                    beta,
                    seed,
                    manifest.digest,
                )
            )
    expected = 2 * len(manifest.ancilla_rungs) * len(manifest.optimizer_seeds) + (
        len(manifest.training_betas) * len(manifest.optimizer_seeds)
    )
    if len(runs) != expected or len({run.run_id for run in runs}) != expected:
        raise ProtocolValidationError(
            f"ancilla campaign must contain {expected} unique fits"
        )
    return tuple(runs)

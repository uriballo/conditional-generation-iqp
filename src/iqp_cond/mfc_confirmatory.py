"""Frozen protocol and data operations for the confirmatory MFC campaign."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import ProtocolValidationError, content_digest


class MFCCoding(StrEnum):
    """Bitstring assignment used for an outgoing-energy bin."""

    BINARY = "binary"
    GRAY = "gray"


class MFCStage(StrEnum):
    """Stage of the two-step MFC campaign."""

    SCREEN = "screen"
    CONDITIONAL = "conditional"


class MFCModelFamily(StrEnum):
    """Models and the deterministic baseline declared by the protocol."""

    CONDITIONED_IQP = "conditioned_iqp"
    NO_CONDITION_IQP = "no_condition_iqp"
    SEPARATE_IQP = "separate_iqp"
    CATEGORICAL_MLP = "categorical_mlp"
    HISTOGRAM_INTERPOLATION = "histogram_interpolation"

    @property
    def is_trained(self) -> bool:
        return self is not MFCModelFamily.HISTOGRAM_INTERPOLATION


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ProtocolValidationError(
            f"{context} keys differ; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _integer_tuple(value: Any, context: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ProtocolValidationError(f"{context} must be a non-empty integer list")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class MFCCampaignManifest:
    """Validated, immutable description of the MFC confirmatory campaign."""

    schema_version: int
    campaign_version: str
    energies_gev: tuple[int, ...]
    bin_counts: tuple[int, ...]
    codings: tuple[MFCCoding, ...]
    total_events_per_energy: int
    train_events_per_observed_energy: int
    validation_events_per_observed_energy: int
    split_root_seed: int
    anchor_energies_gev: tuple[int, ...]
    development_seeds: tuple[int, ...]
    median_validation_tvd_max: float
    worst_validation_tvd_max: float
    held_out_energies_gev: tuple[int, ...]
    confirmatory_seeds: tuple[int, ...]
    bootstrap_replicates: int
    confidence_level: float

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ProtocolValidationError("MFC schema_version must be 1")
        if self.campaign_version not in {
            "mfc_confirmatory_v1",
            "mfc_pseudorapidity_confirmatory_v1",
        }:
            raise ProtocolValidationError("MFC campaign version is not frozen")
        if self.energies_gev != (50, 75, 100, 125, 150, 175, 200):
            raise ProtocolValidationError("MFC energy grid differs")
        if self.bin_counts != (16, 32, 64):
            raise ProtocolValidationError("MFC bin counts must be 16/32/64")
        if self.codings != (MFCCoding.BINARY, MFCCoding.GRAY):
            raise ProtocolValidationError("MFC codings must be binary and Gray")
        if (
            self.total_events_per_energy != 100_000
            or self.train_events_per_observed_energy != 10_000
            or self.validation_events_per_observed_energy != 2_000
        ):
            raise ProtocolValidationError("MFC data split differs")
        if self.split_root_seed < 0:
            raise ProtocolValidationError("split_root_seed must be non-negative")
        if self.anchor_energies_gev != (50, 125, 200):
            raise ProtocolValidationError("MFC screening anchors differ")
        if self.development_seeds != (90, 91, 92):
            raise ProtocolValidationError("MFC development seeds differ")
        if not math.isclose(self.median_validation_tvd_max, 0.10):
            raise ProtocolValidationError("MFC median promotion threshold differs")
        if not math.isclose(self.worst_validation_tvd_max, 0.15):
            raise ProtocolValidationError("MFC worst-seed threshold differs")
        if self.held_out_energies_gev != (75, 100, 125, 150, 175):
            raise ProtocolValidationError("MFC leave-one-out folds differ")
        if self.confirmatory_seeds != tuple(range(100, 110)):
            raise ProtocolValidationError("MFC confirmatory seeds must be 100--109")
        if self.bootstrap_replicates != 10_000 or not math.isclose(
            self.confidence_level, 0.95
        ):
            raise ProtocolValidationError("MFC statistical protocol differs")

    @property
    def test_events_per_energy(self) -> int:
        return self.total_events_per_energy - (
            self.train_events_per_observed_energy
            + self.validation_events_per_observed_energy
        )

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        observable = (
            "outgoing_muon_pseudorapidity"
            if self.campaign_version == "mfc_pseudorapidity_confirmatory_v1"
            else "outgoing_muon_energy/incoming_muon_energy"
        )
        return {
            "schema_version": self.schema_version,
            "campaign_version": self.campaign_version,
            "dataset": {
                "energies_gev": list(self.energies_gev),
                "bin_counts": list(self.bin_counts),
                "codings": [coding.value for coding in self.codings],
                "scaled_observable": observable,
                "total_events_per_energy": self.total_events_per_energy,
            },
            "data_split": {
                "train_events_per_observed_energy": (
                    self.train_events_per_observed_energy
                ),
                "validation_events_per_observed_energy": (
                    self.validation_events_per_observed_energy
                ),
                "test_events_per_energy": self.test_events_per_energy,
                "split_root_seed": self.split_root_seed,
                "method": "numpy_pcg64_permutation_v1",
            },
            "screen": {
                "anchor_energies_gev": list(self.anchor_energies_gev),
                "development_seeds": list(self.development_seeds),
                "model_family": MFCModelFamily.SEPARATE_IQP.value,
                "median_validation_tvd_max": self.median_validation_tvd_max,
                "worst_validation_tvd_max": self.worst_validation_tvd_max,
            },
            "confirmatory": {
                "held_out_energies_gev": list(self.held_out_energies_gev),
                "seeds": list(self.confirmatory_seeds),
                "trained_families": [
                    MFCModelFamily.CONDITIONED_IQP.value,
                    MFCModelFamily.NO_CONDITION_IQP.value,
                    MFCModelFamily.SEPARATE_IQP.value,
                    MFCModelFamily.CATEGORICAL_MLP.value,
                ],
                "evaluation_baseline": (MFCModelFamily.HISTOGRAM_INTERPOLATION.value),
            },
            "statistics": {
                "bootstrap_replicates": self.bootstrap_replicates,
                "confidence_level": self.confidence_level,
                "primary_metrics": [
                    "tvd",
                    (
                        "wasserstein_eta_bin"
                        if self.campaign_version == "mfc_pseudorapidity_confirmatory_v1"
                        else "wasserstein_scaled_energy"
                    ),
                ],
                "diagnostic_metrics": ["mean_error", "variance_error"],
            },
        }

    @classmethod
    def from_json(cls, path: Path) -> MFCCampaignManifest:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolValidationError(f"invalid MFC manifest: {path}") from error
        if not isinstance(value, dict):
            raise ProtocolValidationError("MFC manifest must be an object")
        _exact_keys(
            value,
            {
                "schema_version",
                "campaign_version",
                "dataset",
                "data_split",
                "screen",
                "confirmatory",
                "statistics",
            },
            "MFC manifest",
        )
        dataset = value["dataset"]
        split = value["data_split"]
        screen = value["screen"]
        confirmatory = value["confirmatory"]
        statistics = value["statistics"]
        for item, expected, name in (
            (
                dataset,
                {
                    "energies_gev",
                    "bin_counts",
                    "codings",
                    "scaled_observable",
                    "total_events_per_energy",
                },
                "MFC dataset",
            ),
            (
                split,
                {
                    "train_events_per_observed_energy",
                    "validation_events_per_observed_energy",
                    "test_events_per_energy",
                    "split_root_seed",
                    "method",
                },
                "MFC data split",
            ),
            (
                screen,
                {
                    "anchor_energies_gev",
                    "development_seeds",
                    "model_family",
                    "median_validation_tvd_max",
                    "worst_validation_tvd_max",
                },
                "MFC screen",
            ),
            (
                confirmatory,
                {
                    "held_out_energies_gev",
                    "seeds",
                    "trained_families",
                    "evaluation_baseline",
                },
                "MFC confirmatory plan",
            ),
            (
                statistics,
                {
                    "bootstrap_replicates",
                    "confidence_level",
                    "primary_metrics",
                    "diagnostic_metrics",
                },
                "MFC statistics",
            ),
        ):
            if not isinstance(item, dict):
                raise ProtocolValidationError(f"{name} must be an object")
            _exact_keys(item, expected, name)
        is_eta = value["campaign_version"] == "mfc_pseudorapidity_confirmatory_v1"
        frozen_literals = {
            "scaled_observable": "outgoing_muon_pseudorapidity"
            if is_eta
            else "outgoing_muon_energy/incoming_muon_energy",
            "split_method": "numpy_pcg64_permutation_v1",
            "screen_family": MFCModelFamily.SEPARATE_IQP.value,
            "trained_families": [
                MFCModelFamily.CONDITIONED_IQP.value,
                MFCModelFamily.NO_CONDITION_IQP.value,
                MFCModelFamily.SEPARATE_IQP.value,
                MFCModelFamily.CATEGORICAL_MLP.value,
            ],
            "evaluation_baseline": MFCModelFamily.HISTOGRAM_INTERPOLATION.value,
            "primary_metrics": [
                "tvd",
                "wasserstein_eta_bin" if is_eta else "wasserstein_scaled_energy",
            ],
            "diagnostic_metrics": ["mean_error", "variance_error"],
        }
        actual_literals = {
            "scaled_observable": dataset["scaled_observable"],
            "split_method": split["method"],
            "screen_family": screen["model_family"],
            "trained_families": confirmatory["trained_families"],
            "evaluation_baseline": confirmatory["evaluation_baseline"],
            "primary_metrics": statistics["primary_metrics"],
            "diagnostic_metrics": statistics["diagnostic_metrics"],
        }
        if actual_literals != frozen_literals:
            raise ProtocolValidationError("MFC frozen method declarations differ")
        total = int(dataset["total_events_per_energy"])
        train = int(split["train_events_per_observed_energy"])
        validation = int(split["validation_events_per_observed_energy"])
        if int(split["test_events_per_energy"]) != total - train - validation:
            raise ProtocolValidationError("MFC test split must contain all remainders")
        return cls(
            schema_version=int(value["schema_version"]),
            campaign_version=str(value["campaign_version"]),
            energies_gev=_integer_tuple(dataset["energies_gev"], "MFC energies"),
            bin_counts=_integer_tuple(dataset["bin_counts"], "MFC bin counts"),
            codings=tuple(MFCCoding(item) for item in dataset["codings"]),
            total_events_per_energy=total,
            train_events_per_observed_energy=train,
            validation_events_per_observed_energy=validation,
            split_root_seed=int(split["split_root_seed"]),
            anchor_energies_gev=_integer_tuple(
                screen["anchor_energies_gev"], "MFC screen anchors"
            ),
            development_seeds=_integer_tuple(
                screen["development_seeds"], "MFC development seeds"
            ),
            median_validation_tvd_max=float(screen["median_validation_tvd_max"]),
            worst_validation_tvd_max=float(screen["worst_validation_tvd_max"]),
            held_out_energies_gev=_integer_tuple(
                confirmatory["held_out_energies_gev"], "MFC held-out energies"
            ),
            confirmatory_seeds=_integer_tuple(
                confirmatory["seeds"], "MFC confirmatory seeds"
            ),
            bootstrap_replicates=int(statistics["bootstrap_replicates"]),
            confidence_level=float(statistics["confidence_level"]),
        )


@dataclass(frozen=True, slots=True)
class MFCRunIdentity:
    """Content-addressed identity for one planned fit or evaluation."""

    campaign_version: str
    stage: MFCStage
    bin_count: int
    coding: MFCCoding
    family: MFCModelFamily
    seed: int | None
    anchor_energy_gev: int | None
    held_out_energy_gev: int | None
    requires_promotion: bool
    configuration_digest: str

    def __post_init__(self) -> None:
        if self.campaign_version not in {
            "mfc_confirmatory_v1",
            "mfc_pseudorapidity_confirmatory_v1",
        }:
            raise ProtocolValidationError("unknown MFC campaign version")
        if self.bin_count not in {16, 32, 64}:
            raise ProtocolValidationError("unsupported MFC bin count")
        if len(self.configuration_digest) != 64:
            raise ProtocolValidationError("configuration_digest must be SHA-256")
        if self.stage is MFCStage.SCREEN:
            if (
                self.family is not MFCModelFamily.SEPARATE_IQP
                or self.seed not in {90, 91, 92}
                or self.anchor_energy_gev not in {50, 125, 200}
                or self.held_out_energy_gev is not None
                or self.requires_promotion
            ):
                raise ProtocolValidationError("invalid MFC screening identity")
        elif (
            self.anchor_energy_gev is not None
            or self.held_out_energy_gev not in {75, 100, 125, 150, 175}
            or not self.requires_promotion
        ):
            raise ProtocolValidationError("invalid MFC confirmatory identity")
        if self.stage is MFCStage.CONDITIONAL:
            if self.family.is_trained and self.seed not in range(100, 110):
                raise ProtocolValidationError("trained MFC run needs seed 100--109")
            if not self.family.is_trained and self.seed is not None:
                raise ProtocolValidationError("deterministic baseline has no seed")

    @property
    def run_id(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_version": self.campaign_version,
            "stage": self.stage.value,
            "bin_count": self.bin_count,
            "coding": self.coding.value,
            "family": self.family.value,
            "seed": self.seed,
            "anchor_energy_gev": self.anchor_energy_gev,
            "held_out_energy_gev": self.held_out_energy_gev,
            "requires_promotion": self.requires_promotion,
            "configuration_digest": self.configuration_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MFCRunIdentity:
        expected = {
            "campaign_version",
            "stage",
            "bin_count",
            "coding",
            "family",
            "seed",
            "anchor_energy_gev",
            "held_out_energy_gev",
            "requires_promotion",
            "configuration_digest",
        }
        _exact_keys(value, expected, "MFC run identity")
        for key in ("seed", "anchor_energy_gev", "held_out_energy_gev"):
            item = value[key]
            if item is not None and (
                isinstance(item, bool) or not isinstance(item, int)
            ):
                raise ProtocolValidationError(f"{key} must be an integer or null")
        if not isinstance(value["requires_promotion"], bool):
            raise ProtocolValidationError("requires_promotion must be boolean")
        return cls(
            campaign_version=str(value["campaign_version"]),
            stage=MFCStage(str(value["stage"])),
            bin_count=int(value["bin_count"]),
            coding=MFCCoding(str(value["coding"])),
            family=MFCModelFamily(str(value["family"])),
            seed=value["seed"],
            anchor_energy_gev=value["anchor_energy_gev"],
            held_out_energy_gev=value["held_out_energy_gev"],
            requires_promotion=value["requires_promotion"],
            configuration_digest=str(value["configuration_digest"]),
        )


def plan_mfc_screen_runs(
    manifest: MFCCampaignManifest,
) -> tuple[MFCRunIdentity, ...]:
    runs = tuple(
        MFCRunIdentity(
            manifest.campaign_version,
            MFCStage.SCREEN,
            bin_count,
            coding,
            MFCModelFamily.SEPARATE_IQP,
            seed,
            anchor,
            None,
            False,
            manifest.digest,
        )
        for bin_count in manifest.bin_counts
        for coding in manifest.codings
        for anchor in manifest.anchor_energies_gev
        for seed in manifest.development_seeds
    )
    if len(runs) != 54 or len({run.run_id for run in runs}) != 54:
        raise RuntimeError("MFC screen must contain 54 unique runs")
    return runs


def plan_mfc_conditional_runs(
    manifest: MFCCampaignManifest,
) -> tuple[MFCRunIdentity, ...]:
    trained_families = tuple(family for family in MFCModelFamily if family.is_trained)
    runs: list[MFCRunIdentity] = []
    for bin_count in manifest.bin_counts:
        for coding in manifest.codings:
            for held_out in manifest.held_out_energies_gev:
                for family in trained_families:
                    for seed in manifest.confirmatory_seeds:
                        runs.append(
                            MFCRunIdentity(
                                manifest.campaign_version,
                                MFCStage.CONDITIONAL,
                                bin_count,
                                coding,
                                family,
                                seed,
                                None,
                                held_out,
                                True,
                                manifest.digest,
                            )
                        )
                runs.append(
                    MFCRunIdentity(
                        manifest.campaign_version,
                        MFCStage.CONDITIONAL,
                        bin_count,
                        coding,
                        MFCModelFamily.HISTOGRAM_INTERPOLATION,
                        None,
                        None,
                        held_out,
                        True,
                        manifest.digest,
                    )
                )
    if len(runs) != 1_230 or len({run.run_id for run in runs}) != 1_230:
        raise RuntimeError("MFC conditional plan must contain 1,230 identities")
    return tuple(runs)


def plan_mfc_runs(manifest: MFCCampaignManifest) -> tuple[MFCRunIdentity, ...]:
    """Plan screening and gated confirmatory work without reading results."""
    runs = plan_mfc_screen_runs(manifest) + plan_mfc_conditional_runs(manifest)
    if len(runs) != 1_284 or len({run.run_id for run in runs}) != 1_284:
        raise RuntimeError("MFC campaign must contain 1,284 unique identities")
    return runs


@dataclass(frozen=True, slots=True)
class EventSplit:
    """Read-only indices for one energy's train, validation, and test sets."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def deterministic_split_indices(
    energy_gev: int,
    manifest: MFCCampaignManifest,
) -> EventSplit:
    """Return a stable, energy-specific partition of the 100,000 event rows."""
    if energy_gev not in manifest.energies_gev:
        raise ValueError("energy is not part of the MFC campaign")
    seed = int(
        content_digest(
            {
                "root_seed": manifest.split_root_seed,
                "energy_gev": energy_gev,
                "role": "mfc_event_partition",
            }
        )[:16],
        16,
    )
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(
        manifest.total_events_per_energy
    )
    train_end = manifest.train_events_per_observed_energy
    validation_end = train_end + manifest.validation_events_per_observed_energy
    arrays = (
        permutation[:train_end].copy(),
        permutation[train_end:validation_end].copy(),
        permutation[validation_end:].copy(),
    )
    for array in arrays:
        array.flags.writeable = False
    return EventSplit(*arrays)


def bin_scaled_outgoing_energy(
    outgoing_energy: np.ndarray | Sequence[float],
    incoming_energy_gev: float,
    bin_count: int,
) -> np.ndarray:
    """Map outgoing energies to uniform bins after scaling by incoming energy."""
    if not math.isfinite(incoming_energy_gev) or incoming_energy_gev <= 0.0:
        raise ValueError("incoming energy must be positive and finite")
    if bin_count not in {16, 32, 64}:
        raise ValueError("bin_count must be 16, 32, or 64")
    outgoing = np.asarray(outgoing_energy, dtype=np.float64)
    if outgoing.ndim != 1 or not np.all(np.isfinite(outgoing)):
        raise ValueError("outgoing energies must be a finite one-dimensional array")
    scaled = np.clip(outgoing / incoming_energy_gev, 0.0, 1.0)
    return np.minimum((scaled * bin_count).astype(np.int64), bin_count - 1)


def pooled_quantile_bin_edges(
    events: Mapping[int, np.ndarray], bin_count: int
) -> np.ndarray:
    """Return shared, unbounded-edge quantile bins over all energies."""
    if bin_count not in {16, 32, 64}:
        raise ValueError("bin_count must be 16, 32, or 64")
    values = np.concatenate([np.asarray(v, dtype=np.float64) for v in events.values()])
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("observable values must be finite")
    edges = np.quantile(values, np.arange(1, bin_count) / bin_count, method="linear")
    return np.concatenate(([-np.inf], edges, [np.inf]))


def bin_pseudorapidity(
    pseudorapidity: np.ndarray | Sequence[float], edges: np.ndarray | Sequence[float]
) -> np.ndarray:
    """Assign finite eta values to shared quantile bins."""
    values = np.asarray(pseudorapidity, dtype=np.float64)
    boundaries = np.asarray(edges, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("pseudorapidity must be a finite one-dimensional array")
    if (
        boundaries.ndim != 1
        or boundaries.size not in {17, 33, 65}
        or boundaries[0] != -np.inf
        or boundaries[-1] != np.inf
        or np.any(np.diff(boundaries[1:-1]) < 0)
    ):
        raise ValueError("edges must be unbounded, increasing 16/32/64-bin boundaries")
    return np.searchsorted(boundaries[1:-1], values, side="right").astype(np.int64)


def encode_bin_indices(
    bin_indices: np.ndarray | Sequence[int],
    bin_count: int,
    coding: MFCCoding,
) -> np.ndarray:
    """Return computational-basis labels for physical bin indices."""
    indices = np.asarray(bin_indices)
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("bin indices must be integers")
    if bin_count not in {16, 32, 64} or np.any((indices < 0) | (indices >= bin_count)):
        raise ValueError("bin indices lie outside the requested coding")
    values = indices.astype(np.int64, copy=True)
    if coding is MFCCoding.GRAY:
        values ^= values >> 1
    elif coding is not MFCCoding.BINARY:
        raise ValueError("unsupported MFC coding")
    return values


def decode_basis_labels(
    basis_labels: np.ndarray | Sequence[int],
    bin_count: int,
    coding: MFCCoding,
) -> np.ndarray:
    """Recover physical bin indices from binary or reflected-Gray labels."""
    labels = np.asarray(basis_labels)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("basis labels must be integers")
    if bin_count not in {16, 32, 64} or np.any((labels < 0) | (labels >= bin_count)):
        raise ValueError("basis labels lie outside the requested coding")
    decoded = labels.astype(np.int64, copy=True)
    if coding is MFCCoding.GRAY:
        shift = 1
        while shift < bin_count.bit_length():
            decoded ^= decoded >> shift
            shift *= 2
    elif coding is not MFCCoding.BINARY:
        raise ValueError("unsupported MFC coding")
    return decoded


def decode_probability_vector(
    basis_probabilities: np.ndarray | Sequence[float],
    coding: MFCCoding,
) -> np.ndarray:
    """Reorder basis-label probabilities into increasing physical-bin order."""
    basis = _probabilities(basis_probabilities, "basis probabilities")
    labels = encode_bin_indices(np.arange(basis.size), basis.size, coding)
    return basis[labels]


def _probabilities(value: np.ndarray | Sequence[float], context: str) -> np.ndarray:
    probabilities = np.asarray(value, dtype=np.float64)
    if (
        probabilities.ndim != 1
        or probabilities.size not in {16, 32, 64}
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-10)
    ):
        raise ValueError(f"{context} must be a normalized 16/32/64-bin distribution")
    return probabilities


@dataclass(frozen=True, slots=True)
class MFCMetrics:
    """Primary distances and signed moment errors for one prediction."""

    tvd: float
    wasserstein_scaled_energy: float
    mean_error: float
    variance_error: float


def distribution_metrics(
    target: np.ndarray | Sequence[float],
    prediction: np.ndarray | Sequence[float],
) -> MFCMetrics:
    """Evaluate decoded-bin distances and moments on uniform scaled-energy bins."""
    target_p = _probabilities(target, "target")
    prediction_p = _probabilities(prediction, "prediction")
    if target_p.shape != prediction_p.shape:
        raise ValueError("target and prediction must have the same bin count")
    bin_count = target_p.size
    centres = (np.arange(bin_count, dtype=np.float64) + 0.5) / bin_count
    target_mean = float(target_p @ centres)
    prediction_mean = float(prediction_p @ centres)
    target_variance = float(target_p @ (centres - target_mean) ** 2)
    prediction_variance = float(prediction_p @ (centres - prediction_mean) ** 2)
    return MFCMetrics(
        tvd=float(0.5 * np.abs(target_p - prediction_p).sum()),
        wasserstein_scaled_energy=float(
            np.abs(np.cumsum(target_p - prediction_p)[:-1]).sum() / bin_count
        ),
        mean_error=prediction_mean - target_mean,
        variance_error=prediction_variance - target_variance,
    )


def interpolate_histograms(
    observed: Mapping[int, np.ndarray | Sequence[float]],
    held_out_energy_gev: int,
) -> np.ndarray:
    """Linearly interpolate the two adjacent observed empirical histograms."""
    if held_out_energy_gev in observed:
        raise ValueError("held-out energy must not be present in observed data")
    lower = max(
        (energy for energy in observed if energy < held_out_energy_gev),
        default=None,
    )
    upper = min(
        (energy for energy in observed if energy > held_out_energy_gev),
        default=None,
    )
    if lower is None or upper is None:
        raise ValueError("held-out energy must be bracketed by observed conditions")
    lower_p = _probabilities(observed[lower], "lower histogram")
    upper_p = _probabilities(observed[upper], "upper histogram")
    if lower_p.shape != upper_p.shape:
        raise ValueError("adjacent histograms must use the same bin count")
    upper_weight = (held_out_energy_gev - lower) / (upper - lower)
    return (1.0 - upper_weight) * lower_p + upper_weight * upper_p


def passes_promotion_gate(
    validation_tvd: Mapping[int, Mapping[int, float]],
    manifest: MFCCampaignManifest,
) -> bool:
    """Apply the frozen anchor-median and worst-seed screening thresholds."""
    if set(validation_tvd) != set(manifest.anchor_energies_gev):
        raise ValueError("promotion data must contain exactly the three anchors")
    all_scores: list[float] = []
    for anchor in manifest.anchor_energies_gev:
        scores_by_seed = validation_tvd[anchor]
        if set(scores_by_seed) != set(manifest.development_seeds):
            raise ValueError("each anchor must contain the three development seeds")
        scores = np.asarray(
            [scores_by_seed[seed] for seed in manifest.development_seeds],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("validation TVD values must lie in [0, 1]")
        if float(np.median(scores)) > manifest.median_validation_tvd_max:
            return False
        all_scores.extend(scores.tolist())
    return max(all_scores) <= manifest.worst_validation_tvd_max

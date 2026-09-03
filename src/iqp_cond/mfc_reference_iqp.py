"""Frozen, approximate reproduction of the published MFC benchmark."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier

from .ansatz import complete_ansatz
from .artifacts import sha256_file
from .exact import exact_iqp_probabilities_many
from .training import TrainingConfig, train

ENERGIES = (50, 75, 100, 125, 150, 175, 200)
OBSERVED = (50, 75, 100, 150, 175, 200)
EVALUATION_ENERGIES = (100, 125, 150)
FAMILIES = (
    "conditioned_iqp",
    "no_condition_iqp",
    "separate_iqp",
    "categorical_mlp",
)
PUBLISHED_TABLE_V = {
    "ideal_qcbm": {"100": 0.033, "125": 0.033, "150": 0.016},
    "noisy_qcbm": {"100": 0.067, "125": 0.035, "150": 0.046},
    "hardware_qcbm": {"100": 0.150, "125": 0.094, "150": 0.130},
    "gmmd": {"100": 0.016, "125": 0.034, "150": 0.032},
}


@dataclass(frozen=True)
class ReferenceConfig:
    """The exact schema accepted by the immutable reference campaign."""

    schema_version: int
    energies: tuple[int, ...]
    held_out: int
    bins: int
    n_data_qubits: int
    train_events: int
    test_events: int
    split_root_seed: int
    updates: int
    families: tuple[str, ...]
    seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("energies", "families", "seeds"):
            result[name] = list(result[name])
        return result

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path) -> ReferenceConfig:
    """Load the single accepted protocol configuration."""
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid reference config JSON") from error
    expected = set(ReferenceConfig.__dataclass_fields__)
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("config keys do not match frozen schema")
    integer_names = (
        "schema_version",
        "held_out",
        "bins",
        "n_data_qubits",
        "train_events",
        "test_events",
        "split_root_seed",
        "updates",
    )
    if any(type(raw[name]) is not int for name in integer_names):
        raise ValueError("integer config fields must contain integers")
    if not all(type(value) is int for value in raw["energies"] + raw["seeds"]):
        raise ValueError("energies and seeds must contain integers")
    if not all(type(value) is str for value in raw["families"]):
        raise ValueError("families must contain strings")
    config = ReferenceConfig(
        schema_version=raw["schema_version"],
        energies=tuple(raw["energies"]),
        held_out=raw["held_out"],
        bins=raw["bins"],
        n_data_qubits=raw["n_data_qubits"],
        train_events=raw["train_events"],
        test_events=raw["test_events"],
        split_root_seed=raw["split_root_seed"],
        updates=raw["updates"],
        families=tuple(raw["families"]),
        seeds=tuple(raw["seeds"]),
    )
    frozen = (
        config.schema_version == 1
        and config.energies == ENERGIES
        and config.held_out == 125
        and config.bins == 8
        and config.n_data_qubits == 3
        and config.train_events == 5_120
        and config.test_events == 5_120
        and config.split_root_seed == 20_260_831
        and config.updates == 1_800
        and config.families == FAMILIES
        and config.seeds == tuple(range(100, 110))
    )
    if not frozen:
        raise ValueError("config contains values outside frozen protocol")
    return config


def load_events(path: str | Path) -> dict[int, np.ndarray]:
    """Load the seven declared energy vectors without implicit discovery."""
    archive_path = Path(path)
    if not archive_path.is_file():
        raise ValueError(f"event archive does not exist: {archive_path}")
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            expected = {str(energy) for energy in ENERGIES}
            if set(archive.files) != expected:
                raise ValueError("archive keys differ from the frozen energy grid")
            events = {
                energy: np.asarray(archive[str(energy)], dtype=np.float64).copy()
                for energy in ENERGIES
            }
    except OSError as error:
        raise ValueError(f"could not read event archive: {archive_path}") from error
    validate_events(events)
    return events


def validate_events(events: Mapping[int, np.ndarray]) -> None:
    if set(events) != set(ENERGIES):
        raise ValueError("events differ from the frozen energy grid")
    for energy in ENERGIES:
        values = np.asarray(events[energy])
        if values.ndim != 1 or values.size < 10_240:
            raise ValueError(f"energy {energy} requires at least 10240 events")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"energy {energy} contains invalid values")


def split_indices(
    n_events: int, energy: int, root_seed: int = 20_260_831
) -> tuple[np.ndarray, np.ndarray]:
    if n_events < 10_240:
        raise ValueError("at least 10240 events required")
    permutation = np.random.default_rng(root_seed + energy).permutation(n_events)
    return np.sort(permutation[:5_120]), np.sort(permutation[5_120:10_240])


def condition_angle(energy: float) -> float:
    return float(np.pi * (0.35 - 0.70 * ((energy - 50.0) / 150.0)))


def condition_angles(
    energies: Sequence[int], *, conditioned: bool = True
) -> np.ndarray:
    scalar = [condition_angle(energy) if conditioned else 0.0 for energy in energies]
    return np.repeat(np.asarray(scalar)[:, None], 3, axis=1)


def physical_bins(
    values: np.ndarray, energy: int, bins: int = 8
) -> tuple[np.ndarray, int]:
    scaled = np.asarray(values, dtype=np.float64) / energy
    if scaled.ndim != 1 or not np.all(np.isfinite(scaled)) or np.any(scaled < 0):
        raise ValueError("raw events must be finite and nonnegative")
    labels = np.floor(np.clip(scaled, 0.0, 1.0) * bins).astype(np.int64)
    labels = np.minimum(labels, bins - 1)
    return labels, int(np.count_nonzero(scaled > 1.0))


def target_histogram(
    values: np.ndarray, energy: int, bins: int = 8
) -> tuple[np.ndarray, int]:
    labels, overflow = physical_bins(values, energy, bins)
    counts = np.bincount(labels, minlength=bins).astype(np.float64)
    return counts / counts.sum(), overflow


def bit_codes(labels: np.ndarray) -> np.ndarray:
    """Encode physical bin labels using simulator-compatible little endian."""
    labels = np.asarray(labels, dtype=np.int64)
    return ((labels[:, None] >> np.arange(3)) & 1).astype(np.float64)


def decode_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Return the physical-bin vector (basis index equals little-endian label)."""
    result = np.asarray(probabilities, dtype=np.float64)
    if result.shape != (8,) or np.any(result < -1e-12):
        raise ValueError("probability vector must contain eight nonnegative values")
    result = np.maximum(result, 0.0)
    return result / result.sum()


def run_id(config: ReferenceConfig, family: str, seed: int) -> str:
    return _digest(
        {
            "protocol": "mfc_reference_iqp_v1",
            "config": config.digest,
            "family": family,
            "seed": seed,
        }
    )


def plan(config: ReferenceConfig) -> list[dict[str, Any]]:
    return [
        {"run_id": run_id(config, family, seed), "family": family, "seed": seed}
        for family in config.families
        for seed in config.seeds
    ]


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = decode_probabilities(target)
    prediction = decode_probabilities(prediction)
    centers = (np.arange(8) + 0.5) / 8
    target_mean = float(target @ centers)
    prediction_mean = float(prediction @ centers)
    return {
        "tvd": float(0.5 * np.abs(target - prediction).sum()),
        "w1": float(np.abs(np.cumsum(target - prediction)[:-1]).sum() / 8),
        "mean_error": prediction_mean - target_mean,
        "variance_error": float(
            prediction @ ((centers - prediction_mean) ** 2)
            - target @ ((centers - target_mean) ** 2)
        ),
    }


def _training_config(updates: int) -> TrainingConfig:
    if updates < 1:
        raise ValueError("updates must be positive")
    return TrainingConfig(
        steps=updates,
        batch_size=512,
        learning_rate=0.01,
        final_learning_rate=0.005,
        validation_interval=max(1, min(60, updates)),
        sigmas=(0.01, 0.1, 1.0, 10.0, 100.0),
        n_ops=64,
        n_iqp_samples=2_048,
        validation_n_ops=128,
        validation_n_iqp_samples=4_096,
    )


def _fit_iqp(
    samples: np.ndarray,
    energies: Sequence[int],
    seed: int,
    updates: int,
    *,
    conditioned: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    ansatz = complete_ansatz(3)
    angles = condition_angles(energies, conditioned=conditioned)
    result = train(
        "mfc_reference_iqp_v1",
        ansatz,
        angles,
        samples,
        samples,
        seed,
        _training_config(updates),
    )
    return result.best_params, {
        "n_parameters": ansatz.n_parameters,
        "best_step": result.best_step,
        "best_training_proxy_loss": result.best_validation_loss,
        "checkpoint_data": "training_partition_only",
        "updates": updates,
    }


def _predict_iqp(
    parameters: np.ndarray, energies: Sequence[int], *, conditioned: bool
) -> np.ndarray:
    return exact_iqp_probabilities_many(
        parameters,
        complete_ansatz(3).generators,
        condition_angles(energies, conditioned=conditioned),
    )


def _fit_mlp(
    train_labels: Mapping[int, np.ndarray], seed: int, updates: int
) -> tuple[MLPClassifier, dict[str, Any]]:
    features = np.concatenate(
        [np.full((8, 1), (energy - 50.0) / 150.0) for energy in OBSERVED]
    )
    labels = np.tile(np.arange(8), len(OBSERVED))
    weights = np.concatenate(
        [np.bincount(train_labels[energy], minlength=8) for energy in OBSERVED]
    ).astype(np.float64)
    model = MLPClassifier(
        hidden_layer_sizes=(32, 32),
        activation="relu",
        solver="adam",
        batch_size=48,
        learning_rate_init=0.01,
        max_iter=1,
        shuffle=True,
        random_state=seed,
        tol=0.0,
        warm_start=True,
    )
    best_loss = float("inf")
    best_step = 0
    best_parameters: tuple[list[np.ndarray], list[np.ndarray]] | None = None
    for step in range(1, updates + 1):
        model.partial_fit(features, labels, classes=np.arange(8), sample_weight=weights)
        loss = 0.0
        for energy in OBSERVED:
            probability = model.predict_proba([[(energy - 50.0) / 150.0]])[0]
            counts = np.bincount(train_labels[energy], minlength=8)
            loss -= float(
                counts @ np.log(np.maximum(probability, np.finfo(float).tiny))
            )
        loss /= weights.sum()
        if loss < best_loss:
            best_loss = loss
            best_step = step
            best_parameters = (
                [value.copy() for value in model.coefs_],
                [value.copy() for value in model.intercepts_],
            )
    assert best_parameters is not None
    model.coefs_, model.intercepts_ = best_parameters
    return model, {
        "hidden_layer_sizes": [32, 32],
        "weighted_histogram_training": True,
        "best_step": best_step,
        "best_training_cross_entropy": best_loss,
        "updates": updates,
    }


def _source_digest(repository: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "configs/mfc_reference_iqp_v1.json",
        "src/iqp_cond/ansatz.py",
        "src/iqp_cond/exact.py",
        "src/iqp_cond/mfc_reference_iqp.py",
        "src/iqp_cond/mmd.py",
        "src/iqp_cond/training.py",
        "uv.lock",
    ):
        digest.update(relative.encode())
        digest.update((repository / relative).read_bytes())
    return digest.hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def execute(
    events: Mapping[int, np.ndarray],
    config: ReferenceConfig,
    family: str,
    seed: int,
    *,
    updates: int | None = None,
    event_archive: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one planned fit and return its immutable-artifact payload."""
    validate_events(events)
    if family not in config.families or seed not in config.seeds:
        raise ValueError("run is not part of the frozen plan")
    requested_updates = config.updates if updates is None else updates
    if requested_updates < 1 or requested_updates > config.updates:
        raise ValueError("updates must be within the frozen campaign budget")

    train_labels: dict[int, np.ndarray] = {}
    test_labels: dict[int, np.ndarray] = {}
    clipping: dict[str, dict[str, int]] = {}
    for energy in ENERGIES:
        train_index, test_index = split_indices(
            len(events[energy]), energy, config.split_root_seed
        )
        all_labels, overflow = physical_bins(events[energy], energy, config.bins)
        train_labels[energy] = all_labels[train_index]
        test_labels[energy] = all_labels[test_index]
        clipping[str(energy)] = {"all_events_above_one": overflow}

    if family in {"conditioned_iqp", "no_condition_iqp"}:
        training = np.stack([bit_codes(train_labels[energy]) for energy in OBSERVED])
        conditioned = family == "conditioned_iqp"
        parameters, execution = _fit_iqp(
            training,
            OBSERVED,
            seed,
            requested_updates,
            conditioned=conditioned,
        )
        predictions = _predict_iqp(
            parameters, EVALUATION_ENERGIES, conditioned=conditioned
        )
        evaluated = EVALUATION_ENERGIES
    elif family == "separate_iqp":
        training = bit_codes(train_labels[config.held_out])[None, :, :]
        parameters, execution = _fit_iqp(
            training,
            [config.held_out],
            seed,
            requested_updates,
            conditioned=True,
        )
        predictions = _predict_iqp(parameters, [config.held_out], conditioned=True)
        evaluated = (config.held_out,)
        execution["oracle_capacity_reference"] = True
    else:
        model, execution = _fit_mlp(train_labels, seed, requested_updates)
        predictions = np.stack(
            [
                model.predict_proba([[(energy - 50.0) / 150.0]])[0]
                for energy in EVALUATION_ENERGIES
            ]
        )
        evaluated = EVALUATION_ENERGIES

    results: dict[str, Any] = {}
    for energy, raw_prediction in zip(evaluated, predictions, strict=True):
        target = np.bincount(test_labels[energy], minlength=8).astype(np.float64)
        target /= target.sum()
        prediction = decode_probabilities(raw_prediction)
        results[str(energy)] = {
            "target_probability": target.tolist(),
            "prediction_probability": prediction.tolist(),
            "metrics": metrics(target, prediction),
            "test_events": len(test_labels[energy]),
        }

    repository = Path(__file__).resolve().parents[2]
    archive_provenance = (
        {
            "path": str(Path(event_archive).resolve()),
            "sha256": sha256_file(Path(event_archive)),
        }
        if event_archive is not None
        else {"path": None, "sha256": None}
    )
    return {
        "schema_version": 1,
        "protocol": "mfc_reference_iqp_v1",
        "run_id": run_id(config, family, seed),
        "run": {"family": family, "seed": seed},
        "config": config.to_dict(),
        "config_digest": config.digest,
        "event_archive": archive_provenance,
        "code_revision": _git_revision(repository),
        "source_tree_digest": _source_digest(repository),
        "split": {
            "method": "PCG64(root_seed + energy)",
            "root_seed": config.split_root_seed,
            "train_events": config.train_events,
            "test_events": config.test_events,
            "held_out_125_excluded_from_shared_training": family
            in {"conditioned_iqp", "no_condition_iqp", "categorical_mlp"},
        },
        "binning": {
            "observable": "outgoing_energy / incoming_energy",
            "edges": np.linspace(0.0, 1.0, 9).tolist(),
            "coding": "little_endian_binary",
            "clipping": "clip_above_one_and_count",
            "clipping_counts": clipping,
        },
        "execution": execution,
        "results": results,
        "published_table_v_tvd": PUBLISHED_TABLE_V,
    }

"""Run harness: train an encoder arm, evaluate it exactly, record the run.

A :class:`Study` supplies the problem (conditions, targets, encoder map); the
harness owns the budgets (:class:`~experiments.common.config.RunConfig`), the
training call into the library, exact evaluation through
:mod:`iqp_cond.simulation`, metrics, and the run directory.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import iqp_cond as ciqp
import numpy as np

from .config import RunConfig
from .metrics import bitstring_tvd, mean_weight, moment_errors, profile_tvd

__all__ = ["RESULTS", "Study", "run", "run_all_seeds"]

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"


@dataclass(frozen=True)
class Study:
    """Problem definition shared by every run of a study."""

    name: str
    training_conditions: np.ndarray
    held_out_conditions: np.ndarray
    make_targets: Callable[[np.ndarray, int, int, np.random.Generator], np.ndarray]
    target_probabilities: Callable[[np.ndarray, int], np.ndarray]
    make_encoder: Callable[[np.ndarray, RunConfig], ciqp.Encoder]


def _git(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def repo_state() -> dict[str, object]:
    """Return the paper and library revisions that produced a run."""
    dirty = _git("status", "--porcelain")
    library = REPO / "iqp-cond"
    return {
        "paper_revision": _git("rev-parse", "HEAD"),
        "paper_dirty": bool(dirty) if dirty is not None else None,
        "library_revision": (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=library,
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or None
        ),
        "library_version": importlib.metadata.version("iqp-cond"),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _evaluate(
    study: Study,
    config: RunConfig,
    params: np.ndarray,
    conditions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, object]:
    encoder = study.make_encoder(conditions, config)
    model = ciqp.ConditionalModel(
        ciqp.Ansatz.complete(config.n_total),
        encoder,
        n_visible=config.n_visible if config.n_ancillas else None,
    )
    probabilities = model.probabilities(params)
    target_probabilities = study.target_probabilities(conditions, config.n_visible)
    tvds = bitstring_tvd(probabilities, target_probabilities)
    mean_error, std_error = moment_errors(
        probabilities,
        target_probabilities,
        config.n_visible,
    )
    return {
        "probabilities": probabilities,
        "target_probabilities": target_probabilities,
        "tvd": np.asarray(tvds),
        "profile_tvd": np.asarray(
            profile_tvd(probabilities, target_probabilities, config.n_visible),
        ),
        "mean_error": np.asarray(mean_error),
        "std_error": np.asarray(std_error),
        "mean_weight": np.asarray(mean_weight(probabilities, config.n_visible)),
        "target_mean_weight": np.asarray(
            mean_weight(target_probabilities, config.n_visible),
        ),
    }


def run(
    config: RunConfig,
    study: Study,
    seed: int,
    *,
    output_root: Path | None = None,
) -> dict[str, object]:
    """Train one encoder arm and write its run directory; return the metrics."""
    root = RESULTS if output_root is None else output_root
    samples_rng = np.random.default_rng(seed)
    training_targets = study.make_targets(
        study.training_conditions,
        config.n_visible,
        config.n_train,
        samples_rng,
    )
    validation_targets = study.make_targets(
        study.training_conditions,
        config.n_visible,
        config.n_validation,
        samples_rng,
    )
    held_out_targets = study.make_targets(
        study.held_out_conditions,
        config.n_visible,
        config.n_test,
        samples_rng,
    )

    encoder = study.make_encoder(study.training_conditions, config)
    model = ciqp.ConditionalModel(
        ciqp.Ansatz.complete(config.n_total),
        encoder,
        n_visible=config.n_visible if config.n_ancillas else None,
    )
    start = time.perf_counter()
    result = ciqp.train(
        model,
        training_targets,
        validation_targets,
        seed=seed,
        config=config.training_config(),
    )
    wall_seconds = time.perf_counter() - start

    training = _evaluate(
        study,
        config,
        result.best_params,
        study.training_conditions,
        training_targets,
    )
    held_out = _evaluate(
        study,
        config,
        result.best_params,
        study.held_out_conditions,
        held_out_targets,
    )

    run_id = (
        f"{study.name}-n{config.n_visible}a{config.n_ancillas}"
        f"-{config.encoder}-s{seed}-{config.digest()[:8]}"
    )
    directory = root / study.name / run_id
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "config.json", config.as_dict())
    _write_json(
        directory / "run.json",
        {
            "run_id": run_id,
            "study": study.name,
            "seed": seed,
            "wall_seconds": wall_seconds,
            "best_step": result.best_step,
            "best_validation_loss": float(result.best_validation_loss),
            "state": repo_state(),
        },
    )
    np.savez(
        directory / "checkpoint.npz",
        params=np.asarray(result.best_params),
        train_probabilities=training["probabilities"],
        held_out_probabilities=held_out["probabilities"],
        target_probabilities=training["target_probabilities"],
        held_out_target_probabilities=held_out["target_probabilities"],
    )
    metrics = {
        "study": study.name,
        "run_id": run_id,
        "seed": seed,
        "encoder": config.encoder,
        "n_visible": config.n_visible,
        "n_ancillas": config.n_ancillas,
        "wall_seconds": wall_seconds,
        "best_step": result.best_step,
        "best_validation_loss": float(result.best_validation_loss),
        "train_tvd_mean": float(training["tvd"].mean()),
        "train_tvd_per_condition": training["tvd"].tolist(),
        "held_out_tvd_mean": float(held_out["tvd"].mean()),
        "held_out_tvd_per_condition": held_out["tvd"].tolist(),
        "train_profile_tvd_mean": float(training["profile_tvd"].mean()),
        "held_out_profile_tvd_mean": float(held_out["profile_tvd"].mean()),
        "train_mean_error_mean": float(np.abs(training["mean_error"]).mean()),
        "held_out_mean_error_mean": float(np.abs(held_out["mean_error"]).mean()),
        "train_std_error_mean": float(np.abs(training["std_error"]).mean()),
        "held_out_std_error_mean": float(np.abs(held_out["std_error"]).mean()),
        "config_digest": config.digest(),
    }
    _write_json(directory / "metrics.json", metrics)
    return {**metrics, "directory": str(directory)}


def run_all_seeds(
    config: RunConfig,
    study: Study,
    *,
    output_root: Path | None = None,
) -> list[dict[str, object]]:
    """Run every configured seed and return the metric records."""
    return [run(config, study, seed, output_root=output_root) for seed in config.seeds]

"""Train, evaluate, and record conditional and condition-specific IQP models."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import iqp_cond as ciqp
import jax
import numpy as np

from .config import RunConfig
from .metrics import (
    bitstring_tvd,
    complement_tvd,
    mean_weight,
    moment_errors,
    profile_tvd,
    within_shell_residual,
)

__all__ = ["RESULTS", "Study", "repo_state", "run", "run_all_seeds"]

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"


@dataclass(frozen=True)
class Study:
    """Target family, condition sets, and encoder factory for one study."""

    name: str
    training_conditions: np.ndarray
    held_out_conditions: np.ndarray
    make_targets: Callable[[np.ndarray, int, int, np.random.Generator], np.ndarray]
    target_probabilities: Callable[[np.ndarray, int], np.ndarray]
    make_encoder: Callable[[np.ndarray, RunConfig], ciqp.Encoder]
    condition_protocols: dict[str, np.ndarray] | None = None
    condition_dimension: int = 1
    make_ansatz: Callable[[RunConfig], ciqp.Ansatz] | None = None

    def training_conditions_for(self, config: RunConfig) -> np.ndarray:
        """Resolve the training conditions named by a run config."""
        if config.fit_condition is not None:
            if len(config.fit_condition) != self.condition_dimension:
                raise ValueError("fit_condition has the wrong dimension for this study")
            values = np.asarray([config.fit_condition], dtype=np.float64)
            return values[:, 0] if self.condition_dimension == 1 else values
        if config.condition_protocol == "default":
            return np.asarray(self.training_conditions, dtype=np.float64)
        if self.condition_protocols is None:
            raise ValueError(f"study {self.name!r} has no named protocols")
        try:
            return np.asarray(
                self.condition_protocols[config.condition_protocol],
                dtype=np.float64,
            )
        except KeyError as error:
            choices = ", ".join(sorted(self.condition_protocols))
            raise ValueError(
                f"unknown condition protocol {config.condition_protocol!r}; "
                f"expected one of {choices}",
            ) from error

    def evaluation_conditions_for(self, config: RunConfig) -> np.ndarray:
        """Return the exact-evaluation conditions for a run."""
        if config.model_kind == "independent_direct":
            return self.training_conditions_for(config)
        return np.asarray(self.held_out_conditions, dtype=np.float64)


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


def _model(
    study: Study,
    config: RunConfig,
    conditions: np.ndarray,
) -> ciqp.ConditionalModel:
    encoder = study.make_encoder(conditions, config)
    ansatz = (
        ciqp.Ansatz.complete(config.n_total)
        if study.make_ansatz is None
        else study.make_ansatz(config)
    )
    return ciqp.ConditionalModel(
        ansatz,
        encoder,
        n_visible=config.n_visible if config.n_ancillas else None,
    )


def _paired_complement_errors(
    probabilities: np.ndarray,
    conditions: np.ndarray,
) -> np.ndarray:
    values = np.asarray(conditions, dtype=np.float64).reshape(-1)
    errors: list[float] = []
    for index, condition in enumerate(values):
        if condition > 0.5 + 1e-12:
            continue
        matches = np.flatnonzero(np.isclose(values, 1.0 - condition, atol=1e-12))
        if matches.size:
            errors.append(
                float(complement_tvd(probabilities[index], probabilities[matches[0]])),
            )
    return np.asarray(errors, dtype=np.float64)


def _evaluate(
    study: Study,
    config: RunConfig,
    params: np.ndarray,
    conditions: np.ndarray,
) -> dict[str, np.ndarray]:
    probabilities = _model(study, config, conditions).probabilities(params)
    target_probabilities = study.target_probabilities(conditions, config.n_visible)
    mean_error, std_error = moment_errors(
        probabilities,
        target_probabilities,
        config.n_visible,
    )
    return {
        "probabilities": np.asarray(probabilities),
        "target_probabilities": np.asarray(target_probabilities),
        "tvd": np.asarray(bitstring_tvd(probabilities, target_probabilities)),
        "profile_tvd": np.asarray(
            profile_tvd(probabilities, target_probabilities, config.n_visible),
        ),
        "within_shell_residual": np.asarray(
            within_shell_residual(probabilities, config.n_visible),
        ),
        "mean_error": np.asarray(mean_error),
        "std_error": np.asarray(std_error),
        "mean_weight": np.asarray(mean_weight(probabilities, config.n_visible)),
        "target_mean_weight": np.asarray(
            mean_weight(target_probabilities, config.n_visible),
        ),
        "complement_tvd": _paired_complement_errors(probabilities, conditions),
    }


def _run_id(config: RunConfig, study: Study, seed: int) -> str:
    return (
        f"{study.name}-{config.model_kind}-{config.condition_protocol}"
        f"-{config.budget_mode}"
        f"-n{config.n_visible}a{config.n_ancillas}-{config.encoder}"
        f"-s{seed}-{config.digest()[:10]}"
    )


def _resource_counts(
    config: RunConfig,
    model: ciqp.ConditionalModel,
    n_conditions: int,
    n_evaluation_conditions: int,
    results: list[ciqp.training.TrainingResult],
) -> dict[str, int]:
    training_config = config.training_config()
    n_bandwidths = len(training_config.resolved_sigmas(config.n_visible))
    validation_checks = sum(len(result.validation_steps) for result in results)
    training_reference_draws = len(results) * config.steps * config.mc_samples
    validation_reference_draws = (
        validation_checks * n_conditions * n_bandwidths * config.validation_mc_samples
    )
    return {
        "trainable_parameters_per_circuit": model.ansatz.n_parameters,
        "parameter_instances": model.ansatz.n_parameters,
        "optimized_circuits": 1,
        "optimization_restarts": len(results),
        "optimizer_steps": len(results) * config.steps,
        "training_target_draws": len(results) * config.steps * config.batch_size,
        "training_reference_draws": training_reference_draws,
        "validation_reference_draws": validation_reference_draws,
        "total_reference_draws": training_reference_draws + validation_reference_draws,
        "training_operator_draws": len(results) * config.steps * config.n_ops,
        "test_target_draws": n_evaluation_conditions * config.n_test,
        "test_reference_draws": (
            n_evaluation_conditions * config.validation_mc_samples
        ),
        "test_operator_draws": (
            n_evaluation_conditions * n_bandwidths * config.validation_n_ops
        ),
    }


def _sampled_test_mmd(
    study: Study,
    config: RunConfig,
    params: np.ndarray,
    conditions: np.ndarray,
    targets: np.ndarray,
    seed: int,
) -> float:
    """Evaluate a reproducible sampled MMD diagnostic on held-out targets."""
    training_config = config.training_config()
    model = _model(study, config, conditions)
    result = model.mmd(
        params,
        targets,
        jax.random.PRNGKey(seed + 2_000_003),
        sigmas=training_config.resolved_sigmas(config.n_visible),
        n_ops=config.validation_n_ops,
        n_samples=config.validation_mc_samples,
        shared_samples=True,
        gemm_dtype=training_config.gemm_dtype,
        sample_chunk=config.sample_chunk,
    )
    return float(result.loss)


def run(
    config: RunConfig,
    study: Study,
    seed: int,
    *,
    output_root: Path | None = None,
) -> dict[str, object]:
    """Run one seed, including restart selection, and write immutable artifacts."""
    root = RESULTS if output_root is None else output_root
    run_id = _run_id(config, study, seed)
    directory = root / study.name / run_id
    metrics_path = directory / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        return {**metrics, "directory": str(directory)}
    if directory.exists():
        raise FileExistsError(f"incomplete run directory already exists: {directory}")

    training_conditions = study.training_conditions_for(config)
    evaluation_conditions = study.evaluation_conditions_for(config)
    model = _model(study, config, training_conditions)
    samples_rng = np.random.default_rng(seed)
    training_targets = study.make_targets(
        training_conditions,
        config.n_visible,
        config.n_train,
        samples_rng,
    )
    validation_targets = study.make_targets(
        training_conditions,
        config.n_visible,
        config.n_validation,
        samples_rng,
    )
    test_targets = study.make_targets(
        evaluation_conditions,
        config.n_visible,
        config.n_test,
        np.random.default_rng(seed + 2_000_001),
    )
    exact_training_targets = study.target_probabilities(
        training_conditions,
        config.n_visible,
    )
    selection_model = _model(study, config, training_conditions)

    def exact_training_tvd(params: np.ndarray) -> float:
        probabilities = selection_model.probabilities(params)
        return float(bitstring_tvd(probabilities, exact_training_targets).mean())

    results: list[ciqp.training.TrainingResult] = []
    restart_wall_seconds: list[float] = []
    if config.model_kind == "encoder_only":
        selected_params = np.zeros(model.ansatz.n_parameters, dtype=np.float64)
        selected_restart = None
    else:
        for restart in range(config.restarts):
            initialization_rng = np.random.default_rng(seed + 101 + restart)
            initial_params = initialization_rng.normal(
                loc=0.0,
                scale=config.init_scale,
                size=model.ansatz.n_parameters,
            )
            start = time.perf_counter()
            result = ciqp.train(
                model,
                training_targets,
                validation_targets,
                seed=seed,
                config=config.training_config(),
                initial_params=initial_params,
                selection_metric=(
                    exact_training_tvd
                    if config.selection_mode == "exact_training_tvd"
                    else None
                ),
            )
            restart_wall_seconds.append(time.perf_counter() - start)
            results.append(result)
        selected_restart = int(
            np.argmin([result.best_selection_score for result in results]),
        )
        selected_params = np.asarray(results[selected_restart].best_params)

    training = _evaluate(
        study,
        config,
        selected_params,
        training_conditions,
    )
    evaluation = _evaluate(
        study,
        config,
        selected_params,
        evaluation_conditions,
    )
    held_out_sample_mmd = _sampled_test_mmd(
        study,
        config,
        selected_params,
        evaluation_conditions,
        test_targets,
        seed,
    )
    selected_result = None if selected_restart is None else results[selected_restart]
    resources = _resource_counts(
        config,
        model,
        len(training_conditions),
        len(evaluation_conditions),
        results,
    )
    wall_seconds = float(sum(restart_wall_seconds))

    metrics = {
        "study": study.name,
        "run_id": run_id,
        "seed": seed,
        "model_kind": config.model_kind,
        "budget_mode": config.budget_mode,
        "condition_protocol": config.condition_protocol,
        "fit_condition": config.fit_condition,
        "encoder": config.encoder,
        "n_visible": config.n_visible,
        "n_ancillas": config.n_ancillas,
        "wall_seconds": wall_seconds,
        "selected_restart": selected_restart,
        "best_step": None if selected_result is None else selected_result.best_step,
        "best_validation_loss": (
            None
            if selected_result is None
            else float(selected_result.best_validation_loss)
        ),
        "selection_metric": (
            "mean_exact_training_condition_tvd"
            if config.selection_mode == "exact_training_tvd"
            else "validation_mmd"
        ),
        "best_selection_score": (
            None
            if selected_result is None
            else float(selected_result.best_selection_score)
        ),
        "held_out_sample_mmd": held_out_sample_mmd,
        "train_tvd_mean": float(training["tvd"].mean()),
        "train_tvd_per_condition": training["tvd"].tolist(),
        "held_out_tvd_mean": float(evaluation["tvd"].mean()),
        "held_out_tvd_worst": float(evaluation["tvd"].max()),
        "held_out_tvd_per_condition": evaluation["tvd"].tolist(),
        "train_profile_tvd_mean": float(training["profile_tvd"].mean()),
        "held_out_profile_tvd_mean": float(evaluation["profile_tvd"].mean()),
        "train_within_shell_residual_mean": float(
            training["within_shell_residual"].mean(),
        ),
        "held_out_within_shell_residual_mean": float(
            evaluation["within_shell_residual"].mean(),
        ),
        "train_mean_error_mean": float(np.abs(training["mean_error"]).mean()),
        "held_out_mean_error_mean": float(np.abs(evaluation["mean_error"]).mean()),
        "train_std_error_mean": float(np.abs(training["std_error"]).mean()),
        "held_out_std_error_mean": float(np.abs(evaluation["std_error"]).mean()),
        "train_complement_tvd_mean": (
            float(training["complement_tvd"].mean())
            if training["complement_tvd"].size
            else None
        ),
        "held_out_complement_tvd_mean": (
            float(evaluation["complement_tvd"].mean())
            if evaluation["complement_tvd"].size
            else None
        ),
        "resources": resources,
        "config_digest": config.digest(),
    }

    study_root = root / study.name
    study_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=study_root))
    try:
        _write_json(temporary / "config.json", config.as_dict())
        _write_json(
            temporary / "run.json",
            {
                "run_id": run_id,
                "study": study.name,
                "seed": seed,
                "training_conditions": np.asarray(training_conditions).tolist(),
                "evaluation_conditions": np.asarray(evaluation_conditions).tolist(),
                "restart_wall_seconds": restart_wall_seconds,
                "state": repo_state(),
            },
        )
        _write_json(temporary / "metrics.json", metrics)
        np.savez(
            temporary / "checkpoint.npz",
            params=selected_params,
            training_conditions=training_conditions,
            evaluation_conditions=evaluation_conditions,
            train_probabilities=training["probabilities"],
            held_out_probabilities=evaluation["probabilities"],
            target_probabilities=training["target_probabilities"],
            held_out_target_probabilities=evaluation["target_probabilities"],
        )
        if results:
            np.savez(
                temporary / "restarts.npz",
                restart_seeds=np.asarray(
                    [seed + 101 + index for index in range(len(results))],
                ),
                best_params=np.stack([result.best_params for result in results]),
                final_params=np.stack([result.final_params for result in results]),
                best_steps=np.asarray([result.best_step for result in results]),
                best_validation_losses=np.asarray(
                    [result.best_validation_loss for result in results],
                ),
                best_selection_scores=np.asarray(
                    [result.best_selection_score for result in results],
                ),
                train_steps=np.stack([result.train_steps for result in results]),
                train_losses=np.stack([result.train_losses for result in results]),
                validation_steps=np.stack(
                    [result.validation_steps for result in results],
                ),
                validation_losses=np.stack(
                    [result.validation_losses for result in results],
                ),
                selection_scores=np.stack(
                    [result.selection_scores for result in results],
                ),
                wall_seconds=np.asarray(restart_wall_seconds),
            )
        os.replace(temporary, directory)
    except Exception:
        raise
    return {**metrics, "directory": str(directory)}


def run_all_seeds(
    config: RunConfig,
    study: Study,
    *,
    output_root: Path | None = None,
) -> list[dict[str, object]]:
    """Run every configured seed and return the metric records."""
    return [run(config, study, seed, output_root=output_root) for seed in config.seeds]

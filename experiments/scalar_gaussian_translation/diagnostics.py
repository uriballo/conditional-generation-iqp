"""Restart, threshold, and matched-initialization optimization diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from typing import Any

import iqp_cond as ciqp
import jax
import jax.numpy as jnp
import numpy as np

from experiments.common.config import RunConfig
from experiments.common.metrics import bitstring_tvd

from . import analysis, campaign, protocol
from . import study as study_module

ENCODERS = ("broadcast_affine", "qubit_affine", "anti_block")
DEFAULT_LOSS_LEVELS = (0.2, 0.1, 0.05)
DEFAULT_TVD_LEVELS = (0.3, 0.2, 0.1)


def _curve_crossings(
    steps: np.ndarray,
    values: np.ndarray,
    levels: tuple[float, ...],
    wall_seconds: float,
    total_steps: int,
) -> dict[str, dict[str, float | int | None]]:
    """Return first observed crossing, with linearly estimated elapsed time."""
    x = np.asarray(steps, dtype=int)
    y = np.asarray(values, dtype=float)
    result = {}
    for level in levels:
        found = np.flatnonzero(y <= level)
        if not found.size:
            result[str(level)] = {"updates": None, "estimated_seconds": None}
            continue
        index = int(found[0])
        update = int(x[index])
        result[str(level)] = {
            "updates": update,
            "estimated_seconds": float(wall_seconds * update / total_steps),
        }
    return result


def restart_curve_rows(
    arrays: dict[str, np.ndarray],
    *,
    seed: int,
    encoder: str,
    loss_levels: tuple[float, ...] = DEFAULT_LOSS_LEVELS,
    tvd_levels: tuple[float, ...] = DEFAULT_TVD_LEVELS,
) -> list[dict[str, Any]]:
    """Convert the runner's restarts.npz arrays into JSON-ready restart rows."""
    rows = []
    restart_count = len(arrays["restart_seeds"])
    for index in range(restart_count):
        train_steps = np.asarray(arrays["train_steps"][index], dtype=int)
        train_losses = np.asarray(arrays["train_losses"][index], dtype=float)
        validation_steps = np.asarray(arrays["validation_steps"][index], dtype=int)
        validation_losses = np.asarray(arrays["validation_losses"][index], dtype=float)
        selection_scores = np.asarray(arrays["selection_scores"][index], dtype=float)
        wall = float(arrays["wall_seconds"][index])
        total_steps = int(train_steps[-1]) if train_steps.size else 0
        rows.append(
            {
                "encoder": encoder,
                "optimizer_seed": int(seed),
                "restart": index,
                "restart_seed": int(arrays["restart_seeds"][index]),
                "train_curve": {
                    "steps": train_steps.tolist(),
                    "loss": train_losses.tolist(),
                },
                "validation_curve": {
                    "steps": validation_steps.tolist(),
                    "loss": validation_losses.tolist(),
                },
                "selection_curve": {
                    "steps": validation_steps.tolist(),
                    "score": selection_scores.tolist(),
                    "metric": "mean_exact_training_condition_tvd",
                },
                "best_step": int(arrays["best_steps"][index]),
                "best_selection_score": float(arrays["best_selection_scores"][index]),
                "final_selection_score": float(selection_scores[-1]),
                "wall_seconds": wall,
                "threshold_times": {
                    "training_loss": _curve_crossings(
                        train_steps, train_losses, loss_levels, wall, total_steps
                    ),
                    "validation_loss": _curve_crossings(
                        validation_steps,
                        validation_losses,
                        loss_levels,
                        wall,
                        total_steps,
                    ),
                    "selection_tvd": _curve_crossings(
                        validation_steps,
                        selection_scores,
                        tvd_levels,
                        wall,
                        total_steps,
                    ),
                },
                "elapsed_time_estimate": (
                    "wall_seconds * crossing_updates / total_updates"
                ),
            }
        )
    return rows


def _run_config(path: Path) -> RunConfig:
    values = json.loads(path.read_text())
    values["seeds"] = tuple(values["seeds"])
    if values.get("fit_condition") is not None:
        values["fit_condition"] = tuple(values["fit_condition"])
    values["bodyness_targets"] = tuple(values["bodyness_targets"])
    return RunConfig(**values)


def _exact_restart_metrics(
    entry: dict[str, Any], config: RunConfig, params: np.ndarray
) -> dict[str, Any]:
    job: campaign.CampaignJob = entry["job"]
    study = study_module.build_study(
        job.protocol,
        job.anchor_count,
        job.placement,
        job.layout_seed or protocol.LAYOUT_SEEDS[0],
        job.encoder,
        job.split,
    )
    with np.load(entry["checkpoint"]) as checkpoint:
        train_conditions = np.asarray(checkpoint["training_conditions"])
        held_conditions = np.asarray(checkpoint["evaluation_conditions"])
    train_model = ciqp.ConditionalModel(
        study.make_ansatz(config), study.make_encoder(train_conditions, config)
    )
    held_model = ciqp.ConditionalModel(
        study.make_ansatz(config), study.make_encoder(held_conditions, config)
    )
    train_generated = np.asarray(train_model.probabilities(params))
    held_generated = np.asarray(held_model.probabilities(params))
    train_targets = protocol.target_probabilities(train_conditions)
    held_targets = protocol.target_probabilities(held_conditions)
    train_tvd = np.asarray(bitstring_tvd(train_generated, train_targets))
    held_tvd = np.asarray(bitstring_tvd(held_generated, held_targets))
    return {
        "training_tvd_per_condition": train_tvd.tolist(),
        "training_tvd_mean": float(train_tvd.mean()),
        "held_grid_tvd_per_condition": held_tvd.tolist(),
        "held_grid_tvd_mean": float(held_tvd.mean()),
        "held_grid_tvd_worst": float(held_tvd.max()),
    }


def _gradient_diagnostics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare MMD gradient norms under paired parameters, data, and PRNG keys."""
    by_seed = {int(item["job"].optimizer_seed): item for item in entries}
    reference = next(iter(by_seed.values()))
    ref_job: campaign.CampaignJob = reference["job"]
    ref_config = _run_config(Path(reference["checkpoint"]).parent / "config.json")
    study = study_module.build_study(
        "interpolation", 3, "even", protocol.LAYOUT_SEEDS[0], ENCODERS[0], ref_job.split
    )
    anchors = np.asarray(ref_job.anchors, dtype=float)
    gradient_rows = []
    for seed, entry in sorted(by_seed.items()):
        config = _run_config(Path(entry["checkpoint"]).parent / "config.json")
        targets = study.make_targets(
            anchors, 8, ref_config.n_train, np.random.default_rng(seed)
        )
        restart_file = Path(entry["checkpoint"]).parent / "restarts.npz"
        with np.load(restart_file) as saved:
            restart_seeds = np.asarray(saved["restart_seeds"], dtype=int)
        for restart, init_seed in enumerate(restart_seeds):
            initial = np.random.default_rng(int(init_seed)).normal(
                0.0, config.init_scale, ciqp.Ansatz.complete(8).n_parameters
            )
            key = jax.random.PRNGKey(seed * 1009 + restart + 7331)
            for encoder_name in ENCODERS:
                enc_study = study_module.build_study(
                    "interpolation",
                    3,
                    "even",
                    protocol.LAYOUT_SEEDS[0],
                    encoder_name,
                    ref_job.split,
                )
                model = ciqp.ConditionalModel(
                    enc_study.make_ansatz(config),
                    enc_study.make_encoder(anchors, config),
                )

                objective = partial(
                    _mmd_objective,
                    model=model,
                    targets=jnp.asarray(targets),
                    key=key,
                    config=config,
                )
                loss_value, gradient = jax.value_and_grad(objective)(
                    jnp.asarray(initial)
                )
                norm = float(jnp.linalg.norm(gradient))
                gradient_rows.append(
                    {
                        "encoder": encoder_name,
                        "optimizer_seed": seed,
                        "restart": restart,
                        "restart_seed": int(init_seed),
                        "initialization_sha256": hashlib.sha256(
                            np.asarray(initial).tobytes()
                        ).hexdigest(),
                        "mmd_key": np.asarray(key).tolist(),
                        "mmd_loss": float(loss_value),
                        "gradient_norm": norm,
                    }
                )
    summary = {}
    for encoder_name in ENCODERS:
        values = np.asarray(
            [
                row["gradient_norm"]
                for row in gradient_rows
                if row["encoder"] == encoder_name
            ]
        )
        summary[encoder_name] = {
            "n_matched_initializations": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "protocol": "one paired random initialization per saved restart seed",
        "target_rng": "numpy default_rng(optimizer_seed), shared across encoders",
        "mmd_key_policy": (
            "jax PRNGKey(optimizer_seed*1009 + restart + 7331), shared across encoders"
        ),
        "rows": gradient_rows,
        "summary": summary,
    }


def _mmd_objective(params, *, model, targets, key, config):
    training = config.training_config()
    return model.mmd(
        params,
        targets,
        key,
        sigmas=training.resolved_sigmas(config.n_visible),
        n_ops=config.n_ops,
        n_samples=config.mc_samples,
        shared_samples=True,
        gemm_dtype=training.gemm_dtype,
        sample_chunk=config.sample_chunk,
    ).loss


def _controlled_entries(
    entries: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    selected = [
        entry
        for entry in entries
        if entry["job"].protocol == "interpolation"
        and entry["job"].anchor_count == 3
        and entry["job"].placement == "even"
        and entry["job"].budget_mode == "fixed_total"
        and entry["job"].encoder in ENCODERS
        and entry["job"].split == split
    ]
    expected_seeds = set(protocol.EVAL_SEEDS[split])
    observed = {
        (entry["job"].encoder, entry["job"].optimizer_seed) for entry in selected
    }
    expected = {(encoder, seed) for encoder in ENCODERS for seed in expected_seeds}
    if observed != expected:
        raise ValueError("manifest lacks the full controlled diagnostics subset")
    return sorted(selected, key=lambda e: (e["job"].encoder, e["job"].optimizer_seed))


def build_diagnostics(
    manifest_path: Path,
    *,
    records_dir: Path | None = None,
    success_threshold: float = 0.2,
    loss_levels: tuple[float, ...] = DEFAULT_LOSS_LEVELS,
    tvd_levels: tuple[float, ...] = DEFAULT_TVD_LEVELS,
) -> dict[str, Any]:
    """Read completed validation/audit artifacts and compute diagnostics."""
    manifest, entries = analysis.load_records(manifest_path, records_dir)
    if manifest.get("phase") not in {"validation", "audit"}:
        raise ValueError("diagnostics require a completed validation or audit manifest")
    split = manifest["split"]
    controlled = _controlled_entries(entries, split)
    config_fingerprints = set()
    hyperparameter_ids = set()
    for entry in controlled:
        config_values = json.loads(
            (Path(entry["checkpoint"]).parent / "config.json").read_text()
        )
        config_values["encoder"] = "<encoder>"
        config_values["seeds"] = ["<optimizer_seed>"]
        config_fingerprints.add(
            json.dumps(config_values, sort_keys=True, separators=(",", ":"))
        )
        hyperparameter_ids.add(entry["job"].hyperparameter_id)
    if len(config_fingerprints) != 1 or len(hyperparameter_ids) != 1:
        raise ValueError("controlled subset does not share one training configuration")
    restarts = []
    paired_entries = []
    for entry in controlled:
        job: campaign.CampaignJob = entry["job"]
        run_dir = Path(entry["checkpoint"]).parent
        config = _run_config(run_dir / "config.json")
        with np.load(run_dir / "restarts.npz") as saved:
            arrays = {name: np.asarray(saved[name]) for name in saved.files}
            best_params = np.asarray(saved["best_params"])
            final_params = np.asarray(saved["final_params"])
        metric_file = run_dir / "metrics.json"
        run_metrics = json.loads(metric_file.read_text())
        curves = restart_curve_rows(
            arrays,
            seed=job.optimizer_seed,
            encoder=job.encoder,
            loss_levels=loss_levels,
            tvd_levels=tvd_levels,
        )
        for index, row in enumerate(curves):
            row["best_parameters"] = _exact_restart_metrics(
                entry, config, best_params[index]
            )
            row["final_parameters"] = _exact_restart_metrics(
                entry, config, final_params[index]
            )
            row["run_wall_seconds"] = float(run_metrics["wall_seconds"])
        restarts.extend(curves)
        paired_entries.append(entry)
    by_encoder = {}
    for encoder in ENCODERS:
        selected = [row for row in restarts if row["encoder"] == encoder]
        successes = sum(
            row["best_parameters"]["held_grid_tvd_mean"] <= success_threshold
            for row in selected
        )
        by_encoder[encoder] = {
            "restart_count": len(selected),
            "success_threshold": success_threshold,
            "success_definition": (
                "best-parameter held-grid mean exact TVD <= threshold"
            ),
            "successes": int(successes),
            "success_rate": float(successes / len(selected)),
            "best_selection_score_median": float(
                np.median([row["best_selection_score"] for row in selected])
            ),
            "best_held_grid_tvd_median": float(
                np.median(
                    [row["best_parameters"]["held_grid_tvd_mean"] for row in selected]
                )
            ),
        }
    return {
        "study": "scalar_gaussian_translation",
        "phase": manifest["phase"],
        "split": split,
        "controlled_subset": {
            "protocol": "interpolation",
            "anchor_count": 3,
            "placement": "even",
            "budget_mode": "fixed_total",
            "encoders": list(ENCODERS),
            "optimizer_seeds": list(protocol.EVAL_SEEDS[split]),
        },
        "manifest_protocol_sha256": manifest["protocol_sha256"],
        "success_threshold": success_threshold,
        "encoder_summary": by_encoder,
        "restart_results": restarts,
        "matched_initialization_gradients": _gradient_diagnostics(paired_entries),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--records-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--success-threshold", type=float, default=0.2)
    parser.add_argument(
        "--loss-levels", default=",".join(map(str, DEFAULT_LOSS_LEVELS))
    )
    parser.add_argument("--tvd-levels", default=",".join(map(str, DEFAULT_TVD_LEVELS)))
    args = parser.parse_args(argv)
    payload = build_diagnostics(
        args.manifest,
        records_dir=args.records_dir,
        success_threshold=args.success_threshold,
        loss_levels=tuple(float(x) for x in args.loss_levels.split(",")),
        tvd_levels=tuple(float(x) for x in args.tvd_levels.split(",")),
    )
    analysis._atomic_json(args.output, payload)
    print(f"wrote optimization diagnostics to {args.output}")


if __name__ == "__main__":
    main()

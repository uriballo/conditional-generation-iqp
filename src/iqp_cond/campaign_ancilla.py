"""Campaign-bound provenance checks for the conditional ancilla extension."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .artifacts import CompletionStatus, ResultBundle
from .campaign import CampaignError, CampaignStore, plan_primary
from .protocol import CampaignManifest, ModelFamily, RunIdentity, Stage


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_reused_m0_checkpoint(
    store: CampaignStore,
    manifest: CampaignManifest,
    run: RunIdentity,
) -> ResultBundle:
    """Return the verified corresponding primary m=0 bundle for an ancilla run."""
    if run.stage is not Stage.ANCILLA:
        raise CampaignError("m=0 checkpoint reuse is defined only for ancilla runs")
    decision = store.tuning_decision()
    if decision is None:
        raise CampaignError("ancilla execution requires the frozen tuning decision")
    matches = [
        primary
        for primary in plan_primary(manifest, decision)
        if primary.coupling_seed == run.coupling_seed
        and primary.optimizer_seed == run.optimizer_seed
        and primary.model_family is ModelFamily.FIXED_INPUT
    ]
    if len(matches) != 1:
        raise CampaignError("could not resolve the corresponding primary m=0 run")
    baseline = store.result(matches[0].run_id)
    if (
        baseline is None
        or baseline.completion_status is not CompletionStatus.COMPLETED
        or baseline.checkpoint is None
    ):
        raise CampaignError("corresponding primary m=0 checkpoint is not complete")
    if baseline.run.tuning_candidate_id != run.tuning_candidate_id:
        raise CampaignError("ancilla run and m=0 checkpoint use different tuning")
    checkpoint_path = (store.root / baseline.checkpoint.artifact_path).resolve()
    if not checkpoint_path.is_relative_to(store.root.resolve()):
        raise CampaignError("m=0 checkpoint artifact escapes the campaign root")
    if not checkpoint_path.is_file():
        raise CampaignError("m=0 checkpoint artifact is missing")
    if _sha256_file(checkpoint_path) != baseline.checkpoint.artifact_sha256:
        raise CampaignError("m=0 checkpoint artifact digest mismatch")
    try:
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            required = {
                "best_parameters",
                "final_parameters",
                "generators",
                "best_step",
                "best_validation_loss",
                "train_steps",
                "train_losses",
                "validation_steps",
                "validation_losses",
            }
            if set(archive.files) != required:
                raise CampaignError("m=0 checkpoint artifact has the wrong schema")
            parameters = np.asarray(archive["best_parameters"])
            generators = np.asarray(archive["generators"])
    except (OSError, ValueError) as error:
        raise CampaignError("m=0 checkpoint artifact cannot be loaded") from error
    if parameters.shape != (136,) or generators.shape != (136, 16):
        raise CampaignError("m=0 checkpoint does not use the frozen visible topology")
    exact_rows = baseline.raw_metrics.get("exact_tvd")
    if not isinstance(exact_rows, list) or len(exact_rows) != 9:
        raise CampaignError("m=0 checkpoint bundle lacks all nine exact TVD cells")
    role_counts = {
        role: sum(
            isinstance(row, dict) and row.get("role") == role for row in exact_rows
        )
        for role in ("training", "held_out")
    }
    if role_counts != {"training": 5, "held_out": 4}:
        raise CampaignError("m=0 checkpoint exact TVD roles are incomplete")
    return baseline

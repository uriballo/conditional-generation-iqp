"""Prepare exact targets, calibrated beta grid, and certified finite datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from iqp_cond.campaign import CampaignStore, Gate, GateDecision
from iqp_cond.campaign_finalizers import (
    finalize_sampling_quality,
    finalize_target_calibration,
)
from iqp_cond.ising import (
    BetaGrid,
    CalibrationThresholds,
    SamplerThresholds,
    audit_sampler,
    calibrate_beta_grid,
    exact_ising_target,
    metropolis_chains,
)
from iqp_cond.protocol import CampaignManifest, canonical_json_bytes

SOURCE_PROVENANCE = {
    "dataset": "https://pennylane.ai/datasets/ising",
    "generator_commit": "de96fc2647928c755d15e434c6c8293a0c5b3b60",
    "sampler_commit": "95e5a07e8e9e75ba7e24e67fb32b030112a1309a",
    "canonical_coupling_sha256": (
        "58cd62927e3be95a827e98efb7282af5a62d2232e15953da6362599017989194"
    ),
}

CANDIDATE_INTERVALS = (
    (0.10, 0.45),
    (0.10, 0.50),
    (0.15, 0.50),
    (0.15, 0.55),
    (0.20, 0.50),
    (0.20, 0.55),
    (0.20, 0.60),
    (0.25, 0.55),
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _repository_relative(path: Path) -> str:
    """Return one canonical repository-relative artifact path."""
    try:
        relative = path.resolve().relative_to(REPOSITORY)
    except ValueError as error:
        raise ValueError(
            "campaign preparation output must remain inside the repository",
        ) from error
    return relative.as_posix()


def _atomic_json(path: Path, payload: object) -> None:
    """Create JSON once, accepting only an identical pre-existing payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"existing immutable JSON is invalid: {path}") from error
        if existing != payload:
            raise RuntimeError(f"refusing to rewrite immutable JSON: {path}")
        return
    with NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode())
        stream.write(b"\n")
    try:
        os.link(temporary, path)
    except FileExistsError:
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"existing immutable JSON is invalid: {path}") from error
        if existing != payload:
            raise RuntimeError(f"refusing to rewrite immutable JSON: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> str:
    """Create an NPZ once, accepting only identical typed arrays on rerun."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as existing:
                matches = set(existing.files) == set(arrays) and all(
                    existing[name].dtype == np.asarray(value).dtype
                    and existing[name].shape == np.asarray(value).shape
                    and np.array_equal(existing[name], value)
                    for name, value in arrays.items()
                )
        except (OSError, ValueError) as error:
            raise RuntimeError(f"existing immutable NPZ is invalid: {path}") from error
        if not matches:
            raise RuntimeError(f"refusing to rewrite immutable NPZ: {path}")
        return _sha256_file(path)
    with NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    try:
        os.link(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        return _atomic_npz(path, **arrays)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path)


def _assert_output_version(manifest: CampaignManifest, output: Path) -> None:
    """Refuse to mix artifacts from distinct frozen campaign versions."""
    for name in ("calibration.json", "sampler-audit-summary.json"):
        path = output / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"existing campaign metadata is invalid: {path}",
            ) from error
        if (
            not isinstance(payload, dict)
            or payload.get(
                "campaign_version",
            )
            != manifest.campaign_version
        ):
            raise RuntimeError(
                f"output already belongs to another campaign version: {path}",
            )
        recorded_digest = payload.get("manifest_digest")
        if recorded_digest is not None and recorded_digest != manifest.digest:
            raise RuntimeError(
                f"output already belongs to another manifest revision: {path}",
            )
        if (
            name == "calibration.json"
            and manifest.calibration.artifact_sha256 is not None
            and _sha256_file(path) != manifest.calibration.artifact_sha256
        ):
            raise RuntimeError(
                "existing calibration bytes differ from the frozen manifest",
            )


def _sampler_thresholds(manifest: CampaignManifest) -> SamplerThresholds:
    protocol = manifest.sampler_protocol
    return SamplerThresholds(
        min_ess_per_chain=protocol.min_ess_per_chain,
        max_chain_energy_z=protocol.max_chain_energy_z,
        max_energy_mean_z=protocol.max_energy_mean_z,
        max_magnetization_error=protocol.max_magnetization_error,
        max_correlation_rmse=protocol.max_correlation_rmse,
        max_character_rmse=protocol.max_character_rmse,
    )


def _validate_sampler_configuration(
    manifest: CampaignManifest,
    *,
    n_chains: int,
    burn_in: int,
    thinning: int,
) -> None:
    protocol = manifest.sampler_protocol
    actual = (n_chains, burn_in, thinning)
    expected = (
        protocol.n_chains,
        protocol.burn_in_sweeps,
        protocol.thinning_sweeps,
    )
    if actual != expected:
        raise ValueError(
            "sampler arguments must exactly match the frozen manifest protocol: "
            f"expected {expected}, got {actual}",
        )
    if (
        protocol.training_samples_per_condition % n_chains
        or protocol.validation_samples_per_condition % n_chains
    ):
        raise ValueError(
            "manifest sample counts must be divisible by the frozen chain count",
        )


def _manifest_grid(manifest: CampaignManifest) -> BetaGrid:
    return BetaGrid(
        training=manifest.beta_grid.training,
        heldout=manifest.beta_grid.held_out,
        provenance=manifest.calibration.method,
    )


def prepare_targets(
    manifest: CampaignManifest,
    output: Path,
) -> dict[str, object]:
    """Calibrate from exact Targets and archive all nine target laws."""
    _repository_relative(output)
    _assert_output_version(manifest, output)
    thresholds = CalibrationThresholds()
    calibration = calibrate_beta_grid(
        manifest.coupling_seeds,
        CANDIDATE_INTERVALS,
        thresholds,
    )
    manifest_grid = _manifest_grid(manifest)
    if not np.allclose(
        calibration.grid.training,
        manifest_grid.training,
        atol=1e-14,
        rtol=0.0,
    ):
        raise RuntimeError(
            "manifest beta grid does not equal the target-only calibration result",
        )

    target_records = []
    all_betas = manifest_grid.training + manifest_grid.heldout
    for coupling_seed in manifest.coupling_seeds:
        targets = [exact_ising_target(coupling_seed, beta) for beta in all_betas]
        archive = output / "targets" / f"coupling_{coupling_seed}.npz"
        archive_sha = _atomic_npz(
            archive,
            coupling_seed=np.asarray(coupling_seed, dtype=np.int64),
            betas=np.asarray(all_betas, dtype=np.float64),
            couplings=targets[0].couplings,
            bitstrings=targets[0].bitstrings,
            energies=targets[0].energies,
            probabilities=np.stack([target.probabilities for target in targets]),
        )
        target_records.append(
            {
                "coupling_seed": coupling_seed,
                "archive": _repository_relative(archive),
                "archive_sha256": archive_sha,
                "target_digests": [target.digest for target in targets],
            },
        )

    payload = {
        "schema_version": 1,
        "campaign_version": manifest.campaign_version,
        "source_provenance": SOURCE_PROVENANCE,
        "selection_rule": calibration.rule,
        "thresholds": asdict(thresholds),
        "coupling_seeds": list(manifest.coupling_seeds),
        "selected_interval": list(calibration.selected_interval),
        "training_betas": list(calibration.grid.training),
        "heldout_betas": list(calibration.grid.heldout),
        "candidate_records": list(calibration.candidate_records),
        "target_archives": target_records,
    }
    calibration_path = output / "calibration.json"
    _atomic_json(calibration_path, payload)
    return {
        "calibration_path": _repository_relative(calibration_path),
        "calibration_sha256": _sha256_file(calibration_path),
        "target_archives": target_records,
    }


def prepare_datasets(
    manifest: CampaignManifest,
    output: Path,
    *,
    n_chains: int,
    burn_in: int,
    thinning: int,
) -> dict[str, object]:
    """Generate and certify independent 5,000/2,000 finite-data splits."""
    _repository_relative(output)
    _assert_output_version(manifest, output)
    _validate_sampler_configuration(
        manifest,
        n_chains=n_chains,
        burn_in=burn_in,
        thinning=thinning,
    )
    protocol = manifest.sampler_protocol
    thresholds = _sampler_thresholds(manifest)
    records = []
    all_passed = True
    for coupling_seed in manifest.coupling_seeds:
        base_seed = manifest.data_seed_for(coupling_seed)
        for beta_index, beta in enumerate(manifest.beta_grid.training):
            target = exact_ising_target(coupling_seed, beta)
            training_seed = base_seed * 100 + 2 * beta_index
            validation_seed = training_seed + 1
            training_chains = metropolis_chains(
                target.couplings,
                beta,
                n_chains=n_chains,
                samples_per_chain=(protocol.training_samples_per_condition // n_chains),
                burn_in=burn_in,
                thinning=thinning,
                seed=training_seed,
            )
            validation_chains = metropolis_chains(
                target.couplings,
                beta,
                n_chains=n_chains,
                samples_per_chain=(
                    protocol.validation_samples_per_condition // n_chains
                ),
                burn_in=burn_in,
                thinning=thinning,
                seed=validation_seed,
            )
            training = training_chains.reshape(-1, 16)
            validation = validation_chains.reshape(-1, 16)
            training_audit = audit_sampler(training_chains, target, thresholds)
            validation_audit = audit_sampler(validation_chains, target, thresholds)
            passed = training_audit.passed and validation_audit.passed
            all_passed &= passed

            stem = f"coupling_{coupling_seed}_beta_{beta:.12g}"
            archive = output / "datasets" / f"{stem}.npz"
            archive_sha = _atomic_npz(
                archive,
                training=training,
                validation=validation,
                training_seed=np.asarray(training_seed, dtype=np.int64),
                validation_seed=np.asarray(validation_seed, dtype=np.int64),
                beta=np.asarray(beta, dtype=np.float64),
            )
            record = {
                "coupling_seed": coupling_seed,
                "beta": beta,
                "training_seed": training_seed,
                "validation_seed": validation_seed,
                "n_chains": n_chains,
                "burn_in_sweeps": burn_in,
                "thinning_sweeps": thinning,
                "single_spin_proposals_per_sweep": 16,
                "global_spin_flip_on_retained_samples": True,
                "training_count": len(training),
                "validation_count": len(validation),
                "training_sha256": sha256(training.tobytes()).hexdigest(),
                "validation_sha256": sha256(validation.tobytes()).hexdigest(),
                "archive": _repository_relative(archive),
                "archive_sha256": archive_sha,
                "training_audit": training_audit.to_dict(),
                "validation_audit": validation_audit.to_dict(),
                "passed": passed,
            }
            _atomic_json(output / "sampler-audits" / f"{stem}.json", record)
            records.append(record)

    summary = {
        "schema_version": 1,
        "campaign_version": manifest.campaign_version,
        "manifest_digest": manifest.digest,
        "sampler_protocol": protocol.to_dict(),
        "configuration_digest": sha256(
            canonical_json_bytes(
                {
                    "n_chains": n_chains,
                    "burn_in": burn_in,
                    "thinning": thinning,
                    "symmetry_mix": True,
                },
            ),
        ).hexdigest(),
        "expected_cells": len(manifest.coupling_seeds)
        * len(manifest.beta_grid.training),
        "records": records,
        "passed": all_passed,
    }
    _atomic_json(output / "sampler-audit-summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/quantum_campaign_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/quantum-campaign-v1"),
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path("results/quantum-campaign-v1/campaign"),
    )
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--n-chains", type=int, default=8)
    parser.add_argument("--burn-in", type=int, default=4_000)
    parser.add_argument("--thinning", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = CampaignManifest.from_json(args.manifest)
    store = CampaignStore(args.campaign_root, manifest)
    theorem = store.gate_record(Gate.THEOREM_NOVELTY)
    theorem_passed = theorem is not None and theorem.decision is GateDecision.PASSED
    if not args.targets_only and not theorem_passed:
        verdict = "missing" if theorem is None else theorem.decision.value
        print(
            "refused: dataset preparation and sampling-gate release require a "
            f"passed theorem_novelty gate (current: {verdict})",
            file=sys.stderr,
        )
        return 2

    target_summary = prepare_targets(manifest, args.output)
    print(json.dumps(target_summary, indent=2))
    if theorem_passed:
        canonical_output = (REPOSITORY / manifest.calibration.artifact_path).parent
        if args.output.resolve() != canonical_output.resolve():
            print(
                "refused: gate finalization requires the manifest-bound output "
                f"directory {canonical_output}",
                file=sys.stderr,
            )
            return 2
        if store.gate_record(Gate.TARGET_CALIBRATION) is None:
            finalized = finalize_target_calibration(store, REPOSITORY)
            print(
                json.dumps(
                    {
                        "target_calibration_gate": finalized.record.decision.value,
                        "evidence_digest": finalized.record.evidence_digest,
                    },
                    indent=2,
                ),
            )
    if args.targets_only:
        return 0

    target_gate = store.gate_record(Gate.TARGET_CALIBRATION)
    if target_gate is None or target_gate.decision is not GateDecision.PASSED:
        print(
            "refused: sampling requires a passed target_calibration gate",
            file=sys.stderr,
        )
        return 2
    dataset_summary = prepare_datasets(
        manifest,
        args.output,
        n_chains=args.n_chains,
        burn_in=args.burn_in,
        thinning=args.thinning,
    )
    sampling = store.gate_record(Gate.SAMPLING_QUALITY)
    if sampling is None:
        finalized = finalize_sampling_quality(
            store,
            args.output / "sampler-audit-summary.json",
            REPOSITORY,
        )
        sampling = finalized.record
    print(
        json.dumps(
            {
                "sampler_cells": len(dataset_summary["records"]),
                "sampler_gate_decision": sampling.decision.value,
                "evidence_digest": sampling.evidence_digest,
            },
            indent=2,
        ),
    )
    return 0 if sampling.decision is GateDecision.PASSED else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Frozen n=12 fitting, prediction sealing, and guarded held-out evaluation."""

# Long literals describe the frozen scientific protocol.
# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.artifacts import read_immutable_json, write_immutable_json
from iqp_cond.ashkin_teller_size_baselines import (
    composition_predict,
    expand_orbits,
    fit_baselines,
    orbit_counts,
)
from iqp_cond.bond_reference import BOND_ENCODING
from iqp_cond.conditional_training_loss import sampled_losses
from iqp_cond.encoding import TWO_BLOCK_BRANCH_ENCODING
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many_jax,
    generator_characters,
)
from iqp_cond.exact_tvd_campaign import data_initialize, expected_bodyness_sigmas
from iqp_cond.exact_tvd_runner import _write_immutable_npz
from iqp_cond.exact_tvd_training import CampaignTrainingConfig, train_campaign

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (0.10, 0.22, 0.38, 0.54, 0.70)
TRAIN = np.array(
    [
        (a, b)
        for a in LEVELS
        for b in LEVELS
        if a in (LEVELS[0], LEVELS[-1]) or b in (LEVELS[0], LEVELS[-1])
    ]
)
TEST = np.array([(a, b) for a in LEVELS[1:-1] for b in LEVELS[1:-1]])
SEEDS = tuple(range(3711, 3721))
BITS = enumerate_bitstrings(12)
SPINS = 1 - 2 * BITS
SA = SPINS[:, :6] * np.roll(SPINS[:, :6], -1, axis=1)
SB = SPINS[:, 6:] * np.roll(SPINS[:, 6:], -1, axis=1)
DEFAULT_OUT = ROOT / "artifacts/ashkin-teller-n12-2026-09-07"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _laws(conditions):
    """Analytic targets. Interior calls occur only after the prediction seal."""
    c = np.asarray(conditions)
    logits = (
        c[:, 0, None] * SA.sum(1) + c[:, 1, None] * SB.sum(1) + 0.3 * (SA * SB).sum(1)
    )
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def angles(conditions, encoder):
    values = np.repeat(conditions, 6, axis=1)
    return (
        0.225 * np.pi - 0.20 * np.pi * (values - 0.1) / 0.6
        if encoder == "cat"
        else values
    )


def encoding(encoder):
    return TWO_BLOCK_BRANCH_ENCODING if encoder == "cat" else BOND_ENCODING


def _check_open(out):
    if (out / "evaluation-started.json").exists():
        raise RuntimeError("Held-out evaluation has started; further fitting is closed")


def verify_manifest(out):
    manifest = read_immutable_json(out / "manifest.json")
    for name, expected in manifest["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    if digest(out / "data.npz") != manifest["data_sha256"]:
        raise RuntimeError("Frozen training data changed")
    return manifest


def freeze(out):
    if (out / "manifest.json").exists():
        return verify_manifest(out)
    _check_open(out)
    if (out / "data.npz").exists():
        raise RuntimeError("Partial freeze needs inspection")
    out.mkdir(parents=True, exist_ok=True)
    laws = _laws(TRAIN)  # Do not evaluate TEST here.
    samples = np.empty((10, 2, 16, 1000, 12), np.int8)
    for si, seed in enumerate(SEEDS):
        for role in range(2):
            for c, p in enumerate(laws):
                rng = np.random.default_rng(np.random.SeedSequence([seed, 4, role, c]))
                samples[si, role, c] = BITS[rng.choice(4096, 1000, p=p)]
    full = np.asarray(complete_ansatz(12).generators, np.int8)
    support = (BITS[:, :6].sum(1) % 2 == 0) & (BITS[:, 6:].sum(1) % 2 == 0)
    chars = 1 - 2 * ((BITS[support] @ full.T) % 2)
    kept = []
    mapping = []
    for i in range(len(full)):
        match = next(
            (j for j, k in enumerate(kept) if np.array_equal(chars[:, i], chars[:, k])),
            None,
        )
        if match is None:
            match = len(kept)
            kept.append(i)
        mapping.append(match)
    if len(kept) != 78:
        raise RuntimeError("Unexpected symmetry-sector rank")
    initial = []
    for si, seed in enumerate(SEEDS):
        old = data_initialize(
            samples[si, 0].reshape(-1, 12), full, np.zeros(78, bool), seed
        )
        collapsed = np.bincount(mapping, weights=old, minlength=78)
        np.testing.assert_allclose(chars[:, kept] @ collapsed, chars @ old, atol=1e-12)
        initial.append(collapsed)
    data_hash = _write_immutable_npz(
        out / "data.npz",
        {
            "samples": samples,
            "train_targets": laws,
            "train_conditions": TRAIN,
            "test_coordinates": TEST,
            "generators": full[kept],
            "generators_full": full,
            "mapping": np.array(mapping),
            "initial_params": np.array(initial),
        },
    )
    sources = sorted(
        [
            *ROOT.glob("src/iqp_cond/*.py"),
            *ROOT.glob("tests/test_ashkin_teller*.py"),
            ROOT / "tests/test_conditional_training_loss.py",
            ROOT / "tests/test_ashkin_n12_reference_audit.py",
            ROOT / "scripts/ashkin_teller_n12_audit.py",
            ROOT / "scripts/ashkin_teller_structure_report.py",
            Path(__file__).resolve(),
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
            ROOT / "docs/research/ashkin-teller-n12-protocol-2026-09-07.md",
        ]
    )
    manifest = {
        "campaign": "ashkin_teller_n12_2026-09-07",
        "created_utc": datetime.now(UTC).isoformat(),
        "train_conditions": TRAIN.tolist(),
        "test_coordinates": TEST.tolist(),
        "seeds": list(SEEDS),
        "scope": "compositional interpolation; method developed on related conditions; some perimeter points previously inspected",
        "expected_iqp_fits": 20,
        "expected_classical_bundles": 10,
        "n_visible": 12,
        "n_ancillas": 0,
        "phase_count": 78,
        "kappa": 0.3,
        "flip_masks": [0, 63, 4032, 4095],
        "orbit_count": 1024,
        "smoothing_masses": [6.4, 64.0, 640.0],
        "per_orbit_smoothing": [0.00625, 0.0625, 0.625],
        "log_linear_coefficients": 3069,
        "structural_diagnostics": [
            "TVD(shell_average(q),p)-TVD(q,p)",
            "TVD(q_A*q_B,p)-TVD(q,p)",
            "TVD(shell_average(p),p) descriptive target structure",
        ],
        "encoders": {
            "cat": {
                "encoding": TWO_BLOCK_BRANCH_ENCODING,
                "map": ".225*pi-.20*pi*(c-.1)/.6",
            },
            "bond": {"encoding": BOND_ENCODING, "map": "g=c", "eta": "pi/8"},
        },
        "training": {
            "examples_per_condition": 1000,
            "validation_examples": 1000,
            "steps": 500,
            "batch_size": 100,
            "validation_interval": 25,
            "optimizer": "Adam cosine .01 to .001",
            "n_samples": 2048,
            "n_ops": 512,
            "sigmas": list(expected_bodyness_sigmas(12)),
            "selection": "minimum fixed sampled validation MMD including initial and every25updates",
        },
        "data_rng": "numpy default_rng SeedSequence([seed,4,role,condition_index]); paired across all models",
        "classical": {
            "log_composition": "average log of four corner compositions; 1024 flip-orbit histograms; pseudocount [.00625,.0625,.625]; validation NLL at all perimeter points",
            "log_linear": "1024-orbit reference-category softmax; features [1,(a-.4)/.3,(b-.4)/.3]; 3069 coefficients; mean NLL + lambda/2 sumcoef^2",
            "lambda_grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "solver": "L-BFGS-B zeros per candidate maxiter5000 ftol1e-14 gtol1e-8; require gradinf<=1e-6; same-objective maxiter20000 retry only",
            "selection": "minimum unpenalized validation NLL; ties first grid order",
        },
        "primary_contrast": "bond minus cat mean test TVD, paired over 10 seed means over9conditions",
        "uncertainty": "10000 paired bootstrap resamples of seed indices, percentile95%, RNG seed950701; secondary metrics descriptive/unadjusted",
        "evaluation_guard": "save all20 model and10 baseline prediction archives and seal hashes before first test law; no fitting after evaluation-started marker",
        "no_performance_stopping_gate": True,
        "data_sha256": data_hash,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "versions": {"jax": jax.__version__, "numpy": np.__version__},
    }
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
        for p in sources:
            archive.add(p, arcname=str(p.relative_to(ROOT)))
    manifest["source_archive_sha256"] = digest(out / "source.tar.gz")
    write_immutable_json(out / "manifest.json", manifest)
    return manifest


def _read_complete(out, name):
    path = out / f"{name}.json"
    if not path.exists():
        return None
    r = read_immutable_json(path)
    if r["name"] != name:
        raise RuntimeError(f"Wrong prediction identity: {name}")
    if r["manifest_sha256"] != digest(out / "manifest.json") or r[
        "archive_sha256"
    ] != digest(out / f"{name}.npz"):
        raise RuntimeError(f"Invalid prediction record: {name}")
    return r


def _validate_probabilities(q):
    if not np.isfinite(q).all() or (q < 0).any():
        raise RuntimeError("Invalid predicted probabilities")
    np.testing.assert_allclose(q.sum(-1), 1, atol=1e-11)


def train(out, encoder, seed):
    verify_manifest(out)
    _check_open(out)
    name = f"{encoder}-{seed}"
    if _read_complete(out, name):
        print("RESUMED " + name, flush=True)
        return
    print("START " + name, flush=True)
    start = time.monotonic()
    with np.load(out / "data.npz") as d:
        training, validation = d["samples"][SEEDS.index(seed)]
        initial = d["initial_params"][SEEDS.index(seed)]
        g = d["generators"]
    a = angles(TRAIN, encoder)
    loss, val, bank = sampled_losses(g, a, validation, seed, encoding(encoder), 2048)
    validation_calls = 0

    def progress_val(params):
        nonlocal validation_calls
        value = val(params)
        value.block_until_ready()
        print(
            json.dumps(
                {
                    "name": name,
                    "step": validation_calls * 25,
                    "seconds": time.monotonic() - start,
                }
            ),
            flush=True,
        )
        validation_calls += 1
        return value

    result = train_campaign(
        initial,
        training,
        progress_val,
        loss,
        seed,
        CampaignTrainingConfig(
            steps=500, batch_size=100, n_samples=2048, validation_interval=25
        ),
        include_initial_checkpoint=True,
    )
    selected = min(result.checkpoints, key=lambda c: c.validation_mmd)
    chars = jnp.asarray(generator_characters(g))
    query = np.concatenate([TRAIN, TEST])
    predict = jax.jit(
        lambda p: exact_iqp_probabilities_many_jax(
            p, g, angles(query, encoder), encoding(encoder), chars
        )
    )
    q = np.asarray(predict(selected.params))
    zero = np.asarray(predict(np.zeros(78)))
    _validate_probabilities(q)
    _validate_probabilities(zero)
    arrays = {
        "initial_params": initial,
        "selected_params": selected.params,
        "final_params": result.final_params,
        "checkpoint_params": np.stack([c.params for c in result.checkpoints]),
        "checkpoint_steps": np.array([c.step for c in result.checkpoints]),
        "checkpoint_validation_mmd": np.array(
            [c.validation_mmd for c in result.checkpoints]
        ),
        "train_losses": result.train_losses,
        "validation_steps": result.validation_steps,
        "validation_losses": result.validation_losses,
        "batches": result.batches,
        "train_probabilities": q[:16],
        "test_predictions": q[16:],
        "encoder_only_train": zero[:16],
        "encoder_only_test": zero[16:],
        **bank,
    }
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite {name}/{key}")
    sha = _write_immutable_npz(out / f"{name}.npz", arrays)
    record = {
        "name": name,
        "encoder": encoder,
        "seed": seed,
        "manifest_sha256": digest(out / "manifest.json"),
        "archive_sha256": sha,
        "selected_step": selected.step,
        "validation_mmd": selected.validation_mmd,
        "seconds": time.monotonic() - start,
        "predictions_saved_utc": datetime.now(UTC).isoformat(),
    }
    write_immutable_json(out / f"{name}.json", record)
    print("DONE " + json.dumps(record), flush=True)


def classical(out, seed):
    verify_manifest(out)
    _check_open(out)
    name = f"classical-{seed}"
    if _read_complete(out, name):
        print("RESUMED " + name, flush=True)
        return
    with np.load(out / "data.npz") as d:
        training, validation = d["samples"][SEEDS.index(seed)]
    predictions, metadata = fit_baselines(
        TRAIN, training, validation, np.concatenate([TRAIN, TEST])
    )
    arrays = {}
    for method, q in predictions.items():
        _validate_probabilities(q)
        arrays[method + "_train"] = q[:16]
        arrays[method + "_test"] = q[16:]
    sha = _write_immutable_npz(out / f"{name}.npz", arrays)
    record = {
        "name": name,
        "seed": seed,
        "manifest_sha256": digest(out / "manifest.json"),
        "archive_sha256": sha,
        "models": metadata,
        "predictions_saved_utc": datetime.now(UTC).isoformat(),
    }
    write_immutable_json(out / f"{name}.json", record)
    print("DONE " + name, flush=True)


def seal(out):
    verify_manifest(out)
    with np.load(out / "data.npz") as d:
        g = d["generators"]
        initial = d["initial_params"]
        training = d["samples"][:, 0]
    chars = jnp.asarray(generator_characters(g))
    query = np.concatenate([TRAIN, TEST])
    predictors = {
        e: jax.jit(
            lambda p, e=e: exact_iqp_probabilities_many_jax(
                p, g, angles(query, e), encoding(e), chars
            )
        )
        for e in ("cat", "bond")
    }
    records = {}
    for si, seed in enumerate(SEEDS):
        paired = None
        for model in ("cat", "bond", "classical"):
            name = f"{model}-{seed}"
            record = _read_complete(out, name)
            if record is None:
                raise RuntimeError(f"Missing prediction bundle: {name}")
            with np.load(out / f"{name}.npz") as stored:
                arrays = {k: stored[k] for k in stored.files}
            for value in arrays.values():
                if not np.isfinite(value).all():
                    raise RuntimeError(f"Nonfinite prediction bundle: {name}")
            if model != "classical":
                if arrays["selected_params"].shape != (78,) or arrays[
                    "train_losses"
                ].shape != (500,):
                    raise RuntimeError(f"Incomplete IQP fit: {name}")
                np.testing.assert_array_equal(arrays["initial_params"], initial[si])
                best = np.argmin(arrays["checkpoint_validation_mmd"])
                np.testing.assert_array_equal(
                    arrays["selected_params"], arrays["checkpoint_params"][best]
                )
                if record["selected_step"] != int(arrays["checkpoint_steps"][best]):
                    raise RuntimeError(f"Checkpoint selection mismatch: {name}")
                predicted = np.asarray(predictors[model](arrays["selected_params"]))
                zero = np.asarray(predictors[model](np.zeros(78)))
                for key, expected in (
                    ("train_probabilities", predicted[:16]),
                    ("test_predictions", predicted[16:]),
                    ("encoder_only_train", zero[:16]),
                    ("encoder_only_test", zero[16:]),
                ):
                    _validate_probabilities(arrays[key])
                    np.testing.assert_allclose(arrays[key], expected, atol=1e-12)
                    for flip in (63, 4032):
                        np.testing.assert_allclose(
                            arrays[key],
                            arrays[key][:, np.arange(4096) ^ flip],
                            atol=1e-12,
                        )
                if paired is None:
                    paired = {k: arrays[k] for k in ("masks", "weights", "batches")}
                else:
                    for k in paired:
                        np.testing.assert_array_equal(paired[k], arrays[k])
            else:
                meta = record["models"]
                counts = np.asarray(meta["counts"])
                np.testing.assert_array_equal(counts, orbit_counts(training[si]))
                lam = meta["selected_hyperparams"]["composition_lambda"]
                laws = (counts + lam) / (1000 + 1024 * lam)
                comp = expand_orbits(composition_predict(TRAIN, laws, query))
                features = np.column_stack((np.ones(25), (query - 0.4) / 0.3))
                logits = np.column_stack(
                    (features @ np.asarray(meta["fitted_coef"]), np.zeros(25))
                )
                probs = np.exp(logits - logits.max(1, keepdims=True))
                linear = expand_orbits(probs / probs.sum(1, keepdims=True))
                for method, prediction in (
                    ("log_composition", comp),
                    ("log_linear", linear),
                ):
                    for role, expected in (
                        ("train", prediction[:16]),
                        ("test", prediction[16:]),
                    ):
                        _validate_probabilities(arrays[method + "_" + role])
                        np.testing.assert_allclose(
                            arrays[method + "_" + role], expected, atol=1e-12
                        )
            for suffix in (".json", ".npz"):
                records[name + suffix] = digest(out / (name + suffix))
    payload = {
        "manifest_sha256": digest(out / "manifest.json"),
        "files": records,
        "iqp_fits": 20,
        "classical_bundles": 10,
    }
    path = out / "predictions-seal.json"
    if path.exists():
        if read_immutable_json(path) != payload:
            raise RuntimeError("Prediction seal mismatch")
    else:
        _check_open(out)
        write_immutable_json(path, payload)
    return payload


def metrics(q, p):
    def connected(v):
        return v @ (SA * SB) - (v @ SA) * (v @ SB)

    return {
        "tvd": (0.5 * np.abs(q - p).sum(-1)).tolist(),
        "marginal_a_tvd": (
            0.5
            * np.abs(q.reshape(-1, 64, 64).sum(1) - p.reshape(-1, 64, 64).sum(1)).sum(1)
        ).tolist(),
        "marginal_b_tvd": (
            0.5
            * np.abs(q.reshape(-1, 64, 64).sum(2) - p.reshape(-1, 64, 64).sum(2)).sum(1)
        ).tolist(),
        "correlation_rmse": np.sqrt(
            np.mean((connected(q) - connected(p)) ** 2, axis=-1)
        ).tolist(),
    }


def evaluate(out):
    verify_manifest(out)
    if not (out / "predictions-seal.json").exists():
        raise RuntimeError("Predictions must be sealed before target evaluation")
    seal(out)  # Recheck every saved model/prediction digest before opening targets.
    mark = out / "evaluation-started.json"
    marker = {"seal_sha256": digest(out / "predictions-seal.json")}
    if mark.exists():
        if read_immutable_json(mark) != marker:
            raise RuntimeError("Evaluation marker mismatch")
    else:
        write_immutable_json(mark, marker)
    if (out / "evaluation.json").exists():
        return read_immutable_json(out / "evaluation.json")
    test_targets = _laws(TEST)  # First permitted held-out target access.
    with np.load(out / "data.npz") as d:
        train_targets = d["train_targets"]
    rows = []
    for seed in SEEDS:
        for model in ("cat", "bond"):
            with np.load(out / f"{model}-{seed}.npz") as d:
                for label, prefix in (
                    (model, ""),
                    (model + "_encoder_only", "encoder_only_"),
                ):
                    trainq = (
                        d["train_probabilities"]
                        if not prefix
                        else d["encoder_only_train"]
                    )
                    testq = (
                        d["test_predictions"] if not prefix else d["encoder_only_test"]
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "model": label,
                            "train": metrics(trainq, train_targets),
                            "test": metrics(testq, test_targets),
                        }
                    )
        with np.load(out / f"classical-{seed}.npz") as d:
            for model in ("log_composition", "log_linear"):
                rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "train": metrics(d[model + "_train"], train_targets),
                        "test": metrics(d[model + "_test"], test_targets),
                    }
                )
    sha = _write_immutable_npz(
        out / "evaluation-targets.npz",
        {"test_targets": test_targets, "test_conditions": TEST},
    )
    result = {
        "manifest_sha256": digest(out / "manifest.json"),
        "seal_sha256": digest(out / "predictions-seal.json"),
        "evaluation_targets_sha256": sha,
        "rows": rows,
        "evaluated_utc": datetime.now(UTC).isoformat(),
    }
    write_immutable_json(out / "evaluation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("freeze", "train", "classical", "seal", "evaluate")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--encoder", choices=("cat", "bond"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.output)
    elif args.action == "train":
        for model in (args.encoder,) if args.encoder else ("cat", "bond"):
            for seed in (args.seed,) if args.seed else SEEDS:
                train(args.output, model, seed)
                jax.clear_caches()
    elif args.action == "classical":
        for seed in (args.seed,) if args.seed else SEEDS:
            classical(args.output, seed)
    elif args.action == "seal":
        seal(args.output)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()

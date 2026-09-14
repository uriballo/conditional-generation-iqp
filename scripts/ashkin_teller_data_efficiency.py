#!/usr/bin/env python3
"""Frozen data-size by ancilla-count follow-up on twelve visible AT qubits."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import jax
import numpy as np
from ashkin_teller_data_efficiency_model import (
    exact_predictions,
    generators,
    sampled_losses,
)
from ashkin_teller_n12 import BITS, TEST, TRAIN, _laws, digest, metrics

from iqp_cond.artifacts import read_immutable_json, write_immutable_json
from iqp_cond.ashkin_teller_size_baselines import (
    composition_predict,
    expand_orbits,
    fit_baselines,
    orbit_counts,
)
from iqp_cond.exact_tvd_campaign import data_initialize, expected_bodyness_sigmas
from iqp_cond.exact_tvd_runner import _write_immutable_npz
from iqp_cond.exact_tvd_training import (
    CampaignTrainingConfig,
    shuffled_batches,
    train_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(4711, 4721))
TRAIN_SIZES = (100, 300, 1000)
ANCILLAS = (0, 2, 4)
DEFAULT_OUT = ROOT / "artifacts/ashkin-teller-data-efficiency-2026-09-07"


def arm_initial(training, g, seed):
    return data_initialize(
        training.reshape(-1, 12), g, np.any(g[:, 12:] != 0, axis=1), seed
    )


def draw_data(seed):
    laws = _laws(TRAIN)
    samples = np.empty((2, 16, 1000, 12), np.int8)
    for role in range(2):
        for ci, p in enumerate(laws):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 5, role, ci]))
            samples[role, ci] = BITS[rng.choice(4096, 1000, p=p)]
    return samples


def _check_open(out):
    if (out / "evaluation-started.json").exists():
        raise RuntimeError("Held-out evaluation has started; further fitting is closed")


def verify_manifest(out):
    m = read_immutable_json(out / "manifest.json")
    for name, expected in m["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("Frozen source changed: " + name)
    if digest(out / "data.npz") != m["data_sha256"]:
        raise RuntimeError("Frozen data changed")
    if digest(out / "source.tar.gz") != m["source_archive_sha256"]:
        raise RuntimeError("Frozen source archive changed")
    return m


def freeze(out):
    if (out / "manifest.json").exists():
        return verify_manifest(out)
    _check_open(out)
    if (out / "data.npz").exists():
        raise RuntimeError("Partial freeze needs inspection")
    out.mkdir(parents=True, exist_ok=True)
    samples = np.stack([draw_data(seed) for seed in SEEDS])
    arrays = {
        "samples": samples,
        "train_conditions": TRAIN,
        "test_coordinates": TEST,
        "train_targets": _laws(TRAIN),
    }
    for r in ANCILLAS:
        g = generators(r)
        arrays[f"generators_r{r}"] = g
        arrays[f"initial_r{r}"] = np.stack(
            [
                np.stack(
                    [
                        arm_initial(samples[si, 0, :, :size], g, seed)
                        for size in TRAIN_SIZES
                    ]
                )
                for si, seed in enumerate(SEEDS)
            ]
        )
    sources = sorted(
        set(
            [
                *ROOT.glob("src/iqp_cond/*.py"),
                *ROOT.glob("tests/test_ashkin_teller_data_efficiency*.py"),
                ROOT / "tests/test_exact_tvd_training.py",
                ROOT / "tests/test_ashkin_teller_size_baselines.py",
                ROOT / "scripts/ashkin_teller_n12.py",
                ROOT / "scripts/ashkin_teller_data_efficiency_model.py",
                Path(__file__).resolve(),
                ROOT / "pyproject.toml",
                ROOT / "uv.lock",
                ROOT
                / "docs/research/ashkin-teller-data-efficiency-protocol-2026-09-07.md",
            ]
        )
    )
    data_hash = _write_immutable_npz(out / "data.npz", arrays)
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
        for p in sources:
            archive.add(p, arcname=str(p.relative_to(ROOT)))
    m = {
        "campaign": "ashkin_teller_data_efficiency_2026-09-07",
        "created_utc": datetime.now(UTC).isoformat(),
        "n_visible": 12,
        "ancillas": list(ANCILLAS),
        "train_sizes": list(TRAIN_SIZES),
        "seeds": list(SEEDS),
        "train_conditions": TRAIN.tolist(),
        "test_coordinates": TEST.tolist(),
        "validation_examples_per_condition": 1000,
        "expected_iqp_fits": 180,
        "expected_classical_bundles": 30,
        "expected_prediction_files": 420,
        "phase_counts": {str(r): 78 + 12 * r for r in ANCILLAS},
        "training": {
            "steps": 500,
            "batch_size": 100,
            "validation_interval": 25,
            "n_samples": 2048,
            "n_ops": 512,
            "sigmas": list(expected_bodyness_sigmas(12)),
            "optimizer": "Adam cosine .01 to .001",
            "selection": "first minimum of 21 validation evaluations including initial",
        },
        "reference": "original six+six visible input tensor physical |0> ancillas; post-H uniform ancilla reference",
        "generator_bank": "78 visible one/two-body plus12r visible-ancilla couplings; ancilla-only phases omitted",
        "data_rng": "SeedSequence([seed,5,role,condition_index]); nested prefix sizes; fixed independent1000validation",
        "primary_interaction": "[TVD(r0,s1000)-TVD(r4,s1000)]-[TVD(r0,s100)-TVD(r4,s100)] perencoder",
        "bootstrap": {
            "seed": 950702,
            "repetitions": 10000,
            "unit": "paired seed, allconditions averagedwithinseed",
        },
        "data_sha256": data_hash,
        "source_archive_sha256": digest(out / "source.tar.gz"),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
    }
    write_immutable_json(out / "manifest.json", m)
    return m


def bundle_name(model, size, seed, n_ancillas=0):
    return (
        f"classical-s{size}-{seed}"
        if model == "classical"
        else f"{model}-s{size}-r{n_ancillas}-{seed}"
    )


def _read_complete(out, name):
    path = out / f"{name}.json"
    if not path.exists():
        return None
    m = read_immutable_json(path)
    if (
        m["name"] != name
        or m["manifest_sha256"] != digest(out / "manifest.json")
        or m["archive_sha256"] != digest(out / f"{name}.npz")
    ):
        raise RuntimeError("Prediction identity/hash mismatch: " + name)
    return m


def _validate_probabilities(q):
    if q.ndim != 2 or q.shape[1] != 4096 or not np.isfinite(q).all() or np.any(q < 0):
        raise RuntimeError("Invalid predicted probabilities")
    np.testing.assert_allclose(q.sum(1), 1, atol=1e-11, rtol=0)
    for flip in (63, 4032):
        np.testing.assert_allclose(q, q[:, np.arange(4096) ^ flip], atol=1e-11, rtol=0)


def _load_arm(out, seed, size, r):
    with np.load(out / "data.npz") as d:
        si, ni = SEEDS.index(seed), TRAIN_SIZES.index(size)
        return (
            d["samples"][si, 0, :, :size],
            d["samples"][si, 1],
            d[f"initial_r{r}"][si, ni],
            d[f"generators_r{r}"],
        )


def train(out, encoder, seed, size, n_ancillas):
    verify_manifest(out)
    _check_open(out)
    name = bundle_name(encoder, size, seed, n_ancillas)
    if _read_complete(out, name):
        print("RESUMED " + name, flush=True)
        return
    start = time.monotonic()
    training, validation, initial, g = _load_arm(out, seed, size, n_ancillas)
    loss, val, bank = sampled_losses(g, validation, seed, encoder)
    checkpoint_params = []
    checkpoint_values = []

    def checked_val(params):
        value = float(val(params))
        if not np.isfinite(value) or not np.isfinite(np.asarray(params)).all():
            raise RuntimeError("Nonfinite validation " + name)
        checkpoint_params.append(np.asarray(params).copy())
        checkpoint_values.append(value)
        print(
            f"PROGRESS {name} step={(len(checkpoint_values) - 1) * 25} seconds={time.monotonic() - start:.1f}",
            flush=True,
        )
        return value

    result = train_campaign(
        initial,
        training,
        checked_val,
        loss,
        seed,
        CampaignTrainingConfig(
            steps=500, batch_size=100, n_samples=2048, validation_interval=25
        ),
        include_initial_checkpoint=True,
    )
    selected = int(np.argmin(checkpoint_values))
    selected_params = checkpoint_params[selected]
    query = np.concatenate([TRAIN, TEST])
    q = exact_predictions(selected_params, g, query, encoder)
    zero = exact_predictions(np.zeros(len(g)), g, query, encoder)
    _validate_probabilities(q)
    _validate_probabilities(zero)
    arrays = {
        "initial_params": initial,
        "selected_params": selected_params,
        "final_params": result.final_params,
        "checkpoint_params": np.stack(checkpoint_params),
        "checkpoint_steps": result.validation_steps,
        "checkpoint_validation_mmd": np.asarray(checkpoint_values),
        "validation_steps": result.validation_steps,
        "validation_losses": result.validation_losses,
        "train_losses": result.train_losses,
        "batches": result.batches,
        "train_probabilities": q[:16],
        "test_predictions": q[16:],
        "encoder_only_train": zero[:16],
        "encoder_only_test": zero[16:],
        **bank,
    }
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise RuntimeError("Nonfinite fit " + name)
    sha = _write_immutable_npz(out / f"{name}.npz", arrays)
    write_immutable_json(
        out / f"{name}.json",
        {
            "name": name,
            "encoder": encoder,
            "seed": seed,
            "train_size": size,
            "n_ancillas": n_ancillas,
            "manifest_sha256": digest(out / "manifest.json"),
            "archive_sha256": sha,
            "selected_step": selected * 25,
            "validation_mmd": checkpoint_values[selected],
            "seconds": time.monotonic() - start,
            "predictions_saved_utc": datetime.now(UTC).isoformat(),
        },
    )
    print("DONE " + name, flush=True)


def classical(out, seed, size):
    verify_manifest(out)
    _check_open(out)
    name = bundle_name("classical", size, seed)
    if _read_complete(out, name):
        return
    start = time.monotonic()
    training, validation, _, _ = _load_arm(out, seed, size, 0)
    predictions, meta = fit_baselines(
        TRAIN, training, validation, np.concatenate([TRAIN, TEST])
    )
    arrays = {}
    for model, q in predictions.items():
        _validate_probabilities(q)
        arrays[model + "_train"] = q[:16]
        arrays[model + "_test"] = q[16:]
    sha = _write_immutable_npz(out / f"{name}.npz", arrays)
    write_immutable_json(
        out / f"{name}.json",
        {
            "name": name,
            "seed": seed,
            "train_size": size,
            "manifest_sha256": digest(out / "manifest.json"),
            "archive_sha256": sha,
            "models": meta,
            "seconds": time.monotonic() - start,
            "predictions_saved_utc": datetime.now(UTC).isoformat(),
        },
    )
    print("DONE " + name, flush=True)


def _check_classical(record, arrays, training, validation):
    meta = record["models"]
    counts = orbit_counts(training)
    np.testing.assert_array_equal(meta["counts"], counts)
    query = np.concatenate([TRAIN, TEST])
    lam = meta["selected_hyperparams"]["composition_lambda"]
    comp = expand_orbits(
        composition_predict(
            TRAIN, (counts + lam) / (training.shape[1] + 1024 * lam), query
        )
    )
    features = np.column_stack((np.ones(25), (query - 0.4) / 0.3))
    logits = np.column_stack((features @ np.asarray(meta["fitted_coef"]), np.zeros(25)))
    p = np.exp(logits - logits.max(1, keepdims=True))
    linear = expand_orbits(p / p.sum(1, keepdims=True))
    for model, q in [("log_composition", comp), ("log_linear", linear)]:
        for role, expected in [("train", q[:16]), ("test", q[16:])]:
            _validate_probabilities(arrays[model + "_" + role])
            np.testing.assert_allclose(
                arrays[model + "_" + role], expected, atol=1e-12, rtol=0
            )
    for c in meta["convergence"]:
        if not c["success"] or not np.isfinite(c["grad_inf"]) or c["grad_inf"] > 1e-6:
            raise RuntimeError("Invalid classical convergence")
    for kind, grid, key in [
        ("composition", meta["lambda_grid"], "composition_lambda"),
        ("log_linear", meta["loglinear_lambda_grid"], "loglinear_l2"),
    ]:
        scores = np.asarray(meta["all_validation_NLL"][kind])
        if (
            not np.isfinite(scores).all()
            or meta["selected_hyperparams"][key] != grid[int(np.argmin(scores))]
        ):
            raise RuntimeError("Classical validation selection mismatch")
    vc = orbit_counts(validation)
    # Recompute every composition candidate's held-in validation NLL.
    scores = []
    for candidate in meta["lambda_grid"]:
        q = composition_predict(
            TRAIN, (counts + candidate) / (training.shape[1] + 1024 * candidate), TRAIN
        )
        scores.append(
            float(
                -(vc / vc.sum(1, keepdims=True) * np.log(np.clip(q, 1e-300, None)))
                .sum(1)
                .mean()
            )
        )
    np.testing.assert_allclose(
        scores, meta["all_validation_NLL"]["composition"], atol=1e-12, rtol=0
    )
    orbit_linear = linear[:16].reshape(
        16, 4096
    )  # Recover orbit probabilities through representatives.
    from iqp_cond.ashkin_teller_size_baselines import _orbit_table

    _, reps = _orbit_table(12)
    q = orbit_linear[:, reps] * 4
    nll = float(
        -(vc / vc.sum(1, keepdims=True) * np.log(np.clip(q, 1e-300, None)))
        .sum(1)
        .mean()
    )
    np.testing.assert_allclose(
        nll, min(meta["all_validation_NLL"]["log_linear"]), atol=1e-12, rtol=0
    )


def seal(out):
    verify_manifest(out)
    records = {}
    query = np.concatenate([TRAIN, TEST])
    with np.load(out / "data.npz") as d:
        data = {k: d[k] for k in d.files}
    for si, seed in enumerate(SEEDS):
        shared_bank = None
        controls = {}
        for size in TRAIN_SIZES:
            ni = TRAIN_SIZES.index(size)
            training = data["samples"][si, 0, :, :size]
            validation = data["samples"][si, 1]
            expected_batches = shuffled_batches(
                16, size, CampaignTrainingConfig(steps=500), seed
            )
            for r in ANCILLAS:
                for encoder in ("cat", "bond"):
                    name = bundle_name(encoder, size, seed, r)
                    record = _read_complete(out, name)
                    if record is None:
                        raise RuntimeError("Missing prediction bundle: " + name)
                    if (
                        record["seed"],
                        record["train_size"],
                        record["n_ancillas"],
                        record["encoder"],
                    ) != (seed, size, r, encoder):
                        raise RuntimeError("Wrong arm identity")
                    with np.load(out / f"{name}.npz") as d:
                        a = {k: d[k] for k in d.files}
                    pcount = 78 + 12 * r
                    if a["train_losses"].shape != (500,) or a[
                        "checkpoint_params"
                    ].shape != (21, pcount):
                        raise RuntimeError("Incomplete trajectory: " + name)
                    if any(not np.isfinite(v).all() for v in a.values()):
                        raise RuntimeError("Nonfinite trajectory")
                    np.testing.assert_array_equal(
                        a["checkpoint_steps"], np.arange(0, 501, 25)
                    )
                    np.testing.assert_array_equal(
                        a["validation_steps"], a["checkpoint_steps"]
                    )
                    np.testing.assert_array_equal(
                        a["validation_losses"], a["checkpoint_validation_mmd"]
                    )
                    np.testing.assert_array_equal(
                        a["initial_params"], data[f"initial_r{r}"][si, ni]
                    )
                    np.testing.assert_array_equal(
                        a["checkpoint_params"][0], a["initial_params"]
                    )
                    np.testing.assert_array_equal(
                        a["checkpoint_params"][-1], a["final_params"]
                    )
                    np.testing.assert_array_equal(a["batches"], expected_batches)
                    best = int(np.argmin(a["checkpoint_validation_mmd"]))
                    np.testing.assert_array_equal(
                        a["selected_params"], a["checkpoint_params"][best]
                    )
                    if (
                        record["selected_step"] != best * 25
                        or record["validation_mmd"]
                        != a["checkpoint_validation_mmd"][best]
                    ):
                        raise RuntimeError("Incorrect checkpoint selection")
                    bank = {k: a[k] for k in ("masks", "weights")}
                    if shared_bank is None:
                        shared_bank = bank
                    else:
                        for k in bank:
                            np.testing.assert_array_equal(bank[k], shared_bank[k])
                    g = data[f"generators_r{r}"]
                    q = exact_predictions(a["selected_params"], g, query, encoder)
                    zero = exact_predictions(np.zeros(pcount), g, query, encoder)
                    for key, expected in [
                        ("train_probabilities", q[:16]),
                        ("test_predictions", q[16:]),
                        ("encoder_only_train", zero[:16]),
                        ("encoder_only_test", zero[16:]),
                    ]:
                        _validate_probabilities(a[key])
                        np.testing.assert_allclose(a[key], expected, atol=1e-12, rtol=0)
                    if encoder in controls:
                        np.testing.assert_allclose(
                            zero, controls[encoder], atol=1e-12, rtol=0
                        )
                    else:
                        controls[encoder] = zero
                    for suffix in (".json", ".npz"):
                        records[name + suffix] = digest(out / (name + suffix))
            name = bundle_name("classical", size, seed)
            record = _read_complete(out, name)
            if record is None:
                raise RuntimeError("Missing prediction bundle: " + name)
            if (record["seed"], record["train_size"]) != (seed, size):
                raise RuntimeError("Wrong classical arm identity")
            with np.load(out / f"{name}.npz") as d:
                arrays = {k: d[k] for k in d.files}
            _check_classical(record, arrays, training, validation)
            for suffix in (".json", ".npz"):
                records[name + suffix] = digest(out / (name + suffix))
    payload = {
        "manifest_sha256": digest(out / "manifest.json"),
        "files": records,
        "iqp_fits": 180,
        "classical_bundles": 30,
    }
    path = out / "predictions-seal.json"
    if path.exists():
        if read_immutable_json(path) != payload:
            raise RuntimeError("Prediction seal mismatch")
    else:
        _check_open(out)
        write_immutable_json(path, payload)
    return payload


def evaluate(out):
    verify_manifest(out)
    if not (out / "predictions-seal.json").exists():
        raise RuntimeError("Predictions must be sealed before target evaluation")
    sealed = seal(out)
    marker = {"seal_sha256": digest(out / "predictions-seal.json")}
    path = out / "evaluation-started.json"
    if path.exists():
        if read_immutable_json(path) != marker:
            raise RuntimeError("Evaluation marker mismatch")
    else:
        write_immutable_json(path, marker)
    if (out / "evaluation.json").exists():
        return read_immutable_json(out / "evaluation.json")
    targets = _laws(TEST)
    with np.load(out / "data.npz") as d:
        train_targets = d["train_targets"]
    rows = []
    for seed in SEEDS:
        for size in TRAIN_SIZES:
            for encoder in ("cat", "bond"):
                for r in ANCILLAS:
                    with np.load(
                        out / (bundle_name(encoder, size, seed, r) + ".npz")
                    ) as d:
                        rows.append(
                            {
                                "seed": seed,
                                "train_size": size,
                                "n_ancillas": r,
                                "model": encoder,
                                "train": metrics(
                                    d["train_probabilities"], train_targets
                                ),
                                "test": metrics(d["test_predictions"], targets),
                            }
                        )
                        if r == 0:
                            rows.append(
                                {
                                    "seed": seed,
                                    "train_size": size,
                                    "n_ancillas": 0,
                                    "model": encoder + "_encoder_only",
                                    "train": metrics(
                                        d["encoder_only_train"], train_targets
                                    ),
                                    "test": metrics(d["encoder_only_test"], targets),
                                }
                            )
            with np.load(out / (bundle_name("classical", size, seed) + ".npz")) as d:
                for model in ("log_composition", "log_linear"):
                    rows.append(
                        {
                            "seed": seed,
                            "train_size": size,
                            "n_ancillas": None,
                            "model": model,
                            "train": metrics(d[model + "_train"], train_targets),
                            "test": metrics(d[model + "_test"], targets),
                        }
                    )
    sha = _write_immutable_npz(
        out / "evaluation-targets.npz",
        {"test_targets": targets, "test_conditions": TEST},
    )
    result = {
        "manifest_sha256": sealed["manifest_sha256"],
        "seal_sha256": marker["seal_sha256"],
        "evaluation_targets_sha256": sha,
        "rows": rows,
        "evaluated_utc": datetime.now(UTC).isoformat(),
    }
    write_immutable_json(out / "evaluation.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action", choices=("freeze", "train", "classical", "seal", "evaluate")
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--size", type=int, choices=TRAIN_SIZES)
    p.add_argument("--ancillas", type=int, choices=ANCILLAS)
    p.add_argument("--encoder", choices=("cat", "bond"))
    a = p.parse_args()
    if a.action == "freeze":
        freeze(a.output)
    elif a.action == "seal":
        seal(a.output)
    elif a.action == "evaluate":
        evaluate(a.output)
    else:
        for seed in (a.seed,) if a.seed is not None else SEEDS:
            for size in (a.size,) if a.size is not None else TRAIN_SIZES:
                if a.action == "classical":
                    classical(a.output, seed, size)
                else:
                    for r in (a.ancillas,) if a.ancillas is not None else ANCILLAS:
                        for encoder in (a.encoder,) if a.encoder else ("cat", "bond"):
                            train(a.output, encoder, seed, size, r)
                            jax.clear_caches()


if __name__ == "__main__":
    main()

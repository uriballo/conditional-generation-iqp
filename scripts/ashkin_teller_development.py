#!/usr/bin/env python3
"""Reproducible development-only sampled/exact comparison; no held-out targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.artifacts import read_immutable_json, write_immutable_json
from iqp_cond.bond_reference import BOND_ENCODING
from iqp_cond.encoding import TWO_BLOCK_BRANCH_ENCODING
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many_jax,
    generator_characters,
)
from iqp_cond.exact_tvd_campaign import data_initialize, expected_bodyness_sigmas
from iqp_cond.exact_tvd_runner import _derived_seed, _write_immutable_npz
from iqp_cond.exact_tvd_training import (
    CampaignTrainingConfig,
    sample_nonidentity_masks,
    train_campaign,
)
from iqp_cond.mmd import _target_terms, exact_conditional_mmd, exact_target_fouriers
from iqp_cond.simulator import _estimate_pauli_z_unchecked

ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = np.array(
    [
        (0.1, 0.1),
        (0.3, 0.1),
        (0.5, 0.1),
        (0.7, 0.1),
        (0.1, 0.3),
        (0.1, 0.5),
        (0.1, 0.7),
        (0.5, 0.5),
        (0.5, 0.7),
        (0.7, 0.5),
        (0.7, 0.7),
    ]
)
SEEDS = (1701, 1702, 1703)
BITS = enumerate_bitstrings(8)
SPINS = 1 - 2 * BITS
SA = SPINS[:, :4] * np.roll(SPINS[:, :4], -1, axis=1)
SB = SPINS[:, 4:] * np.roll(SPINS[:, 4:], -1, axis=1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def targets():
    logw = (
        CONDITIONS[:, 0, None] * SA.sum(1)
        + CONDITIONS[:, 1, None] * SB.sum(1)
        + 0.3 * (SA * SB).sum(1)
    )
    p = np.exp(logw - logw.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def bundle(out):
    if (out / "manifest.json").exists():
        manifest = read_immutable_json(out / "manifest.json")
        for path, expected in manifest["source_sha256"].items():
            if digest(ROOT / path) != expected:
                raise RuntimeError(f"Frozen source changed: {path}")
        if digest(out / "data.npz") != manifest["data_sha256"]:
            raise RuntimeError("Frozen data changed")
        return manifest
    if (out / "data.npz").exists():
        raise RuntimeError("Partial freeze exists; inspect before proceeding")
    out.mkdir(parents=True, exist_ok=True)
    laws = targets()
    samples = np.empty((3, 2, 11, 1000, 8), dtype=np.int8)
    for si, seed in enumerate(SEEDS):
        for role in range(2):
            for ci, p in enumerate(laws):
                rng = np.random.default_rng(np.random.SeedSequence([seed, role, ci]))
                samples[si, role, ci] = BITS[rng.choice(256, 1000, p=p)]
    full = np.asarray(complete_ansatz(8).generators, dtype=np.int8)
    support = (BITS[:, :4].sum(1) % 2 == 0) & (BITS[:, 4:].sum(1) % 2 == 0)
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
    assert len(kept) == 30
    initial = []
    for si, seed in enumerate(SEEDS):
        old = data_initialize(
            samples[si, 0].reshape(-1, 8), full, np.zeros(36, bool), seed
        )
        p = np.bincount(mapping, weights=old, minlength=30)
        np.testing.assert_allclose(chars[:, kept] @ p, chars @ old, atol=1e-13)
        initial.append(p)
    data = {
        "targets": laws,
        "samples": samples,
        "generators": full[kept],
        "generators_full": full,
        "mapping": np.array(mapping),
        "initial_params": np.array(initial),
        "conditions": CONDITIONS,
    }
    data_hash = _write_immutable_npz(out / "data.npz", data)
    sources = sorted(
        [*ROOT.glob("src/iqp_cond/*.py"), Path(__file__).resolve(), ROOT / "uv.lock"]
    )
    manifest = {
        "campaign": "ashkin_teller_development_2026-09-07",
        "conditions": CONDITIONS.tolist(),
        "condition_role": "development fitting and validation only",
        "seeds": list(SEEDS),
        "kappa": 0.3,
        "n_train": 1000,
        "n_validation": 1000,
        "data_rng": "numpy default_rng SeedSequence([seed,role,condition_index])",
        "encoders": {
            "cat": {
                "encoding": TWO_BLOCK_BRANCH_ENCODING,
                "map": ".225*pi-.20*pi*(c-.1)/.6",
            },
            "bond": {"encoding": BOND_ENCODING, "map": "g=c", "eta": "pi/8"},
        },
        "phases": 30,
        "steps": 500,
        "batch_size": 100,
        "validation_interval": 25,
        "optimizer": "Adam cosine .01 to .001",
        "n_ops_per_bandwidth_condition": 512,
        "sigmas": list(expected_bodyness_sigmas(8)),
        "reference_samples": 2048,
        "fallback_reference_samples": 4096,
        "checkpoint": (
            "minimum fixed sampled validation MMD among initial and each "
            "25-update checkpoint; both modes same validation"
        ),
        "exact_mode": (
            "full population Fourier MMD; ignores minibatch targets during optimization"
        ),
        "sampled_mode": (
            "unbiased sample-pair MMD; fixed sampled nonidentity masks, fresh "
            "reference samples each update"
        ),
        "gate": (
            "each encoder median paired mean-TVD(sampled-exact)<=.03; no "
            "unexplained estimator failure"
        ),
        "fallback": (
            "only if gap fails: 4096 reference samples for both encoders, same "
            "data/maps/bank/optimizer; reselect exact controls with matching "
            "validation"
        ),
        "data_sha256": data_hash,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "versions": {"jax": jax.__version__, "numpy": np.__version__},
    }
    write_immutable_json(out / "manifest.json", manifest)
    return manifest


def sampled_losses(g, angles, validation, seed, encoding, n_samples):
    """Same mask law/U-statistic as campaign runner, sequential compiled map.

    Padding adds zero-weight masks only. lax.map bounds live transfer tensors
    while avoiding 33 copies of the reference sampler in the compiled graph.
    """
    banks = []
    for band, sigma in enumerate(expected_bodyness_sigmas(8)):
        for c in range(11):
            key = jax.random.PRNGKey(_derived_seed(seed, band, c) & (2**32 - 1))
            banks.append(sample_nonidentity_masks(key, 512, 8, sigma))
    width = max(len(x[0]) for x in banks)
    masks = jnp.asarray(
        np.stack([np.pad(x, ((0, width - len(x)), (0, 0))) for x, _ in banks])
    )
    weights = jnp.asarray(np.stack([np.pad(w, (0, width - len(w))) for _, w in banks]))
    condition = jnp.tile(jnp.arange(11), 3)
    g = jnp.asarray(g)
    angles = jnp.asarray(angles)

    def estimate(p, target, key):
        keys = jax.random.split(key, 33)

        def one(args):
            ops, w, c, k = args
            model = _estimate_pauli_z_unchecked(
                p, g, ops, angles[c][None, :], k, n_samples, encoding
            )
            means, squares = _target_terms(target[c][None, :, :], ops)
            sums = jnp.sum(model.sample_values, axis=1)
            model_square = (sums**2 - jnp.sum(model.sample_values**2, axis=1)) / (
                n_samples * (n_samples - 1)
            )
            return jnp.sum(w * (model_square - 2 * model.values * means + squares)[0])

        return jnp.mean(jax.lax.map(one, (masks, weights, condition, keys)))

    validation = jnp.asarray(validation)
    fixed = jax.random.PRNGKey(_derived_seed(seed, "validation") & (2**32 - 1))
    return (
        estimate,
        jax.jit(lambda p: estimate(p, validation, fixed)),
        {"masks": np.asarray(masks), "weights": np.asarray(weights)},
    )


def metrics(q, p):
    def c(v):
        return v @ (SA * SB) - (v @ SA) * (v @ SB)

    return {
        "tvd": (0.5 * np.abs(q - p).sum(1)).tolist(),
        "marginal_a_tvd": (
            0.5
            * np.abs(q.reshape(-1, 16, 16).sum(1) - p.reshape(-1, 16, 16).sum(1)).sum(1)
        ).tolist(),
        "marginal_b_tvd": (
            0.5
            * np.abs(q.reshape(-1, 16, 16).sum(2) - p.reshape(-1, 16, 16).sum(2)).sum(1)
        ).tolist(),
        "correlation_rmse": np.sqrt(np.mean((c(q) - c(p)) ** 2, axis=1)).tolist(),
        "flip_a_defect": (0.5 * np.abs(q - q[:, np.arange(256) ^ 15]).sum(1)).tolist(),
        "flip_b_defect": (0.5 * np.abs(q - q[:, np.arange(256) ^ 240]).sum(1)).tolist(),
    }


def run_fit(out, encoder, mode, seed, n_samples):
    identity = f"{encoder}-{mode}-{seed}-s{n_samples}"
    record = out / f"{identity}.json"
    manifest_hash = digest(out / "manifest.json")
    if record.exists():
        old = read_immutable_json(record)
        if old["manifest_sha256"] != manifest_hash or old[
            "checkpoint_sha256"
        ] != digest(out / f"{identity}.npz"):
            raise RuntimeError("Resume artifact mismatch")
        print(f"RESUMED {identity}", flush=True)
        return
    print(f"START {identity}", flush=True)
    start = time.monotonic()
    with np.load(out / "data.npz") as d:
        g = d["generators"]
        p = d["targets"]
        train, val = d["samples"][SEEDS.index(seed)]
        initial = d["initial_params"][SEEDS.index(seed)]
    angles = np.repeat(CONDITIONS, 4, axis=1)
    encoding = BOND_ENCODING if encoder == "bond" else TWO_BLOCK_BRANCH_ENCODING
    if encoder == "cat":
        angles = 0.225 * np.pi - 0.20 * np.pi * (angles - 0.1) / 0.6
    chars = jnp.asarray(generator_characters(g))
    fourier = exact_target_fouriers(p)
    sampled, validation, bank = sampled_losses(
        g, angles, val, seed, encoding, n_samples
    )

    def exact(params, batch, key):
        return exact_conditional_mmd(
            params, g, angles, fourier, expected_bodyness_sigmas(8), encoding, chars
        ).loss

    config = CampaignTrainingConfig(
        steps=500, batch_size=100, n_samples=n_samples, validation_interval=25
    )
    result = train_campaign(
        initial,
        train,
        validation,
        sampled if mode == "sampled" else exact,
        seed,
        config,
        include_initial_checkpoint=True,
    )
    selected = min(result.checkpoints, key=lambda x: x.validation_mmd)
    arrays = {
        "initial_params": initial,
        "selected_params": selected.params,
        "final_params": result.final_params,
        "checkpoint_params": np.stack([x.params for x in result.checkpoints]),
        "checkpoint_steps": np.array([x.step for x in result.checkpoints]),
        "checkpoint_validation_mmd": np.array(
            [x.validation_mmd for x in result.checkpoints]
        ),
        "train_losses": result.train_losses,
        "validation_steps": result.validation_steps,
        "validation_losses": result.validation_losses,
        "batches": result.batches,
        **bank,
    }
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite {name} in {identity}")
    evaluate = jax.jit(
        lambda x: exact_iqp_probabilities_many_jax(x, g, angles, encoding, chars)
    )
    q = np.asarray(evaluate(selected.params))
    zero = np.asarray(evaluate(np.zeros(30)))
    arrays.update(probabilities=q, encoder_only_probabilities=zero)
    checkpoint_hash = _write_immutable_npz(out / f"{identity}.npz", arrays)
    data = {
        "identity": identity,
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
        "encoder": encoder,
        "mode": mode,
        "seed": seed,
        "n_samples": n_samples,
        "selected_step": selected.step,
        "selected_validation_mmd": selected.validation_mmd,
        "metrics": metrics(q, p),
        "encoder_only": metrics(zero, p),
        "seconds": time.monotonic() - start,
    }
    write_immutable_json(record, data)
    print(
        "DONE "
        + json.dumps(
            {
                "identity": identity,
                "mean_tvd": np.mean(data["metrics"]["tvd"]),
                "seconds": data["seconds"],
            }
        ),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/ashkin-teller-development-2026-09-07",
    )
    ap.add_argument("--freeze-only", action="store_true")
    ap.add_argument("--encoder", choices=("cat", "bond"))
    ap.add_argument("--mode", choices=("exact", "sampled"))
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--n-samples", type=int, choices=(2048, 4096), default=2048)
    a = ap.parse_args()
    bundle(a.output)
    if a.freeze_only:
        print(a.output / "manifest.json")
        return
    for encoder in (a.encoder,) if a.encoder else ("cat", "bond"):
        for seed in (a.seed,) if a.seed else SEEDS:
            for mode in (a.mode,) if a.mode else ("exact", "sampled"):
                run_fit(a.output, encoder, mode, seed, a.n_samples)
                jax.clear_caches()


if __name__ == "__main__":
    main()

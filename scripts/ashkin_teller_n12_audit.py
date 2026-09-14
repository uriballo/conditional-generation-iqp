"""Training-only n=12 engineering smoke; never construct interior target laws."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.artifacts import write_immutable_json
from iqp_cond.bond_reference import BOND_ENCODING
from iqp_cond.conditional_training_loss import sampled_losses
from iqp_cond.encoding import TWO_BLOCK_BRANCH_ENCODING
from iqp_cond.exact import enumerate_bitstrings
from iqp_cond.exact_tvd_campaign import data_initialize
from iqp_cond.exact_tvd_runner import _write_immutable_npz
from iqp_cond.exact_tvd_training import CampaignTrainingConfig, train_campaign

LEVELS = (0.1, 0.22, 0.38, 0.54, 0.7)
TRAIN = np.array(
    [(a, b) for a in LEVELS for b in LEVELS if a in (0.1, 0.7) or b in (0.1, 0.7)]
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("cat", "bond"), required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ashkin-teller-n12-audit-2026-09-07"),
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{args.encoder}.json").exists():
        raise RuntimeError("Audit already recorded")
    start = time.monotonic()
    bits = enumerate_bitstrings(12)
    s = 1 - 2 * bits
    ea = s[:, :6] * np.roll(s[:, :6], -1, axis=1)
    eb = s[:, 6:] * np.roll(s[:, 6:], -1, axis=1)
    logits = (
        TRAIN[:, 0, None] * ea.sum(1)
        + TRAIN[:, 1, None] * eb.sum(1)
        + 0.3 * (ea * eb).sum(1)
    )
    laws = np.exp(logits - logits.max(1, keepdims=True))
    laws /= laws.sum(1, keepdims=True)
    samples = np.empty((2, 16, 1000, 12), np.int8)
    for role in range(2):
        for c, p in enumerate(laws):
            rng = np.random.default_rng(np.random.SeedSequence([3700, 4, role, c]))
            samples[role, c] = bits[rng.choice(4096, 1000, p=p)]
    g = np.asarray(complete_ansatz(12).generators)
    initial = data_initialize(samples[0].reshape(-1, 12), g, np.zeros(78, bool), 3700)
    angle = np.repeat(TRAIN, 6, axis=1)
    enc = BOND_ENCODING
    if args.encoder == "cat":
        angle = 0.225 * np.pi - 0.20 * np.pi * (angle - 0.1) / 0.6
        enc = TWO_BLOCK_BRANCH_ENCODING
    loss, val, bank = sampled_losses(g, angle, samples[1], 3700, enc, 2048)
    val_times = []

    def timed_val(params):
        t = time.monotonic()
        value = val(params)
        value.block_until_ready()
        val_times.append(time.monotonic() - t)
        print(
            json.dumps(
                {
                    "encoder": args.encoder,
                    "validation_call": len(val_times),
                    "elapsed_s": time.monotonic() - start,
                }
            ),
            flush=True,
        )
        return value

    result = train_campaign(
        initial,
        samples[0],
        timed_val,
        loss,
        3700,
        CampaignTrainingConfig(
            steps=50, batch_size=100, n_samples=2048, validation_interval=25
        ),
        include_initial_checkpoint=True,
    )
    arrays = {
        "initial_params": initial,
        "final_params": result.final_params,
        "train_losses": result.train_losses,
        "validation_losses": result.validation_losses,
        "batches": result.batches,
        **bank,
    }
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError("Nonfinite smoke trajectory")
    sha = _write_immutable_npz(out / f"{args.encoder}.npz", arrays)
    record = {
        "encoder": args.encoder,
        "seed": 3700,
        "steps": 50,
        "n_visible": 12,
        "phase_count": 78,
        "conditions": TRAIN.tolist(),
        "target_access": "training perimeter only",
        "seconds": time.monotonic() - start,
        "validation_seconds": val_times,
        "archive_sha256": sha,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passed": "finite complete trajectory; no performance gate",
    }
    write_immutable_json(out / f"{args.encoder}.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

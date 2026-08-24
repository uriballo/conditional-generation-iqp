#!/usr/bin/env python3
"""Two-block (Z2 x Z2) Curie--Weiss encoding screen.

The target is two independent Curie--Weiss blocks (8 spins each) at the same
inverse temperature beta, so it is invariant under independently flipping each
block (Z2 x Z2) and under permuting within each block.  We compare three
reference states at fixed condition angles:

    angle             broadcast RY product (no symmetry)
    spin_flip_branch  single two-branch, Z2 (total even parity; wrong group)
    two_block_branch  two-block two-branch, Z2 x Z2 (matches the target)

The condition is a scalar beta mapped to a broadcast angle via the frozen
piecewise-linear Curie--Weiss map; only the encoding interpretation differs.
TVD (raw and 2D magnetization profile) is the objective metric.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from iqp_cond.ansatz import complete_ansatz
from iqp_cond.curie_weiss_data import exact_two_block_curie_weiss_target
from iqp_cond.encoding import (
    ANGLE_ENCODING,
    SPIN_FLIP_BRANCH_ENCODING,
    TWO_BLOCK_BRANCH_ENCODING,
)
from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many,
    total_variation_distance,
)
from iqp_cond.training import TrainingConfig, train

M_A = 8
M_B = 8
N = M_A + M_B
TRAINING_BETAS = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
HELD_OUT_BETAS = (0.6, 0.8, 1.0, 1.2, 1.4)
ALL_BETAS = tuple(sorted((*TRAINING_BETAS, *HELD_OUT_BETAS)))

ENCODINGS = {
    "angle": ANGLE_ENCODING,
    "spin_flip_branch": SPIN_FLIP_BRANCH_ENCODING,
    "two_block_branch": TWO_BLOCK_BRANCH_ENCODING,
}

# Frozen broadcast map: alpha(beta) in units of pi, piecewise linear in beta.
PIECES = [
    {"start": 0.5, "end": 0.9, "slope": 0.175, "anchor": -0.14},
    {"start": 0.9, "end": 1.0, "slope": 0.70, "anchor": -0.07},
    {"start": 1.0, "end": 1.2, "slope": 0.40, "anchor": 0.0},
    {"start": 1.2, "end": 1.5, "slope": 0.20, "anchor": 0.08},
]

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results/two-block-screen-v1"
ROWS_DIR = OUT_DIR / "rows"


def alpha(beta: float) -> float:
    for piece in PIECES:
        if piece["start"] <= beta <= piece["end"]:
            return math.pi * (
                piece["anchor"] + piece["slope"] * (beta - piece["start"])
            )
    raise ValueError(f"beta {beta} outside the frozen map")


def condition_angles(betas: tuple[float, ...]) -> np.ndarray:
    return np.asarray([[alpha(beta)] * N for beta in betas])


def sample_bitstrings(beta: float, count: int, seed: int) -> np.ndarray:
    probabilities = exact_two_block_curie_weiss_target(M_A, M_B, beta)
    bits = enumerate_bitstrings(N)
    indices = np.random.default_rng(seed).choice(2**N, count, p=probabilities)
    return bits[indices]


def build_datasets(seed: int) -> tuple[np.ndarray, np.ndarray]:
    training = [
        sample_bitstrings(beta, 10_000, seed + i)
        for i, beta in enumerate(TRAINING_BETAS)
    ]
    validation = [
        sample_bitstrings(beta, 2_000, seed + 100 + i)
        for i, beta in enumerate(TRAINING_BETAS)
    ]
    return np.stack(training), np.stack(validation)


def training_config(updates: int | None, lightweight: bool) -> TrainingConfig:
    return TrainingConfig(
        steps=2000 if updates is None else updates,
        batch_size=75,
        learning_rate=0.01,
        final_learning_rate=0.001,
        validation_interval=min(25, updates or 25),
        sigmas=(1.0, 2.0, 4.0),
        n_ops=256,
        n_iqp_samples=16 if lightweight else 8192,
        validation_n_ops=256,
        validation_n_iqp_samples=16 if lightweight else 16384,
    )


def two_block_profile(probabilities: np.ndarray) -> np.ndarray:
    bits = enumerate_bitstrings(N)
    spins = 2.0 * bits - 1.0
    m_a = spins[:, :M_A].sum(axis=1).astype(np.int64) + M_A
    m_b = spins[:, M_A:].sum(axis=1).astype(np.int64) + M_B
    profile = np.zeros((2 * M_A + 1, 2 * M_B + 1))
    np.add.at(profile, (m_a, m_b), np.asarray(probabilities))
    return profile


def evaluate(params, generators, encoding: str, betas: tuple[float, ...]) -> dict:
    angles = condition_angles(betas)
    model = exact_iqp_probabilities_many(params, generators, angles, encoding)
    raw_tvd = []
    profile_tvd = []
    for index, beta in enumerate(betas):
        target = exact_two_block_curie_weiss_target(M_A, M_B, beta)
        raw_tvd.append(total_variation_distance(target, model[index]))
        profile_tvd.append(
            total_variation_distance(
                two_block_profile(target).ravel(),
                two_block_profile(model[index]).ravel(),
            )
        )
    return {
        "betas": list(betas),
        "raw_tvd": raw_tvd,
        "profile_tvd": profile_tvd,
    }


def _seed_list(spec: str) -> list[int]:
    if "," in spec:
        return [int(s) for s in spec.split(",")]
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def run(args: argparse.Namespace) -> int:
    for encoding in args.encoding:
        for seed in _seed_list(args.seeds):
            row_path = ROWS_DIR / encoding / f"seed{seed}.json"
            if row_path.exists():
                print(f"skip {encoding} seed{seed}", flush=True)
                continue
            print(f"start {encoding} seed{seed}", flush=True)
            started = time.perf_counter()
            training, validation = build_datasets(seed)
            ansatz = complete_ansatz(N)
            result = train(
                f"two_block_{encoding}_{seed}",
                ansatz,
                condition_angles(TRAINING_BETAS),
                training,
                validation,
                seed,
                training_config(args.updates, args.lightweight),
                encoding=ENCODINGS[encoding],
            )
            metrics = evaluate(
                result.best_params,
                np.asarray(ansatz.generators, dtype=np.int8),
                ENCODINGS[encoding],
                ALL_BETAS,
            )
            metrics.update(
                {
                    "encoding": encoding,
                    "seed": seed,
                    "best_validation_loss": result.best_validation_loss,
                    "best_step": result.best_step,
                    "runtime_seconds": time.perf_counter() - started,
                }
            )
            row_path.parent.mkdir(parents=True, exist_ok=True)
            row_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
            held_out = [
                v
                for b, v in zip(metrics["betas"], metrics["raw_tvd"], strict=True)
                if b in HELD_OUT_BETAS
            ]
            print(
                f"done  {encoding} seed{seed} held-out raw TVD={np.mean(held_out):.4f}",
                flush=True,
            )
    return 0


def report(args: argparse.Namespace) -> int:
    print(
        "\n# Two-block Curie--Weiss encoding screen (held-out TVD, lower is better)\n"
    )
    print("| encoding | held-out raw TVD | held-out 2D-profile TVD |")
    print("|---|---|---|")
    for encoding in ENCODINGS:
        rows = []
        for path in (ROWS_DIR / encoding).glob("seed*.json"):
            rows.append(json.loads(path.read_text()))
        if not rows:
            continue
        raw = []
        profile = []
        for row in rows:
            mask = [b in HELD_OUT_BETAS for b in row["betas"]]
            raw.append(np.mean(np.asarray(row["raw_tvd"])[mask]))
            profile.append(np.mean(np.asarray(row["profile_tvd"])[mask]))
        raw_se = np.std(raw, ddof=1) / np.sqrt(len(raw))
        profile_se = np.std(profile, ddof=1) / np.sqrt(len(profile))
        print(
            f"| {encoding} | {np.mean(raw):.4f} ± {raw_se:.4f} "
            f"| {np.mean(profile):.4f} ± {profile_se:.4f} |"
        )
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument(
        "--encoding", nargs="+", choices=sorted(ENCODINGS), required=True
    )
    run_p.add_argument("--seeds", default="0-2")
    run_p.add_argument("--updates", type=int, default=None)
    run_p.add_argument("--lightweight", action="store_true")
    run_p.set_defaults(func=run)
    rep_p = sub.add_parser("report")
    rep_p.set_defaults(func=report)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""PROTOTYPE: non-Curie control for the ancilla-purified input encoder.

The control reuses the paper's target-independent Gaussian location path at
eight visible qubits.  It withholds the same normalized middle interval as the
Curie--Weiss critical-gap protocol.  No target-dependent feature is added to
the encoder.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from prototype_ancilla_field_encoding import (
    N_VISIBLE,
    _coordinate,
    _fit,
    _model_builder,
    _parse_integers,
    _profile,
    _summarize,
)

from iqp_cond.exact import enumerate_bitstrings, total_variation_distance
from iqp_cond.targets import GaussianHammingTarget

TRAINING_CONDITIONS = (0.5, 0.7, 0.8, 1.2, 1.3, 1.5)
TEST_CONDITIONS = (0.9, 1.0, 1.1)
WIDTH = math.sqrt(N_VISIBLE) / 2.0


def _target(condition: float) -> GaussianHammingTarget:
    t = float(_coordinate(np.asarray(condition)))
    mean = N_VISIBLE * (0.15 + 0.70 * t)
    return GaussianHammingTarget.create(N_VISIBLE, mean, WIDTH)


def _target_matrix(conditions: tuple[float, ...]) -> jnp.ndarray:
    bitstrings = enumerate_bitstrings(N_VISIBLE)
    return jnp.asarray(
        np.stack(
            [
                _target(value).bitstring_probabilities(bitstrings)
                for value in conditions
            ]
        ),
        dtype=jnp.float64,
    )


def _evaluate(
    encoding: str,
    ancillas: int,
    params: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    _, model_probabilities = _model_builder(encoding, ancillas)
    conditions = (*TRAINING_CONDITIONS, *TEST_CONDITIONS)
    model = np.asarray(
        model_probabilities(
            jnp.asarray(params),
            jnp.asarray(conditions, dtype=jnp.float64),
        )
    )
    bitstrings = enumerate_bitstrings(N_VISIBLE)
    weights = bitstrings.sum(axis=1)
    rows = []
    for index, condition in enumerate(conditions):
        target = _target(condition)
        target_probability = target.bitstring_probabilities(bitstrings)
        probability = model[index]
        rows.append(
            {
                "encoding": encoding,
                "ancillas": ancillas,
                "seed": seed,
                "beta": condition,
                "role": "training" if condition in TRAINING_CONDITIONS else "test",
                "profile_tvd": total_variation_distance(
                    target.shell_probabilities,
                    _profile(probability),
                ),
                "raw_tvd": total_variation_distance(target_probability, probability),
                "absolute_magnetization_error": abs(
                    target.actual_mean / N_VISIBLE
                    - float(weights @ probability) / N_VISIBLE
                ),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    targets = _target_matrix(TRAINING_CONDITIONS)
    rows: list[dict[str, object]] = []
    fits = []
    for encoding in args.encoding:
        ancillas = 0 if encoding == "affine_broadcast" else args.ancillas
        for seed in _parse_integers(args.seeds):
            params, best_loss, best_step = _fit(
                encoding,
                ancillas,
                TRAINING_CONDITIONS,
                targets,
                seed,
                args.updates,
                "iid",
                args.phase_scale,
            )
            fits.append(
                {
                    "encoding": encoding,
                    "ancillas": ancillas,
                    "seed": seed,
                    "parameter_count": int(params.size),
                    "best_training_mmd_squared": best_loss,
                    "best_step": best_step,
                }
            )
            rows.extend(_evaluate(encoding, ancillas, params, seed))
    return {
        "status": "prototype",
        "target_family": "Gaussian Hamming-weight location path",
        "visible_qubits": N_VISIBLE,
        "training_conditions": TRAINING_CONDITIONS,
        "test_conditions": TEST_CONDITIONS,
        "width": WIDTH,
        "mean_path": "E[K|t]/n=0.15+0.70t",
        "encodings": list(args.encoding),
        "ancillas": args.ancillas,
        "fits": fits,
        "rows": rows,
        "summary": _summarize(rows),
        "leakage_audit": {
            "target_specific_encoder_feature": False,
            "test_targets_accessed_before_checkpoint_freeze": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoding",
        nargs="+",
        choices=("affine_broadcast", "latent_input_linear", "visible_capacity"),
        default=("affine_broadcast", "latent_input_linear", "visible_capacity"),
    )
    parser.add_argument("--ancillas", type=int, default=5)
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--updates", type=int, default=5_000)
    parser.add_argument("--phase-scale", type=float, default=0.1 / math.sqrt(36.0))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ancillas < 1 or args.updates < 1:
        parser.error("ancillas and updates must be positive")
    payload = run(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

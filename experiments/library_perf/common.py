"""Shared setup for the `iqp-cond` performance harness.

Run from the repo root as

    PYTHONPATH=packages/iqp-cond/src .venv/bin/python experiments/library_perf/bench.py

with the research tree *not* on `sys.path`, so the library under test is the
standalone package.  Numbers are saved to `results/` as JSON.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import jax
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
PYTHON = REPO / ".venv" / "bin" / "python"


def block(tree) -> None:
    """Block on every array leaf of a pytree."""
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def timeit(fn, *args, warmup: int = 2, repeats: int = 7) -> dict[str, float]:
    """Return median/min/max milliseconds for a fully blocking call."""
    for _ in range(warmup):
        block(fn(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        block(fn(*args))
        samples.append((time.perf_counter() - start) * 1e3)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def save(name: str, payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / name
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[wrote {out}]", file=sys.stderr)


def problem(
    n_visible: int = 12,
    n_conditions: int = 5,
    n_ancillas: int = 0,
    n_targets: int = 2048,
    seed: int = 0,
    family: str = "product",
):
    """Build a complete-ansatz problem: ansatz, encoders, target samples."""
    import iqp_cond as iqc

    n_qubits = n_visible + n_ancillas
    ansatz = iqc.complete_ansatz(n_qubits)
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.4, 1.6, size=(n_conditions, n_qubits))
    if family == "spin_flip":
        angles = np.broadcast_to(angles[:, :1], angles.shape)
    encoders = iqc.encoder(family, angles)
    targets = rng.integers(0, 2, size=(n_conditions, n_targets, n_visible)).astype(
        np.float64,
    )
    return ansatz, encoders, targets

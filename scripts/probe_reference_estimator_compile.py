#!/usr/bin/env python3
"""Sidecar compile/lowering probe for the reference estimator.

This intentionally does not alter the production estimator runner or simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from run_reference_estimator_study import _build_functions, _key, _source_hashes

jax.config.update("jax_enable_x64", True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetition", type=int, default=0)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    case = next(
        (item for item in manifest["cases"] if item["case_id"] == args.case_id), None
    )
    if case is None:
        raise ValueError(f"unknown case-id: {args.case_id}")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    sampled, _ = _build_functions(case, 8192)
    params = jnp.asarray(case["params"], dtype=jnp.float64)
    key = _key(int(manifest["root_seed"]), str(case["case_id"]), 8192, args.repetition)

    t0 = time.perf_counter()
    lowered = sampled.lower(params, key)
    lowering_seconds = time.perf_counter() - t0
    before = lowered.compiler_ir("hlo").as_hlo_text()
    (output / "sampled_before.hlo.txt").write_text(before)

    t0 = time.perf_counter()
    compiled = lowered.compile()
    compilation_seconds = time.perf_counter() - t0
    after = compiled.as_text()
    (output / "sampled_after.hlo.txt").write_text(after)

    t0 = time.perf_counter()
    values, gradient = compiled(params, key)
    jax.block_until_ready((values, gradient))
    first_seconds = time.perf_counter() - t0
    warm_seconds = []
    for _ in range(5):
        t0 = time.perf_counter()
        values, gradient = compiled(params, key)
        jax.block_until_ready((values, gradient))
        warm_seconds.append(time.perf_counter() - t0)

    cost_analysis = _jsonable(compiled.cost_analysis())
    metadata = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "case_id": case["case_id"],
        "sample_budget": 8192,
        "repetition": args.repetition,
        "n_qubits": case["n_qubits"],
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "persistent_compilation_cache_disabled": os.environ.get(
            "JAX_ENABLE_COMPILATION_CACHE", "true"
        ).lower()
        == "false",
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("XLA_FLAGS", "OMP_NUM_THREADS", "JAX_NUM_GENERATED_THREADS")
        },
        "timing_seconds": {
            "lowering": lowering_seconds,
            "compilation": compilation_seconds,
            "first_execution": first_seconds,
            "warm_executions": warm_seconds,
        },
        "hlo_text_lengths": {"before": len(before), "after": len(after)},
        "cost_analysis": cost_analysis,
        "source_hashes": _source_hashes(),
        "artifacts": {
            "before_hlo_sha256": _sha256(output / "sampled_before.hlo.txt"),
            "after_hlo_sha256": _sha256(output / "sampled_after.hlo.txt"),
        },
    }
    temporary = output / ".probe.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "probe.json")
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

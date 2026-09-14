#!/usr/bin/env python3
"""Run a reproducible reference-sampling estimator accuracy study."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.exact import exact_iqp_probabilities_many_jax, generator_characters
from iqp_cond.simulator import _estimate_pauli_z_unchecked

jax.config.update("jax_enable_x64", True)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"


def _case_key(case_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in case_id)


def _key(root_seed: int, case_id: str, budget: int, repetition: int) -> jax.Array:
    digest = hashlib.sha256(
        _canonical([root_seed, case_id, budget, repetition])
    ).digest()
    seed = int.from_bytes(digest[:4], "big")
    return jax.random.PRNGKey(seed)


def _validate_case(case: dict[str, Any]) -> dict[str, Any]:
    required = {
        "case_id",
        "encoder",
        "n_qubits",
        "encoding",
        "angles",
        "generators",
        "params",
        "ops",
        "phase_regime",
        "condition",
    }
    missing = required - set(case)
    if missing:
        raise ValueError(f"case missing fields: {sorted(missing)}")
    n = int(case["n_qubits"])
    generators = np.asarray(case["generators"], dtype=np.float64)
    ops = np.asarray(case["ops"], dtype=np.float64)
    angles = np.asarray(case["angles"], dtype=np.float64)
    params = np.asarray(case["params"], dtype=np.float64)
    if generators.ndim != 2 or generators.shape[1] != n:
        raise ValueError(f"{case['case_id']}: generators shape mismatch")
    if ops.ndim != 2 or ops.shape[1] != n:
        raise ValueError(f"{case['case_id']}: ops shape mismatch")
    if angles.shape != (n,) or params.shape != (generators.shape[0],):
        raise ValueError(f"{case['case_id']}: angles or params shape mismatch")
    if not np.isfinite(
        np.concatenate((generators.ravel(), ops.ravel(), angles, params))
    ).all():
        raise ValueError(f"{case['case_id']}: non-finite inputs")
    if not np.isin(generators, (0.0, 1.0)).all() or not np.isin(ops, (0.0, 1.0)).all():
        raise ValueError(f"{case['case_id']}: generators and ops must be binary")
    return case


def _build_functions(case: dict[str, Any], budget: int):
    generators = jnp.asarray(case["generators"], dtype=jnp.float64)
    ops = jnp.asarray(case["ops"], dtype=jnp.float64)
    angles = jnp.asarray(case["angles"], dtype=jnp.float64)
    encoding = str(case["encoding"])
    chars = jnp.asarray(
        generator_characters(np.asarray(case["generators"])), dtype=jnp.float64
    )
    bits = np.asarray(
        [
            [(state >> q) & 1 for q in range(int(case["n_qubits"]))]
            for state in range(2 ** int(case["n_qubits"]))
        ],
        dtype=np.float64,
    )
    bit_chars = jnp.asarray(
        1.0 - 2.0 * ((bits @ np.asarray(case["ops"], dtype=np.float64).T) % 2.0)
    )

    def sampled_values(params, key):
        return _estimate_pauli_z_unchecked(
            params, generators, ops, angles[None, :], key, budget, encoding
        ).values[0]

    def sampled_with_aux(params, key):
        values = sampled_values(params, key)
        return values, values

    sampled_jacobian = jax.jacrev(sampled_with_aux, argnums=0, has_aux=True)

    sampled = jax.jit(
        lambda params, key: (
            sampled_jacobian(params, key)[1],
            sampled_jacobian(params, key)[0],
        )
    )

    def exact_values(params):
        probabilities = exact_iqp_probabilities_many_jax(
            params, generators, angles[None, :], encoding, chars
        )[0]
        return jnp.real(probabilities) @ bit_chars

    exact = jax.jit(
        lambda params: (exact_values(params), jax.jacrev(exact_values)(params))
    )
    return sampled, exact


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = {path for path in (root / "src" / "iqp_cond").rglob("*.py")}
    paths.add(Path(__file__).resolve())
    return {str(path.relative_to(root)): _sha256_file(path) for path in sorted(paths)}


def run_case(
    case: dict[str, Any],
    budgets: list[int],
    repetitions: int,
    root_seed: int,
    output: Path,
    manifest_sha: str,
    manifest: dict[str, Any],
    pilot: bool,
) -> None:
    case = _validate_case(case)
    for budget in budgets:
        stem = f"{_case_key(str(case['case_id']))}__b{budget}"
        metadata_path, arrays_path = output / f"{stem}.json", output / f"{stem}.npz"
        identity = {
            "manifest_sha256": manifest_sha,
            "case_id": case["case_id"],
            "sample_budget": budget,
            "repetitions": repetitions,
        }
        if metadata_path.exists() or arrays_path.exists():
            if not (metadata_path.exists() and arrays_path.exists()):
                raise RuntimeError(f"incomplete existing output for {stem}")
            existing = json.loads(metadata_path.read_text())
            current_sources = _source_hashes()
            stored_npz_sha = _sha256_file(arrays_path)
            if (
                existing.get("identity") != identity
                or existing.get("source_hashes") != current_sources
                or existing.get("npz_sha256") != stored_npz_sha
            ):
                raise RuntimeError(f"stale or mismatched output for {stem}")
            print(
                json.dumps(
                    {"case_id": case["case_id"], "budget": budget, "status": "resumed"}
                ),
                flush=True,
            )
            continue
        setup_start = time.perf_counter()
        sampled, exact = _build_functions(case, budget)
        setup_seconds = time.perf_counter() - setup_start
        params = jnp.asarray(case["params"], dtype=jnp.float64)
        t0 = time.perf_counter()
        exact_compiled = exact.lower(params).compile()
        exact_compile_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        exact_values, exact_gradient = exact_compiled(params)
        jax.block_until_ready((exact_values, exact_gradient))
        exact_first_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        sampled_compiled = sampled.lower(
            params, _key(root_seed, str(case["case_id"]), budget, -1)
        ).compile()
        sampled_compile_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        warm_values, warm_gradient = sampled_compiled(
            params, _key(root_seed, str(case["case_id"]), budget, -1)
        )
        jax.block_until_ready((warm_values, warm_gradient))
        sampled_first_seconds = time.perf_counter() - t0
        values, gradients, wall = [], [], []
        for repetition in range(repetitions):
            t0 = time.perf_counter()
            value, gradient = sampled_compiled(
                params, _key(root_seed, str(case["case_id"]), budget, repetition)
            )
            jax.block_until_ready((value, gradient))
            wall.append(time.perf_counter() - t0)
            values.append(np.asarray(value))
            gradients.append(np.asarray(gradient))
        arrays = {
            "sample_values": np.asarray(values),
            "sample_gradients": np.asarray(gradients),
            "exact_values": np.asarray(exact_values),
            "exact_gradients": np.asarray(exact_gradient),
        }
        n_ops = len(case["ops"])
        n_params = len(case["params"])
        expected_shapes = {
            "sample_values": (repetitions, n_ops),
            "sample_gradients": (repetitions, n_ops, n_params),
            "exact_values": (n_ops,),
            "exact_gradients": (n_ops, n_params),
        }
        for name, shape in expected_shapes.items():
            if arrays[name].shape != shape:
                raise ValueError(
                    f"unexpected {name} shape {arrays[name].shape}; expected {shape}"
                )
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError(f"non-finite estimator output for {stem}")
        npz_buffer = io.BytesIO()
        np.savez_compressed(npz_buffer, **arrays)
        npz_bytes = npz_buffer.getvalue()
        metadata = {
            "identity": identity,
            "study_id": manifest["study_id"],
            "case": case,
            "pilot": pilot,
            "npz_sha256": _sha256_bytes(npz_bytes),
            "source_hashes": _source_hashes(),
            "environment": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "device": [str(device) for device in jax.devices()],
                "jax_version": jax.__version__,
                "jax_backend": jax.default_backend(),
                "thread_env": {
                    name: os.environ.get(name)
                    for name in (
                        "XLA_FLAGS",
                        "OMP_NUM_THREADS",
                        "JAX_NUM_GENERATED_THREADS",
                    )
                },
            },
            "timing": {
                "setup_seconds": setup_seconds,
                "exact_compile_seconds": exact_compile_seconds,
                "exact_first_call_seconds": exact_first_seconds,
                "sampled_compile_seconds": sampled_compile_seconds,
                "sampled_first_call_seconds": sampled_first_seconds,
                "wall_seconds": wall,
            },
            "rss_highwater_bytes": _rss_bytes(),
        }
        _atomic_write(arrays_path, npz_bytes)
        _atomic_write(metadata_path, _json_bytes(metadata))
        print(
            json.dumps(
                {"case_id": case["case_id"], "budget": budget, "status": "complete"}
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-ids")
    parser.add_argument("--budgets")
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha = _sha256_bytes(manifest_bytes)
    budgets = (
        [int(x) for x in args.budgets.split(",")]
        if args.budgets
        else [int(x) for x in manifest["sample_budgets"]]
    )
    repetitions = (
        args.repetitions
        if args.repetitions is not None
        else int(manifest["repetitions"])
    )
    pilot = args.budgets is not None or args.repetitions is not None
    if any(b < 2 for b in budgets) or repetitions < 1:
        raise ValueError("budgets must be >= 2 and repetitions >= 1")
    selected = set(args.case_ids.split(",")) if args.case_ids else None
    cases = [
        _validate_case(case)
        for case in manifest["cases"]
        if selected is None or case["case_id"] in selected
    ]
    if not cases or (selected and {case["case_id"] for case in cases} != selected):
        raise ValueError("case-ids selected no known cases")
    for case in cases:
        run_case(
            case,
            budgets,
            repetitions,
            int(manifest["root_seed"]),
            args.output,
            manifest_sha,
            manifest,
            pilot,
        )


if __name__ == "__main__":
    main()

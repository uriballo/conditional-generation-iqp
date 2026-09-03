#!/usr/bin/env python3
"""Run exact-memory and worker-throughput preflights for the Strix hosts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iqp_cond.ansatz import complete_ansatz  # noqa: E402
from iqp_cond.artifacts import write_immutable_json  # noqa: E402
from iqp_cond.exact import exact_iqp_visible_probabilities  # noqa: E402
from iqp_cond.exact_tvd_campaign import load_manifest, manifest_digest  # noqa: E402
from iqp_cond.exact_tvd_runner import _source_tree_digest  # noqa: E402


def _peak_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-qubits", type=int, default=16)
    parser.add_argument("--ancillas", type=int, default=10)
    parser.add_argument("--limit-gib", type=float, default=96.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/exact_tvd_ancilla_v1.json",
    )
    args = parser.parse_args()
    config = load_manifest(args.manifest)
    total = args.visible_qubits + args.ancillas
    ansatz = complete_ansatz(total)
    params = np.random.default_rng(0).normal(0.0, 0.01, ansatz.n_parameters)
    angles = np.zeros((1, total), dtype=np.float64)
    started = time.monotonic()
    probabilities = exact_iqp_visible_probabilities(
        params,
        np.asarray(ansatz.generators),
        angles,
        args.visible_qubits,
        args.ancillas,
    )
    payload = {
        "schema_version": 1,
        "preflight": "exact_26_qubit_visible_marginal",
        "configuration_digest": manifest_digest(config),
        "source_tree_sha256": _source_tree_digest(ROOT),
        "hostname": socket.gethostname(),
        "visible_qubits": args.visible_qubits,
        "ancillas": args.ancillas,
        "total_qubits": total,
        "runtime_seconds": time.monotonic() - started,
        "peak_rss_bytes": _peak_bytes(),
        "memory_limit_bytes": int(args.limit_gib * 2**30),
        "normalization_error": abs(float(probabilities.sum()) - 1.0),
        "pid": os.getpid(),
    }
    payload["passed"] = (
        payload["peak_rss_bytes"] < payload["memory_limit_bytes"]
        and payload["normalization_error"] < 1e-10
    )
    if args.output:
        write_immutable_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

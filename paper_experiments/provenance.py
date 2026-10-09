"""Content and environment identities for resumable manuscript campaigns.

A changed source tree requires a new output directory. Historical campaigns
must resume from their frozen source snapshot rather than this working tree.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import jax


def source_identity(protocol_path: Path, protocol_sha256: str):
    """Bind resumed jobs to their code, dependencies and actual numerical backend."""
    import iqp_cond

    root = Path(__file__).resolve().parents[1]
    library = Path(iqp_cond.__file__).parent
    files = [
        *sorted((root / "paper_experiments").glob("*.py")),
        protocol_path,
        root / "run_experiments.py",
    ]
    result = dict(
        protocol_sha256=protocol_sha256,
        paper_sources={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        library_sources={
            str(p.relative_to(library)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(library.rglob("*.py"))
        },
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "optax", "numpy", "optuna")
        },
        numerical_environment={
            key: os.environ.get(key)
            for key in (
                "JAX_ENABLE_X64",
                "JAX_PLATFORMS",
                "XLA_FLAGS",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "HIP_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "JAX_COMPILATION_CACHE_DIR",
            )
        },
        platform=platform.platform(),
        backend=jax.default_backend(),
        devices=[str(x) for x in jax.devices()],
    )
    if result["dependencies"]["optuna"] != "5.0.0":
        raise ValueError("accepted sampler requires locked Optuna 5.0.0")
    lock = root / "uv.lock"
    if lock.exists():
        result["dependency_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    try:
        result["paper_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["library_revision"] = subprocess.check_output(
            ["git", "-C", str(root / "iqp-cond"), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        deployment = root / "deployment.json"
        result["deployment"] = (
            json.loads(deployment.read_text()) if deployment.exists() else None
        )
    return result

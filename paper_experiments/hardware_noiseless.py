"""Exact noiseless magnetization profiles of the deployed IQM Emerald fits.

The state is held as a (2,)*n tensor, with axis q for logical qubit q, so the
30-qubit fits need about 40 GB of memory (complex128 state plus temporaries)
instead of the dense bit table of hardware_report.model_probabilities.

Run: python -m paper_experiments.hardware_noiseless --size 30 --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from .ansatz import rectangular_lattice_ansatz
from .hardware_report import BUNDLE, load_runs
from .scaling_hpo import schedule


def _axis(n, axes, values):
    shape = [1] * n
    for q in axes:
        shape[q] = 2
    return np.asarray(values).reshape(shape)


def model_profile(fit, beta):
    """Exact distribution of the Hamming weight |x| of a lattice fit's output."""
    n = fit["size"]
    t = float(schedule(fit["family"], fit["hyperparameters"], [beta], n)[0])
    c, s = np.cos(t / 2), np.sin(t / 2)
    psi = np.ones((2,) * n, dtype=np.complex128)
    parity = np.zeros((2,) * (n - 1), dtype=np.uint8)
    for q in range(n - 1):
        psi *= _axis(n, [q], [c, -1j * s])
        parity ^= _axis(n - 1, [q], np.array([0, 1], dtype=np.uint8))
    # The parity encoder sets the last qubit to the parity of the others.
    psi[..., 0] *= parity == 0
    psi[..., 1] *= parity == 1
    del parity
    gens = np.asarray(rectangular_lattice_ansatz(n).generators, dtype=np.int64)
    for g, theta in zip(gens, fit["params"], strict=True):
        support = list(np.flatnonzero(g))
        signs = np.array([1.0])
        for _ in support:
            signs = np.multiply.outer(signs, [1.0, -1.0]).reshape(-1)
        psi *= _axis(n, support, np.exp(1j * theta * signs))
    for q in range(n):
        lo = psi[(slice(None),) * q + (0,)]
        hi = psi[(slice(None),) * q + (1,)]
        diff = lo - hi
        lo += hi
        hi[...] = diff
        del diff
    weight = np.zeros((2,) * n, dtype=np.uint8)
    for q in range(n):
        weight += _axis(n, [q], np.array([0, 1], dtype=np.uint8))
    prob = np.abs(psi) ** 2
    del psi
    prob /= 1 << n
    return np.bincount(weight.reshape(-1), prob.reshape(-1), n + 1)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _revision():
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for group in load_runs(args.bundle):
        meta = group["meta"]
        n, seed = meta["size"], meta["seed"]
        if n != args.size:
            continue
        path = args.bundle / "models" / f"n{n}-s{seed}.json"
        fit = json.loads(path.read_text())
        profiles = {}
        for beta in sorted(group["counts"]):
            start = time.time()
            prof = model_profile(fit, beta)
            profiles[str(beta)] = dict(
                profile=prof.tolist(),
                norm_error=float(abs(prof.sum() - 1)),
                seconds=time.time() - start,
            )
            seconds = profiles[str(beta)]["seconds"]
            print(f"n={n} seed={seed} beta={beta}: {seconds:.0f} s", flush=True)
        record = dict(
            size=n,
            seed=seed,
            fit=str(path.relative_to(args.bundle)),
            fit_sha256=_sha256(path),
            script_sha256=_sha256(__file__),
            revision=_revision(),
            host=platform.node(),
            numpy=np.__version__,
            dtype="complex128",
            profiles=profiles,
        )
        out = args.output / f"n{n}-s{seed}-noiseless-profiles.json"
        out.write_text(json.dumps(record, indent=1) + "\n")
        print(out)


if __name__ == "__main__":
    main()

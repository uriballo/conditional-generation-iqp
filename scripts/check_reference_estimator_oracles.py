#!/usr/bin/env python3
"""Check estimator study exact references against physical input construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from iqp_cond.exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many_jax,
    fwht,
    generator_characters,
)


def physical_input(case):
    n = case["n_qubits"]
    bits = enumerate_bitstrings(n)
    angles = np.asarray(case["angles"])
    if case["encoder"] == "product":
        return np.prod(np.where(bits, np.sin(angles / 2), np.cos(angles / 2)), axis=1)
    m = n // 2
    if case["encoder"] == "cat":
        blocks = []
        for sl in [slice(0, m), slice(m, n)]:
            a = angles[sl]
            b = bits[:, sl]
            product = np.prod(np.where(b, np.sin(a / 2), np.cos(a / 2)), axis=1)
            flipped = np.prod(np.where(1 - b, np.sin(a / 2), np.cos(a / 2)), axis=1)
            blocks.append((product + flipped) / np.sqrt(2 * (1 + np.prod(np.sin(a)))))
        return blocks[0] * blocks[1]
    spins = 1 - 2 * bits
    ba = spins[:, :m] * np.roll(spins[:, :m], -1, axis=1)
    bb = spins[:, m:] * np.roll(spins[:, m:], -1, axis=1)
    a = np.exp(
        0.5 * (angles[0] * ba.sum(1) + angles[m] * bb.sum(1))
        + 1j * np.pi / 8 * (ba.sum(1) + bb.sum(1))
    )
    return a / np.linalg.norm(a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    results = []
    for case in data["cases"]:
        n = case["n_qubits"]
        params = np.asarray(case["params"])
        g = np.asarray(case["generators"])
        chars = generator_characters(g)
        physical = physical_input(case)
        assert np.isclose(np.linalg.norm(physical), 1, atol=1e-12)
        ref = fwht(physical) / np.sqrt(2**n)
        phased = ref * np.exp(1j * (chars @ params))
        out = fwht(phased) / np.sqrt(2**n)
        q = np.abs(out) ** 2
        ops = np.asarray(case["ops"])
        obs = 1 - 2 * ((enumerate_bitstrings(n) @ ops.T) % 2)
        ids = np.unique(np.linspace(0, len(g) - 1, 4, dtype=int))
        basis = np.eye(len(g))[ids]

        def oracle(
            delta, params=params, basis=basis, g=g, case=case, chars=chars, obs=obs
        ):
            return exact_iqp_probabilities_many_jax(
                jnp.asarray(params) + delta @ jnp.asarray(basis),
                g,
                np.array([case["angles"]]),
                case["encoding"],
                chars,
            )[0] @ jnp.asarray(obs)

        moments = oracle(jnp.zeros(len(ids)))
        grad = jax.jacfwd(oracle)(jnp.zeros(len(ids)))
        expected_grad = []
        for j in ids:
            derivative = 1j * fwht(chars[:, j] * phased) / np.sqrt(2**n)
            expected_grad.append((2 * np.real(out.conj() * derivative)) @ obs)
        errors = [
            float(np.max(np.abs(np.asarray(moments) - q @ obs))),
            float(np.max(np.abs(np.asarray(grad) - np.array(expected_grad).T))),
        ]
        if max(errors) > 1e-9:
            raise RuntimeError((case["case_id"], errors))
        results.append(
            {
                "case_id": case["case_id"],
                "moment_max_error": errors[0],
                "gradient_max_error": errors[1],
                "checked_gradient_indices": ids.tolist(),
            }
        )
        print(case["case_id"], errors, flush=True)
        jax.clear_caches()
    payload = {
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "checks": results,
        "method": (
            "Independent physical input, NumPy Hadamards and analytic phase "
            "derivatives versus JAX on 4 selected gradient coordinates."
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()

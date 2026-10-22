"""Classical <m^2> estimates for the selected scaling fits on a temperature grid.

Replays the pair-correlation estimator of the scaling physics probe (uniform
distinct pairs, reference-state Pauli-Z estimates) for the selected checkpoint
of each final scaling fit. The estimator, budget and seed derivation match the
probe, so ``--namespace probe --betas 0.4 0.6 0.8 0.95 1 1.05 1.2 1.4 1.6``
reproduces the bundled nine-temperature pair estimates.

Run: python -m paper_experiments.scaling_observables FITS_ROOT --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

NAMESPACES = {"probe": 1100000000, "dense": 1200000000, "final": 1300000000}
DENSE_BETAS = np.round(np.linspace(0.0, 2.0, 41), 10)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimate(fit, betas, namespace="dense", banks=5, masks=256, draws=4096):
    """Mean distinct-pair <Z_i Z_j> per bank and temperature for one fit."""
    import iqp_cond as ciqp

    from .parity_encoder import ParityEncoder

    fit = Path(fit)
    request = json.loads((fit / "request.json").read_text())
    receipt = json.loads((fit / "receipt.json").read_text())
    for name in ("checkpoint.npz",):
        if _sha(fit / name) != receipt["artifacts_sha256"][name]:
            raise ValueError(f"fit artifact changed: {name}")
    with np.load(fit / "checkpoint.npz") as data:
        generators, params = data["generators"], data["params"]
    n, seed = int(request["size"]), int(request["seed"])
    encoder = ParityEncoder.from_betas(
        np.asarray(betas, dtype=float),
        n,
        scale=float(request["candidate"]["encoder_scale"]),
    )
    model = ciqp.ConditionalModel(ciqp.Ansatz(generators), encoder)
    pair_means, bank_seeds = pair_estimates(
        model, params, n, seed, len(betas), namespace, banks, masks, draws
    )
    return dict(
        job_id=f"n{n}-s{seed}",
        size=n,
        seed=seed,
        betas=[float(b) for b in betas],
        pair_means=pair_means.tolist(),
        m2=(1 / n + (n - 1) / n * pair_means).mean(axis=0).tolist(),
        bank_seeds=bank_seeds,
        budget=dict(banks=banks, masks=masks, draws=draws, namespace=namespace),
        fit_receipt_sha256=_sha(fit / "receipt.json"),
        script_sha256=_sha(__file__),
    )


def pair_estimates(model, params, n, seed, n_betas, namespace, banks, masks, draws):
    """Bank means of uniform distinct-pair <Z_i Z_j> for every condition."""
    import jax
    import jax.numpy as jnp
    from iqp_cond.estimator import estimate_pauli_z_unchecked

    @jax.jit
    def ev(m, s):
        return estimate_pauli_z_unchecked(
            jnp.asarray(params),
            model.generators,
            model.encoder,
            m,
            samples=s,
            gemm_dtype=jnp.float32,
        ).values

    pair_means, bank_seeds = np.empty((banks, n_betas)), []
    for b in range(banks):
        bank_seed = NAMESPACES[namespace] + n * 10000 + seed * 10 + b
        rng = np.random.default_rng(bank_seed)
        pairs = np.stack([np.sort(rng.choice(n, 2, False)) for _ in range(masks)])
        if namespace == "probe":  # consume the quadruple draws of the probe
            for _ in range(masks):
                rng.choice(n, 4, False)
        mask = np.zeros((masks, n))
        mask[np.arange(masks)[:, None], pairs] = 1
        refs = model.encoder.sample(jax.random.PRNGKey(bank_seed + 1000000), draws)
        total = np.zeros((n_betas, masks))
        for k in range(0, draws, 256):
            total += np.asarray(ev(jnp.asarray(mask), refs[:, k : k + 256])) * min(
                256, draws - k
            )
        pair_means[b] = (total / draws).mean(axis=1)
        bank_seeds.append(bank_seed)
    return pair_means, bank_seeds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fits", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--namespace", choices=sorted(NAMESPACES), default="dense")
    parser.add_argument("--betas", type=float, nargs="+")
    parser.add_argument("--jobs", nargs="+")
    args = parser.parse_args(argv)
    betas = np.asarray(args.betas) if args.betas else DENSE_BETAS
    args.output.mkdir(parents=True, exist_ok=True)
    for fit in sorted(args.fits.iterdir()):
        if args.jobs and fit.name not in args.jobs:
            continue
        out = args.output / f"{fit.name}.json"
        if out.exists():
            continue
        result = estimate(fit, betas, args.namespace)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(out, flush=True)


if __name__ == "__main__":
    main()

"""Sixteen-qubit study behind the scaling selection score.

Trains 120 random configurations (60 linear S0, 60 saturating S1) of the
parity-input 4x4 lattice model and records at every checkpoint the dev-grid
score of the scaling search, the validation MMD squared at the training
temperatures, and the exact full-distribution TVD on 40 test temperatures.
``analyze`` ranks every checkpoint of every configuration by each score and
reports the Spearman correlation with the exact TVD, aggregated over the test
grid by the same (2 max + trapezoidal mean) / 3 rule.

The settings and seeds are those of the 2026-10-15 development run that chose
the score for ``scaling_hpo.py``; ``run`` reproduces its configurations.

Run:   python -m paper_experiments.scaling_score_study run OUT [--trials 60]
       python -m paper_experiments.scaling_score_study collect OUT
       python -m paper_experiments.scaling_score_study analyze
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from paper_experiments.scaling_hpo import make_targets, r_gamma

N = 16
TRAIN = np.array([0.6, 0.8, 1.2, 1.4])
DEV = np.round(np.arange(0.3, 1.71, 0.2), 10)
TEST = np.round(0.025 + 0.05 * np.arange(40), 10)
STEPS, INTERVAL = 2000, 100
EVAL = dict(n_ops=1024, n_samples=2048, banks=2)
DATA_SEED, STREAM_SEED = 20261015, 20261016
FAMILIES = ("S0", "S1")
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "scaling" / "score_study"


def sample_trial(family, index):
    """Scaling search space at the reference size (alpha = gamma = 0)."""
    rng = np.random.default_rng([DATA_SEED, ord(family[-1]), index])

    def logu(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    hp = dict(
        learning_rate=logu(1e-4, 0.05),
        init_scale=logu(0.005, 0.3),
        b1=float(rng.uniform(0.8, 0.95)),
        b2=float(rng.uniform(0.95, 0.999)),
        final_ratio=logu(0.05, 1.0),
        n_ops=int(rng.choice([512, 1024, 2048])),
        mc_samples=int(rng.choice([1024, 2048])),
    )
    if family == "S0":
        hp["lam"] = float(rng.uniform(0.25, 1.5))
    else:
        hp.update(
            tmax=float(rng.uniform(0.5, np.pi)),
            k=logu(1.0, 20.0),
            b0=float(rng.uniform(0.5, 1.5)),
        )
    return hp


def schedule(family, hp, betas):
    b = np.asarray(betas, dtype=float)
    if family == "S0":
        t = hp["lam"] * b
    else:
        s0 = 1 / (1 + np.exp(hp["k"] * hp["b0"]))
        t = hp["tmax"] * (1 / (1 + np.exp(-hp["k"] * (b - hp["b0"]))) - s0) / (1 - s0)
    return np.clip(t, 0, np.pi)


class Context:
    """Targets, exact target distributions and the exact model kernel."""

    def __init__(self):
        import jax
        import jax.numpy as jnp

        from paper_experiments.ansatz import rectangular_lattice_ansatz
        from paper_experiments.target import curie_weiss_target

        self.ansatz = rectangular_lattice_ansatz(N)
        rng = np.random.default_rng(DATA_SEED)
        self.training = make_targets(TRAIN, N, 10000, rng)
        self.validation = make_targets(TRAIN, N, 2000, rng)
        self.dev_targets = make_targets(DEV, N, 2000, rng)
        bits = ((np.arange(1 << N)[:, None] >> np.arange(N)) & 1).astype(np.int8)
        self.p_test = np.stack(
            [curie_weiss_target(N, b).bitstring_probabilities(bits) for b in TEST]
        )
        gens = np.asarray(self.ansatz.generators, dtype=np.int64)
        chars = jnp.asarray(1 - 2 * ((bits.astype(np.int64) @ gens.T) % 2), jnp.float64)

        @jax.jit
        def kernel(theta, reference):
            # Diagonal phases on the reference amplitudes, then H^{(x)n} (fast WHT).
            state = reference * jnp.exp(1j * (chars @ theta))[None, :]
            batch = reference.shape[0]
            for bit in range(N):
                blocks = state.reshape(batch, -1, 2, 1 << bit)
                left, right = blocks[:, :, 0, :], blocks[:, :, 1, :]
                state = jnp.stack((left + right, left - right), axis=2)
                state = state.reshape(batch, -1)
            return jnp.abs(state) ** 2 / (1 << N)

        self.kernel = kernel


def run_trial(ctx, family, index):
    import iqp_cond as ciqp
    import jax
    import jax.numpy as jnp

    from paper_experiments.config import RunConfig
    from paper_experiments.parity_encoder import ParityEncoder
    from paper_experiments.training import make_trainer

    hp = sample_trial(family, index)
    config = RunConfig(
        study="scaling_score_study",
        encoder="parity",
        n_visible=N,
        n_train=10000,
        n_validation=2000,
        n_test=2,
        steps=STEPS,
        batch_size=450,
        n_ops=hp["n_ops"],
        mc_samples=hp["mc_samples"],
        bodyness_targets=(2.0, 4.0),
        learning_rate=hp["learning_rate"],
        final_learning_rate=hp["learning_rate"] * hp["final_ratio"],
        validation_interval=INTERVAL,
        gemm_dtype="float32",
        sample_chunk=256,
        validation_n_ops=2,
        validation_mc_samples=2,
        init_scale=hp["init_scale"],
        restarts=1,
    )
    tc = config.training_config()

    def encoder(grid):
        return ParityEncoder.from_betas(schedule(family, hp, grid), N, scale=1.0)

    train_model = ciqp.ConditionalModel(ctx.ansatz, encoder(TRAIN))
    dev_model = ciqp.ConditionalModel(ctx.ansatz, encoder(DEV))
    test_reference = encoder(TEST).post_hadamard_amplitudes()
    keys = jnp.stack(
        [jax.random.PRNGKey(STREAM_SEED + 7 + b) for b in range(EVAL["banks"])]
    )

    def mmd(model, targets):
        def one(theta, key):
            return model.mmd(
                theta,
                jnp.asarray(targets),
                key,
                sigmas=tc.sigmas,
                n_ops=EVAL["n_ops"],
                n_samples=EVAL["n_samples"],
                shared_samples=True,
                gemm_dtype=jnp.float32,
                sample_chunk=256,
            ).per_condition

        return jax.jit(jax.vmap(one, in_axes=(None, 0)))

    mmd_train, mmd_dev = (
        mmd(train_model, ctx.validation),
        mmd(dev_model, ctx.dev_targets),
    )
    records, cache = [], {}

    def evaluate(theta):
        q = np.asarray(ctx.kernel(jnp.asarray(theta), test_reference))
        cache.update(
            theta=np.asarray(theta).copy(),
            rec=dict(
                val_mmd_train=np.asarray(mmd_train(jnp.asarray(theta), keys)).mean(0),
                mmd_dev=np.asarray(mmd_dev(jnp.asarray(theta), keys)).mean(0),
                test_tvd=0.5 * np.abs(q - ctx.p_test).sum(1),
            ),
        )
        return float(cache["rec"]["val_mmd_train"].mean())

    def callback(step, theta):
        if "theta" not in cache or not np.array_equal(cache["theta"], theta):
            evaluate(theta)
        records.append(dict(step=int(step), **cache["rec"]))

    initial = (
        np.random.default_rng([STREAM_SEED, 1]).normal(0, 1, ctx.ansatz.n_parameters)
        * hp["init_scale"]
    )
    trainer = make_trainer(
        dict(
            name="adam",
            learning_rate=hp["learning_rate"],
            b1=hp["b1"],
            b2=hp["b2"],
            eps=1e-8,
            final_learning_rate_ratio=hp["final_ratio"],
            compute_validation_mmd=False,
            rng_seeds=dict(
                batches=STREAM_SEED + 1,
                bands=STREAM_SEED + 2,
                conditions=STREAM_SEED + 3,
                masks=STREAM_SEED + 4,
                reference=STREAM_SEED + 5,
            ),
        )
    )
    started = time.perf_counter()
    trainer(
        train_model,
        ctx.training,
        ctx.validation,
        seed=STREAM_SEED,
        config=tc,
        initial_params=initial,
        checkpoint_callback=callback,
        selection_metric=evaluate,
    )
    return dict(
        family=family,
        index=index,
        hyperparameters=hp,
        seconds=time.perf_counter() - started,
        steps=[r["step"] for r in records],
        **{
            k: np.stack([r[k] for r in records]).tolist()
            for k in records[0]
            if k != "step"
        },
    )


def collect(root):
    """Per-checkpoint arrays of a run directory as one JSON list of trials.

    Reads the ``run`` output and the 2026-10-15 development run (``.json``
    metadata plus ``.npz`` checkpoint arrays) alike.
    """
    arrays = ("val_mmd_train", "mmd_dev", "test_tvd")
    trials = []
    for path in sorted(Path(root).glob("S*-*.json")):
        meta = json.loads(path.read_text())
        npz = path.with_suffix(".npz")
        if npz.exists():
            z = np.load(npz)
            meta.update(
                steps=z["steps"].tolist(),
                **{k: z[k].tolist() for k in arrays},
            )
        keys = ("family", "index", "hyperparameters", "steps", *arrays)
        trials.append({k: meta[k] for k in keys})
    return trials


def spearman(x, y):
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def analyze(trials):
    rows = [
        dict(
            family=t["family"],
            trial=f"{t['family']}-{t['index']:03d}",
            step=step,
            dev_score=r_gamma(dev, DEV),
            train_mmd=float(np.mean(train)),
            tvd=r_gamma(tvd, TEST),
            mean_tvd=float(np.mean(tvd)),
        )
        for t in trials
        for step, dev, train, tvd in zip(
            t["steps"], t["mmd_dev"], t["val_mmd_train"], t["test_tvd"], strict=True
        )
    ]

    def correlations(subset):
        cols = {k: np.array([r[k] for r in subset]) for k in subset[0]}
        return dict(
            configurations=len(set(cols["trial"])),
            checkpoints=len(subset),
            dev_score_vs_tvd=spearman(cols["dev_score"], cols["tvd"]),
            train_mmd_vs_tvd=spearman(cols["train_mmd"], cols["tvd"]),
            dev_score_vs_mean_tvd=spearman(cols["dev_score"], cols["mean_tvd"]),
            train_mmd_vs_mean_tvd=spearman(cols["train_mmd"], cols["mean_tvd"]),
        )

    by_family = {f: [r for r in rows if r["family"] == f] for f in FAMILIES}
    return dict(
        all=correlations(rows),
        **{f: correlations(subset) for f, subset in by_family.items()},
        every_200_steps=correlations([r for r in rows if r["step"] % 200 == 0]),
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "collect", "analyze"])
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--trials", type=int, default=60)
    args = parser.parse_args(argv)
    if args.command == "run":
        args.out.mkdir(parents=True, exist_ok=True)
        ctx = Context()
        for index in range(args.trials):
            for family in FAMILIES:
                path = args.out / f"{family}-{index:03d}.json"
                if path.exists():
                    continue
                trial = run_trial(ctx, family, index)
                path.write_text(json.dumps(trial) + "\n")
                print(family, index, f"{trial['seconds']:.1f}s", flush=True)
        return
    if args.command == "collect":
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        trials = collect(args.out)
        (DATA_DIR / "checkpoints.json").write_text(json.dumps(trials) + "\n")
        print(f"{len(trials)} trials -> {DATA_DIR / 'checkpoints.json'}")
        return
    trials = json.loads((DATA_DIR / "checkpoints.json").read_text())
    result = analyze(trials)
    (DATA_DIR / "spearman.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

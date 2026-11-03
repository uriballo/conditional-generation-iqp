"""Random-search HPO for the redesigned parity-input lattice scaling models.

Each trial trains the same hyperparameters at every protocol size and records,
at every checkpoint, the dev-grid MMD that defines the objective together with
the old training-temperature validation MMD and a sampled dev-grid estimate of
the squared magnetization. The protocol is frozen in
``scaling_hpo_protocol.json``; the sixteen-qubit study that chose the objective
is ``scaling_score_study.py``.

Run:   python -m paper_experiments.scaling_hpo run OUT [--family S1 S0]
       python -m paper_experiments.scaling_hpo report OUT
       python -m paper_experiments.scaling_hpo confirm OUT --search SEARCH_OUT
       python -m paper_experiments.scaling_hpo report-confirm OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

PROTOCOL_PATH = Path(__file__).with_name("scaling_hpo_protocol.json")
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
TRAIN = np.asarray(PROTOCOL["training_betas"])
DEV = np.asarray(PROTOCOL["dev_betas"])
SEEDS = PROTOCOL["seeds"]
FAMILY_CODE = {"S0": 0, "S1": 1}


def make_targets(conditions, n_visible, n_samples, rng):
    """Curie--Weiss training samples, one block per inverse temperature."""
    from paper_experiments.target import curie_weiss_target

    return np.stack(
        [
            curie_weiss_target(n_visible, float(beta)).sample(n_samples, rng)
            for beta in np.asarray(conditions).ravel()
        ]
    ).astype(np.float64)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def schedule(family, hp, betas, n):
    """Encoder angle for each inverse temperature at size n."""
    b = np.asarray(betas, dtype=float)
    if family == "S0":
        t = hp["lam"] * b
    elif family == "S1":
        k = hp["k"] * (n / PROTOCOL["reference_size"]) ** hp["gamma"]
        s0 = _sigmoid(-k * hp["b0"])
        t = hp["tmax"] * (_sigmoid(k * (b - hp["b0"])) - s0) / (1 - s0)
    else:
        raise ValueError(family)
    if np.any(t < 0) or np.any(t > np.pi + 1e-12):
        raise ValueError("encoder angle outside [0, pi]")
    return np.clip(t, 0, np.pi)


def sample_trial(family, index):
    rng = np.random.default_rng([SEEDS["hyperparameters"], FAMILY_CODE[family], index])
    space = PROTOCOL["search"]

    def draw(spec):
        if spec[-1] == "choice":
            return spec[0][int(rng.integers(len(spec[0])))]
        lo, hi, scale = spec
        if scale == "log":
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return float(rng.uniform(lo, hi))

    hp = {k: draw(v) for k, v in space.items() if k not in FAMILY_CODE}
    hp.update({k: draw(v) for k, v in space[family].items()})
    return hp


def grown(value, n, exponent):
    scaled = value * (n / PROTOCOL["reference_size"]) ** exponent
    return int(max(64, 64 * round(scaled / 64)))


def r_gamma(e, grid):
    e, grid = np.asarray(e, float), np.asarray(grid, float)
    avg = np.trapezoid(e, grid) / (grid[-1] - grid[0])
    return float((2 * np.max(e) + avg) / 3)


def make_ansatz(n, kind="lattice"):
    """Open rectangular lattice, all one- and two-body generators, or a device graph.

    A device graph is a JSON file whose ``edges`` list the logical qubit pairs
    (for example the calibrated couplers of a processor).
    """
    from paper_experiments.ansatz import rectangular_lattice_ansatz

    if kind == "lattice":
        return rectangular_lattice_ansatz(n)
    if kind == "complete":
        import iqp_cond as ciqp

        iu = np.triu_indices(n, 1)
        rows = np.zeros((n + len(iu[0]), n))
        rows[np.arange(n), np.arange(n)] = 1
        rows[n + np.arange(len(iu[0])), iu[0]] = 1
        rows[n + np.arange(len(iu[0])), iu[1]] = 1
        return ciqp.Ansatz(rows, name="complete_2local")
    if str(kind).endswith(".json"):
        import iqp_cond as ciqp

        graph = json.loads(Path(kind).read_text())
        if len(graph["physical_qubits"]) != n:
            raise ValueError("device graph size differs from n")
        edges = sorted(tuple(sorted(e)) for e in graph["edges"])
        rows = np.zeros((n + len(edges), n))
        rows[np.arange(n), np.arange(n)] = 1
        for i, (a, b) in enumerate(edges, start=n):
            rows[i, [a, b]] = 1
        return ciqp.Ansatz(rows, name=f"device_{Path(kind).stem}")
    raise ValueError(kind)


class SizeContext:
    """Targets, pair masks and the trial-independent evaluation budget at one size."""

    def __init__(self, n, smoke=False, seed=None, ansatz="lattice"):
        self.n = n
        self.ansatz = make_ansatz(n, ansatz)
        self.ansatz_kind = ansatz
        ev = PROTOCOL["evaluation"]
        g = ev["growth_exponent"]
        self.eval = dict(
            n_ops=grown(ev["n_ops"], n, g),
            n_samples=grown(ev["n_samples"], n, g),
            draws=round(ev["dev_target_draws"] * (n / PROTOCOL["reference_size"]) ** g),
            banks=ev["banks"],
            pair_samples=grown(ev["pair_samples"], n, g),
        )
        if smoke:
            self.eval.update(n_ops=8, n_samples=16, draws=32, pair_samples=16)
        data = PROTOCOL["data"]
        rng = np.random.default_rng([SEEDS["targets"], n])
        draws = 32 if smoke else None
        self.training = make_targets(TRAIN, n, draws or data["n_train"], rng)
        self.validation = make_targets(TRAIN, n, draws or data["n_validation"], rng)
        self.dev_targets = make_targets(DEV, n, self.eval["draws"], rng)
        pairs = set()
        while len(pairs) < min(ev["pair_masks"], n * (n - 1) // 2):
            i, j = sorted(rng.choice(n, 2, replace=False).tolist())
            pairs.add((i, j))
        self.pairs = np.asarray(sorted(pairs))
        self.pair_masks = np.zeros((len(self.pairs), n))
        self.pair_masks[np.arange(len(self.pairs)), self.pairs[:, 0]] = 1
        self.pair_masks[np.arange(len(self.pairs)), self.pairs[:, 1]] = 1
        if seed is not None:
            # Confirmation seeds redraw the training pools; the dev pools and
            # pair masks stay shared so candidates are compared on equal terms.
            rng = np.random.default_rng([SEEDS["targets"], n, seed])
            self.training = make_targets(TRAIN, n, draws or data["n_train"], rng)
            self.validation = make_targets(TRAIN, n, draws or data["n_validation"], rng)


def run_size(
    ctx, family, hp, smoke=False, seed=None, steps=None, restart=0, interval=None
):
    import iqp_cond as ciqp
    import jax
    import jax.numpy as jnp
    from iqp_cond.estimator import estimate_pauli_z_unchecked

    from paper_experiments.config import RunConfig
    from paper_experiments.parity_encoder import ParityEncoder
    from paper_experiments.training import make_trainer

    n, data = ctx.n, PROTOCOL["data"]
    stream = SEEDS["streams"] + n + (0 if seed is None else 100_000 * (seed + 1))
    steps = 20 if smoke else steps or PROTOCOL["steps"]
    interval = 10 if smoke else interval or PROTOCOL["checkpoint_interval"]
    n_ops = 8 if smoke else grown(hp["n_ops"], n, hp["alpha"])
    mc = 16 if smoke else grown(hp["mc_samples"], n, hp["alpha"])
    config = RunConfig(
        study="scaling_hpo",
        encoder="parity",
        n_visible=n,
        n_train=len(ctx.training[0]),
        n_validation=len(ctx.validation[0]),
        n_test=2,
        steps=steps,
        batch_size=8 if smoke else data["batch_size"],
        n_ops=n_ops,
        mc_samples=mc,
        bodyness_targets=tuple(PROTOCOL["bodyness_targets"]),
        learning_rate=hp["learning_rate"],
        final_learning_rate=hp["learning_rate"] * hp["final_ratio"],
        validation_interval=interval,
        gemm_dtype="float32",
        sample_chunk=256,
        validation_n_ops=2,
        validation_mc_samples=2,
        init_scale=hp["init_scale"],
        restarts=1,
    )
    tc = config.training_config()

    def model(grid):
        encoder = ParityEncoder.from_betas(schedule(family, hp, grid, n), n, scale=1.0)
        return ciqp.ConditionalModel(ctx.ansatz, encoder)

    train_model, dev_model = model(TRAIN), model(DEV)
    ev = ctx.eval
    keys = jnp.stack([jax.random.PRNGKey(stream + 7 + b) for b in range(ev["banks"])])

    def mmd(m, targets):
        def one(theta, key):
            return m.mmd(
                theta,
                jnp.asarray(targets),
                key,
                sigmas=tc.sigmas,
                n_ops=ev["n_ops"],
                n_samples=ev["n_samples"],
                shared_samples=True,
                gemm_dtype=jnp.float32,
                sample_chunk=256,
            ).per_condition

        return jax.jit(jax.vmap(one, in_axes=(None, 0)))

    mmd_train = mmd(train_model, ctx.validation)
    mmd_dev = mmd(dev_model, ctx.dev_targets)
    pair_refs = dev_model.encoder.sample(
        jax.random.PRNGKey(stream + 99), ev["pair_samples"]
    )

    pair_masks = jnp.asarray(ctx.pair_masks)

    @jax.jit
    def _pair_est(theta, masks, refs):
        return estimate_pauli_z_unchecked(
            theta,
            dev_model.generators,
            dev_model.encoder,
            masks,
            samples=refs,
            gemm_dtype=jnp.float32,
        ).values

    def pair_est(theta):
        return _pair_est(theta, pair_masks, pair_refs)

    records, params, cache = [], [], {}

    def evaluate(theta):
        theta = jnp.asarray(theta)
        corr = np.asarray(pair_est(theta))
        rec = dict(
            val_mmd_train=np.asarray(mmd_train(theta, keys)).mean(0),
            mmd_dev=np.asarray(mmd_dev(theta, keys)).mean(0),
            m2_est_dev=1 / n + (n - 1) / n * corr.mean(1),
        )
        rec["objective"] = r_gamma(rec["mmd_dev"], DEV)
        cache.update(theta=np.asarray(theta).copy(), rec=rec)
        return rec["objective"]

    def callback(step, theta):
        if "theta" not in cache or not np.array_equal(cache["theta"], theta):
            evaluate(theta)
        records.append(dict(step=int(step), **cache["rec"]))
        params.append(np.asarray(theta).copy())

    initial = (
        np.random.default_rng([stream, 1 + restart]).normal(
            0, 1, ctx.ansatz.n_parameters
        )
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
                batches=stream + 1,
                bands=stream + 2,
                conditions=stream + 3,
                masks=stream + 4,
                reference=stream + 5,
            ),
        )
    )
    started = time.perf_counter()
    status = "ok"
    try:
        trainer(
            train_model,
            ctx.training,
            ctx.validation,
            seed=stream,
            config=tc,
            initial_params=initial,
            checkpoint_callback=callback,
            selection_metric=evaluate,
        )
    except FloatingPointError as error:
        status = f"numerical_failure: {error}"
    arrays = (
        {k: np.stack([r[k] for r in records]) for k in records[0] if k != "step"}
        if records
        else {}
    )
    arrays["steps"] = np.array([r["step"] for r in records])
    arrays["params"] = np.stack(params) if params else np.zeros((0,))
    budget = dict(train_n_ops=n_ops, train_mc_samples=mc, **ev)
    return status, time.perf_counter() - started, arrays, budget


def trial_score(per_size):
    return float(
        np.mean([np.log(max(float(np.min(a["objective"])), 1e-8)) for a in per_size])
    )


def run_trial(ctxs, family, index, out, smoke=False, fixed_budget=None):
    import jax

    hp = sample_trial(family, index)
    if fixed_budget:
        # Fixed after sampling so the other hyperparameters keep their draws.
        hp.update(n_ops=fixed_budget[0], mc_samples=fixed_budget[1])
    sizes, statuses, arrays = {}, [], []
    for ctx in ctxs:
        status, seconds, a, budget = run_size(ctx, family, hp, smoke)
        np.savez_compressed(out / f"{family}-{index:03d}-n{ctx.n}.npz", **a)
        statuses.append(status)
        arrays.append(a)
        sizes[str(ctx.n)] = dict(
            status=status,
            seconds=seconds,
            budget=budget,
            best_objective=float(np.min(a["objective"])) if len(a["steps"]) else None,
            best_step=int(a["steps"][np.argmin(a["objective"])])
            if len(a["steps"])
            else None,
        )
    ok = all(s == "ok" for s in statuses)
    meta = dict(
        family=family,
        index=index,
        hyperparameters=hp,
        ansatz=ctxs[0].ansatz_kind,
        status="ok" if ok else "failed",
        score=trial_score(arrays) if ok else None,
        sizes=sizes,
        smoke=smoke,
        protocol_sha256=hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        devices=[str(d) for d in jax.devices()],
    )
    (out / f"{family}-{index:03d}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def report(out):
    rows = []
    for path in sorted(Path(out).glob("S?-???.json")):
        meta = json.loads(path.read_text())
        if meta["status"] == "ok":
            rows.append(meta)
    rows.sort(key=lambda m: m["score"])
    for fam in ("S1", "S0"):
        subset = [m for m in rows if m["family"] == fam]
        print(f"{fam}: {len(subset)} completed")
        for m in subset[:5]:
            best = {
                n: f"{s['best_objective']:.3g}@{s['best_step']}"
                for n, s in m["sizes"].items()
            }
            hp = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in m["hyperparameters"].items()
            }
            print(f"  {fam}-{m['index']:03d} score={m['score']:.3f} {best} {hp}")


def confirm(out, search, smoke=False):
    """Retrain the shortlisted search trials with independent seeds."""
    import jax

    spec = PROTOCOL["confirmation"]
    sizes = [16, 36] if smoke else PROTOCOL["sizes"]
    steps = 20 if smoke else spec["steps"]
    for seed in spec["seeds"]:
        ctxs = [SizeContext(n, smoke, seed) for n in sizes]
        for name in spec["candidates"]:
            path = out / f"{name}-s{seed}.json"
            if path.exists():
                continue
            source = json.loads((search / f"{name}.json").read_text())
            family, hp = source["family"], source["hyperparameters"]
            expected = sample_trial(family, source["index"])
            if hp.keys() != expected.keys() or not all(
                np.isclose(hp[k], expected[k], rtol=1e-12, atol=0) for k in hp
            ):
                raise ValueError(f"{name} hyperparameters differ from the protocol")
            sizes_meta = {}
            for ctx in ctxs:
                status, seconds, a, budget = run_size(
                    ctx, family, hp, smoke, seed=seed, steps=steps
                )
                np.savez_compressed(out / f"{name}-s{seed}-n{ctx.n}.npz", **a)
                i = int(np.argmin(a["objective"])) if len(a["steps"]) else None
                sizes_meta[str(ctx.n)] = dict(
                    status=status,
                    seconds=seconds,
                    budget=budget,
                    best_objective=None if i is None else float(a["objective"][i]),
                    best_step=None if i is None else int(a["steps"][i]),
                )
            meta = dict(
                candidate=name,
                family=family,
                seed=seed,
                hyperparameters=hp,
                steps=steps,
                sizes=sizes_meta,
                status="ok"
                if all(v["status"] == "ok" for v in sizes_meta.values())
                else "failed",
                smoke=smoke,
                protocol_sha256=hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                devices=[str(d) for d in jax.devices()],
            )
            path.write_text(json.dumps(meta, indent=2) + "\n")
            secs = sum(v["seconds"] for v in sizes_meta.values())
            print(name, seed, meta["status"], f"{secs:.1f}s", flush=True)


def final(
    out,
    sizes,
    search=None,
    confirmation=None,
    smoke=False,
    ansatz="lattice",
    trial=None,
    selection=None,
):
    """Train the selected configuration at every requested size.

    ``selection`` is a stored final-fit receipt whose candidate and
    hyperparameters are trained without the search and confirmation stages.

    Sizes run in ascending order with seeds inside sizes, so an interrupted
    queue leaves complete smaller sizes. Restarts share data and optimizer
    streams and differ in the initial parameters; the restart and checkpoint
    with the lowest dev objective are selected.
    """
    import jax

    spec = PROTOCOL["final"]
    if smoke:
        family, hp = "S1", sample_trial("S1", 5)
    elif selection:
        source = json.loads(Path(selection).read_text())
        name, family, hp = (
            source["candidate"],
            source["family"],
            source["hyperparameters"],
        )
    else:
        # The all-to-all ablation takes its configuration directly from its
        # own single-size search instead of the lattice confirmation stage.
        name = trial or report_confirm(confirmation)
        source = json.loads((search / f"{name}.json").read_text())
        family, hp = source["family"], source["hyperparameters"]
        if family != spec["family"]:
            raise ValueError("selected candidate is not from the eligible family")
    steps = 20 if smoke else spec["steps"]
    for n in sizes:
        for seed in spec["seeds"]:
            path = out / f"n{n}-s{seed}.json"
            if path.exists():
                continue
            ctx = SizeContext(n, smoke, seed, ansatz)
            restarts = []
            for restart in range(spec["restarts"]):
                status, seconds, a, budget = run_size(
                    ctx,
                    family,
                    hp,
                    smoke,
                    seed=seed,
                    steps=steps,
                    restart=restart,
                    interval=spec["checkpoint_interval"],
                )
                np.savez_compressed(out / f"n{n}-s{seed}-r{restart}.npz", **a)
                i = int(np.argmin(a["objective"])) if len(a["steps"]) else None
                restarts.append(
                    dict(
                        restart=restart,
                        status=status,
                        seconds=seconds,
                        best_objective=None if i is None else float(a["objective"][i]),
                        best_step=None if i is None else int(a["steps"][i]),
                        best_params=None if i is None else a["params"][i].tolist(),
                    )
                )
                print(n, seed, restart, status, f"{seconds:.1f}s", flush=True)
            ok = [r for r in restarts if r["status"] == "ok"]
            best = min(ok, key=lambda r: r["best_objective"]) if ok else None
            meta = dict(
                size=n,
                seed=seed,
                ansatz=ansatz,
                ansatz_graph=json.loads(Path(ansatz).read_text())
                if str(ansatz).endswith(".json")
                else None,
                candidate=None if smoke else name,
                family=family,
                hyperparameters=hp,
                schedule_angles=dict(
                    training=schedule(family, hp, TRAIN, n).tolist(),
                    dev=schedule(family, hp, DEV, n).tolist(),
                ),
                steps=steps,
                budget=budget,
                restarts=restarts,
                selected_restart=None if best is None else best["restart"],
                selected_step=None if best is None else best["best_step"],
                selected_objective=None if best is None else best["best_objective"],
                params=None if best is None else best["best_params"],
                status="ok" if best is not None else "failed",
                smoke=smoke,
                protocol_sha256=hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                devices=[str(d) for d in jax.devices()],
            )
            path.write_text(json.dumps(meta, indent=2) + "\n")


def evaluate_fit(receipt_path, out, smoke=False):
    """Held-out MMD trajectory and dense <m^2> estimates of one final fit.

    Every bank draws fresh held-out targets and estimator keys from a namespace
    disjoint from training, dev selection and the confirmation stage; banks are
    paired across checkpoints.
    """
    import iqp_cond as ciqp
    import jax
    import jax.numpy as jnp

    from paper_experiments.config import RunConfig
    from paper_experiments.parity_encoder import ParityEncoder
    from paper_experiments.scaling_observables import DENSE_BETAS, pair_estimates

    spec = PROTOCOL["final_evaluation"]
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    n, seed, family = receipt["size"], receipt["seed"], receipt["family"]
    hp = receipt["hyperparameters"]
    ansatz = make_ansatz(n, receipt.get("ansatz", "lattice"))
    with np.load(
        receipt_path.with_name(f"n{n}-s{seed}-r{receipt['selected_restart']}.npz")
    ) as data:
        steps, params = data["steps"], data["params"]
    selected = int(np.flatnonzero(steps == receipt["selected_step"])[0])
    if not np.allclose(params[selected], receipt["params"], rtol=0, atol=0):
        raise ValueError("selected parameters differ from the receipt")
    g = PROTOCOL["evaluation"]["growth_exponent"]
    budget = dict(
        n_ops=8 if smoke else grown(spec["mmd_ops"], n, g),
        n_samples=16 if smoke else grown(spec["mmd_references"], n, g),
        draws=32 if smoke else round(spec["mmd_targets"] * (n / 64) ** g),
        banks=2 if smoke else spec["mmd_banks"],
    )
    tc = RunConfig(
        study="scaling_hpo",
        encoder="parity",
        n_visible=n,
        bodyness_targets=tuple(PROTOCOL["bodyness_targets"]),
    ).training_config()
    held = np.asarray(spec["held_out_betas"])

    def model(grid):
        encoder = ParityEncoder.from_betas(schedule(family, hp, grid, n), n, scale=1.0)
        return ciqp.ConditionalModel(ansatz, encoder)

    held_model = model(held)
    base = spec["namespace"] + 10_000 * n + 10 * seed
    banks = []
    for b in range(budget["banks"]):
        rng = np.random.default_rng([base, b])
        banks.append(
            (
                jnp.asarray(make_targets(held, n, budget["draws"], rng)),
                jax.random.PRNGKey(base + b),
            )
        )

    @jax.jit
    def mmd(theta, targets, key):
        return held_model.mmd(
            theta,
            targets,
            key,
            sigmas=tc.sigmas,
            n_ops=budget["n_ops"],
            n_samples=budget["n_samples"],
            shared_samples=True,
            gemm_dtype=jnp.float32,
            sample_chunk=256,
        ).per_condition

    every = spec["trajectory_interval"]
    wanted = sorted({int(t) for t in steps if t % every == 0} | {int(steps[selected])})
    trajectory = []
    for t in wanted:
        theta = jnp.asarray(params[int(np.flatnonzero(steps == t)[0])])
        values = np.stack([np.asarray(mmd(theta, x, k)) for x, k in banks])
        trajectory.append(dict(step=t, per_bank_per_condition=values.tolist()))
    dense_model = model(DENSE_BETAS)
    pair_means, bank_seeds = pair_estimates(
        dense_model,
        receipt["params"],
        n,
        seed,
        len(DENSE_BETAS),
        "final",
        2 if smoke else spec["dense_banks"],
        8 if smoke else spec["dense_masks"],
        16 if smoke else spec["dense_draws"],
    )
    result = dict(
        job_id=f"n{n}-s{seed}",
        size=n,
        seed=seed,
        held_out_betas=held.tolist(),
        mmd_budget=budget,
        selected_step=int(steps[selected]),
        trajectory=trajectory,
        betas=DENSE_BETAS.tolist(),
        pair_means=pair_means.tolist(),
        m2=(1 / n + (n - 1) / n * pair_means).mean(axis=0).tolist(),
        bank_seeds=bank_seeds,
        dense_budget=dict(
            banks=spec["dense_banks"],
            masks=spec["dense_masks"],
            draws=spec["dense_draws"],
            namespace="final",
        ),
        fit_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        smoke=smoke,
    )
    (out / f"n{n}-s{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def evaluate(out, fits, smoke=False, wait=False):
    """Evaluate every completed final fit, optionally waiting for the queue."""
    expected = {
        f"n{n}-s{s}"
        for n in PROTOCOL["final"]["sizes"]
        for s in PROTOCOL["final"]["seeds"]
    }
    while True:
        done = {p.stem for p in out.glob("n*-s*.json")}
        ready = sorted(
            (p for p in fits.glob("n*-s*.json") if p.stem not in done),
            key=lambda p: json.loads(p.read_text())["size"],
        )
        for path in ready:
            started = time.perf_counter()
            evaluate_fit(path, out, smoke)
            print(path.stem, f"{time.perf_counter() - started:.1f}s", flush=True)
        if not wait or expected <= {p.stem for p in out.glob("n*-s*.json")}:
            return
        time.sleep(60)


def bundle(out, fits, evaluations, smoke=False):
    """Assemble the figure layout from final fits and evaluations."""
    import shutil

    spec = PROTOCOL["final"]
    sizes = sorted({json.loads(p.read_text())["size"] for p in fits.glob("n*-s*.json")})
    if not smoke and sizes != spec["sizes"]:
        print("warning: incomplete sizes", sizes)
    (out / "physics" / "fits").mkdir(parents=True, exist_ok=True)
    (out / "physics" / "dense").mkdir(parents=True, exist_ok=True)
    curves, steps, candidate = {}, None, None
    for n in sizes:
        per_seed = []
        for seed in spec["seeds"]:
            fit = fits / f"n{n}-s{seed}.json"
            ev = evaluations / f"n{n}-s{seed}.json"
            if not fit.exists() or not ev.exists():
                raise ValueError(f"missing fit or evaluation for n{n}-s{seed}")
            record = json.loads(ev.read_text())
            if (
                record["fit_receipt_sha256"]
                != hashlib.sha256(fit.read_bytes()).hexdigest()
            ):
                raise ValueError("evaluation does not belong to the fit receipt")
            receipt = json.loads(fit.read_text())
            candidate = candidate or receipt.get("candidate")
            every = PROTOCOL["final_evaluation"]["trajectory_interval"]
            rows = [t for t in record["trajectory"] if t["step"] % every == 0]
            these = [t["step"] for t in rows]
            if steps is None:
                steps = these
            elif these != steps:
                raise ValueError("trajectory steps differ across fits")
            per_seed.append([float(np.mean(t["per_bank_per_condition"])) for t in rows])
            shutil.copyfile(fit, out / "physics" / "fits" / fit.name)
            dense = {
                k: record[k]
                for k in (
                    "job_id",
                    "size",
                    "seed",
                    "betas",
                    "pair_means",
                    "m2",
                    "bank_seeds",
                    "fit_receipt_sha256",
                    "script_sha256",
                )
            }
            dense["budget"] = record["dense_budget"]
            (out / "physics" / "dense" / ev.name).write_text(
                json.dumps(dense, indent=2) + "\n"
            )
        curves[str(n)] = {"held_out": {"overall_per_seed": per_seed}}
    payload = dict(
        sizes=sizes,
        seeds=spec["seeds"],
        steps=steps,
        curves=curves,
        model_metadata=dict(
            encoder="RX-CNOT parity encoder with saturating size-aware schedule",
            ansatz_by_size={str(n): ["rectangular lattice"] for n in sizes},
            candidate=dict(
                id=candidate,
                **json.loads(next(fits.glob("n*-s*.json")).read_text())[
                    "hyperparameters"
                ],
            ),
            aggregation="mean over held-out banks and temperatures per seed; "
            "mean and sample standard deviation across seeds in renderer",
        ),
    )
    (out / "figure_data.json").write_text(json.dumps(payload, indent=2) + "\n")
    protocol = dict(PROTOCOL, training_betas=PROTOCOL["training_betas"])
    (out / "scaling_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return out


def report_confirm(out):
    """Candidate score: worst size of the seed-averaged best dev objective."""
    spec = PROTOCOL["confirmation"]
    results = {}
    for name in spec["candidates"]:
        runs = [
            json.loads(p.read_text()) for p in sorted(Path(out).glob(f"{name}-s*.json"))
        ]
        runs = [r for r in runs if r["status"] == "ok"]
        if not runs:
            continue
        per_size = {
            n: float(np.mean([r["sizes"][n]["best_objective"] for r in runs]))
            for n in runs[0]["sizes"]
        }
        results[name] = dict(
            seeds=len(runs), per_size=per_size, score=max(per_size.values())
        )
    eligible = [n for n in results if n.startswith(spec["eligible_family"])]
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["score"]):
        sizes = {n: round(v, 5) for n, v in r["per_size"].items()}
        print(f"{name} seeds={r['seeds']} worst={r['score']:.5f} {sizes}")
    if len(eligible) != len(
        [c for c in spec["candidates"] if c.startswith(spec["eligible_family"])]
    ) or any(results[n]["seeds"] != len(spec["seeds"]) for n in results):
        raise ValueError("confirmation is incomplete")
    selected = min(eligible, key=lambda n: results[n]["score"])
    print("selected:", selected)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "run",
            "report",
            "confirm",
            "report-confirm",
            "final",
            "evaluate",
            "bundle",
        ],
    )
    parser.add_argument("--search", type=Path)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--fits", type=Path)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument(
        "--ansatz", default="lattice", help="lattice, complete or a device graph JSON"
    )
    parser.add_argument("--trial")
    parser.add_argument(
        "--selection", type=Path, help="final: train a stored fit receipt's candidate"
    )
    parser.add_argument("--evaluations", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--family", nargs="+", default=PROTOCOL["families"])
    parser.add_argument("--trials", type=int, default=PROTOCOL["trials_per_family"])
    parser.add_argument("--sizes", type=int, nargs="+", default=PROTOCOL["sizes"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--budget",
        type=int,
        nargs=2,
        metavar=("N_OPS", "MC_SAMPLES"),
        help="fix the training budget instead of searching it",
    )
    args = parser.parse_args(argv)
    if args.command == "report":
        return report(args.out)
    if args.command == "report-confirm":
        return report_confirm(args.out)
    if args.command == "bundle":
        args.out.mkdir(parents=True, exist_ok=True)
        return bundle(args.out, args.fits, args.evaluations, args.smoke)
    if args.command == "evaluate":
        args.out.mkdir(parents=True, exist_ok=True)
        return evaluate(args.out, args.fits, args.smoke, args.wait)
    if args.command == "final":
        args.out.mkdir(parents=True, exist_ok=True)
        return final(
            args.out,
            args.sizes,
            args.search,
            args.confirmation,
            args.smoke,
            args.ansatz,
            args.trial,
            args.selection,
        )
    if args.command == "confirm":
        args.out.mkdir(parents=True, exist_ok=True)
        return confirm(args.out, args.search, args.smoke)
    ablation = args.ansatz == "complete" and args.sizes == [64]
    if not args.smoke and args.sizes != PROTOCOL["sizes"] and not ablation:
        raise SystemExit("non-smoke runs must use the protocol sizes")
    args.out.mkdir(parents=True, exist_ok=True)
    ctxs = [SizeContext(n, args.smoke, ansatz=args.ansatz) for n in args.sizes]
    for index in range(args.trials):
        for family in args.family:
            if (args.out / f"{family}-{index:03d}.json").exists():
                continue
            meta = run_trial(ctxs, family, index, args.out, args.smoke, args.budget)
            secs = sum(s["seconds"] for s in meta["sizes"].values())
            print(
                family, index, meta["status"], meta["score"], f"{secs:.1f}s", flush=True
            )


if __name__ == "__main__":
    main()

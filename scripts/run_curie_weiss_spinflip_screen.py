#!/usr/bin/env python3
"""Matched production screen of the spin-flip two-branch encoding on Curie--Weiss.

Runs the full production arm structure for ``spin_flip_branch``, matched to the
frozen v2 protocol (13 qubits, 91 phases, 2000 updates, 10 seeds, same datasets
and checkpoint rule).  TVD is the objective metric; MMD is only the training loss.

Arms:
    conditioned   10 seeds (already available from the encoding benchmark)
    no_condition  10 seeds
    reference     10 seeds x 6 training temperatures = 60 single-temperature models

The report reproduces the paper's headline comparisons in TVD:
conditioned vs no-condition (training / held-out / cold / critical) and
conditioned vs adjacent-mixture / single-temperature reference.

Subcommands:
    run     train/evaluate one family over seeds (writes one immutable row per run).
    report  aggregate rows into the headline TVD comparisons.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from iqp_cond.curie_weiss_ancilla_campaign import CurieWeissAncillaCampaignStore
from iqp_cond.curie_weiss_ancilla_protocol import (
    CurieWeissAncillaFamily,
    CurieWeissAncillaManifest,
    plan_curie_weiss_ancilla_runs,
)
from iqp_cond.curie_weiss_ancilla_runner import (
    evaluate_run,
    load_reused_dataset,
    train_model,
    training_config,
)
from iqp_cond.curie_weiss_data import exact_curie_weiss_target
from iqp_cond.encoding import SPIN_FLIP_BRANCH_ENCODING
from iqp_cond.exact import (
    enumerate_bitstrings,
    shell_probabilities,
    total_variation_distance,
)
from iqp_cond.gaussian_statistics import paired_bootstrap_interval

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "configs/curie_weiss_m5_foundations_v2.json"
BASELINE_STORE = REPO / "results/curie-weiss-m5-foundations-v2/store"
OUT_DIR = REPO / "results/curie-weiss-spinflip-screen-v1"
ROWS_DIR = OUT_DIR / "rows"

FAMILIES = {
    "conditioned": CurieWeissAncillaFamily.CONDITIONED,
    "no_condition": CurieWeissAncillaFamily.NO_CONDITION,
    "reference": CurieWeissAncillaFamily.TEMPERATURE_REFERENCE,
}

TRAINING_BETAS = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5)
HELD_OUT_BETAS = (0.6, 0.8, 1.0, 1.2, 1.4)
CRITICAL_BETAS = (0.9, 1.0, 1.1)
COLD_BETAS = (1.3, 1.4, 1.5)
BOOTSTRAP_SEED = 20_260_815


def _tvd_vectors(
    target: np.ndarray, model: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    bits = enumerate_bitstrings(8)
    raw = np.asarray(
        [total_variation_distance(t, m) for t, m in zip(target, model, strict=True)]
    )
    profile = np.asarray(
        [
            total_variation_distance(
                shell_probabilities(t, bits), shell_probabilities(m, bits)
            )
            for t, m in zip(target, model, strict=True)
        ]
    )
    return raw, profile


def _seed_list(spec: str) -> list[int]:
    if "," in spec:
        return [int(s) for s in spec.split(",")]
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def _row_path(family: str, run) -> Path:
    if family == "reference":
        name = f"seed{run.optimizer_seed}_beta{run.reference_beta:.1f}.json"
    else:
        name = f"seed{run.optimizer_seed}.json"
    return ROWS_DIR / family / name


def _run_one(
    manifest: CurieWeissAncillaManifest,
    run,
    family: str,
    *,
    updates: int | None,
    lightweight: bool,
) -> dict:
    started = time.perf_counter()
    dataset = load_reused_dataset(REPO, manifest, run)
    config = training_config(manifest, run, updates=updates, lightweight=lightweight)
    result = train_model(
        manifest, run, dataset, config, encoding=SPIN_FLIP_BRANCH_ENCODING
    )
    _metrics, arrays = evaluate_run(
        manifest,
        run,
        result.best_params,
        result.generators,
        encoding=SPIN_FLIP_BRANCH_ENCODING,
    )
    betas = np.asarray(arrays["betas"], dtype=np.float64)
    target = np.asarray(arrays["target_probabilities"], dtype=np.float64)
    model = np.asarray(arrays["model_probabilities"], dtype=np.float64)
    raw, profile = _tvd_vectors(target, model)
    row = {
        "family": family,
        "seed": run.optimizer_seed,
        "reference_beta": run.reference_beta,
        "betas": betas.tolist(),
        "raw_tvd": raw.tolist(),
        "profile_tvd": profile.tolist(),
        "best_validation_loss": result.best_validation_loss,
        "best_step": result.best_step,
        "runtime_seconds": time.perf_counter() - started,
    }
    if family == "reference":
        row["model_probabilities"] = model.tolist()
    return row


def run(args: argparse.Namespace) -> int:
    manifest = CurieWeissAncillaManifest.from_json(MANIFEST_PATH)
    family = FAMILIES[args.family]
    planned = plan_curie_weiss_ancilla_runs(manifest)
    runs = [
        r
        for r in planned
        if r.family is family and r.ancillas == manifest.headline_ancillas
    ]
    if args.seeds is not None:
        seeds = set(_seed_list(args.seeds))
        runs = [r for r in runs if r.optimizer_seed in seeds]
    for run in runs:
        row_path = _row_path(args.family, run)
        if row_path.exists():
            print(f"skip {args.family} {row_path.stem}", flush=True)
            continue
        print(f"start {args.family} {row_path.stem}", flush=True)
        row = _run_one(
            manifest,
            run,
            args.family,
            updates=args.updates,
            lightweight=args.lightweight,
        )
        row_path.parent.mkdir(parents=True, exist_ok=True)
        row_path.write_text(json.dumps(row, indent=2, sort_keys=True))
        print(f"done  {args.family} {row_path.stem}", flush=True)
    return 0


def _load_rows(family: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not (ROWS_DIR / family).exists():
        return out
    for path in (ROWS_DIR / family).glob("*.json"):
        row = json.loads(path.read_text())
        out[row["seed"]] = row
    return out


def _per_beta_tvd(row: dict, betas: tuple[float, ...]) -> dict[float, float]:
    out = {}
    for beta, profile in zip(row["betas"], row["profile_tvd"], strict=True):
        if beta in betas:
            out[beta] = profile
    return out


def _bootstrap(diffs: dict[int, float]) -> dict[str, float]:
    interval = paired_bootstrap_interval(diffs, replicates=10_000, seed=BOOTSTRAP_SEED)
    return interval.to_dict()


def _cond_vs_nocond(
    conditioned: dict[int, dict],
    no_condition: dict[int, dict],
    betas: tuple[float, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    profile, raw = {}, {}
    for seed, cond in conditioned.items():
        nc = no_condition.get(seed)
        if nc is None:
            continue
        cp = np.mean(
            [
                cond["profile_tvd"][cond["betas"].index(b)]
                for b in betas
                if b in cond["betas"]
            ]
        )
        np_ = np.mean(
            [nc["profile_tvd"][nc["betas"].index(b)] for b in betas if b in nc["betas"]]
        )
        cr = np.mean(
            [
                cond["raw_tvd"][cond["betas"].index(b)]
                for b in betas
                if b in cond["betas"]
            ]
        )
        nr = np.mean(
            [nc["raw_tvd"][nc["betas"].index(b)] for b in betas if b in nc["betas"]]
        )
        profile[seed] = np_ - cp  # positive = conditioning helps
        raw[seed] = nr - cr
    return _bootstrap(profile), _bootstrap(raw)


def _load_reference_rows() -> dict[tuple[int, float], dict]:
    out: dict[tuple[int, float], dict] = {}
    if not (ROWS_DIR / "reference").exists():
        return out
    for path in (ROWS_DIR / "reference").glob("*.json"):
        row = json.loads(path.read_text())
        out[(row["seed"], float(row["reference_beta"]))] = row
    return out


def _adjacent(beta: float) -> tuple[float, float]:
    for i in range(len(TRAINING_BETAS) - 1):
        if np.isclose(beta, 0.5 * (TRAINING_BETAS[i] + TRAINING_BETAS[i + 1])):
            return TRAINING_BETAS[i], TRAINING_BETAS[i + 1]
    raise ValueError(f"{beta} is not a training midpoint")


def _cond_vs_mixture(
    conditioned: dict[int, dict], refs: dict[tuple[int, float], dict]
) -> tuple[dict[str, float], dict[str, float]]:
    bits = enumerate_bitstrings(8)
    profile, raw = {}, {}
    for seed, cond in conditioned.items():
        cprof, craw, mprof, mraw = [], [], [], []
        for b in HELD_OUT_BETAS:
            left, right = _adjacent(b)
            lr, rr = refs.get((seed, left)), refs.get((seed, right))
            if lr is None or rr is None:
                continue
            target = exact_curie_weiss_target(8, b).probabilities
            mixture = 0.5 * (
                np.asarray(lr["model_probabilities"])[0]
                + np.asarray(rr["model_probabilities"])[0]
            )
            mprof.append(
                total_variation_distance(
                    shell_probabilities(target, bits),
                    shell_probabilities(mixture, bits),
                )
            )
            mraw.append(total_variation_distance(target, mixture))
            cprof.append(cond["profile_tvd"][cond["betas"].index(b)])
            craw.append(cond["raw_tvd"][cond["betas"].index(b)])
        if cprof:
            profile[seed] = float(np.mean(mprof) - np.mean(cprof))
            raw[seed] = float(np.mean(mraw) - np.mean(craw))
    return _bootstrap(profile), _bootstrap(raw)


def _cond_vs_reference(
    conditioned: dict[int, dict], refs: dict[tuple[int, float], dict]
) -> tuple[dict[str, float], dict[str, float]]:
    profile, raw = {}, {}
    for seed, cond in conditioned.items():
        cprof, craw, rprof, rraw = [], [], [], []
        for b in TRAINING_BETAS:
            r = refs.get((seed, b))
            if r is None:
                continue
            cprof.append(cond["profile_tvd"][cond["betas"].index(b)])
            craw.append(cond["raw_tvd"][cond["betas"].index(b)])
            rprof.append(r["profile_tvd"][0])
            rraw.append(r["raw_tvd"][0])
        if cprof:
            profile[seed] = float(np.mean(rprof) - np.mean(cprof))
            raw[seed] = float(np.mean(rraw) - np.mean(craw))
    return _bootstrap(profile), _bootstrap(raw)


def _baseline_conditioned_tvd(
    store: CurieWeissAncillaCampaignStore,
) -> dict[str, float]:
    planned = plan_curie_weiss_ancilla_runs(store.manifest)
    conditioned = [
        r
        for r in planned
        if r.family is CurieWeissAncillaFamily.CONDITIONED and r.ancillas == 5
    ]
    profile, raw = [], []
    for run in conditioned:
        bundle = store.load_result(run)
        with np.load(bundle.probabilities_path, allow_pickle=False) as archive:
            betas = np.asarray(archive["betas"], dtype=np.float64)
            target = np.asarray(archive["target_probabilities"], dtype=np.float64)
            model = np.asarray(archive["model_probabilities"], dtype=np.float64)
        raw_v, profile_v = _tvd_vectors(target, model)
        mask = np.isin(betas, HELD_OUT_BETAS)
        profile.append(profile_v[mask].mean())
        raw.append(raw_v[mask].mean())
    return {
        "held_out_profile_mean": float(np.mean(profile)),
        "held_out_raw_mean": float(np.mean(raw)),
    }


def report(args: argparse.Namespace) -> int:
    conditioned = _load_rows("conditioned")
    no_condition = _load_rows("no_condition")
    references = _load_reference_rows()

    manifest = CurieWeissAncillaManifest.from_json(MANIFEST_PATH)
    store = CurieWeissAncillaCampaignStore(BASELINE_STORE, manifest)
    baseline = _baseline_conditioned_tvd(store)

    # Absolute conditioned held-out TVD (spin-flip) for context.
    c_prof_ho = np.mean(
        [
            np.mean([r["profile_tvd"][r["betas"].index(b)] for b in HELD_OUT_BETAS])
            for r in conditioned.values()
        ]
    )

    print(
        "\n# spin_flip_branch matched production screen "
        "(TVD; positive Δ = conditioning/encoding wins)\n"
    )
    baseline_profile = baseline["held_out_profile_mean"]
    baseline_raw = baseline["held_out_raw_mean"]
    print(
        "baseline (angle) conditioned held-out: "
        f"profile {baseline_profile:.4f}, raw {baseline_raw:.4f}"
    )
    print(f"spin_flip_branch conditioned held-out:   profile {c_prof_ho:.4f}\n")

    rows = []
    for label, betas in (
        ("training", TRAINING_BETAS),
        ("held-out", HELD_OUT_BETAS),
        ("cold", COLD_BETAS),
        ("critical", CRITICAL_BETAS),
    ):
        p, r = _cond_vs_nocond(conditioned, no_condition, betas)
        profile_effect = f"+{p['estimate']:.4f} [{p['lower']:+.4f},{p['upper']:+.4f}]"
        raw_effect = f"+{r['estimate']:.4f} [{r['lower']:+.4f},{r['upper']:+.4f}]"
        rows.append(
            f"| conditioned vs no-condition ({label}) | "
            f"profile {profile_effect} | raw {raw_effect} |"
        )
    p, r = _cond_vs_mixture(conditioned, references)
    profile_effect = f"+{p['estimate']:.4f} [{p['lower']:+.4f},{p['upper']:+.4f}]"
    raw_effect = f"+{r['estimate']:.4f} [{r['lower']:+.4f},{r['upper']:+.4f}]"
    rows.append(
        "| conditioned vs adjacent mixture (held-out) | "
        f"profile {profile_effect} | raw {raw_effect} |"
    )
    p, r = _cond_vs_reference(conditioned, references)
    profile_effect = f"+{p['estimate']:.4f} [{p['lower']:+.4f},{p['upper']:+.4f}]"
    raw_effect = f"+{r['estimate']:.4f} [{r['lower']:+.4f},{r['upper']:+.4f}]"
    rows.append(
        "| conditioned vs single-temp reference (training) | "
        f"profile {profile_effect} | raw {raw_effect} |"
    )
    print("\n| comparison | profile TVD effect | raw TVD effect |")
    print("|---|---|---|")
    for row in rows:
        print(row)

    payload = {
        "baseline": baseline,
        "spin_flip_conditioned_held_out_profile": c_prof_ho,
        "comparisons": rows,
    }
    out = OUT_DIR / "summary.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--family", choices=sorted(FAMILIES), required=True)
    run_p.add_argument("--seeds", default=None)
    run_p.add_argument("--updates", type=int, default=None)
    run_p.add_argument("--lightweight", action="store_true")
    run_p.set_defaults(func=run)
    rep_p = sub.add_parser("report")
    rep_p.set_defaults(func=report)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

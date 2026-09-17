"""Performance harness for the `iqp-cond` library.

Subcommands:

    step       jitted loss+gradient (fixed bank, host rebuild, forward only)
    validate   one fixed-bank validation pass
    e2e        full `train()` run at the shipped defaults
    memory     peak RSS of one case, in a subprocess
    all        everything above at the small and production sizes

Every timing is a median over warm, fully blocking calls; memory is
`ru_maxrss` of a dedicated subprocess, so it includes the JAX runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from common import PYTHON, REPO, block, problem, save, timeit
from iqp_cond.loss import banked_mmd, operator_bank, sampled_mmd

import iqp_cond as iqc

TRAIN = iqc.TrainingConfig()
PRODUCTION = dict(n_visible=12, n_conditions=5, n_targets=2048)
SMALL = dict(n_visible=8, n_conditions=3, n_targets=1024)
PAPER = dict(
    n_visible=12,
    n_conditions=10,
    n_ancillas=10,
    n_targets=2048,
    family="spin_flip",
)


def _losses(problem_case: dict):
    """Build the two training objectives (fresh masks, fixed bank) on one problem."""
    ansatz, encoders, targets = problem(**problem_case)
    config = TRAIN
    n_visible = problem_case["n_visible"]
    bank = operator_bank(
        config.sigmas,
        config.n_ops,
        problem_case["n_conditions"],
        n_visible,
        seed=0,
    )
    params = iqc.initialize_parameters(ansatz.n_parameters, 0)
    generators = jnp.asarray(ansatz.generators)
    batch = jnp.asarray(targets[:, : config.batch_size])

    def fresh_loss(params, target_batch, key):
        return sampled_mmd(
            params,
            generators,
            encoders,
            target_batch,
            key,
            sigmas=config.sigmas,
            n_ops=config.n_ops,
            n_samples=config.n_iqp_samples,
        ).loss

    def banked_loss(params, target_batch, key):
        # Mirrors the trainer's per-step path: shared reference samples.
        return banked_mmd(
            params,
            generators,
            encoders,
            target_batch,
            bank,
            config.n_iqp_samples,
            key,
            shared_samples=True,
        ).loss

    return {
        "ansatz": ansatz,
        "encoders": encoders,
        "targets": targets,
        "bank": bank,
        "params": params,
        "generators": generators,
        "batch": batch,
        "fresh_loss": fresh_loss,
        "banked_loss": banked_loss,
    }


def bench_step(case: dict) -> dict:
    context = _losses(case)
    config = TRAIN
    key = jax.random.PRNGKey(0)
    arms = {}
    for name in ("fresh_loss", "banked_loss"):
        loss = context[name]
        value_and_grad = jax.jit(jax.value_and_grad(loss))
        forward = jax.jit(loss)
        arms[name] = {
            "step_loss_grad": timeit(
                lambda fn=value_and_grad: fn(context["params"], context["batch"], key),
            ),
            "step_forward": timeit(
                lambda fn=forward: fn(context["params"], context["batch"], key),
            ),
        }
    out = {
        "case": case,
        **arms,
        "bank_build": timeit(
            lambda: operator_bank(
                config.sigmas,
                config.n_ops,
                case["n_conditions"],
                case["n_visible"],
                seed=0,
            ),
            warmup=1,
            repeats=3,
        ),
    }
    return out


def bench_validate(case: dict) -> dict:
    context = _losses(case)
    config = TRAIN
    validation_targets = jnp.asarray(context["targets"])
    validation_bank = operator_bank(
        config.sigmas,
        config.validation_n_ops,
        case["n_conditions"],
        case["n_visible"],
        seed=1,
    )

    def validation(params):
        return banked_mmd(
            params,
            context["generators"],
            context["encoders"],
            validation_targets,
            validation_bank,
            config.validation_n_iqp_samples,
            jax.random.PRNGKey(2),
        ).loss

    return {
        "case": case,
        "validation_pass": timeit(jax.jit(validation), context["params"]),
    }


def bench_e2e(case: dict, steps: int, refresh: str = iqc.STEP) -> dict:
    ansatz, encoders, targets = problem(**case)
    config = iqc.TrainingConfig(steps=steps)
    validation = targets[:, : max(1, targets.shape[1] // 4)]
    start = time.perf_counter()
    result = iqc.train(
        ansatz,
        encoders,
        targets,
        validation,
        seed=0,
        config=config,
        mask_refresh=refresh,
    )
    wall = time.perf_counter() - start
    return {
        "case": case,
        "refresh": refresh,
        "steps": steps,
        "wall_s": wall,
        "per_step_ms": 1e3 * wall / steps,
        "n_banks": int(result.n_banks),
        "best_validation_loss": float(result.best_validation_loss),
    }


def _memory_case(name: str, case: dict) -> dict:
    import resource

    context = _losses(case)
    config = TRAIN
    value_and_grad = jax.jit(jax.value_and_grad(context["banked_loss"]))

    if name == "validate":
        validation_bank = operator_bank(
            config.sigmas,
            config.validation_n_ops,
            case["n_conditions"],
            case["n_visible"],
            seed=1,
        )

        def call(params):
            return banked_mmd(
                params,
                context["generators"],
                context["encoders"],
                jnp.asarray(context["targets"]),
                validation_bank,
                config.validation_n_iqp_samples,
                jax.random.PRNGKey(2),
            ).loss

        fn = jax.jit(call)
        args = (context["params"],)
    elif name == "jax":
        fn = jax.jit(jax.value_and_grad(context["fresh_loss"]))
        args = (context["params"], context["batch"], jax.random.PRNGKey(0))
    else:
        fn = value_and_grad
        args = (context["params"], context["batch"], jax.random.PRNGKey(0))

    start = time.perf_counter()
    block(fn(*args))
    compile_s = time.perf_counter() - start
    # ru_maxrss is KiB on Linux and bytes on macOS.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
    return {
        "case": case,
        "mode": name,
        "compile_and_first_call_s": compile_s,
        "peak_rss_mib": rss / divisor,
    }


def memory(cases: list[tuple[str, dict]]) -> list[dict]:
    results = []
    for name, case in cases:
        payload = {"name": name, "case": case}
        script = Path(__file__).resolve()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO / "packages" / "iqp-cond" / "src")
        proc = subprocess.run(
            [
                str(PYTHON),
                str(script),
                "memory-case",
                name,
                json.dumps(case),
            ],
            capture_output=True,
            text=True,
            cwd=str(script.parent),
            env=env,
            check=True,
        )
        results.append({**payload, **json.loads(proc.stdout)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["step", "validate", "e2e", "memory", "all", "memory-case"],
    )
    parser.add_argument("mode", nargs="?", default="train")
    parser.add_argument("case", nargs="?", default="{}")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--refresh", choices=[iqc.STEP, iqc.CACHE], default=iqc.STEP)
    args = parser.parse_args()

    if args.command == "memory-case":
        print(json.dumps(_memory_case(args.mode, json.loads(args.case))))
        return

    payload: dict = {
        "label": args.label,
        "config": {
            "steps": TRAIN.steps,
            "batch_size": TRAIN.batch_size,
            "sigmas": TRAIN.sigmas,
            "n_ops": TRAIN.n_ops,
            "n_iqp_samples": TRAIN.n_iqp_samples,
            "validation_n_ops": TRAIN.validation_n_ops,
            "validation_n_iqp_samples": TRAIN.validation_n_iqp_samples,
        },
    }

    if args.command in ("step", "all"):
        payload["step_production"] = bench_step(PRODUCTION)
        payload["step_small"] = bench_step(SMALL)
    if args.command in ("validate", "all"):
        payload["validate_production"] = bench_validate(PRODUCTION)
    if args.command in ("e2e", "all"):
        payload["e2e_production"] = bench_e2e(PRODUCTION, args.steps, args.refresh)
        payload["e2e_small"] = bench_e2e(SMALL, args.steps, args.refresh)
    if args.command in ("memory", "all"):
        payload["memory"] = memory(
            [
                ("train", PRODUCTION),
                ("jax", PRODUCTION),
                ("validate", PRODUCTION),
                ("paper", PAPER),
            ],
        )

    name = f"{args.label}-{args.command}.json"
    save(name, payload)


if __name__ == "__main__":
    main()

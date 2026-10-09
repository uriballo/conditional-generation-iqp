"""Single entry point for the paper's experiments, searches and figures."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _execute(arguments):
    from paper_experiments.runtime import execute_job

    job, output, smoke = arguments
    return execute_job(job, output, smoke=smoke)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("reported", "hpo", "figures"), default="reported")
    p.add_argument("--hpo-phase", choices=("random", "tpe"), default="random")
    p.add_argument(
        "--experiment",
        choices=(
            "all",
            "gaussian",
            "curie-weiss",
            "curie-weiss-followup",
            "curie-weiss-historical",
            "gaussian2d-phase",
            "scaling",
            "switching16",
        ),
        default="all",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("results/paper"))
    p.add_argument(
        "--data-root",
        type=Path,
        help="compact data directory, or completed reported/HPO output",
    )
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be positive")
    cw_choices = ("all", "curie-weiss", "curie-weiss-followup")
    if args.hpo_phase == "tpe" and not (
        args.mode == "hpo" and args.experiment in cw_choices
    ):
        p.error("--hpo-phase tpe applies only to Curie-Weiss HPO")
    if args.mode == "hpo":
        if args.experiment not in cw_choices:
            p.error("HPO applies only to the current Curie-Weiss campaign")
        from paper_experiments.cw_followup_hpo import protocol

        if args.hpo_phase == "tpe":
            from paper_experiments.cw_followup_tpe import run_all
        else:
            from paper_experiments.cw_followup_hpo import run_all
    if args.mode == "reported":
        if args.experiment in ("curie-weiss", "curie-weiss-followup"):
            p.error("Curie-Weiss final configurations are pending; use --mode hpo")
        if args.experiment in (
            "gaussian2d-phase",
            "switching16",
            "curie-weiss-historical",
        ):
            p.error(
                "this experiment supports --mode figures; frozen training sources "
                "remain in the local research archives"
            )
        from paper_experiments.campaign import inventory
    if args.dry_run:
        if args.mode == "reported":
            plan = inventory(args.experiment, smoke=args.smoke)
        elif args.mode == "hpo":
            plan = protocol(args.smoke)
            if args.hpo_phase == "tpe":
                plan = {"phase": "tpe", "protocol": plan}
        else:
            plan = dict(
                data=str(args.data_root or Path("paper/data")),
                output=str(args.output / "figures"),
                experiment=args.experiment,
            )
        print(json.dumps(plan, indent=2))
        return 0
    namespace = args.output / (
        "smoke" if args.smoke else "reported" if args.mode == "reported" else args.mode
    )
    if args.mode == "figures":
        from paper_experiments.figures import render_all

        paths = render_all(
            args.data_root or Path(__file__).parent / "paper/data",
            namespace,
            experiment=(
                "curie-weiss"
                if args.experiment == "curie-weiss-followup"
                else args.experiment
            ),
        )
        print("\n".join(str(path) for path in paths))
        return 0
    if args.mode == "hpo":
        run_all(
            namespace / "hpo" if args.smoke else namespace,
            smoke=args.smoke,
            workers=args.workers,
        )
        message = (
            "TPE HPO complete, paused for review"
            if args.hpo_phase == "tpe"
            else "Random HPO complete, paused for review"
        )
        print(f"{message}: {namespace}")
        return 0
    jobs = inventory(args.experiment, smoke=args.smoke)
    arguments = [(job, namespace, args.smoke) for job in jobs]
    if args.workers == 1:
        records = [_execute(a) for a in arguments]
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            records = list(pool.map(_execute, arguments))
    namespace.mkdir(parents=True, exist_ok=True)
    (namespace / "reported-results.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(f"Completed {len(records)} paper jobs: {namespace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

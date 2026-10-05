"""Single entry point for the paper's experiments, searches and figures."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from paper_experiments.campaign import inventory
from paper_experiments.runtime import execute_job


def _execute(arguments):
    job, output, smoke = arguments
    return execute_job(job, output, smoke=smoke)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("reported", "hpo", "figures"), default="reported")
    p.add_argument(
        "--experiment",
        choices=("all", "gaussian", "curie-weiss", "scaling", "symmetry"),
        default="all",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("results/paper"))
    p.add_argument(
        "--data-root",
        type=Path,
        help="compact data directory, or completed reported-run output",
    )
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("--workers must be positive")
    if args.mode == "hpo" and args.experiment not in ("all", "curie-weiss"):
        p.error("HPO applies only to curie-weiss")
    if args.dry_run:
        if args.mode == "reported":
            plan = inventory(args.experiment, smoke=args.smoke)
        elif args.mode == "hpo":
            from paper_experiments.hpo import protocol

            plan = protocol(args.smoke)
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
            experiment=args.experiment,
        )
        print("\n".join(str(path) for path in paths))
        return 0
    if args.mode == "hpo":
        from paper_experiments.hpo import run_all

        run_all(
            namespace / "hpo" if args.smoke else namespace,
            smoke=args.smoke,
            workers=args.workers,
        )
        print(f"HPO complete: {namespace}")
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

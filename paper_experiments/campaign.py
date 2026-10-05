"""Job inventories for the manuscript's reported experiments."""

from .runtime import (
    BASELINE_SEEDS,
    GAUSSIAN_SEEDS,
    REPORT_BETAS,
    SCALING_SEEDS,
    SCALING_SIZES,
    SHARED_SEEDS,
    build_job,
)


def inventory(experiment="all", *, smoke=False):
    jobs = []
    if experiment in ("all", "gaussian"):
        jobs.extend(
            build_job("gaussian", s)
            for s in (GAUSSIAN_SEEDS[:1] if smoke else GAUSSIAN_SEEDS)
        )
    if experiment in ("all", "curie-weiss"):
        jobs.extend(
            build_job("shared", s)
            for s in (SHARED_SEEDS[:1] if smoke else SHARED_SEEDS)
        )
        for kind in ("conditional", "unconditional"):
            jobs.extend(
                build_job(kind, s, beta=b)
                for s in (BASELINE_SEEDS[:1] if smoke else BASELINE_SEEDS)
                for b in REPORT_BETAS
            )
    if experiment in ("all", "scaling"):
        jobs.extend(
            build_job("scaling", s, size=n)
            for n in (SCALING_SIZES[:1] if smoke else SCALING_SIZES)
            for s in (SCALING_SEEDS[:1] if smoke else SCALING_SEEDS)
        )
    if experiment in ("all", "symmetry"):
        jobs.append(build_job("symmetry", 0))
    return jobs

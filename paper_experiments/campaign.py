"""Job inventories for the manuscript's reported experiments."""

from .runtime import (
    GAUSSIAN_SEEDS,
    SCALING_SEEDS,
    SCALING_SIZES,
    build_job,
)


def inventory(experiment="all", *, smoke=False):
    if experiment not in ("all", "gaussian", "scaling"):
        raise ValueError("unsupported paper experiment")
    jobs = []
    if experiment in ("all", "gaussian"):
        jobs.extend(
            build_job("gaussian", s)
            for s in (GAUSSIAN_SEEDS[:1] if smoke else GAUSSIAN_SEEDS)
        )
    if experiment in ("all", "scaling"):
        jobs.extend(
            build_job("scaling", s, size=n)
            for n in (SCALING_SIZES[:1] if smoke else SCALING_SIZES)
            for s in (SCALING_SEEDS[:1] if smoke else SCALING_SEEDS)
        )
    return jobs

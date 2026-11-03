"""Values of the 64-qubit lattice vs all-to-all table from fit receipts and evaluations.

Usage: python -m paper_experiments.scaling_locality_table FITS_DIR, where FITS_DIR holds
n64-sSEED.json and n64-sSEED-evaluation.json for seeds 100-102.
"""

import json
import sys
from pathlib import Path

import numpy as np

from paper_experiments.target import curie_weiss_target


def exact_m2(n, beta):
    t = curie_weiss_target(n, beta)
    return float(t.shell_probabilities @ t.magnetizations**2)


fits = Path(sys.argv[1])
params, mmd, delta = [], [], []
for seed in (100, 101, 102):
    fit = json.loads((fits / f"n64-s{seed}.json").read_text())
    ev = json.loads((fits / f"n64-s{seed}-evaluation.json").read_text())
    selected = next(t for t in ev["trajectory"] if t["step"] == ev["selected_step"])
    betas = np.asarray(ev["betas"])
    params.append(len(fit["params"]))
    mmd.append(float(np.mean(selected["per_bank_per_condition"])))
    delta.append(np.asarray(ev["m2"]) - [exact_m2(64, b) for b in betas])
d = np.mean(delta, axis=0)


def at(beta):
    return float(d[np.argmin(abs(betas - beta))])


window = (betas >= 0.8 - 1e-9) & (betas <= 1.3 + 1e-9)
table = {
    "trainable_parameters": params[0],
    "test_mmd2": float(np.mean(mmd)),
    "test_mmd2_per_seed": mmd,
    **{f"delta_m2_beta_{b}": at(b) for b in (0.8, 1.0, 1.2, 1.7, 2.0)},
    "mean_abs_delta_m2_0.8_to_1.3": float(np.mean(abs(d[window]))),
    "delta_m2_per_seed": {
        b: [float(x[np.argmin(abs(betas - b))]) for x in delta]
        for b in (0.8, 1.0, 1.2, 1.7, 2.0)
    },
    "note": (
        "means over seeds 100-102; mean |delta| is over the seed-mean curve"
        " at the 11 dense temperatures in [0.8, 1.3]"
    ),
}
print(json.dumps(table, indent=2))

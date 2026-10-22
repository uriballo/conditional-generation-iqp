"""Render the two target families of Sec. 4 from their closed forms.

uv run python -m paper_experiments.target_families --output results/figures
"""

import argparse
from math import comb
from pathlib import Path

import numpy as np

from paper_experiments.figures import COLORS, MM, _save, plt

N = 16


def hamming_profile(u, v, n=N):
    k = np.arange(n + 1)
    binom = np.array([comb(n, j) for j in k], dtype=float)
    w = binom * np.exp(
        -((k - 8 - 3 * u) ** 2) / (8 * (1 + 0.2 * v) ** 2) + (k - 8) ** 2 / 8
    )
    return k, w / w.sum()


def cw_profile(beta, n=N):
    k = np.arange(n + 1)
    binom = np.array([comb(n, j) for j in k], dtype=float)
    w = binom * np.exp(beta * (2 * k - n) ** 2 / (2 * n))
    return (2 * k - n) / n, w / w.sum()


def render(output):
    fig, (a, b) = plt.subplots(1, 2, figsize=(170 * MM, 68 * MM), layout="constrained")
    # Reference, mean shift, width change and both combined.
    for i, (u, v) in enumerate(((0, 0), (1, 0), (0, -1), (-1, 1))):
        k, rho = hamming_profile(u, v)
        a.plot(k, rho, "-o", color=COLORS[i], label=rf"$u={u}$, $v={v}$")
    a.set_xlabel(r"Hamming weight $k$")
    a.set_ylabel(r"$\rho_{u,v}(k)$")
    a.set_title("(a) Hamming-weight family", loc="left")
    a.set_xticks(range(0, N + 1, 4))
    a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)

    for i, beta in enumerate((0.4, 1.0, 2.0)):
        m, p = cw_profile(beta)
        b.plot(m, p, "-o", color=COLORS[i], label=rf"$\beta={beta:g}$")
    b.set_xlabel(r"Magnetization $m$")
    b.set_ylabel("Probability mass")
    b.set_title("(b) Curie--Weiss model", loc="left")
    b.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    return _save(fig, output, "target_families")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in render(args.output):
        print(path)


if __name__ == "__main__":
    main()

"""Figures and tables for the IQM Emerald deployment of the scaling models.

Reads the bundle in data/hardware/iqm-emerald: the calibration snapshot,
the deployed lattice fits with their classical <m^2> estimates, exact
noiseless profiles of the 30-qubit fit (hardware_noiseless.py), and the raw
counts of every hardware job. Jobs with the same fit, physical layout and
calibration are pooled. Counts keys index bits from logical qubit 0.

Run: python -m paper_experiments.hardware_report --output DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from .ansatz import rectangular_lattice_ansatz
from .figures import COLORS, MM, _save, plt
from .scaling_hpo import schedule

BUNDLE = Path(__file__).resolve().parents[1] / "data/hardware/iqm-emerald"


def load_runs(bundle=BUNDLE):
    groups = {}
    for path in sorted((bundle / "runs").glob("*.json")):
        meta = json.loads(path.read_text())
        key = (meta["size"], meta["seed"], tuple(meta["layout"]))
        group = groups.setdefault(key, dict(meta=meta, jobs=[], counts={}))
        group["jobs"].append(meta["job_id"])
        for rec in meta["runs"]:
            merged = group["counts"].setdefault(rec["beta"], {})
            for k, v in rec["counts"].items():
                merged[k] = merged.get(k, 0) + v
    return [groups[k] for k in sorted(groups)]


def target_profile(n, beta):
    k = np.arange(n + 1)
    logw = np.array(
        [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in k]
    ) + beta * (2 * k - n) ** 2 / (2 * n)
    w = np.exp(logw - logw.max())
    return w / w.sum()


def model_probabilities(fit, beta):
    """Exact output distribution of a lattice parity-encoder fit (n <= 20)."""
    n = fit["size"]
    if n > 20:
        raise ValueError("exact probabilities require n <= 20")
    t = float(schedule(fit["family"], fit["hyperparameters"], [beta], n)[0])
    bits = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int64)
    leaves = bits[:, :-1].sum(1)
    amp = np.cos(t / 2) ** (n - 1 - leaves) * (-1j * np.sin(t / 2)) ** leaves
    amp = np.where(bits[:, -1] == leaves % 2, amp, 0)
    gens = np.asarray(rectangular_lattice_ansatz(n).generators, dtype=np.int64)
    chars = 1 - 2 * ((bits @ gens.T) % 2)
    psi = amp * np.exp(1j * chars @ np.asarray(fit["params"]))
    for q in range(n):
        psi = psi.reshape(-1, 2, 1 << q)
        psi = np.stack([psi[:, 0] + psi[:, 1], psi[:, 0] - psi[:, 1]], 1).reshape(-1)
    return np.abs(psi) ** 2 / (1 << n)


def counts_profile(counts, n):
    prof = np.zeros(n + 1)
    for key, v in counts.items():
        prof[key.count("1")] += v
    return prof / prof.sum()


def m2_from_counts(counts, n):
    k = np.array([key.count("1") for key in counts])
    w = np.array(list(counts.values()), dtype=float)
    m2 = ((2 * k - n) / n) ** 2
    mean = float(w @ m2 / w.sum())
    se = float(np.sqrt(w @ (m2 - mean) ** 2 / (w.sum() - 1) / w.sum()))
    return mean, se


def calibration(bundle=BUNDLE):
    path = next(bundle.glob("calibration-*.json"))
    data = json.loads(path.read_text())
    qubit, cz = {}, {}
    fields = {
        r"characterization\.model\.(QB\d+)\.t1_time$": "t1",
        r"characterization\.model\.(QB\d+)\.t2_time$": "t2",
        r"characterization\.model\.(QB\d+)\.t2_echo_time$": "t2e",
        r"metrics\.ssro\.measure\.constant\.(QB\d+)\.fidelity$": "readout",
        r"metrics\.ssro\.measure\.constant\.(QB\d+)\.error_0_to_1$": "e01",
        r"metrics\.ssro\.measure\.constant\.(QB\d+)\.error_1_to_0$": "e10",
        r"metrics\.rb\.prx\.drag_crf_sx\.(QB\d+)\.fidelity": "prx",
    }
    for o in data["observations"]:
        if o.get("invalid"):
            continue
        for pattern, name in fields.items():
            m = re.match(pattern, o["dut_field"])
            if m:
                qubit.setdefault(m.group(1), {})[name] = float(o["value"])
        m = re.match(
            r"metrics\.irb\.cz\.[^.]+\.(QB\d+)__(QB\d+)\.fidelity", o["dut_field"]
        )
        if m:
            cz[frozenset(m.groups())] = float(o["value"])
    return path, data["created_timestamp"], qubit, cz


def _lattice_couplers(n, layout_qubits):
    from .ansatz import open_rectangle_shape

    h, w = open_rectangle_shape(n)
    edges = [(r * w + c, r * w + c + 1) for r in range(h) for c in range(w - 1)]
    edges += [(r * w + c, (r + 1) * w + c) for r in range(h - 1) for c in range(w)]
    return [(layout_qubits[a], layout_qubits[b]) for a, b in sorted(edges)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    groups = load_runs(args.bundle)
    cal_path, cal_time, qcal, czcal = calibration(args.bundle)
    rows, summary = [], []
    betas = sorted(groups[0]["counts"])
    fig, ax = plt.subplots(
        len(groups),
        len(betas),
        figsize=(170 * MM, 78 * MM),
        layout="constrained",
        sharex=True,
        sharey="row",
    )
    for r, group in enumerate(groups):
        meta = group["meta"]
        n, seed = meta["size"], meta["seed"]
        fit = json.loads((args.bundle / "models" / f"n{n}-s{seed}.json").read_text())
        obs = json.loads(
            (args.bundle / "models" / f"n{n}-s{seed}-observables.json").read_text()
        )
        # Exact profiles beyond dense simulation come from hardware_noiseless.
        stored_path = args.bundle / "models" / f"n{n}-s{seed}-noiseless-profiles.json"
        stored = None
        if stored_path.exists():
            stored = json.loads(stored_path.read_text())
            if (
                stored["fit_sha256"]
                != hashlib.sha256(
                    (args.bundle / "models" / f"n{n}-s{seed}.json").read_bytes()
                ).hexdigest()
            ):
                raise ValueError(f"{stored_path.name} does not match its fit")
        dense = np.asarray(obs["betas"])
        bank = 1 / n + (1 - 1 / n) * np.asarray(obs["pair_means"])
        m = (2 * np.arange(n + 1) - n) / n
        for c, beta in enumerate(betas):
            counts = group["counts"][beta]
            shots = sum(counts.values())
            prof = counts_profile(counts, n)
            tp = target_profile(n, beta)
            hw, hw_se = m2_from_counts(counts, n)
            i = int(np.argmin(abs(dense - beta)))
            row = dict(
                n=n,
                seed=seed,
                beta=beta,
                shots=shots,
                jobs=group["jobs"],
                m2_hardware=hw,
                m2_hardware_se=hw_se,
                m2_model=float(obs["m2"][i]),
                m2_model_se=float(bank[:, i].std(ddof=1) / np.sqrt(len(bank))),
                m2_target=float(tp @ m**2),
                profile_tvd_hardware_target=float(0.5 * abs(prof - tp).sum()),
            )
            a = ax[r, c]
            a.bar(
                m, prof, width=1.8 / n, color=COLORS[0], alpha=0.55, label="IQM Emerald"
            )
            a.plot(m, tp, "k--", lw=1.0, label="Target")
            pm = None
            if stored is not None:
                pm = np.asarray(stored["profiles"][str(beta)]["profile"])
            elif n <= 20:
                q = model_probabilities(fit, beta)
                ones = np.array([bin(k).count("1") for k in range(1 << n)])
                pm = np.bincount(ones, q, n + 1)
            if pm is not None:
                a.plot(m, pm, color=COLORS[1], lw=1.2, label="Noiseless model")
                row.update(
                    profile_tvd_hardware_model=float(0.5 * abs(prof - pm).sum()),
                    profile_tvd_model_target=float(0.5 * abs(pm - tp).sum()),
                )
            rows.append(row)
            if r == 0:
                a.set_title(rf"$\beta={beta:g}$", fontsize=9)
            a.text(
                0.5,
                0.99,
                f"TVD {row['profile_tvd_hardware_target']:.2f}",
                transform=a.transAxes,
                va="top",
                ha="center",
                fontsize=7,
            )
            a.spines[["top", "right"]].set_visible(False)
            if c == 0:
                a.set_ylabel(f"{n} qubits", fontsize=9)
        summary.append(
            dict(
                n=n,
                seed=seed,
                jobs=group["jobs"],
                layout_qubits=meta["layout_qubits"],
                isa_cz=meta["runs"][0]["isa_cz"],
                isa_depth=meta["runs"][0]["isa_depth"],
                encoder_angles={
                    str(b["beta"]): b["encoder_angle"] for b in meta["runs"]
                },
            )
        )
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.supxlabel(r"Magnetization $m$", fontsize=9)
    fig.supylabel("Probability", fontsize=9)
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=8)
    paths = _save(fig, args.output, "iqm_emerald_profiles")

    def pm(x, s):
        if round(s, 3) == 0:
            return f"${x:.4f}\\pm{s:.4f}$"
        return f"${x:.3f}\\pm{s:.3f}$"

    m2_lines = [
        f"{x['n']} & {x['beta']:g} & {pm(x['m2_hardware'], x['m2_hardware_se'])}"
        f" & {pm(x['m2_model'], x['m2_model_se'])} & {x['m2_target']:.3f}"
        f" & {x['profile_tvd_hardware_target']:.3f} \\\\"
        for x in rows
    ]
    (args.output / "iqm_emerald_m2.tex").write_text(
        "\\begin{tabular}{rrcccc}\n\\toprule\n"
        "$n$ & $\\beta$ & Hardware $\\langle m^2\\rangle$"
        " & Model $\\langle m^2\\rangle$ & Target & Profile TVD \\\\\n\\midrule\n"
        + "\n".join(
            line + ("\n\\midrule" if k == len(betas) - 1 else "")
            for k, line in enumerate(m2_lines)
        )
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    # Calibration tables for every qubit and coupler that the largest run uses.
    big = max(summary, key=lambda s: s["n"])
    small = min(summary, key=lambda s: s["n"])
    used16 = set(small["layout_qubits"])
    q_lines = []
    for q in sorted(big["layout_qubits"], key=lambda s: int(s[2:])):
        d = qcal[q]
        mark = "" if q in used16 else "$^{\\dagger}$"
        q_lines.append(
            f"{q}{mark} & {d['t1'] * 1e6:.1f} & {d['t2'] * 1e6:.1f}"
            f" & {d['t2e'] * 1e6:.1f} & {d['prx']:.4f} & {d['readout']:.4f}"
            f" & {d['e01']:.4f} & {d['e10']:.4f} \\\\"
        )
    (args.output / "iqm_emerald_calibration_qubits.tex").write_text(
        "\\begin{tabular}{lrrrcccc}\n\\toprule\n"
        "Qubit & $T_1$ ($\\mu$s) & $T_2^{*}$ ($\\mu$s)"
        " & $T_2^{\\mathrm{echo}}$ ($\\mu$s) & PRX fidelity & Readout fidelity"
        " & $\\epsilon_{0\\to1}$ & $\\epsilon_{1\\to0}$ \\\\\n"
        "\\midrule\n" + "\n".join(q_lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    couplers = _lattice_couplers(big["n"], big["layout_qubits"])
    small_couplers = {
        frozenset(e) for e in _lattice_couplers(small["n"], small["layout_qubits"])
    }
    entries = []
    for a, b in sorted(couplers, key=lambda e: sorted(int(x[2:]) for x in e)):
        a, b = sorted((a, b), key=lambda s: int(s[2:]))
        mark = "" if frozenset((a, b)) in small_couplers else "$^{\\dagger}$"
        entries.append(f"{a}--{b}{mark} & {czcal[frozenset((a, b))]:.4f}")
    half = (len(entries) + 2) // 3
    cols = [entries[i * half : (i + 1) * half] for i in range(3)]
    c_lines = [
        " & ".join(col[k] if k < len(col) else "&" for col in cols) + " \\\\"
        for k in range(half)
    ]
    (args.output / "iqm_emerald_calibration_couplers.tex").write_text(
        "\\begin{tabular}{lclclc}\n\\toprule\n"
        "Coupler & CZ fidelity & Coupler & CZ fidelity & Coupler & CZ fidelity \\\\\n"
        "\\midrule\n" + "\n".join(c_lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    record = dict(
        calibration=cal_path.name,
        calibration_created=cal_time,
        calibration_sha256=hashlib.sha256(cal_path.read_bytes()).hexdigest(),
        runs=summary,
        rows=rows,
        median_used=dict(
            cz=float(np.median([czcal[frozenset(e)] for e in couplers])),
            readout=float(
                np.median([qcal[q]["readout"] for q in big["layout_qubits"]])
            ),
            prx=float(np.median([qcal[q]["prx"] for q in big["layout_qubits"]])),
            t1_us=float(np.median([qcal[q]["t1"] for q in big["layout_qubits"]]) * 1e6),
            t2_us=float(np.median([qcal[q]["t2"] for q in big["layout_qubits"]]) * 1e6),
        ),
    )
    (args.output / "iqm_emerald_summary.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record["median_used"], indent=1))
    for x in rows:
        print(
            {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in x.items()
                if k != "jobs"
            }
        )
    return paths


if __name__ == "__main__":
    main()

# Classical training of conditional IQP Born machines

Code and data for *Classical training of conditional instantaneous quantum
polynomial-time Born machines* by Oriol Balló-Gimbernat, Emiliano Tolotti, Laia
Domingo and Fernando Vilariño.

A conditional IQP Born machine encodes the condition in the input state and
shares one IQP unitary across conditions. The models are trained classically
with an unbiased estimator of a conditional maximum mean discrepancy. This
repository contains the experiment code, the protocol of every experiment, and
the result bundles behind every numerical figure and table of the paper.

## Layout

```
reproduce.py            all experiments, figures and tables in one run
run_experiments.py      figure replay and Curie–Weiss searches
paper_experiments/      experiment code and *_protocol.json files
data/                   result bundles behind the paper's figures and tables
tests/                  test suite
iqp-cond/               iqp-cond library (git submodule)
```

## Installation

Python 3.13 and [uv](https://docs.astral.sh/uv/):

```sh
git clone --recurse-submodules https://github.com/uriballo/conditional-generation-iqp.git
cd conditional-generation-iqp
uv sync --locked --extra hpo
```

## Reproducing the paper

One command trains every model of the paper, in order, and then renders every
figure and table:

```sh
uv run python reproduce.py --output results/reproduce --workers 2
```

It runs the Gaussian-shaped family (conditional IQP and the four diagonal
conditioning variants), the Curie–Weiss parity protocol, the twisted parity
encoder with its per-temperature model and κ = 0 ablation, D-IQP, the classical
baselines at both capacities, lattice scaling at 64–2048 qubits with the
64-qubit all-to-all comparison, and the classical training of the 16- and
30-qubit models deployed on IQM Emerald. The figures and tables are then
rendered from the new bundles into `results/reproduce/figures/`.

Each experiment writes to its own folder under the output root, and an
interrupted run resumes from its completed stages when restarted with the same
command. Campaigns record the hash of the source tree, so do not edit
`paper_experiments/` while a run is in progress. `--only EXPERIMENT` runs one
experiment (`gaussian`, `curie-weiss`, `curie-weiss-twisted`,
`curie-weiss-diagonal`, `curie-weiss-classical`, `scaling`, `hardware` or
`figures`); the hardware fits reuse the scaling search and confirmation.
`--smoke` runs every stage with reduced budgets under `OUTPUT/smoke` and renders
the figures from `data/`, because the reporting code accepts only full
campaigns. The output root must lie under `results/`.

**Hardware.** The device counts are read from `data/hardware/iqm-emerald`, and
the hardware figures and tables come from those runs and the stored deployed
fits. Rerunning the circuits on the device requires an IQM access token and is
not part of the script.

**Runtime.** The original campaigns ran on an AMD Ryzen AI Max+ 395 machine
(Radeon 8060S GPU through ROCm JAX, 128 GB memory). Recorded wall-clock times
were about 16 h for the Curie–Weiss parity protocol (GPU, two workers), 17 h
for the twisted campaigns and ablation (CPU), 14 h for the
classical baselines (GPU, both capacities concurrently), 17 h for lattice
scaling including 7 h for the 2048-qubit fits (GPU), under 1 h for the
Gaussian-shaped family (CPU) and about 20 min for the hardware fits and the
30-qubit noiseless profile. No timing was recorded for D-IQP. Run in sequence,
the full pipeline takes three to four days on comparable hardware. Scaling at
1024 and 2048 qubits needs a GPU in practice, and the exact 30-qubit noiseless
profile needs about 40 GB of memory. A smoke run takes about seven minutes on an Apple M-series laptop CPU. When
`latex` is on the path the figures are typeset with LaTeX, and the classical
replay figures also need the `cm-super` fonts; without `latex` the figures fall
back to Matplotlib's mathtext.

**Comparison with `data/`.** After a full run,

```sh
uv run python reproduce.py --output results/reproduce --check
```

compares every numeric value of the new bundles with the stored ones (relative
tolerance 1e-5) and lists the files that differ. Timings, hashes, paths and
other strings are provenance and are skipped. The seeds and random streams are
fixed, but floating-point reductions can differ across devices and library
versions, so agreement is expected on the original software stack rather than
bit for bit everywhere.

To run a single stage, call its module directly, for example
`uv run python -m paper_experiments.scaling_hpo --help`; `reproduce.py` shows
the arguments used for each stage. Figures from the stored bundles alone:

```sh
uv run python run_experiments.py --mode figures
uv run python -m paper_experiments.hardware_report --output results/hardware
uv run python -m paper_experiments.target_families --output results/figures
```

## Experiments

| Experiment | Code | Protocol | Data |
| --- | --- | --- | --- |
| Gaussian-shaped family, conditional IQP | `gaussian2d_lattice*.py`, `gaussian2d_hpo.py` | `gaussian2d_lattice_protocol.json` | `data/gaussian_lattice/` |
| Gaussian-shaped family, diagonal conditioning | `gaussian2d_diagonal*.py`, `gaussian2d_hpo.py` | `gaussian2d_diagonal*_protocol.json` | `data/gaussian2d-diagonal/`, `data/gaussian2d-diagonal-variants/` |
| Curie–Weiss, C-IQP (P) and per-temperature IQP | `cw_followup_*.py` | `cw_followup_protocol.json` | `data/curie-weiss-followup/` |
| Curie–Weiss, C-IQP (TP) and κ = 0 ablation | `twisted_parity_encoder.py`, `cw_encoder_hpo.py`, `cw_kappa0_control.py` | `cw_twisted*_protocol.json` | `data/curie-weiss-twisted/` |
| Curie–Weiss, D-IQP | `cw_diagonal_fit.py`, `cw_encoder_hpo.py` | `cw_diagonal_k4_protocol.json` | `data/curie-weiss-diagonal/` |
| Curie–Weiss, classical baselines | `cw_classical_*.py` | `cw_classical_protocol.json` | `data/curie-weiss-classical/` |
| Lattice scaling, 64–2048 qubits | `scaling_hpo.py`, `scaling_observables.py`, `scaling_score_study.py` | `scaling_hpo_protocol.json` | `data/scaling/` |
| IQM Emerald, 16 and 30 qubits | `scaling_hpo.py`, `hardware_noiseless.py`, `hardware_report.py` | — | `data/hardware/` |

## Training

Each stage writes to the given output root and resumes only with the same
configuration and source. Search stages stop after each phase. `--smoke` (or
`--stage smoke`) runs a reduced pipeline check.

**Gaussian-shaped family.** Five campaigns under the Curie–Weiss search
protocol (60 random and 60 TPE proposals, confirmation with five repeats and
final fits with ten repeats, each with three starts), with fit streams and
target pools paired across campaigns: input conditioning (`input`), the affine
diagonal baseline (`diagonal`) and its variants (`diagonal_quadratic`,
`diagonal_complete`, `diagonal_complete_quadratic`):

```sh
for phase in random tpe confirmation final; do
  uv run python -m paper_experiments.gaussian2d_hpo --campaign input --root results/gaussian/input --phase $phase --workers 6
done
uv run python -m paper_experiments.gaussian2d_lattice_report --campaign results/gaussian/input --output data/gaussian_lattice
uv run python -m paper_experiments.gaussian2d_diagonal_report --campaign results/gaussian/diagonal --output data/gaussian2d-diagonal
uv run python -m paper_experiments.gaussian2d_diagonal_report --five-way --output data/gaussian2d-diagonal-variants/five_way.json
```

The variant bundles go to `data/gaussian2d-diagonal-variants/{lattice-quadratic,complete-affine,complete-quadratic}`.

**Curie–Weiss, parity and per-temperature models.** 60 random and 60 TPE
proposals per family, then confirmation (five repeats) and final fits (ten
repeats), each with three starts:

```sh
uv run python run_experiments.py --mode hpo --experiment curie-weiss --hpo-phase random --workers 2 --output results/cw
uv run python run_experiments.py --mode hpo --experiment curie-weiss --hpo-phase tpe --workers 2 --output results/cw
uv run python run_experiments.py --mode hpo --experiment curie-weiss --hpo-phase reporting --workers 2 --output results/cw
```

**Curie–Weiss, twisted parity encoder and D-IQP.** Single-family campaigns under
the same protocol, with fit streams paired to the parity fits:

```sh
for phase in random tpe confirmation final; do
  uv run python -m paper_experiments.cw_encoder_hpo --campaign twisted --root results/cw-twisted --phase $phase
done
```

`--campaign twisted_independent` gives the per-temperature twisted model and
`--campaign diagonal_k4` gives D-IQP. Bundle and ablation:

```sh
uv run python -m paper_experiments.cw_twisted_bundle results/cw-twisted results/cw-twisted-independent --output results/cw-twisted-bundle
uv run python -m paper_experiments.cw_kappa0_control run results/cw-kappa0 --winner results/cw-twisted/final/campaign.json
uv run python -m paper_experiments.cw_kappa0_control bundle results/cw-kappa0 --output results/cw-twisted-bundle/kappa0_control.json
```

**Curie–Weiss, classical baselines.** For each capacity (`small`, `large`):

```sh
for phase in random tpe confirmation final; do
  uv run python run_experiments.py --mode hpo --experiment curie-weiss-classical \
    --classical-capacity small --hpo-phase $phase --output results/cw-classical
done
uv run python run_experiments.py --mode figures --experiment curie-weiss-classical \
  --data-root results/cw-classical/hpo/small --output results/cw-classical-replay
```

**Lattice scaling.** Search at 64 and 256 qubits, confirmation, final fits,
observables and bundle:

```sh
uv run python -m paper_experiments.scaling_hpo run results/scaling/search
uv run python -m paper_experiments.scaling_hpo report results/scaling/search
uv run python -m paper_experiments.scaling_hpo confirm results/scaling/confirm --search results/scaling/search
uv run python -m paper_experiments.scaling_hpo report-confirm results/scaling/confirm
uv run python -m paper_experiments.scaling_hpo final results/scaling/final --search results/scaling/search --confirmation results/scaling/confirm --sizes 64 128 256 512 1024 2048
uv run python -m paper_experiments.scaling_hpo evaluate results/scaling/evaluations --fits results/scaling/final
uv run python -m paper_experiments.scaling_hpo bundle results/scaling/bundle --fits results/scaling/final --evaluations results/scaling/evaluations
```

The 64-qubit all-to-all comparison of the paper uses `--ansatz complete
--sizes 64` with the training budget of the lattice fits fixed by `--budget 512
1024`; its commands and results are in
`data/scaling/locality/all-to-all-64-budget512/`. The earlier comparison in
`data/scaling/locality/all-to-all-64/`, whose search also varied the budget, is
kept for reference. The sixteen-qubit selection-score study is described in
`data/scaling/score_study/`.

## Tests

```sh
uv run pytest
uv run ruff check paper_experiments tests run_experiments.py reproduce.py
```

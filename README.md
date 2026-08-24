# Conditional IQP generation

This repository studies classically trainable conditional generation with IQP
circuits. A condition enters only through a classically tractable input state.
The Gaussian campaign uses the product-state encoding

\[
|\psi(\boldsymbol{\alpha})\rangle
=\bigotimes_i R_Y(\alpha_i)|0\rangle,
\]

while every condition shares the same IQP parameters. The Curie--Weiss v3
campaign instead uses a spin-flip two-branch input whose basis sampling and
amplitude ratios remain exact and tractable. The implementation extends the
probabilistic expectation-estimation strategy of den Nest to these input
families and trains with unbiased, class-balanced Conditional MMD.

The current manuscript combines the completed `gaussian_foundations_v1` study
with the completed `curie_weiss_m5_foundations_v3` extension under
the title **“Conditional Generation with Classically Trainable IQP Circuits.”**
It defines fixed-circuit input-state conditioning, narrows “classically
trainable” to selected Pauli moments, Conditional MMD, and its gradients, and
reports complete performance surfaces without an empirical pass/fail
threshold. The campaigns contain 1,020 Gaussian and 80 fixed-$r=5$
Curie--Weiss atomic fits. Positive, mixed, and negative comparisons all remain
visible.

Earlier manuscript lines, manifests, evidence, and PDFs remain immutable under
`archive/` and their original result stores. They are repository history, not
source material for the current manuscript.

## Project map

- `src/iqp_cond/simulator.py`: product-state Pauli-\(Z\) estimator.
- `src/iqp_cond/encoding.py`: fixed angle and noncommuting reupload encoders.
- `src/iqp_cond/mmd.py`: unbiased multiscale Conditional MMD.
- `src/iqp_cond/targets.py`: calibrated Gaussian Hamming-weight targets.
- `src/iqp_cond/ansatz.py`: heavy-hex, lattice, and complete generators.
- `src/iqp_cond/training.py`: matched-seed JAX/Optax training.
- `src/iqp_cond/exact.py`: exact 19-qubit evaluation.
- `src/iqp_cond/gaussian_protocol.py`: typed manifest, run identity, and exact
  840+180 planner.
- `src/iqp_cond/gaussian_data.py`: role-isolated deterministic master pools and
  nested prefix subsets.
- `src/iqp_cond/gaussian_campaign.py`: isolated immutable ledger, resource
  preflight, and result envelopes.
- `src/iqp_cond/gaussian_runner.py`: resumable training, validation-only
  checkpointing, and exact post-selection evaluation.
- `src/iqp_cond/gaussian_statistics.py`: paired bootstrap and held-out bank
  baselines.
- `src/iqp_cond/gaussian_report.py`: all-cell profile/bitstring/moment/MMD
  synthesis.
- `src/iqp_cond/curie_weiss_ancilla_protocol.py`: versioned eight-visible
  protocol and exact fixed-$r=5$ 80-fit planner.
- `src/iqp_cond/curie_weiss_ancilla_runner.py`: unrestricted conditioned,
  no-condition, and five-ancilla temperature-reference fits.
- `src/iqp_cond/curie_weiss_ancilla_report.py`: both TVDs, physical response,
  metric-specific oracles, and paired-bootstrap synthesis.
- `src/iqp_cond/interpolation.py`: held-out angle interpolation protocol.
- `src/iqp_cond/sparse_interpolation.py`: sparse-grid interpolation with ancillas.
- `src/iqp_cond/scaling.py`: frozen Condition-count scaling protocol.
- `src/iqp_cond/symmetry_fidelity.py`: symmetry-aligned fidelity audit.
- `docs/experiment-design.md`: preregistered experimental design.
- `docs/interpolation-design.md`: frozen interpolation path and decision rules.
- `docs/condition-count-scaling-design.md`: frozen K=2/3/5/9 stress test.
- `docs/symmetry-aligned-fidelity-design.md`: frozen output-action audit.
- `docs/encoding-family-research.md`: tractability criterion and encoding ladder.
- `results/summary/success_criterion.md`: generated condition-wise decision audit.
- `paper/main.tex`: internal arXiv foundations manuscript source.
- `paper/supplement.tex`: complete derivations and reproducibility supplement.
- `results/gaussian-foundations-v1/`: isolated campaign data, evidence, store,
  report, and claim registry.
- `results/curie-weiss-m5-foundations-v3/`: active calibrated-map 80-fit
  spin-flip campaign store, visible laws, report, and claim registry.
- `results/curie-weiss-capacity-control-v2/`: ten-seed 91-versus-92-phase
  historical diagnostic for the v2 product-input model; it is not a v3
  matched-capacity control.
- `docs/curie-weiss-ancilla-claim-evidence-registry-v3.md`: active
  Curie--Weiss claim-to-evidence register.
- `docs/claim-evidence-registry.md`: historical failed-Ising claim register,
  not an active manuscript specification.
- `output/pdf/conditional_iqp_generation.pdf`: current local coauthor-review
  PDF; it is not an upload or submission artifact.
- `CONTEXT.md`: canonical domain vocabulary.

## Setup and verification

```bash
uv sync --group dev
uv run pytest
uv run ruff check src scripts tests
```

## Gaussian foundations campaign

The frozen manifest digest is
`386ce98be93501f078d45b29e7fa498db1600c44d0e22f4483c3c8d5c7ec0d8c`.
Prepare data, record the resource preflight, inspect the plan, and execute one
or more resumable shards:

```bash
uv run python scripts/prepare_gaussian_campaign.py
uv run python scripts/run_gaussian_campaign.py preflight
uv run python scripts/freeze_gaussian_execution_source.py
uv run python scripts/run_gaussian_campaign.py run --dry-run
uv run python scripts/run_gaussian_campaign.py run \
  --shard-count 5 --shard-index 0
```

Run shard indices 1--4 analogously. Completed identities are verified and
skipped; incompatible partial outputs are refused. After all 1,020 identities
complete, finalize evidence and build the paper:

```bash
uv run python scripts/summarize_gaussian_campaign.py
uv run python scripts/finalize_gaussian_claims.py
uv run python scripts/generate_gaussian_tables.py
uv run python scripts/plot_gaussian_campaign.py
uv run python scripts/build_gaussian_arxiv_archive.py
uv run python scripts/audit_gaussian_release.py
```

The last command compiles from a clean extracted source tree, rejects missing
references, overfull boxes, blank pages, excluded campaign language, or a
checksum mismatch, then promotes the canonical PDF. It records
`upload_ready: false` until the research team supplies verified author and
release metadata and completes a human proof.

## Eight-visible Curie--Weiss v3 ancilla extension

The active campaign uses the selected spin-flip two-branch input, reuses the
hashed eight-qubit datasets, and has a distinct manifest and store. Prepare it,
record the 100-update worst-case preflight, and run resumable shards with:

```bash
uv run python scripts/run_curie_weiss_ancilla_campaign.py prepare
uv run python scripts/run_curie_weiss_ancilla_campaign.py preflight
uv run python scripts/run_curie_weiss_ancilla_campaign.py run \
  --shard-count 5 --shard-index 0
```

After all 80 fits complete:

```bash
uv run python scripts/finalize_curie_weiss_ancilla_campaign.py
uv run python scripts/generate_curie_weiss_ancilla_tables.py
uv run python scripts/plot_curie_weiss_ancilla_campaign.py
uv run python scripts/build_gaussian_arxiv_archive.py
uv run python scripts/audit_curie_weiss_ancilla_release.py
```

The previous three-size `curie_weiss_foundations_v1` campaign and its 270 fits
remain immutable repository history and are excluded from the active paper.
The interrupted six-rung ancilla campaign and the 18-fit $m=0,1,3$
development study are likewise preserved; only the latter motivates the
prospective compute--capacity choice of $r=5$.

## Role-separated Curie--Weiss encoding comparison

This frozen comparison reuses the ten completed v2 conditioned fits and trains
twenty paired fits for linear and quadratic ancilla maps. The visible-qubit map,
datasets, IQP ansatz, optimizer, and estimator streams remain fixed.

```bash
uv run python scripts/run_curie_weiss_encoding_comparison.py prepare
uv run python scripts/run_curie_weiss_encoding_comparison.py preflight
uv run python scripts/run_curie_weiss_encoding_comparison.py run \
  --shard-count 5 --shard-index 0
```

Run shard indices 1 through 4 analogously. After all twenty fits complete:

```bash
uv run python scripts/finalize_curie_weiss_encoding_comparison.py
uv run python scripts/plot_curie_weiss_encoding_comparison.py
```

The preregistered decision requires a strictly positive paired improvement in
held-out profile TVD and a bitstring-TVD degradation upper bound below 0.01.

## Archived research lines

### Frozen Quantum campaign v1

The manifest digest is
`8cf5380b7a658381c8b61b3069f921c938f92be814c091418e46d4d905f6f0f3`.
Inspect the stopped campaign and regenerate its deterministic report:

```bash
uv run python scripts/prepare_ising_campaign.py --targets-only
uv run python scripts/run_ising_campaign.py --stage primary --dry-run
uv run python scripts/build_campaign_report.py
```

The target-only command rechecks the immutable exact archives without releasing
a gate. The primary dry run is expected to refuse the downstream stage before
any run reservation, data load, or trainer call. Running the preparation command
without `--targets-only` likewise refuses before dataset generation because the
theorem gate failed. In a future campaign with a passed theorem gate, that same
command derives and archives the target-calibration and sampling decisions via
their sealed finalizers. The permitted synthetic estimator audit consumes no
Ising training budget:

```bash
uv run python scripts/run_estimator_scaling.py \
  --output /tmp/iqp-cond-estimator-scaling.json
```

The production response-spectrum preflight is also expected to stop with exit
status 2 because both frozen primary checkpoints are absent and the theorem
gate failed:

```bash
uv run python scripts/audit_response_spectrum.py \
  --output /tmp/iqp-cond-response-spectrum.json
```

Regenerate the manuscript evidence and compile the paper:

```bash
uv run python scripts/generate_quantum_draft_figures.py
(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

The commands below reproduce legacy studies; they do not release or supply
evidence for `quantum-campaign-v1`.

Run the estimator calibration:

```bash
uv run python scripts/calibrate_estimators.py
```

Run or resume the full experiment matrix:

```bash
uv run python scripts/run_experiments.py \
  --models all \
  --seeds 0-9 \
  --skip-existing
```

Summarize completed results and generate paper figures:

```bash
uv run python scripts/summarize_results.py
```

Run the held-out conditional-interpolation study:

```bash
uv run python scripts/run_interpolation.py \
  --seeds 0-9 \
  --skip-existing
uv run python scripts/summarize_interpolation.py
```

Its five training angles, four interleaved held-out angles, target path, and
frozen decision rules are specified in `docs/interpolation-design.md`.

Run the sparse-grid interpolation study, which trains on only two or three
peak positions and checks interpolation between them with 0, 2, 3, 5, or 7
ancilla qubits (2, 3, 5, 7 in the primary sweep; the 2-peak m=0 row reuses the
endpoint-only primary model):

```bash
uv run python scripts/run_sparse_interpolation.py \
  --grid 3 --ancillas 0,2,3,5,7 --seeds 0-9 --skip-existing
uv run python scripts/run_sparse_interpolation.py \
  --grid 2 --ancillas 2,3,5,7 --seeds 0-9 --skip-existing
uv run python scripts/summarize_sparse_interpolation.py
```

The two grids, shared held-out angles, oracle adjacent-angle lookup, and
frozen decision rules are specified in `docs/sparse-interpolation-design.md`.
Six showcase figures (Target family heatmap plus angle-to-mean path,
held-out model-vs-Target-vs-Oracle profiles, quantitative held-out TVD
summary, interpolation vs Oracle lookup, ancilla effect, supervision curve)
are generated by `scripts/plot_interpolation_variants.py` into
`results/sparse-interpolation/variants/` and included individually in the
paper appendix.

Run the full Condition-count stress test and strict audit:

```bash
uv run python scripts/run_condition_scaling.py \
  --conditions 2,3,5,9 \
  --tracks fixed_total,fixed_per_condition \
  --methods shared_angle,no_condition,separate_bank \
  --seeds 0-9 \
  --skip-existing
uv run python scripts/summarize_condition_scaling.py
```

Run the endpoint-matched noncommuting reupload study and compare it with the
angle runs:

```bash
uv run python scripts/run_interpolation.py \
  --encoding symmetric_reupload \
  --seeds 0-9 \
  --output results/encoding/runs \
  --skip-existing
uv run python scripts/summarize_encodings.py
```

Run the Hopfield-beta study (dense model, no-condition and separate-model
baselines) and summarize it:

```bash
uv run python scripts/run_hopfield.py --seeds 0-9 --skip-existing
uv run python scripts/summarize_hopfield.py
```

Try the ancilla-only condition encoding on the three-class Iris dataset:

```bash
uv run python scripts/run_iris_experiment.py --seeds 0-2
```

The four thresholded Iris features are the output distribution on four data
qubits. Class labels are binary-encoded on two ancillas only; the complete
2-local ansatz acts on all six qubits and the ancillas are marginalized at
evaluation. This is an exploratory generative benchmark, not a claim of
supervised classification accuracy.

The training betas \(0.5,0.75,1,1.25\), held-out log-midpoints, frozen pattern
pair, Angle-to-beta path, exact training objective, and decision rules are
specified in `docs/hopfield-design.md`. Measure the representability boundary
(about one hour):

```bash
uv run python scripts/analyze_hopfield_boundary.py
```

Build the manuscript from `paper/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -output-directory=../tmp/pdfs main.tex
```

## Legacy archive at a glance

One shared 190-parameter Angle-conditioned circuit produces two distinct
Gaussian Hamming-weight regimes. Its mean-Condition profile TVD is 0.096,
versus 0.365 and 0.364 for the No-condition and Basis-condition controls. The
frozen condition-wise rule passes three of four comparisons because Basis
conditioning fits Condition A's profile better; that failure is not replaced
by the balanced aggregate.

Five-angle training reduces mean Held-out profile TVD from 0.160 for
endpoint-only training to 0.061, but fails one of four oracle midpoint tests.
The sparse-grid study measures the same interpolation capacity at matched
ancilla counts (2, 3, 5, 7): with only three trained peaks, median Held-out
profile TVD stays 0.067--0.080 and Bitstring TVD falls monotonically from
0.345 at $m=0$ to 0.214 at $m=7$, beating the oracle adjacent-angle lookup at
every Held-out angle and every ancilla count; a two-peak model with no
between-peak supervision passes all but one oracle cell, and matched three-peak
supervision beats two-peak supervision at every $m$; see
`results/sparse-interpolation/summary/`.
The fixed-capacity Condition-count audit passes every profile rule through
K=9, where Shared Angle reaches 0.066 mean profile TVD versus 0.242 for No
condition. Its overall scaling verdict is nevertheless FAIL because 49 of 90
K=9 Bitstring comparisons fail, above the frozen limit of 45.

Exact untwirled Bitstring TVD exposes within-shell nonuniformity. Marginalizing
over traced-out ancilla qubits reduces it monotonically: Held-out Bitstring
TVD falls from 0.354 at m=0 to 0.208 at m=7, while a matched-capacity m=0
control with 3-local generators stays at or above the baseline at every
parameter count, attributing the gain to the marginalization rather than to
parameter count. For the known
shell-uniform Targets, a parameter-free permutation twirl reduces endpoint
Bitstring TVD from 0.352 to 0.121 on A and from 0.275 to 0.072 on B, while
preserving profiles and moments. A matched XOR-pair action fails the complete
rule. This supports the complete symmetry-aligned generation protocol, not a
claim that the circuit learned the symmetry.

### Additional boundary study

The Hopfield-beta study (ten seeds, exact zero-variance Conditional MMD)
passes its profile-level memorization test at all four training betas
(dense median profile TVD 0.031 vs 0.140 for the no-condition control) but
fails every bitstring-level test: dense bitstring TVD 0.541 matches the
no-condition 0.538, model pattern mass is below \(10^{-4}\) at every beta
against target masses up to 0.063, and dense interpolation loses to oracle
adjacent-beta lookup at all three held-out betas. A preregistered
three-barrier representability analysis (estimator, MMD localization,
product-envelope) explains the boundary; see `docs/hopfield-design.md`.

## Claim boundary

The repository demonstrates efficient classical estimation and
backpropagation for the observables used in MMD training. It does not claim
that these particular 19-qubit circuits demonstrate classically hard sampling
or quantum advantage; see `docs/adr/0001-limit-the-claim-to-classical-training.md`.

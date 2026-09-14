# Complete eight- and twelve-qubit results

The 4 September 2026 snapshot contains all 2,600 in-scope fits and exact
evaluations: Gaussian n=8 and n=12 have 700 each; Curie--Weiss n=8 and n=12
have 600 each. Each case includes ancillas 0, 1, 3, 6, 10, seeds 100--109,
the shared models, the independent training bank, and direct held-out fits.
The original manifest is unchanged. The 16-qubit extension is deferred for
compute/noise constraints, not excluded based on a comparison of final results.

## Integrity and numerical data

- `provenance.json` records envelope, checkpoint, and probability checksums for
  every retained identity and the frozen manifest checksum.
- `summary.json` contains per-condition comparisons, seed-level averages,
  median/IQR summaries, 95% paired bootstrap intervals from 10,000 resamples,
  ancilla effects, and separate non-landmark CW comparisons.
- `seed_summary.csv` is the tabular seed-level export.
- Profile CSVs contain the exact values underlying the refreshed CW12 figures.

The sync from both hosts copied 556 missing files without overwriting existing
files. All overlapping files agreed. Probability archives and checkpoints were
verified against the immutable envelopes. The complete subset was then checked
against the frozen production identities. Raw checkpoints/probabilities remain
in the local ignored `results/exact-tvd-ancilla-v1` store and the two hosts, not
in Git. These numerical summaries reproduce the scaling figures without those
raw archives; regenerating the profile figures requires the retained archives.

## Reproduction

From the repository root, with the project environment and retained results:

```sh
export PATH=/Library/TeX/texbin:$PATH  # macOS TeX installation, if needed
.venv/bin/python scripts/process_paper_results.py --output docs/paper_8_12
.venv/bin/python scripts/plot_paper_ancilla_scaling.py --summary docs/paper_8_12/summary.json --output Overleaf/figures
.venv/bin/python scripts/plot_qualitative_ancilla_profiles.py
.venv/bin/python scripts/plot_curie_weiss_n12_partial.py
```

The last command retains its historical filename for compatibility, but now
requires the completed r=0,3,10 profiles and uses r=10 for both the shared model
and the independent interpolation bank. Its output directory is
`artifacts/curie_weiss_n12_complete`. Copy the corresponding PDF/PNG profile
assets from there and `artifacts/qualitative_ancilla_profiles` to
`Overleaf/figures`, then compile `Overleaf/main.tex` with `latexmk -pdf`.
Gaussian midpoint probes are post-training evaluations of already selected
checkpoints, not the registered eight-condition headline test.

## Findings and interpretation

| Case | Held-out shared TVD, r=0 | Held-out shared TVD, r=10 | Interpolated bank, r=10 | Direct oracle, r=10 |
| --- | ---: | ---: | ---: | ---: |
| Gaussian n=8 | 0.219 | 0.102 | 0.122 | 0.085 |
| Gaussian n=12 | 0.292 | 0.174 | 0.175 | 0.128 |
| CW n=8 | 0.190 | 0.047 | 0.059 | 0.083 |
| CW n=12 | 0.275 | 0.076 | 0.112 | 0.146 |

Values are medians after averaging held-out conditions within each seed. CW
rows include beta=1. Excluding that supplied landmark, r=10 shared versus
interpolated-bank TVDs are 0.047 versus 0.058 at n=8 and 0.081 versus 0.117 at
n=12. Paired median advantages are 0.0119 [0.0080, 0.0136] and
0.0363 [0.0299, 0.0433], respectively, with positive advantages in all ten seeds.

Claim-evidence review:

- Ancillas reduce median shared raw TVD in all four cases: supported by the
  five-point ladders. This is not monotonicity of every seed or condition.
- Shared conditioning always beats independent interpolation: unsupported.
  It loses at low ancilla counts and is statistically comparable for Gaussian
  n=12 at r=10. The manuscript does not make the universal claim.
- CW learns an unknown transition location: unsupported. The map supplies the
  landmark; non-landmark interpolation is the relevant comparison.
- Shared models exceed a fundamental oracle capacity bound: unsupported.
  Direct fits are finite-budget trained references, not globally optimized
  bounds. Shared training also accesses several condition datasets.
- Accurate profiles imply accurate raw bitstring distributions: unsupported.
  Profile aggregation can conceal errors within a shell.

## Current manuscript review status

The draft now includes Gaussian, Curie–Weiss and the completed spatial Ashkin–Teller study. Abstract, Introduction, Methods, Results, Discussion and Conclusions have been reconciled and assistant-reviewed. The CW development-ablation artifact pointer is verified; the three-seed scope remains explicit.

See the [first supervisor draft review](../supervisor-draft-review-2026-09-07.md) for the PDF, figure map, validation and remaining author decisions. Author scientific review, authorship/funding confirmation and public archival deposition remain pending. Preserve the distinction between tractable observable estimation and output sampling, and between supplied input structure and interpolation. No quantum advantage, general extrapolation or scaling claim follows from these sizes.

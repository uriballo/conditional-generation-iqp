# Twelve-qubit data-size and ancilla interaction result

All 180 IQP fits and 30 classical bundles completed under the frozen protocol. The coordinator verified and sealed all 420 prediction files before evaluating the nine interior targets, then independently recomputed all 300 seed/size/model metric records. Current frozen sources, archived sources, data, prediction and evaluation identities match. Ten fresh paired seeds and every outcome are retained.

The six-panel main-body Figure 9 shows exact joint TVD, connected-bond RMSE and paired ancilla benefit, with shared limits within each metric. All means first average nine conditions within each seed; 95% percentile intervals use 10,000 shared paired-seed resamples, without multiplicity correction. The report script reproduces both vector PDF and PNG from verified artifacts.

| Primary interaction (four-ancilla benefit at 1,000 minus at 100 training examples) | Mean | Descriptive 95% interval |
| --- | ---: | --- |
| Symmetry-only IQP | -0.008787 | [-0.021926, 0.003152] |
| Bond-informed IQP | -0.001216 | [-0.076778, 0.073148] |

Neither encoder establishes that ancilla benefit grows with training data. This is unresolved interaction, not equivalence or proof of no interaction. The secondary two-ancilla intervals also include zero. Symmetry-only four-ancilla benefits remain positive at each measured size (0.0203, 0.0037, 0.0115); bond-informed mean TVDs differ by less than 0.0003 across r at 300/1,000 examples.

All learned arms have lower mean TVD at 1,000 than at 100 examples, but the curves need not be monotonic. Symmetry-only r=0 changes 0.2981 -> 0.2725 -> 0.2774. Bond-informed r=0 changes 0.2256 -> 0.1927 -> 0.1920. Two isolated low-data bond fits remain in the analysis: seed4718 r=0 TVD0.45045 and seed4714 r=4 TVD0.44359; other fits in those arms lie in [0.19345,0.20719]. Their selected validation MMDs are also higher than their paired successful alternatives, but no mechanistic or causal failure explanation is established and no fit is replaced.

The pooled log-linear model improves 0.2840 -> 0.1611 -> 0.1105 and is more accurate in joint TVD than every quantum arm at 300 and 1,000 examples. Its connected-bond RMSE improves 0.0464 -> 0.0141, versus quantum means 0.1094–0.1236. Every quantum arm improves mean joint and connected-bond errors over its supplied input; this is not an every-seed statement.

Interpretation is restricted to this procedure's training-data sensitivity on the previously studied interpolation domain. Validation remains 1,000 examples/condition (16,000 additional labels) and optimization remains 500 updates, so epochs vary. Active phase counts grow 78/102/126. There is no few-label, parameter-count-matched, optimal sample-complexity, out-of-domain, or quantum-advantage claim.

Artifacts: `artifacts/ashkin-teller-data-efficiency-2026-09-07/{manifest.json,predictions-seal.json,evaluation.json,report/summary.json,report/report.md,report/independent-interaction-check.json}`. Immutable manifest SHA256: `ded6ad230319b14256eda0c1255c13c5d79faf3d6e0e754b844dd305a11649f5`.

Paper: `Overleaf/main.tex`, sections `sec:data-ancilla-followup` and `sec:data-ancilla-results`, equation `eq:data-ancilla-interaction`, main-body figure `fig:data-ancilla`, and Discussion. Verified integration snapshot: `ashkin-teller-data-ancilla-integration-2026-09-08.json`. Scientific author/supervisor review remains pending.

Regenerate figures and numerical audit:

```sh
PYTHONPATH=src .venv/bin/python scripts/report_ashkin_teller_data_efficiency.py --directory artifacts/ashkin-teller-data-efficiency-2026-09-07 --output artifacts/ashkin-teller-data-efficiency-2026-09-07/report
```

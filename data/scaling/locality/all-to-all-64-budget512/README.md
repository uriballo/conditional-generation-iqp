# All-to-all ansatz at 64 qubits, lattice estimator budget

Comparison model for the large-scale experiment: the parity-input conditional IQP
model with all one- and two-body phases (2080 parameters) at 64 qubits, trained with
the per-update estimator budget of the selected lattice configuration (512 masks,
1024 reference samples). It replaces `../all-to-all-64/`, whose search let the budget
vary and selected 1024 masks and 2048 reference samples.

## What was run

- Search: 60 saturating-schedule (S1) configurations at 64 qubits, drawn from the
  scaling search space and stream of `scaling_hpo_protocol.json`; every draw equals
  the original all-to-all search except the training budget, fixed with `--budget 512 1024`.
  Selection: lowest search score. Winner: S1-017 (score −4.945), the same trial as before.
- Final fits: seeds 100, 101, 102, 5000 updates, data, streams and evaluation of the
  lattice fits.
- Selected configuration (S1-017): learning rate 0.00169, init width 0.0127,
  β1 0.934, β2 0.982, final learning-rate ratio 0.335, t_max 1.159, τ 4.117,
  β0 1.018, γ 0.189.

## Results (`table.json`, means over the three fits)

| Quantity | Value |
| --- | --- |
| Trainable parameters | 2080 |
| Test MMD² | 0.00097 |
| Δ⟨m²⟩ at β = 0.8 / 1.0 / 1.2 | −0.027 / −0.013 / +0.004 |
| Mean \|Δ⟨m²⟩\|, 0.8 ≤ β ≤ 1.3 | 0.012 |
| Δ⟨m²⟩ at β = 1.7 / 2.0 | −0.106 / −0.184 |

Δ⟨m²⟩ is the classically estimated model value minus the exact target. The mean
|Δ⟨m²⟩| is taken over the seed-averaged curve at the 11 dense temperatures in [0.8, 1.3].

## Reproduce

```sh
uv run python -m paper_experiments.scaling_hpo run results/a2a-512/search --ansatz complete --sizes 64 --family S1 --budget 512 1024
uv run python -m paper_experiments.scaling_hpo final results/a2a-512/fits --search results/a2a-512/search --trial S1-017 --ansatz complete --sizes 64
uv run python -m paper_experiments.scaling_hpo evaluate results/a2a-512/evaluations --fits results/a2a-512/fits
uv run python -m paper_experiments.scaling_locality_table data/scaling/locality/all-to-all-64-budget512/fits
```

Runs on CPU in about 41 s per search trial and 90 s per final fit on an idle Apple M3 Max (CPU JAX).

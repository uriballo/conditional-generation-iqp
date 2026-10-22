# Curie–Weiss, classical baselines

Final results for conditional RBM, autoregressive Transformer and VAE
generators at two capacities, under `paper_experiments/cw_classical_protocol.json`.
They use the training temperatures, target pools, update budget, starts, search
sizes and selection rule of the IQP models. The RBM and Transformer train on
the exact likelihood and the VAE on the ELBO; VAE probabilities are integrated
numerically over the latent prior.

| Capacity | RBM | Transformer | VAE, training / generation |
|---|---:|---:|---:|
| Small | 140 | 137 | 134 / 92 |
| Large | 986 | 971 | 950 / 548 |

- `{small,large}/figure_data.json`: selected configurations, architectures, curves and per-repeat scores.
- `{small,large}/confirmation.json`, `{small,large}/shortlist.json`: confirmation stage and shortlist.
- `comparison.{csv,json}`: both capacities with the C-IQP references and paired bootstrap intervals.
- `validation.json`: bundle checks.

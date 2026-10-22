# Curie–Weiss, parity protocol

Final results of the four-family Curie–Weiss protocol
(`paper_experiments/cw_followup_protocol.json`): shared parity (C-IQP (P)),
shared product, per-temperature parity and per-temperature standard-input
models. The paper reports C-IQP (P) and the per-temperature standard-input
model.

Shared models train at β ∈ {0.1, 0.7, 1.3, 1.95}. Checkpoints are selected on
the A17 grid and configurations on A17 ∪ B20; the F101 grid is used only for
reporting. The test temperatures β = 0.4, 1 and 2 are excluded from shared
fitting and selection, and per-temperature models fit directly at them. Each
family had 60 random and 60 TPE proposals, confirmation with five repeats and
final fits with ten repeats, each with three starts. Training uses the sampled
squared MMD; selection uses the full-TVD score J, with the physics score P as a
tie-breaker within J + 0.01.

- `figure_data.json`: final curves, profiles, targets and metrics.
- `review-ready.json`, `summary.json`, `REPORTING_AUDIT.json`: completion status, winners and checks.
- `provenance/`: confirmation and final campaign manifests and results.

Magnetization is m = (2k − 16)/16 for Hamming weight k.

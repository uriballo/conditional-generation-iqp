# Interval-based condition encoding prototype

## Hypothesis

The current Gaussian model maps each Condition (c) to one broadcast angle

\[
\alpha(c)=\pi(0.35-0.70c).
\]

This prototype replaces that row by independent per-qubit angles sampled from
\([alpha(c)-w(c),\alpha(c)+w(c)]\). The stochastic arm draws fresh angles
during training and inference, so the generated law includes this classical
input randomness.

The script reports both one fresh draw and an average over several fresh draws.
The latter estimates the actual distribution produced by repeated randomized
generation.

## Run

```bash
uv run python scripts/run_interval_encoding_prototype.py \
  --seeds 0-2 --steps 500 --n-qubits 8 \
  --half-width-pi 0 0.05 0.10 \
  --output results/interval-encoding-prototype-v1.json
```

The first screen should use the same `--seeds`, optimizer settings, target
width, and qubit count for every arm. Compare held-out rows by
`profile_tvd` first, then inspect `bitstring_tvd` as the limitation metric.
The zero-width arm is the present broadcast baseline, recovered by resampling
with a zero interval.

The range is a modeling choice, not something inferred from held-out target
data. A useful follow-up is to choose (w(c)) using training Conditions only,
then freeze it before evaluating held-out Conditions.

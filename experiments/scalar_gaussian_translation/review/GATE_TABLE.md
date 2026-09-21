# Corrected locked-audit gate table

This table reports the threshold-preserving re-evaluation of the saved audit
rows. The correction excludes mirrored protocols from every qualification
check and evaluates the maximum of the three layout-seed medians rather than a
single median pooled across layouts. No threshold, checkpoint, or training run
changed. Both general encoders pass all 28 checks.

For upper-bound checks, passing requires observed \(\leq\) threshold. For the
encoder-improvement and mean-movement checks, passing requires observed
\(\geq\) threshold. The full per-seed and per-layout values are stored in
[`audit_gate_check.json`](audit_gate_check.json).

| Check | Threshold | Broadcast observed | Broadcast | Qubit observed | Qubit |
| --- | ---: | ---: | :---: | ---: | :---: |
| `encoder_only_improvement.median` | 0.3000 | 0.5194 | pass | 0.5097 | pass |
| `encoder_only_improvement.worst_group` | 0.2500 | 0.3747 | pass | 0.3743 | pass |
| `far_extrapolation.median_seed_mean` | 0.4000 | 0.3421 | pass | 0.3534 | pass |
| `far_extrapolation.worst_condition` | 0.8200 | 0.7907 | pass | 0.6874 | pass |
| `far_extrapolation.worst_seed_mean` | 0.4300 | 0.3573 | pass | 0.3844 | pass |
| `held_out_interpolation.median_seed_mean` | 0.3200 | 0.2795 | pass | 0.2833 | pass |
| `held_out_interpolation.worst_condition` | 0.6000 | 0.5358 | pass | 0.5301 | pass |
| `held_out_interpolation.worst_seed_mean` | 0.3400 | 0.2839 | pass | 0.2843 | pass |
| `mean_movement.correct_sign_fraction` | 0.9500 | 1.0000 | pass | 1.0000 | pass |
| `mean_movement.median_ratio` | 0.5000 | 0.6263 | pass | 0.6409 | pass |
| `mean_movement.worst_seed_median_ratio` | 0.4500 | 0.6194 | pass | 0.6394 | pass |
| `near_extrapolation.median_seed_mean` | 0.3300 | 0.2774 | pass | 0.2739 | pass |
| `near_extrapolation.worst_condition` | 0.5500 | 0.4466 | pass | 0.4338 | pass |
| `near_extrapolation.worst_seed_mean` | 0.3500 | 0.2790 | pass | 0.2775 | pass |
| `reliability.optimizer_seed_range` | 0.1000 | 0.0153 | pass | 0.0153 | pass |
| `reliability.random_layout.far_extrapolation` | 0.4500 | 0.3154 | pass | 0.3945 | pass |
| `reliability.random_layout.held_out_interpolation` | 0.3500 | 0.3018 | pass | 0.3053 | pass |
| `reliability.random_layout.near_extrapolation` | 0.3800 | 0.2711 | pass | 0.2676 | pass |
| `reliability.random_layout.training` | 0.3300 | 0.2998 | pass | 0.2949 | pass |
| `structure.profile` | 0.2200 | 0.1594 | pass | 0.1654 | pass |
| `structure.profile_worst_seed_mean` | 0.2800 | 0.2074 | pass | 0.2119 | pass |
| `structure.shell` | 0.2000 | 0.1486 | pass | 0.1561 | pass |
| `structure.shell_worst_seed_mean` | 0.2500 | 0.1723 | pass | 0.1767 | pass |
| `training.median_seed_mean` | 0.3000 | 0.2631 | pass | 0.2629 | pass |
| `training.worst_condition` | 0.5200 | 0.4741 | pass | 0.4648 | pass |
| `training.worst_seed_mean` | 0.3200 | 0.2666 | pass | 0.2646 | pass |
| `width.median` | 0.1800 | 0.1157 | pass | 0.1212 | pass |
| `width.worst_seed_mean` | 0.2800 | 0.1927 | pass | 0.2125 | pass |

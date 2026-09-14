# Noisy bars-and-stripes quick capacity screen

2026-09-07. Development-only result, not manuscript evidence.

## Decision

Do not promote this target to a conditional campaign with the current setup.
The predeclared three-ancilla gate required median exact raw TVD <=0.15 and
worst-case TVD <=0.25. Sampled training gave median 0.5419 and worst 0.5469.
The exact-population-MMD control also failed, with median 0.5479 and worst
0.5495. Sampling noise is therefore not the sole obstacle in this screen.
This is not a proof of a representational limitation or impossibility.

## Frozen screen

The target is the [proposed density-conditioned noisy BAS law](conditional-bars-and-stripes-proposal.md):
2x4 images, horizontal/vertical orientation equiprobable, independent
Bernoulli(q) stripe bits, independent 5% pixel flips. Densities are 0.2, 0.5,
0.8. Each condition receives its own target-aware complete two-local IQP fit,
with 0 or 3 ancillas and fixed zero input angles. No condition is held out and
no shared conditional model is trained.

One development seed, 300 Adam-cosine updates, learning rate 0.01 to 0.001,
batch size 100, 1,000 training samples and 1,000 independent validation
samples per condition. Data are paired across ancilla counts and modes.
Initialization uses the repository's data-based visible initializer and
small ancilla-supported noise. The sampled mode uses the existing isotropic
three-bandwidth loss, 256 nonidentity draws per bandwidth and 1,024 model
estimator samples. This is a reduced budget, not the production protocol.

The population control uses the full analytic target and exact MMD gradient,
with the same initialization, optimizer and update count. It removes target
sampling error and operator/model estimation noise together; it is not a
finite-data comparison or a deployable classical training claim.

Candidate checkpoints are initial, final, and lowest validation-MMD during
training. Selection among them uses exact TVD against the independent
empirical validation distribution. Analytic target TVD does not select the
checkpoint. Population-MMD validation generates one candidate in the
population control and is explicitly target-aware.

The protocol was written before training. Its SHA256 is
`39c282253e31790aafb963fff8656dc54ecd0c5c3124a0d08f7899b6abd99de9`.
If the sampled gate passed, seeds 1 and 2 would repeat the same screen.
It failed, so no additional seeds or hyperparameter search were run.

## Exact target TVD

| Density q | Sampled r=0 | Sampled r=3 | Population r=0 | Population r=3 |
|---|---:|---:|---:|---:|
| 0.2 | 0.5461 | 0.5469 | 0.5495 | 0.5495 |
| 0.5 | 0.3928 | 0.3406 | 0.3840 | 0.2799 |
| 0.8 | 0.5413 | 0.5419 | 0.5473 | 0.5479 |

At q=0.5, three ancillas help: sampled TVD improves from 0.3928 to 0.3406,
and population TVD from 0.3840 to 0.2799. These remain outside the capacity
gate. At q=0.2 and 0.8, both modes plateau at much larger error.

The sampled q=0.5/r=3 fit reduces pair-correlation RMSE from 0.2952 to 0.1390,
and population MMD from 0.03053 to 0.00612. The loss responds to spatial
structure, but this does not produce a sufficiently accurate full law.
A training-histogram baseline has exact TVD 0.1091, 0.1374, 0.1167 at the three
conditions. A fitted independent-pixel baseline has TVD 0.5353, 0.6515,
0.5246. These small-state-space controls are descriptive, not quantum-advantage
comparisons.

## Interpretation and checks

An exact kernel can distinguish the target while the optimization still fails
at the selected budget and initialization. This experiment leaves model
capacity, poor initialization, local minima and optimization time unresolved.
It does not justify claiming that isotropic MMD cannot learn spatial structure.
Given two previous grid-Ising failures, adding another encoder/tuning campaign
solely to obtain a third positive dataset is not recommended here.

All 12 fits completed. The target formula was independently cross-checked by
explicit enumeration of latent stripe bits. All candidate output laws were
normalized, the empirical-validation argmin was checked, and selected exact
TVD and population MMD were independently recomputed from saved probabilities.
No production code, frozen campaign, manuscript claim or existing result store
was modified. The temporary prototype and checkpoints were deleted after the
outcome was recorded; no permanent benchmark implementation was added.

[Machine-readable settings, metrics and controls](noisy-bas-capacity-screen-2026-09-07.json)
record source commit, runner hash and runtime versions. This durable summary
supports a development decision; it is not an archived reproducible campaign.

# Curie--Weiss claim--evidence registry

This registry governs the completed `curie_weiss_foundations_v1` extension.
Every numerical statement below is bound to the immutable 270-fit report with
SHA-256 `4bf0fd1e8a2b1376cc403544fcab68bf7d1ac939da0a79bfa80318b01c012740`
and manifest digest
`b272abfdea1b7f18ab1e0e3b1a885d16530445816f27974662e5fabd936e4367`.
Positive effects mean lower profile TVD for the symmetry-aligned conditioned
model. There is no empirical pass gate.

| Candidate claim | Immutable evidence | Status and permitted wording |
|---|---|---|
| Conditioning improves on the matched no-condition model. | Training effects: n=8 `-0.044 [-0.063,-0.022]`; n=12 `+0.139 [+0.133,+0.144]`; n=16 `+0.147 [+0.145,+0.150]`. Held-out effects: n=8 `-0.027 [-0.045,-0.005]`; n=12 `+0.142 [+0.133,+0.152]`; n=16 `+0.143 [+0.141,+0.144]`. | **Mixed across sizes.** Say that temperature conditioning improves on no conditioning at 12 and 16 spins, but not at 8. Do not pool sizes. |
| The model interpolates more accurately than adjacent references. | Held-out oracle effects are `-0.158`, `-0.191`, `-0.239`; mixture effects are `-0.158`, `-0.176`, `-0.224`, all with intervals strictly below zero for n=8,12,16. | **Unsupported.** The conditioned model exhibits within-range response but is less accurate than both adjacent references at every size. |
| The unseen critical point is represented faithfully. | No-condition effects at beta=1: n=8 `-0.122 [-0.170,-0.061]`; n=12 `+0.135 [+0.123,+0.146]`; n=16 `+0.006 [-0.005,+0.018]`. Mean absolute profile TVD is 0.438, 0.361, and 0.532. | **Mixed and low fidelity.** The critical comparison is favorable only at n=12, inconclusive at n=16, and adverse at n=8. |
| Exchange-symmetric sharing improves fidelity. | Unrestricted-minus-aligned held-out effects are `-0.137`, `-0.145`, `-0.162`, with intervals strictly below zero at every size. | **Unsupported.** The tied circuit preserves exchange symmetry, but the unrestricted phase ablation is more accurate. Describe the two-coefficient bank as an overly restrictive bias in this experiment. |
| The model reproduces thermodynamic response. | Exact response rows and 95% seed intervals for absolute magnetization, susceptibility, Binder cumulant, energy, and heat capacity at all 11 temperatures. | **Qualitative only.** Curves respond to temperature, but deviations are substantial near the critical and cold regimes. Do not claim thermal-state preparation or critical scaling. |

The report classifies 5 comparisons as supported, 1 as mixed, and 24 as
unsupported. This count is descriptive; it is not a publication criterion.
The Abstract may state the size-dependent no-condition result and the negative
adjacent-reference and unrestricted-ablation results. It may not call the
Curie--Weiss model a high-fidelity interpolator.

Prohibited claims: state-of-the-art thermal-state preparation, exact critical
scaling, quantum advantage, extrapolation, and general many-body thermal
generation. The direct-TVD global search is a supplement-only capacity
diagnostic and cannot support the classical-training claim.

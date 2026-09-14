# Ashkin–Teller information and resource accounting

Companion to the [paper study plan](ashkin-teller-paper-study-plan-2026-09-07.md). This table describes the frozen designs, independently of the pending twelve-qubit outcomes. It is a proposed manuscript resource table, not a computational-advantage comparison.

| Model | Supplied structure and condition dependence | Fitted quantities at n=8 / n=12 |
| --- | --- | --- |
| Symmetry-only input | Two layer labels, independent global layer flips, fixed affine cat-angle map of each coupling | None |
| Symmetry-only IQP | Same input; shared complete one-/two-body phase bank | 30 / 78 independent real phases |
| Bond-informed input | Same flips and partition, ring adjacency, full uncoupled coherent Ising law with g(c)=c and fixed phase eta=pi/8 | None |
| Bond-informed IQP | Same correlated input; same phase bank as symmetry-only IQP within each size | 30 / 78 independent real phases |
| Empirical log composition | Flip-orbit histograms at the perimeter and declared log-composition rule | 16×63 / 16×1023 histogram degrees of freedom; one validation-selected smoothing level |
| Pooled log-linear | Flip orbits, affine conditional log odds; no supplied ring-energy features | 189 / 3069 coefficients; one validation-selected penalty |

The histogram degrees of freedom and log-linear coefficient counts describe different parameterizations and should not be interpreted as equivalent capacity measures. The log-linear family is well specified for the target's affine coupling dependence. This is a more direct explanation of its suitability than parameter count alone.

Both fitted quantum models use 16,000 training examples and 16,000 independent validation examples per seed, 500 sampled-MMD updates, and ten paired seeds at each size. The phase vector is shared across conditions but trained afresh at each size and seed. The circuit bank is matched between encoders within a size; it grows between sizes. Encoder-only predictions are deterministic controls, not ten independent fits. Classical selection uses the same perimeter validation data. No model selects settings using interior target laws.

For n=2m, the cat reference is represented by two m+1 shell-amplitude tables, with sampling through shell weights and uniform strings within a shell. The bond reference uses fixed-dimension transfer contractions: 2×2 matrices for amplitudes and 4×4 matrices for sequential reference probabilities. Bond amplitude queries and reference draws each require O(m) arithmetic after O(m) suffix preparation. These statements concern reference access for the training estimator; they do not assert efficient classical sampling from a general trained IQP output.

The bond input has a constructive preparation with O(m) bounded-control rotations and O(m) diagonal bond gates per ring, without additional logical ancillas. That operation count assumes access to the specified controls; routing and synthesis precision are additional costs. The [reference-method derivation](bond-reference-method.md) separates this construction from classical reference sampling and records the finite-precision limitations. The present experiments use classical simulation, with exact full-distribution evaluation over 256 or 4096 states after fitting. Exact evaluation is an exponential-size verification step, not part of a claimed scalable output-sampling procedure.

Both quantum families enforce independent global layer flips for every phase vector. Their unrestricted banks do not enforce all ring translations or reflections. The comparison therefore tests complete encoder families with declared prior information; it does not isolate a causal benefit from correct topology or symmetry enforcement.

The paired mean-TVD contrast is computed separately within each size. Structural transformations assess the contribution of the saved output's arrangement and dependence relative to its own profiles and marginals. They are not independently optimized comparator models. These distinctions should remain in the figure captions and discussion when the results are inserted into the paper.

# Scalar Gaussian translation benchmark

This directory freezes the protocol. The author-approved n=8 production
campaign completed on 2026-09-21 through the external authorization path; the
proposal remains disabled for the legacy runner by design. The register has
eight visible qubits, no ancillas, and `Ansatz.complete(8)` (36 parameters).
Targets use the shared width-0.5 truncated Gaussian.

The general broadcast map is `theta(c)=pi*(0.32-0.68*c)`. The wire-dependent
map uses `t_j=j/7` and
`theta_j(c)=pi*((0.28+0.06*t_j)-(0.62+0.08*t_j)*c)`. Both stay within
`(-pi/2, pi/2)`, so neither has a zero local post-Hadamard amplitude.
Their coefficients were frozen before validation and audit, and the maps do
not consume target moments at runtime. The tracked design record does not
establish a target-family-independent derivation, so review claims call them
predeclared runtime-target-agnostic maps rather than target-independent maps.

The anti-block map assigns paired angles `pi*c` and `pi*(1-c)`. It prepares
`|+>^8` at `c=0.5`, preserves one expected input excitation per pair, and is
complement-covariant under `c -> 1-c`. These properties are intentional
controls. Because the construction changes several coupled properties, its
lower TVD is attributed to the complete anti-block construction rather than
to complement symmetry alone.

`protocol.json` records intervals, anchors, immutable evaluation splits, seeds,
and the bounded two-stage search. The exponential encoder is a fallback with
only the listed `(k, s)` choices; the raw eight-wire range is excluded because
of oscillation and aliasing. No 12-qubit configuration is part of this study.
The primary comparison uses the two affine RY maps and `anti_block`; the
condition-blind model is a separate baseline. Exponential fallback is eligible
only if both affine RY maps fail. The default runner evaluation split is dev;
validation and audit require an explicit split selection.
Advance an affine RY encoder only if all predeclared Stage 1 dev-seed gates
pass: median training TVD and median held-interpolation TVD at most 0.45,
worst-seed held mean TVD at most 0.60, median improvement of at least 0.05 over
its own encoder-only held mean TVD, and correctly signed generated-mean
movement of at least 40% of target movement. Consider the exponential fallback
only if both affine RY maps fail these gates.

`build_study()` returns a runner-compatible descriptor. `protocol.py` exposes
encoder construction, anchor layouts, evaluation grids, target probabilities
and moments, condition classification, and matched-exposure update scaling.

## Review outcome

Both general affine RY encoders pass all 28 frozen locked-audit checks at n=8
after a threshold-preserving post-audit evaluator correction. The correction
excludes mirrored protocols from qualification and takes the maximum of the
three random-layout medians; it does not change training or thresholds.
The qubit-dependent formula did not materially improve on the broadcast map.
The symmetry-informed anti-block encoder was substantially more accurate, while
the mirrored far diagnostic exposed directional bias in both general encoders.
This is a proof-of-concept result, not a competitive-performance or quantum-
advantage claim.

The audit evidence, risks, decision, and reproduction commands are in the
[`review report`](review/REPORT.md). The completed campaign and its limits are
in the [`production report`](review/PRODUCTION_REPORT.md). Machine-readable
artifacts include the
[`frozen gates`](review/frozen_gates.json),
[`audit provenance root`](review/audit_provenance_root.json),
[`audit result`](review/audit_gate_check.json),
[`audit attestation`](review/audit_attestation.json),
[`review summaries`](review/review_summaries.json),
[`bounded-search decision`](review/selected_config.json), and
[`optimization diagnostics`](review/optimization_diagnostics.json),
[`production authorization`](review/production_authorization.json), and
[`production attestation`](review/production_attestation.json). The four audit
figure classes are in `review/figures/`; production figures and the 3,120 raw
runs are under `results/scalar_gaussian_translation_production/`.

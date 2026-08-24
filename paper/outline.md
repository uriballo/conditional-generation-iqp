# Submission-draft outline

## Thesis

The paper introduces conditional generation inside the classically
observable-trainable IQP construction. A condition changes a tractable input
state, while the compiled IQP unitary, ordered generator bank, and learned
phase vector are shared. The preparation-plus-circuit pipeline is not called a
strict IQP circuit. Product inputs support the general estimator result; the
Curie--Weiss benchmark uses a tractable spin-flip two-branch input.

“Classically trainable” applies only to the selected Pauli observables, the
Hamming-kernel conditional MMD estimator, and its sampled phase gradients. It
does not imply efficient output sampling, guaranteed optimization, or quantum
advantage.

## Six-section main-text narrative

1. **Introduction.** Establish the conditional-generation problem, distinguish
   the contribution from earlier conditional QCBMs, and position it as an
   extension of the classically trainable IQP construction.
2. **Conditional IQP model.** Define tractable input-state preparation followed
   by one fixed compiled IQP unitary. Use (r) for ancillas and (m) for learned
   generators or phases.
3. **Classical MMD training.** State the mask representation and the corrected
   amplitude-ratio result: second moment and variance are bounded, without a
   zero-variance or optimizer-efficiency claim.
4. **Gaussian-mixture experiments.** Report the complete 1,020-fit campaign,
   including the sharing–fidelity tradeoff and the condition-count result.
5. **Curie--Weiss experiments.** Report the completed 80-fit v3 campaign with
   the spin-flip two-branch input. The calibrated condition map is the only
   active encoder.
6. **Discussion.** Treat this as a foundational characterization, state the
   shared-circuit fidelity limitation directly, and restrict conclusions to
   the demonstrated finite systems.

## Three main figures

1. Gate-level product-state conditioning schematic and representative Gaussian
   targets.
2. Gaussian conditional profiles and the shared-versus-independent fidelity
   tradeoff.
3. Curie--Weiss profiles and response observables, including independently
   trained per-β IQP references at the six training temperatures.

## Supplement

- Finite response and estimator derivations.
- Compact Gaussian factorial and condition-count results.
- Curie--Weiss v3 protocol, calibrated-map provenance, and paired effects.
- Historical v2 matched-capacity diagnostic, explicitly not a v3 control.
- Post-core two-block and radial-shell symmetry screens, with their design
  limitations stated explicitly.
- Negative encoding comparison and implementation diagnostics.
- Finite 21-case estimator-consistency and resource audit, without asymptotic
  scaling language.
- Content-addressed Gaussian execution-source snapshot and regeneration notes.

## Release boundary

The endpoint is a local package for coauthor review. It contains no upload
script and performs no submission. Draft metadata are advisory only; approval
and submission remain solely with the authors.

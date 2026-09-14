# Ashkin–Teller paper study: motivation, experiments and discussion

2026-09-07. Proposed paper study following the completed [confirmatory interpolation experiment](ashkin-teller-confirmation-result-2026-09-07.md). This is a planning document, not an executable protocol freeze. No new fits or diagnostics were run to write it. Existing manifests, results and test roles remain unchanged. Numerical and scientific author review are distinct.

## Recommendation

Use Ashkin–Teller as the paper's demonstration of spatially structured, two-input conditional generation with two tractable encoder families. Retain the completed eight-qubit result as the primary comparison. Add two inexpensive structural diagnostics of its saved outputs, then a separately frozen twelve-qubit replication with the same two encoders and classical comparators. Do not start an ancilla ladder, encoder-map search or four-body campaign unless the paper adopts a specific additional claim requiring it.

The minimum defensible inclusion already has training and held-out evidence at n=8. The n=12 experiment is recommended because six-site rings reduce dependence on the special four-site geometry and test the transfer-reference construction at a second trained size. It is not needed merely to make every benchmark have identical plots, and it would not establish asymptotic scaling. If resources preclude it, an explicitly n=8 example remains defensible.

## Scientific motivation

The existing Gaussian and Curie–Weiss targets depend on Hamming weight or total magnetization. They establish conditional response within exchangeable distribution families. This third case addresses a different question: can input-conditioned IQP generation represent labelled spatial correlations, with one fixed learned unitary responding to two independently varied inputs?

Define two periodic rings with m sites each, n=2m spins, and

\[
 A_i=s_i s_{i+1},\qquad B_i=t_i t_{i+1},\qquad
 p_{a,b}(s,t)=Z(a,b)^{-1}\exp\!\left[a\sum_i A_i+b\sum_i B_i+\kappa\sum_i A_iB_i\right],\quad \kappa=0.3.
\]

The couplings a and b are dimensionless conditions, not independently fitted IQP phases. This is a finite one-dimensional periodic Ashkin–Teller-type benchmark: two Ising layers with an interaction between aligned bond energies. The coupled-Ising/four-spin formulation is standard; our graph and conditional split are explicit benchmark choices. See Giuliani, [Universality and non-universality in the Ashkin–Teller model](https://arxiv.org/abs/cond-mat/0412176), and Aoun, Dober and Glazman, [Phase Diagram of the Ashkin–Teller Model](https://doi.org/10.1007/s00220-023-04925-0). Those works study other lattice/phase-diagram settings; their critical phenomena are not claims about these small rings.

Three features motivate inclusion:

1. **Spatial information beyond magnetization.** Ring energies and aligned interlayer bonds distinguish configurations with the same layer Hamming weights. Joint weight profiles alone cannot certify recovery of the full target.
2. **Two-input composition.** Perimeter training exposes each individual test coordinate, while withholding its joint interior combination. The question is compositional interpolation within the domain, not extrapolation or a claim of a hard statistical task.
3. **A broader tractable input interface.** Independent layer flips give Z2 x Z2 symmetry. Both a product of two cat states and a product of two correlated coherent-ring states satisfy the sampling-and-ratio interface. The latter extends the demonstrated input family beyond weight-only amplitude formulas through fixed-dimension transfer contractions.

Independent global flips of either ring leave the target unchanged. Both encoders and the IQP unitary enforce those flips for every phase vector. Neither unrestricted IQP bank guarantees translation/reflection symmetry, or layer exchange at fixed unequal a,b. Swapping layers is a covariance with (a,b) also swapped. Do not describe the ansatz as enforcing every symmetry of the Hamiltonian.

The isotropic Hamming kernel does not force a permutation-invariant learned distribution: simultaneous permutation invariance of k(x,y) is not averaging each distribution over permutations. Population MMD with the finite-bandwidth Hamming-RBF kernel can distinguish spatial distributions. A finite frozen mask bank and sampled optimization still have finite-budget limitations; kernel distinguishability is not an optimization guarantee.

## Claims and essential experiments

| Question | Evidence / experiment | New fits | Role in paper |
| --- | --- | ---: | --- |
| Can both encoder families train through the proposed classical interface? | Existing reference/gradient/preparation checks and 12 sampled-versus-exact development fits | 0 | Methods and appendix; engineering evidence, not held-out generalization |
| Can one fixed IQP unitary predict new joint input combinations? | Completed n=8: two encoders, ten paired seeds, 16 perimeter train / nine interior test conditions | 0 | Primary result; preserve the frozen prediction seal |
| What does supplying the uncoupled spatial law change? | Same n=8 circuit/data budget; encoder-only versus trained outputs; full classical context | 0 | Main encoder-family comparison, not topology-only causality |
| Do the outputs capture spatial arrangement and dependence beyond their marginal profiles? | Two descriptive transformations of existing saved distributions, below | 0 | Main diagnostic panel plus appendix tables; explicitly post hoc at n=8 |
| Does the demonstration extend beyond four-site rings? | New n=12 replication: two encoders, ten paired seeds; same split/domain/kappa | 20 | Recommended second-size result; report either outcome |

### E1. Preserve the completed eight-qubit primary result

Use the frozen [stage-3 protocol](ashkin-teller-confirmation-protocol-2026-09-07.md), not another rerun with favorable seeds. Both models have 30 independent two-local phases, no ancillas, 500 sampled-MMD updates, 1,000 training and 1,000 independent validation samples per training condition. Predictions from all 20 IQP fits and ten classical bundles were sealed before test-target access.

Keep all six curves/rows: symmetry-only input, symmetry-only IQP, bond-informed input, bond-informed IQP, empirical log composition and pooled log-linear. Mean test TVD is respectively .33999, .13271, .27707, .09325, .09881 and .02595. The paired bond-minus-cat TVD difference is −.03946, percentile bootstrap 95% interval [−.04175,−.03703]. Both circuits improve joint TVD and connected-correlation error over their inputs in all 90 seed/condition pairs. These are exact evaluation metrics after sampled training.

The informed input supplies the full uncoupled ring law, ring labels, g(c)=c and fixed eta=pi/8. At kappa=.3 that input is not the product of the true coupled-target marginals. The symmetry-only input knows the partition, two flips and fixed affine condition map. Neither input is prior-free. Changing their families changes more than topology, including their response to the condition and their input phases.

### E2. Diagnose structure without retraining

Use every saved n=8 seed and every test condition. Keep predictions and target laws immutable. These diagnostics are chosen after seeing n=8 results and must be called descriptive post hoc analyses. Predeclare them for n=12 before its target evaluation.

**Spatial arrangement at a fixed joint weight profile.** Let E q uniformly redistribute probability within each (K_A,K_B) shell, preserving the complete joint layer-weight distribution. This is the average over independent within-layer site permutations and can be computed by shell sums, without enumerating permutations. Report

\[
 \Delta_{\rm spatial}=\operatorname{TVD}(E q,p)-\operatorname{TVD}(q,p).
\]

A positive value means the model's labelled arrangement improves target accuracy beyond its own joint weight profile; a negative value means removing its spatial variation improves accuracy. This is an output diagnostic, not an optimally refitted exchangeable baseline. Also show TVD(E p,p) as a target-only measure of departure from exchangeability, not as the optimal exchangeable error or a predictive comparator. Retain raw TVD as primary; do not replace it with profile TVD.

**Interlayer dependence at fixed full marginals.** Form r(s,t)=q_A(s)q_B(t) from each trained output and report

\[
 \Delta_{\rm dependence}=\operatorname{TVD}(r,p)-\operatorname{TVD}(q,p).
\]

This preserves every within-layer statistic and removes interlayer dependence. A positive value shows that the output dependence improves joint accuracy relative to its own marginals. It does not compare with the best independently trained product model or causally isolate particular circuit gates. Report both layer-marginal TVDs, the existing aligned connected-bond RMSE, and the signed diagnostic even if unfavorable. Encoder-only inputs factorize and provide a zero-dependence sanity check.

For each diagnostic average conditions within seed, then summarize ten paired seeds. Include condition means and counts of negative seed/condition values. The existing connected statistic C_i=E[A_i B_i]−E[A_i]E[B_i] remains essential: cross-layer spin correlations E[s_i t_j] vanish by independent flips and would be a misleading test of whether dependence was learned.

### E3. Recommended twelve-qubit replication

Run the same finite-ring family at m=6 sites per layer. This is a new trained size and new target laws, not transfer of learned n=8 parameters. Retain L={.10,.22,.38,.54,.70}, the 16 perimeter conditions, nine interior tests, kappa=.3, both maps, eta, no ancillas and the complete two-local bank. There should be 78 independent phases at n=12; verify character deduplication explicitly instead of hardcoding the n=8 reduction.

Proposed main budget: two encoders x ten new paired seeds = **20 IQP fits**; ten paired data archives and ten classical bundles containing both baseline families. Use 1,000 train / 1,000 independent validation examples per training condition, 500 updates, batch 100, 2,048 reference draws, 512 masks per band/condition, the same Adam schedule and the existing expected-bodyness bandwidth rule evaluated at n=12. Share data, initialization, batches and mask banks between encoders within seed. Choose checkpoints on training-condition validation MMD only. Initialize and train from scratch; do not warm-start from n=8 or select seeds based on results.

Before committing this budget, audit the six-site reference against exact amplitudes, moments and gradients at predeclared perimeter conditions and nontrivial phase vectors, verify support/symmetry/preparation identities and phase-bank equivalence, and time a short sampled-training smoke run for each encoder at training conditions only. Reserve one engineering seed, e.g. 3700, and 50 updates per encoder (two short runs); use distinct proposed confirmation seeds 3711–3720 after checking collisions. This is a correctness/runtime gate, not an accuracy screen. Do not open any n=12 interior target laws during this work. Existing m=6 checks may be reused where they already establish an identity; add only missing size-specific coverage.

Freeze executable choices, independent data, source, environment and all condition roles after the audit and before the main fits. If numerical correctness fails, stop and document the issue before designing a repair. If only accuracy is poor, report the full frozen result; do not increase samples, steps, phase count or change maps using test scores. The smoke timing supplies a resource estimate; no reliable n=12 wall-clock estimate is claimed here.

**Classical scaling must be explicit.** At n=12 there are 1,024 flip orbits. Pooled affine orbit logits have 3(1,024−1)=3,069 coefficients; perimeter histograms have 16(1,024−1)=16,368 degrees of freedom. With 1,000 examples per condition sparsity matters. For log composition, preserve the n=8 grid of total pseudocount mass {6.4,64,640}, hence pseudocount per orbit {.00625,.0625,.625}. This avoids silently multiplying prior mass sixteenfold. Keep the pooled model's six-candidate lambda grid and objective normalization from n=8, validating convergence at training conditions before freeze. Select all hyperparameters with training-condition validation likelihood only. Record candidate diagnostics and every selected value. This size-aware classical specification is new and must not alter the completed n=8 result.

Use exact bitstring evaluation over 4,096 states, all nine conditions, both encoder-only controls and both classical predictors. Seal every prediction bundle before test targets. Apply the existing primary bond-minus-cat mean-TVD analysis within n=12 (ten paired seeds, 10,000 bootstrap draws); other metrics are descriptive. Present n=8 and n=12 separately. The new study is a follow-up informed by n=8, not a second untouched design-selection exercise, and the two sizes are not extra seed replications. Do not claim matched capacity across sizes: phases rise from 30 to 78 while the data budget is held fixed.

## Experiments to leave out of the core study

- **Ancilla ladder:** already a central axis in the other benchmarks. Here it changes capacity and conditioning and dilutes the encoder-interface question. Add only if adopting a distinct ancilla claim for this family.
- **New four-body campaign:** the existing 2x2 development ablation can remain in the appendix, labelled development. If later testing a circuit effect, freeze both encoders with both banks on a new evaluation design; adding only bond/four-body after seeing confirmation is not a matched comparison.
- **Wrong-topology or generic product controls:** necessary only for a causal claim about correct ring alignment or symmetry enforcement. The proposed claim compares encoder families with declared information. Four-site cycles also make wrong-topology controls weak.
- **Independent quantum banks and target-aware fits:** needed for a sharing-cost or advantage-over-independent-interpolation claim on this benchmark, not for showing one shared circuit's response. Do not generalize the existing Gaussian/Curie–Weiss comparator conclusions to this case.
- **Kappa sweep or another extrapolation search:** changes the question and adds design freedom. Retain the failed extrapolation as a limitation; do not seek a replacement favorable split. Conditioning on kappa is not tested.
- **Hardware/noise or quantum-advantage demonstrations:** outside this study's scope. Efficient reference access and a preparation construction do not establish efficient classical output sampling or an executed hardware implementation.

## Proposed paper presentation

**Main text.** Add a short motivation/target subsection, one two-column figure, and a compact information/resource table. The figure should show: (a) perimeter/interior condition roles; (b) all six models' exact mean test TVD with seed uncertainty, separated by size; (c) connected-bond error with the same model coverage; (d) the two structural diagnostics for trained encoders, if legible. Split the last panel into a separate figure rather than shrink text if necessary. The resource table distinguishes supplied information, trainable phase count, classical coefficient count, reference-access complexity and quantum preparation assumptions. Show encoder-only points as deterministic controls, not independent seed replications.

**Appendix.** Transfer-reference and preparation derivations; support/gradient checks; sampled-versus-exact development comparison; all condition-level TVD, marginals and connected errors; full seed summaries and frozen baseline tuning; original exploratory/development history and failed extrapolation. The existing full condition grids should remain available even if the main figure uses summary panels. No favorable-condition filtering.

Keep this experiment's convention explicit: mean-over-condition, then mean-over-seed with paired bootstrap intervals. The earlier paper campaign emphasizes medians; either explain the differing predeclared summaries in captions or provide supplementary alternate summaries labelled descriptive. Do not silently replace the primary n=8 statistic for stylistic consistency.

## Draft motivation paragraph

“The Gaussian and Curie–Weiss examples probe conditional families whose probabilities depend only on collective magnetization. We next consider two coupled periodic Ising rings, where labelled bond structure distinguishes configurations with the same magnetization and two couplings index the target family. This example tests whether the same input-conditioned IQP construction can represent spatial and interlayer dependence at new combinations of known coupling values. We compare two tractable inputs that both enforce independent layer-flip symmetries: a symmetry-only cat construction and a bond-informed coherent-ring construction. The latter supplies uncoupled within-layer structure, making it necessary to measure separately what the learned shared unitary contributes.”

## Draft discussion based on established n=8 evidence

“The eight-qubit experiment extends the demonstration from exchangeable families to a spatially structured target with two condition variables. Both encoder families improve joint distribution error and the measured connected bond correlations relative to their untrained inputs at every tested seed and condition. Bond-informed inputs yield lower mean held-out TVD under the same 30-phase IQP bank and sampling budget. This comparison concerns the complete encoder families and their supplied information; it does not isolate a topology-only effect. The bond construction already contains the uncoupled Ising law and its coupling dependence, but supplies no interlayer dependence.”

“The pooled classical log-linear predictor is substantially more accurate. Its affine conditional log odds are well suited to this target's exponential-family form, and it uses a different parameterization and larger parameter budget. These results demonstrate a conditional quantum construction with a tractable training interface rather than predictive or computational quantum advantage. Lower average quantum error also does not imply uniformly better dependence recovery: at (a,b)=(.54,.54), symmetry-only IQP has lower connected-bond error than bond-informed IQP, while both retain their largest condition-mean correlation errors near .15 at (.22,.22).”

“The held-out conditions test new combinations within the training domain, not extrapolation, critical phenomena or learning unknown physical laws. Successful interpolation does not resolve the previously observed extrapolation failure. Reference access is efficient in the stated transfer-contraction arithmetic model; state preparation, numerical precision, hardware routing and output sampling are separate questions.”

The exponential-family interpretation is a direct calculation from the declared target: with fixed kappa, log orbit odds are affine in a,b. It explains favorable model specification; it does not prove a finite-sample baseline must win. Do not attribute its result solely to parameter count. Add n=12 conclusions only after its results exist; a weaker second-size result would delimit the claim rather than trigger suppression or tuning.

## Claim-to-evidence and execution order

| Proposed claim | Evidence | Figure/table | Limitation | Status |
| --- | --- | --- | --- | --- |
| Conditional generation on spatial two-input targets | Sealed n=8 report; proposed n=12 replication | Six-model TVD and condition split | Within-domain combination holdout | n=8 verified; author review pending |
| Correlated tractable inputs fit the estimator interface | Reference method and numerical identities | Methods/resource table; appendix | Precision/routing assumptions, no hardware claim | Implemented and checked |
| Learned output structure adds beyond input and weight profiles | Encoder-only comparison; proposed E2 | C-error and structural diagnostics | E2 post hoc at n=8; no best-product/exchangeable claim | Input gains verified; E2 planned |
| Bond-informed family lowers n=8 mean TVD | Frozen paired difference | Primary six-model table | Information and encoder form change together | Verified at n=8 |
| Robustness at a second trained size | Proposed E3 | Separate n=12 results | Not size transfer, asymptotic scaling or matched capacity | Planned |
| Quantum advantage / topology-only causality / uniform recovery | No supporting isolated experiment | None | Outside the proposed claim | Not claimed |

1. Agree the core scientific scope and paper emphasis in this plan.
2. Implement E2 as a read-only diagnostic report, verifying shell redistribution, marginal preservation and independence identities; label n=8 outputs post hoc. Preserve original confirmation summaries.
3. Audit/time n=12 using training conditions only; make the full configuration and budget concrete, then freeze the new study identity.
4. Execute all 20 n=12 fits and classical bundles, seal, evaluate and report regardless of ranking. If the resource audit makes this impractical, retain the n=8 scope and state the limitation.
5. Assemble the main figure/resource table and integrate the text around the actual evidence. Reconcile Abstract, Introduction, Results, Discussion and Conclusions so claims no longer say all evaluated targets are exchangeable. Preserve the distinct primary statistics and existing campaign denominators.

Only this plan is produced now. Steps 2–5 remain proposed work; no experimental launch or manuscript insertion is implied by writing it.


## Execution update

The author authorized the proposed experiments. E2 is complete and explicitly post hoc at n=8: [structural results](ashkin-teller-structure-result-2026-09-07.md). Removing spatial arrangement or interlayer dependence increases TVD for both IQP encoders at every seed/condition pair. These are transformed-output diagnostics, not refitted baselines.

The n=12 engineering audit passed, including a stable full suite of 243 tests, both training-only 50-update smoke trajectories, and all six classical solver candidates. The [new protocol](ashkin-teller-n12-protocol-2026-09-07.md) is frozen under `artifacts/ashkin-teller-n12-2026-09-07/`; 20 IQP fits are running and ten classical bundles are saved. No n=12 held-out target laws have been evaluated. Paper text integration follows the complete verified result.

## Completed twelve-qubit evidence

The separately frozen replication is complete: all 20 IQP fits and ten classical bundles passed sealing before held-out target access, followed by independent numerical verification. See the [n=12 primary report](ashkin-teller-n12-result-2026-09-07.md) and [predeclared structural diagnostics](ashkin-teller-n12-structure-result-2026-09-07.md). All four generated figures were inspected as PNGs and rendered PDFs. Manuscript insertion and author scientific review remain separate.

Mean test TVD is .27367 for symmetry-only IQP and .18930 for bond-informed IQP; the paired difference is −.08437, with 95% bootstrap interval [−.08919,−.08016]. Their inputs have TVD .63845 and .24999, respectively. Both trained models improve joint TVD and connected-bond error over their own input in all 90 seed/condition pairs. Empirical log composition gives .31454 and pooled log-linear .10883. Keep all six models in the paper.

Spatial arrangement helps both quantum outputs in all 90 pairs. Mean spatial/dependence TVD increases after transformation are .12190/.01033 for symmetry-only IQP and .17557/.05541 for bond-informed IQP. The dependence contribution is not uniformly helpful for symmetry-only IQP: 13/90 pairs have negative deltas, and the condition mean at (.54,.54) is −.00197. Bond-informed IQP has positive dependence deltas in all 90 pairs. These are transformations of each output, not refitted baseline comparisons. The bond input itself already has a large spatial delta (.18480); the trained delta being smaller does not contradict its lower raw TVD, because the transformed distributions also change.

Connected-bond errors remain substantial: .11622 for symmetry-only and .11272 for bond-informed IQP, versus .01536 for pooled log-linear. Bond-informed IQP has higher mean correlation error than symmetry-only IQP at four of the nine conditions. Its layer-marginal errors can also worsen relative to its input (29/90 A-marginal and 28/90 B-marginal pairs), despite every joint-TVD comparison improving. State the metric when claiming improvement.

The two sizes support the encoder-interface demonstration and compositional interpolation, with clear limitations. Absolute trained quantum errors increase at n=12 under the fixed sample and update budgets. The larger bond-versus-cat gap does not establish superior scaling: the sizes have different phase counts, target distributions, and input errors. No settings were changed in response to n=12 outcomes. The classical result and the previous failed extrapolation remain part of the discussion.

### Figure and resource table ready for manuscript review

[Combined-size figure](../../artifacts/ashkin-teller-n12-2026-09-07/paper-figures/ashkin_teller_paper_summary.pdf) and [information/resource table](ashkin-teller-paper-resources-2026-09-07.md).

Suggested caption: “Conditional generation on two coupled periodic Ising rings. (a) Sixteen perimeter training conditions and nine held-out interior combinations, shared by both sizes. (b,c) Exact held-out joint TVD and connected-bond correlation RMSE after sampled-MMD training. Circles denote n=8 and squares n=12; parameters are trained separately at each size. Points average nine conditions within each seed and then ten seeds; error bars show percentile 95% bootstrap intervals over seeds and may be smaller than the markers. Grey encoder-only inputs are deterministic controls. Cat and bond denote symmetry-only and bond-informed input families. The classical predictors share the sampled training/validation data but use different parameterizations. No predictive or computational quantum advantage is claimed.”

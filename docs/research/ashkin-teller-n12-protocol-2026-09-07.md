# Twelve-qubit Ashkin–Teller replication protocol

Proposed execution record to be frozen after the training-only engineering audit and before fitting. The immutable manifest and source archive in `artifacts/ashkin-teller-n12-2026-09-07` are authoritative. This is a follow-up informed by the n=8 results, with fresh n=12 target laws and independently trained parameters. It is not size transfer or a scaling theorem.

## Fixed design

Two periodic six-site rings, 12 visible qubits, no ancillas, kappa=.3. Target log mass is a sum of layer bond energies weighted by (a,b), plus .3 times the sum of aligned cross-layer bond products. L={.10,.22,.38,.54,.70}; train at the 16 perimeter points of L squared and test at the nine interior combinations {.22,.38,.54} squared. Lexicographic ordering. No interior target evaluation before the full prediction seal.

Symmetry-only: product of two cat inputs, alpha(c)=.225*pi−.20*pi*(c−.1)/.6, repeated across each six-site layer. Bond-informed: product of coherent Ising rings with g(c)=c and eta=pi/8. Both enforce independent layer flips; neither unrestricted bank enforces all target spatial symmetries. Same complete one-/two-body bank: verify 78 independent phase characters on even-even support. Use all 78; no four-body additions or phase tying.

Ten paired confirmation seeds3711–3720. Training-only engineering seed3700 is excluded. Draw 1,000 training and 1,000 independent validation examples at each perimeter condition using NumPy default_rng SeedSequence([seed,4,role,condition_index]). All models receive the same data per seed. Initialize phases from pooled training examples, verifying deduplication preserves full phase action.

500 Adam-cosine updates, .01 to .001; batch100/condition; 2,048 fresh reference draws per band/condition/update; 512 masks per band/condition; existing three expected-bodyness bandwidths evaluated at n=12. Same batches, mask banks and initial phases across paired encoders. Fixed sampled validation MMD at training conditions only, evaluated initially and every25updates; select its minimum, including initial. Freeze all settings and source before main fitting. Record full histories, checkpoints, batches, masks and selected parameters.

## Engineering gate and resource scope

Before freeze: six-site physical reference/amplitude/phase-gradient, parity, preparation and bank-rank checks; one 50-update smoke per encoder on training conditions only; one training-only classical bundle checking all candidate solver convergence. Require finite complete trajectories and numerical identities, not favorable fit accuracy. These runs do not evaluate n=12 interior laws. Timing informs scheduling only. If numerical checks fail, diagnose before freeze. No performance-triggered expansion of sample count, maps, phases, steps or seed selection.

## Classical baselines

Use the independent flip group masks {0,63,4032,4095}, 1,024 equal four-state orbits. The empirical predictor uses actual perimeter histograms and
L(a,b)=.5 logq(a,.1)+.5 logq(a,.7)+.5 logq(.1,b)+.5 logq(.7,b)−.25 sum_four_corners logq,
then orbit softmax and divide each orbit mass by four. Pseudocount total mass grid {6.4,64,640}, giving per-orbit lambda {.00625,.0625,.625}. Choose minimum perimeter-validation orbit NLL, grid-order ties. This preserves total prior strength from n=8, not per-orbit strength. Histograms have16*1023 degrees of freedom.

Pooled orbit log-linear model: 3,069 coefficients, features [1,(a−.4)/.3,(b−.4)/.3], last orbit logit zero. Mean-per-condition NLL +lambda/2 sumcoef² including intercepts. Lambda {1e−6,1e−5,1e−4,1e−3,1e−2,1e−1}. Zero initialization per candidate, L-BFGS-B maxiter5000, ftol1e−14, gtol1e−8. Require finite success and gradient infinity norm<=1e−6; failed candidate retries same objective from last iterate with maxiter20000; persistent failure stops evaluation for diagnosis. Select unpenalized perimeter-validation NLL, first grid-order tie. Retain every convergence record, validation score, selected lambda, coefficient and training count. No hidden ring sufficient statistics.

## Seal, evaluation and uncertainty

Save all20 IQP prediction bundles and ten classical bundles, including selected phases and zero-phase inputs. Verify source/data hashes, complete finite histories, selection, paired initialization/batches/masks, normalization, flips, and recomputed quantum/classical predictions. Seal all60 JSON/NPZ files. Only then write evaluation-started marker and open the nine interior target laws. The marker closes further fitting. Retry read-only evaluation only with unchanged seal; no seed filtering or outcome stopping rule.

Exact full-bitstring TVD, both layer-marginal TVDs and RMSE of six connected aligned cross-layer bond correlations. Six models: two inputs, two trained IQP encoders and two classical predictors. Report every condition and seed, seed means and worst-condition errors. Primary contrast: bond-minus-cat mean test TVD, first average nine conditions within each seed. Bootstrap ten paired seed indices10,000times, RNG950701, percentile95% interval. Per-model and secondary intervals descriptive, no multiplicity-adjusted significance claim. Analyze separately from n=8; do not pool sizes as replications.

Predeclared secondary structure diagnostics for all saved model outputs:
- Delta_spatial=TVD(Eq,p)−TVD(q,p), with Eq uniform within each joint layer-Hamming-weight shell and preserving that shell mass.
- Delta_dependence=TVD(q_A*q_B,p)−TVD(q,p), preserving full within-layer marginals.
- TVD(Ep,p), target-only departure from exchangeability, not a predictive comparator or optimal exchangeable error.
Report signed seed/condition results, condition means, negative counts and seed-bootstrap summaries. These transformations are not refitted optimal baselines or isolated gate/topology interventions. They were post hoc at n=8 and are predeclared here.

## Interpretation

Report the full study regardless of which encoder or classical predictor wins. Compare supplied information explicitly: bond supplies the uncoupled ring law and map, cat supplies layer symmetry/partition and its map. Do not attribute an encoder-family difference solely to topology. Preserve correlation exceptions, classical superiority, numerical/resource assumptions and previous failed extrapolation. This study tests compositional interpolation and a second trained size, not quantum advantage, critical scaling, conditioning on kappa, unknown-physics discovery, hardware execution or uniform dependence recovery. Author scientific review and manuscript text integration follow verified evidence.

# Reference-estimator precision and cost study

The author selected the estimator study first; the data-efficiency study is deferred. This study evaluates the current sampling-and-ratio interface, without new generative fitting, target selection, or manuscript claims inferred from optimization loss.

## Fixed fixtures

`artifacts/reference-estimator-study-2026-09-07/manifest.json` fixes 60 cases and every parameter, angle, generator and observable. SHA-256: `6e90ce608eaa31e8377bffca5d3b9e040b2cfc30b61b48812ffd4b04935c43f5`.

- 54 synthetic cases: 8/12/16 total qubits; product, two-block symmetry-only and two-ring bond inputs; coupling coordinates (.1,.7), (.4,.4), (.7,.1); Gaussian phase draws at standard deviations .1 and .4. Within each width/regime the same phase vector and complete one/two-body bank are used for all encoders and conditions. Product and cat angle maps are explicitly recorded. No ancillas.
- Six trained fixtures: smallest declared production seed at each of n=8/12 for Gaussian product r=0 and AT cat/bond. Gaussian c=.5 and AT (.38,.38) were chosen before inspecting estimator errors. Actual selected checkpoints and bank artifacts are hash-identified; n=8 AT retains its 30-generator symmetry-sector bank, not the redundant 36-generator full bank. These are descriptive trained checks, not representative samples of trained-model variability.
- Sixteen fixed Pauli masks per width span local, cross-layer and higher-body observables. Identity and parity-forbidden controls are retained and reported separately from active-mask error. This is a fixed observable audit, not an estimate of mask-sampling error or full-kernel MMD accuracy.
- Reference budgets 128, 512, 2048, 8192; 32 independent Monte Carlo repetitions at each case/budget. All moment and phase-gradient coordinates are retained. Total: 240 case/budget bundles and 7,680 repeated joint moment/Jacobian evaluations.

## Reference and gates

Exact moments and full gradients use enumerated probabilities and JAX differentiation. Before the production run, a separate check constructs each physical input directly (product amplitudes, normalized two-block cats, phased classical ring amplitudes), applies NumPy Hadamard transforms, and compares moments and four analytic phase derivatives with the JAX reference. Required absolute agreement: 1e-9 for all 60 fixtures. This is a correctness gate; estimator accuracy has no favorable-result gate.

Pilots use the same predetermined central-condition initial-scale cases at n=8 on Halo 1 and n=16 on Halo 2, budgets 128/8192 and four repetitions. They test finite output, compilation, timing and memory; their estimates are excluded from production summaries. No fixture or budget is selected using favorable pilot error.

## Execution and measurement

Both machines are AMD Ryzen AI MAX+ 395 systems with 124 GiB RAM. Existing matched Python 3.13.14/JAX 0.11.0/NumPy 2.5.1 CPU environments are used. Pin one sequential worker per machine to CPUs 0–15, OMP_NUM_THREADS=16, OPENBLAS_NUM_THREADS=1, JAX_PLATFORMS=cpu. No GPU acceleration is claimed. Synthetic phase regimes are split across machines so every encoder/width is represented on both. Trained cases are balanced separately. Archive the current estimator source and fixtures before the full run; output identity checks reject changed sources/manifests or altered NPZ files on resume.

Compile using lower/compile separately from execution; synchronize exact, initial and repeated sampled executions. Report setup, exact-oracle compile/first execution, sampled compile/first execution and warmed sampled calls separately. The warmed joint call includes reference draws, ratios, moments and all phase derivatives. Host PRNG-key construction also lies inside the timed call wrapper and is a fixed overhead. RSS is the process high-water mark, includes exact/JIT allocations and prior cases, and must not be described as isolated estimator memory scaling.

## Analysis and scope

Use absolute errors. Compute per-case RMSE across independent repetitions and active observable/gradient coordinates. For synthetic summary curves, average case MSE equally over the six condition/regime fixtures for each encoder/width, then take the square root. Bootstrap entire replicate vectors within each case, independently across cases, for percentile 95% intervals (2,000 resamples, fixed analysis seed). Observables/gradient coordinates are not independent replications. Show trained checks separately. Empirical bias estimates are descriptive and do not prove unbiasedness.

Report time medians and descriptive case spread without removing slow cases. Width changes also change the number of phase derivatives. Controls, outliers and any failure remain recorded. Success means complete verified artifacts and honest precision/cost curves, not a particular slope or encoder ordering. These measurements do not establish optimization convergence, generalization, output-sampling tractability, asymptotic scaling, or quantum advantage.

Planned outputs: accuracy figure (moment/gradient error versus reference count), cost figure (warmed execution and compilation versus width), full machine-readable results, trained-check table, and a concise interpretation for manuscript review.

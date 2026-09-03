# Large-scale IQP training and data-based initialization

## Scope and bottom line

This report uses primary sources only. It fixes the scaling paper at arXiv v2, revised 6 February 2026, the official reproduction repository at commit [`8e5620c`](https://github.com/XanaduAI/scaling-gqml/tree/8e5620cfa71f9a4e4254597b2dfe1692231d3723), and IQPopt at commit [`bf20750`](https://github.com/XanaduAI/iqpopt/tree/bf207502c317364746bd56ae0dd1b4d04e492c7b). The reproduction repository pins `iqpopt==2024.7.0` in its [plotting environment](https://github.com/XanaduAI/scaling-gqml/blob/8e5620cfa71f9a4e4254597b2dfe1692231d3723/paper/plots/requirements.txt#L1-L2). It has no tagged paper release, while the paper links an older code snapshot, `b907bb1`, for its hyperparameter search.

The paper establishes that IQP loss and gradient estimators can train unconditional binary models with 484 to 1,000 qubits on classical compute. It does not establish large-circuit output sampling, quantum-hardware deployment, conditional generation, or a scalar covariance-recovery result. Its covariance evidence consists of heatmaps. These limits should remain explicit when positioning a 196-qubit conditional covariance study. [Paper](https://arxiv.org/html/2503.02934v2) [PDF](https://arxiv.org/pdf/2503.02934v2)

## What the large experiments did

Every final large IQP used every one-body and two-body $X$-string generator. The parameter count was therefore

\[
P=n+\binom{n}{2}=\frac{n(n+1)}{2}.
\]

| Data | Samples and construction | Final IQP | Reported outcome and qualification |
|---|---|---|---|
| D-Wave | 484-bit configurations from a D-Wave Advantage Pegasus system after a $100\,\mu\mathrm{s}$ anneal; 10,000 train and 60,000 test | 117,370 parameters; 2,500 Adam updates | No model captured the target well. IQP had much lower test MMD than the bitflip, RBM, and EBM comparisons, but its correlations were weaker than the data. The D-Wave machine generated the dataset. It did not run the trained IQP. |
| Binarized MNIST | Thresholded and flattened 784-pixel images; 50,000 train and 10,000 test in the paper and repository | 307,720 parameters; 3,000 updates | Test MMD was similar to the paper's $p=0.3$ corruption control. The covariance pattern resembled the data at lower amplitude. The authors did not generate IQP images and state that sampling would be needed to confirm image quality. MNIST had no RBM or EBM comparison. |
| Scale-free Ising | 1,000 spins on a Barabasi-Albert graph; samples came from eight Markov chains; 20,000 train and 20,000 test | 500,500 parameters; graph-local hyperparameter search followed by an all-to-all final fit; the current YAML specifies 2,000 updates | IQP was the only model with reasonable results across most bandwidths, but the bitflip model was best at the smallest bandwidth. All recovered correlations were weaker than the target. KGEL failed for every model. |
| Genomic | 805 biallelic SNPs; 5,008 haplotypes from 2,504 individuals; one-third test split | 324,415 parameters; 1,000 updates | IQP test MMD lay between published GAN and RBM results depending on bandwidth, and its correlation pattern was weaker. The published comparisons were not equal-budget baselines. The paper reports that they likely used train and test data, that the RBM samples were correlated, and that RBM sampling used unavailable extra genomic data. |

The paper reports the dataset construction and interpretations in Section 8. The exact final configurations are in [`best_hyperparameters.yaml`](https://github.com/XanaduAI/scaling-gqml/blob/8e5620cfa71f9a4e4254597b2dfe1692231d3723/paper/training/best_hyperparameters.yaml#L67-L152). Two artifact inconsistencies matter. Table 3 marks the scale-free IQP training budget as unavailable, while the current YAML specifies 2,000 updates. The paper calls 50,000 and 10,000 images the standard MNIST split, but [`download_data.py`](https://github.com/XanaduAI/scaling-gqml/blob/8e5620cfa71f9a4e4254597b2dfe1692231d3723/paper/datasets/MNIST/download_data.py#L8-L14) takes both subsets from the canonical 60,000-image training set and does not use the canonical 10,000-image test set. The repository does not commit the generated covariance CSV files or the full six-dataset gradient-result pickles, so those results cannot be reprocessed directly from raw committed outputs. Its MMD CSV files also mix bandwidth grids, including values that differ slightly from the current YAML. Any numerical reuse should select rows explicitly rather than aggregate whole files.

### Training protocol

The large IQP fits shared the following protocol.

- The loss was the mean of three Gaussian-kernel MMD-squared terms. Their bandwidths were D-Wave `{7.762, 6.151, 3.928}`, MNIST `{9.887, 7.391, 3.390}`, scale-free `{11.169, 8.348, 3.825}`, and genomic `{10.019, 7.701, 4.269}`. These probe average Pauli weights `{2,4,8}`, `{2,4,17}`, `{2,4,17}`, and `{2,4,11}`, respectively.
- Each update used 1,000 sampled Pauli operators, 1,000 Monte Carlo samples for the IQP expectation estimator, the full training set, JAX differentiation, and Adam. All four final large-model configurations used a learning rate of $10^{-3}$. [`train_iqp.py`](https://github.com/XanaduAI/scaling-gqml/blob/8e5620cfa71f9a4e4254597b2dfe1692231d3723/paper/training/train_iqp.py#L51-L122)
- Single-body angles used the data marginals. Two-body angles used covariance after mapping bits to $\{-1,+1\}$. The final two-body scale was `0.001` for D-Wave, MNIST, and genomic, and `0.01` for scale-free. Higher-body noise was zero for these models.
- Hyperparameter search used one train-validation split. The authors did not run repeated cross-validation. They selected the lowest mean validation MMD over the training bandwidths and then refit once on all training data.
- The IQP and stochastic bitflip models shared gate sets, initialization, loss, and optimizer where the comparison was made. The other baselines were scikit-learn `BernoulliRBM` and a feedforward neural EBM trained by contrastive divergence. Uniform data and corruption controls appeared where applicable. The paper calls the RBM and EBM non-state-of-the-art.
- All reported calculations ran on CPU except the D-Wave EBM baseline, which used a GPU.

### What covariance evaluation measured

The paper defines the covariance of the $\{-1,+1\}$ variables as

\[
C_{ij}=\langle Z_iZ_j\rangle-\langle Z_i\rangle\langle Z_j\rangle.
\]

For an IQP model, IQPopt estimates all upper-triangular one-body and two-body expectations and sets $C_{ii}=1-\langle Z_i\rangle^2$. For sample-based models, the paper code maps `0/1` to `+1/-1` and calls `np.cov`, whose default is the sample covariance. The reproduction script uses 3,000 Monte Carlo samples for each IQP expectation batch. It then plots heatmaps on a fixed `[-1, 1]` color range. [`construct_convariance_matrix`](https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/utils.py#L228-L269) [`calc_cov_matrix.py`](https://github.com/XanaduAI/scaling-gqml/blob/8e5620cfa71f9a4e4254597b2dfe1692231d3723/paper/plots/calc_cov_matrix.py#L18-L197) [`plot_cov_matrix.py`](https://github.com/XanaduAI/scaling-gqml/blob/8e5620cfa71f9a4e4254597b2dfe1692231d3723/paper/plots/plot_cov_matrix.py#L79-L84)

The paper uses these matrices for visual inspection. It reports no covariance RMSE, normalized Frobenius error, upper-triangle correlation, confidence interval, or repeated-estimator analysis. Quantitative evaluation instead uses held-out MMD-squared and KGEL. A covariance heatmap can show a correct pattern while hiding amplitude shrinkage, which the authors observed in every large experiment.

## Limits that should carry into our paper

The following statements are supported.

- Classical estimators can optimize and evaluate selected observables of large unconditional IQP models.
- Data-scaled initialization allowed the authors to obtain trainable large-model runs under their protocol.
- A 196-qubit complete two-local model is smaller than the models trained in the paper.

The following statements are not supported by that work.

- The trained large IQP models were deployed or sampled on quantum hardware.
- Their generated samples have the quality suggested by MMD or covariance estimates.
- The method learns conditional distributions or interpolates to held-out conditions.
- The heatmaps establish quantitatively accurate covariance recovery.
- Covariance initialization itself, rather than parameter scale or another property of the ansatz, prevents barren plateaus.
- The comparisons establish an advantage over modern classical generative models or a quantum advantage.

Further constraints strengthen these claim limits. The final fits do not provide a multi-seed performance study. MMD is both the IQP training objective and the main evaluation score, unlike the RBM and EBM objectives. The classical baselines are weak by the authors' account. The random-initialization evidence consists of gradient histograms and failed training attempts, and the authors could not show that the apparently large random-initialization gradients differed statistically from zero. IQP correlations remained weaker than the targets. The paper also proves that the model is not universal without ancillas and notes that adding all higher-body gates becomes combinatorial.

## Exact IQPopt data initialization

### Public API and shapes

The relevant imports are:

```python
from iqpopt import IqpSimulator, Trainer
from iqpopt.gen_qml import mmd_loss_iqp
from iqpopt.utils import initialize_from_data, local_gates
```

`iqpopt.__init__` exports `IqpSimulator` and `Trainer`; it does not export the initialization helpers. The core signatures are documented by the official source:

```python
IqpSimulator(n_qubits, gates, device="lightning.qubit",
             spin_sym=False, init_gates=None, sparse=False, bitflip=False)
Trainer(optimizer, loss, stepsize, opt_jit=False)
local_gates(n_qubits, max_weight=2)
initialize_from_data(gates_list, data, scale=-1, param_noise=0.0)
```

[`IqpSimulator`](https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/iqp_optimizer.py#L11-L43) [`Trainer`](https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/training.py#L12-L43) [`utils.py`](https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/utils.py#L10-L24)

`data` must be a finite binary array with shape `(N, d)`. In practice, `N >= 2` and `d >= 2` are needed for a matrix-valued sample covariance. The implementation does not validate binary values or these dimensions. Each outer element of `gates_list` denotes one trainable parameter. Each inner list denotes an $X$-string generator through its qubit indices. For example, `[[[0]], [[1]], [[0, 1]]]` has three parameters. `local_gates(d, 2)` returns $d(d+1)/2$ such entries. The initializer returns a JAX array with shape `(len(gates_list),)`.

### Implemented mathematics

Let $X\in\{0,1\}^{N\times d}$, $z=2X-1$, and let $C=\operatorname{np.cov}(z^\top)$, which uses the $N-1$ sample-covariance denominator. IQPopt zeroes the diagonal of $C$, defines $M=\max_{ij}|C_{ij}|$, and computes binary means $\mu_i=N^{-1}\sum_r X_{ri}$. It initializes:

\[
\theta_i=\arcsin\sqrt{\mu_i},\qquad
\theta_{ij}=C_{ij}\,s\,\frac{\pi}{M}.
\]

Here `s` is the explicit `scale` argument. The singleton formula makes $\sin^2\theta_i=\mu_i$ for the official circuit's unconditional $|0\rangle$ input when all other angles are zero. The source implements this formula even though its docstring incorrectly says `mean * pi/2`.

With the default `scale=-1`, IQPopt sets

\[
s=\sqrt{\frac{2d}{\lVert C\rVert_F^2}}\frac{M}{\pi},
\qquad
\theta_{ij}=C_{ij}\sqrt{\frac{2d}{\lVert C\rVert_F^2}}.
\]

Any gate with one generator of weight at least three, and any parameter shared across multiple generators, receives an independent normal draw with standard deviation `param_noise`. [`initialize_from_data`](https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/utils.py#L174-L225)

### Minimal use

```python
import numpy as np

from iqpopt import IqpSimulator, Trainer
from iqpopt.gen_qml import mmd_loss_iqp
from iqpopt.utils import initialize_from_data, local_gates

X_train = np.array(
    [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]],
    dtype=np.int8,
)  # shape (N=4, d=3), with nonzero off-diagonal covariance

d = X_train.shape[1]
gates = local_gates(d, max_weight=2)
theta0 = initialize_from_data(
    gates, X_train, scale=1e-3, param_noise=0.0
)
assert theta0.shape == (d * (d + 1) // 2,)

circuit = IqpSimulator(d, gates)
trainer = Trainer("Adam", mmd_loss_iqp, stepsize=1e-3)
trainer.train(
    n_iters=10,
    loss_kwargs={
        "params": theta0,
        "iqp_circuit": circuit,
        "ground_truth": X_train,
        "sigma": [1.0],
        "n_ops": 1_000,
        "n_samples": 1_000,
    },
    random_state=666,
)
```

Three implementation hazards need local guards.

1. If every off-diagonal covariance is zero, the two-body formula divides by zero and can return `NaN`.
2. Higher-body noise uses NumPy's global random state. The function accepts no random generator or seed.
3. [`gates_from_covariance`](https://github.com/XanaduAI/iqpopt/blob/bf207502c317364746bd56ae0dd1b4d04e492c7b/src/iqpopt/utils.py#L144-L163) intends to select the strongest pairs, but the current duplicate check compares tuple indices with nested lists. It can return both `[i, j]` and `[j, i]`. Use `local_gates` or deduplicate and test the result.

The initializer has no condition argument. Its marginal-matching interpretation assumes the official unconditional $|0\rangle$ input. It does not automatically survive a condition-dependent product input. Any condition-aware use in this repository is a new method and should use training data only.

## The normalization confound

A later author repository, [`iqp_initializations`](https://github.com/XanaduAI/iqp_initializations/tree/f6c90ba47ddfcd58aaaf9c22a267a6e75e0585df), is separate from the scaling paper's arXiv v2 evidence. Its committed `training_results.pkl` contains 150-qubit genomic runs with five trials and 3,000 updates. At mean operator weight 2, initialization scale 1, and uniform noise, the median final training MMD was:

| Initialization | No parameter normalization | Divide noise scale by $\sqrt{P}$ | Divide noise scale by $P$ |
|---|---:|---:|---:|
| Random | 0.05466 | 0.000683 | 0.000500 |
| Data covariance | 0.000964 | 0.000714 | 0.000624 |
| Data means without covariance | 0.05375 | 0.000583 | 0.000594 |

These are training losses, not held-out scores. Once the random angles use size-dependent normalization, covariance initialization is not uniquely best in this artifact. The code defines the normalization factors and initialization arms in [`common.py`](https://github.com/XanaduAI/iqp_initializations/blob/f6c90ba47ddfcd58aaaf9c22a267a6e75e0585df/common.py#L56-L144). The result still does not isolate covariance cleanly because the `from_data_nocov` arm also adds noise to one-body angles. The current YAML specifies 6 to 16 qubits and 1,000 trials, which does not reproduce the committed 150-qubit, five-trial pickle. [`mmd_variance_config.yaml`](https://github.com/XanaduAI/iqp_initializations/blob/f6c90ba47ddfcd58aaaf9c22a267a6e75e0585df/mmd_variance_config.yaml#L9-L35)

The defensible conclusion is narrow: unscaled random initialization is a poor control at large $P$. A covariance-specific trainability claim requires random, mean-only, and covariance arms with matched one-body offsets, parameter RMS or norm, noise law, optimizer, data, estimator streams, and at least several independent seeds.

## Implications for a 196-qubit conditional covariance study

A complete two-local 196-qubit IQP has 19,306 angles and 19,110 distinct off-diagonal covariances. This is within the classical optimization scale demonstrated by the unconditional paper. The conditional input and held-out-condition test remain new.

The study should make the following design choices explicit.

- Build initialization statistics from training conditions only. Equalize the number of rows per condition before pooling. Otherwise well-sampled conditions dominate.
- Decide whether the two-body initializer represents total pooled covariance or mean within-condition covariance. Pooled covariance contains both within-condition dependence and covariance caused by condition-dependent mean shifts. IQPopt implements only the pooled raw-data form.
- Compare covariance initialization with a norm-matched random arm and a mean-only arm. Keep one-body angles and every other source of noise identical when isolating the covariance term.
- Evaluate covariance numerically at each held-out condition. Report off-diagonal RMSE and normalized off-diagonal Frobenius error. Upper-triangle correlation is useful for pattern recovery but must accompany an amplitude-sensitive score.
- Report uncertainty across independent training seeds and independent data units. Separate target-data uncertainty from the IQP Monte Carlo estimator error. Reusing one 3,000-sample heatmap is not enough.
- Include a conditional independent-Bernoulli baseline that matches each bit's conditional mean but has zero cross-covariance, plus the matched bitflip model. These controls distinguish mean recovery from dependence recovery.
- Keep held-out MMD or another distributional score beside covariance. Correct two-body moments do not establish full-distribution fidelity.

If the results pass these checks, the paper may claim that a classically trained 196-qubit conditional IQP recovers specified held-out conditional pair correlations under the reported estimator budget. It should not claim high-quality samples, hardware deployment, a full distributional match, quantum advantage, or demonstrated scaling of conditional learning to 1,000 qubits.

## Public real-data replacement for the negative MFC track

The best first-party-supported candidate found in this audit is NOAA NCEI's [Global Historical Climatology Network Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily). GHCN-Daily integrates quality-controlled daily observations from more than 100,000 stations in 180 countries and exposes the station files, inventory, and metadata through an [official public archive](https://www.ncei.noaa.gov/pub/data/ghcn/daily/). The official [format and field documentation](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt) defines `PRCP` in tenths of a millimeter, missing values as `-9999`, and the quality flag. It gives DOI [`10.7289/V5D21VHZ`](https://doi.org/10.7289/V5D21VHZ).

A defensible derived benchmark would select 196 stations with high unflagged precipitation completeness, using training years only. Each complete day would yield a 196-bit vector with bit $j=1$ when station $j$ records at least 1 mm of precipitation. Day of year would supply a seasonal condition. A cyclic pair, sine and cosine of day of year, avoids the discontinuity between December and January. If the model must keep one scalar condition, restrict the study to a non-wrapping seasonal window and use normalized day index.

Freeze the raw archive version, access date, station list, metadata, and file hashes. Split by whole calendar years because adjacent days share weather systems. Select stations and any learned thresholds from training years only. Withhold prespecified day-of-year bins within training years if the goal is interpolation to unseen conditions, then evaluate the same bins on test years. Use week-level or year-level block bootstrap rather than treating days as independent.

This construction is a proposed benchmark, not an existing NOAA benchmark. It retains the useful properties that MFC lacked after bin encoding: real binary vectors, many physically related bits, nonzero spatial covariance, and a meaningful condition. Its risks are missing-not-at-random station records, station and instrument changes, spatial selection effects, and temporal dependence. NOAA states that GHCN-Daily is continually reprocessed and not homogenized for station or instrument changes. Before committing to the track, an eligibility audit must confirm that 196 stations meet a prespecified completeness threshold without imputation. If they do not, reduce the station count or reject this dataset rather than select a favorable panel after model results are known.

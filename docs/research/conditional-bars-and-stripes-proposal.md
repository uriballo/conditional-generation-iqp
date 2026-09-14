# Third benchmark proposal: density-conditioned noisy bars and stripes

Proposed on 2026-09-05. This is a benchmark recommendation, not a frozen
protocol or evidence that the IQP model can learn the target.

## Why this target

The Gaussian and Curie-Weiss targets are uniform inside each Hamming-weight
shell. A third benchmark should require correlations between particular bits.
The local grid-Ising alternative already failed two target-aware capacity
pilots; see [product pilot](grid-ising-capacity-pilot-results.md) and
[equivariant pilot](grid-ising-equivariant-capacity-pilot-results.md).
Those failures do not prove a representational impossibility, but do not
justify another full campaign without a new mechanism.

Bars and stripes is established in quantum generative benchmarking:
[Benedetti et al.](https://www.nature.com/articles/s41534-019-0157-8).
[Liu and Wang](https://arxiv.org/abs/1804.04168) trained quantum circuit Born
machines on BAS using MMD. Neither result establishes trainability for this
project's fixed two-local IQP circuit. The following conditional variant is
our proposal, distinct from the standard uniform BAS law.

## Proposed conditional law

Use 2x4 binary images at n=8 and 3x4 at n=12, flattened in a fixed order.
At condition q:

1. Choose horizontal or vertical orientation with probability one half.
2. For horizontal images, draw one independent Bernoulli(q) bit per row and
   repeat it across that row. For vertical images, draw one per column and
   repeat it down that column.
3. Independently flip every pixel with a fixed probability epsilon=0.05.

The condition controls stripe density. Noise remains fixed; the suggested
value is a design choice, not a selected optimum. Uniform images can arise
through either orientation, so their probability includes both routes.
Even at q=0.5 the clean target is generally not uniform over unique BAS images.

An exact likelihood follows by summing over each stripe's latent bit. Let
L_b(x)=1-epsilon when x=b and epsilon otherwise. For image x,

p(x|q) = 0.5 product_rows [(1-q) product_columns L_0(x) + q product_columns L_1(x)]
       + 0.5 product_columns [(1-q) product_rows L_0(x) + q product_rows L_1(x)].

This has full support and permits exact enumeration of 256 or 4096 outcomes.
It is easily sampled classically and has no implied computational advantage.
The response is nonlinear in q. Conditioning only on orientation mixture
weight would instead be exactly solved by mixing two fixed output laws.

## What it measures

The target tests nonexchangeable spatial patterns at the existing qubit sizes.
It changes the relative probabilities of many stripe configurations as q
changes. Exact raw TVD should remain the headline metric, supplemented by
row/column correlation errors and image panels. A Hamming-weight profile
alone can hide the entire spatial failure.

Direct enumeration of the proposed epsilon=0.05 law at q=0.5 gives TVD 0.4722
for n=8 and 0.6697 for n=12 against a distribution with identical shell masses
but uniform probabilities inside each shell. The midpoint law differs from
the equal mixture of q=0.2 and q=0.8 endpoints by TVD 0.2232 and 0.2707,
respectively. These are target-only calculations, not training results.

## Smallest useful decision experiment

First test independent target-aware fits at n=8, q in {0.2,0.5,0.8},
r in {0,3}, and three development seeds: 18 fits. Use the current MMD,
initialization and checkpoint rules. Freeze an acceptance gate before fitting;
for example, the previous grid screen's median TVD <=0.15 and worst <=0.25
for r=3 can serve as a proposed capacity gate. Do not require ancillas to win
as a condition for reporting a legitimate benchmark. If capacity fails, stop
before conditional encoder design or a broad tuning campaign.

If it passes, compare one shared circuit with the adjacent-output independent
bank and target-aware oracle at interleaved q values. Keep the affine
condition map generic and fixed. Select widths or encoder choices using
training conditions only. The exact classical latent-stripe generator is a
reference showing the target's simplicity, not a learned competitive baseline.

This would strengthen the paper's spatial-correlation coverage. It would still
be a synthetic benchmark, and it would not establish real-data usefulness or
quantum advantage. Keeping the present two-benchmark paper is preferable to
adding an uninformative or poorly learned third task.

## Isotropic-MMD follow-up, 2026-09-05

Isotropy does not identify all images in the same Hamming-weight shell. The
kernel is invariant under a common coordinate permutation applied to both
arguments; that is different from randomly permuting each image independently.
For k(x,y)=exp(-d_H(x,y)/(2 sigma^2)), let rho=(1-exp(-1/(2 sigma^2)))/2.
The exact population squared MMD is

sum_S rho^|S| (1-rho)^(n-|S|) (E_P chi_S - E_Q chi_S)^2,

where chi_S(x)=product_{i in S}(-1)^x_i. All weights are positive for finite
positive sigma. Equal weights for equally sized subsets do not merge their
individual moments. Thus the population loss distinguishes all distributions
on the finite hypercube. This is a direct Fourier expansion of the repository's
kernel in src/iqp_cond/mmd.py, not a convergence guarantee.

For the proposed q=0.5, epsilon=0.05 target, E[s_i s_j]=0.405 for distinct
pixels sharing a row or column, and zero for pixels sharing neither. All
one-body means are zero. Exact enumeration confirms these values.

Against the distribution preserving each target Hamming-shell mass but uniform
inside shells, the three bandwidths from expected_bodyness_sigmas give:

| n | Expected mask weights | Exact squared MMD per bandwidth | Two-body share of each gap |
|---|---|---|---|
| 8 | 1, 2.46, 3.92 | 0.008325, 0.018197, 0.018391 | 94.74%, 64.46%, 25.84% |
| 12 | 1, 3.44, 5.88 | 0.008190, 0.016108, 0.011261 | 95.34%, 46.71%, 6.81% |

At the lowest-body bandwidth, 2,000 nonidentity draws contain about 598 and
593 two-body draws in expectation for n8/n12, counting repetitions. These
are target-versus-shell-control calculations. They do not measure model
gradients, finite-target/model estimator noise or training success.

The appropriate conclusion is that isotropy is not a fundamental obstacle
and the chosen target has accessible low-order spatial signal. Population
identifiability, finite-budget statistical efficiency, representability by
the fixed IQP family, and optimization success remain separate questions.
Prior MMD-trained BAS results use other circuit families, so they cannot
settle the latter two questions here. Run the capacity screen before promotion.

## Screening outcome, 2026-09-07

The [quick capacity screen](noisy-bas-capacity-screen-2026-09-07.md) failed its
predeclared gate under both sampled and exact-population MMD. Do not promote
this proposal to a conditional campaign with the current setup. This is a
one-seed development decision, not proof of unlearnability.

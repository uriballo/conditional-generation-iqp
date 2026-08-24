# Hamming-ball center-count experiment

## Question

Can the product-state angle encoding control the number of centers (mixture
components) of a learned distribution, and does that control generalize
between the trained center counts?

## Target family

For \(n=19\), a class is a mixture of \(K\) planted binary centers with
bit-flip noise \(p=0.08\):

\[
p(x)=\frac1K\sum_{k=1}^{K}p^{d_k}(1-p)^{n-d_k},
\qquad d_k=d_H(x,c_k).
\]

Every center has exactly \(9\) ones, so the Hamming-weight marginal of a ball
depends only on the shared center weight. As a result the shell profile is
identical for every \(K\); \(K\) is visible only in bitstring-level
multi-modality. Shell TVD is therefore meaningless here and is not reported.

Centers are nested prefixes of one seeded stream (`pattern_seed=0`): the
\(K=8\) class contains the \(K=4\) centers, which contain the \(K=2\) centers,
which contain the \(K=1\) center.

## Angle path

The four classes map to encoding angles on a log-\(K\) path:

\[
\alpha(K)=\frac\pi4\frac{\log_2 K}{3},\qquad
K(1,2,4,8)\mapsto(0,\pi/12,\pi/6,\pi/4),
\]

so the encoding controls \(\log_2 K\) linearly, with inverse
\(K(\alpha)=2^{12\alpha/\pi}\).

## Held-out targets

\(K\) is an integer; no exact \(K=5.6\)-center class exists. At each held-out
angle (the midpoint of two adjacent training angles) the target is the equal
distribution-space mixture of the two adjacent classes:

\[
p=\tfrac12 p_{K_{\mathrm{left}}}+\tfrac12 p_{K_{\mathrm{right}}}.
\]

Held-out angles: \(\pi/24,\pi/8,5\pi/24\), corresponding to \(\log_2K\)
midpoints \(0.5,1.5,2.5\).

## Split

- Training: 10,000 samples per class; validation: 2,000 disjoint samples per
  class.
- Held-out mixtures: 2,000 samples per midpoint, used only for the final
  held-out MMD diagnostic; never for optimization or checkpoint selection.
- Model: the 190-parameter complete ansatz with the frozen optimizer, kernel
  bandwidths, estimator budgets, update count, and ten matched seeds from the
  primary experiment. The encoding is a parameter (angle or symmetric
  reupload).

## Readouts

- **Bitstring TVD** against each angle's target (primary distributional
  metric).
- **Effective center count:** the model's exact distribution is fitted as a
  nonnegative mixture of the eight planted ball distributions plus a uniform
  background column (scipy NNLS). Effective \(K\) is the number of ball
  weights carrying at least 5% of the fitted mass; the relative L2 residual
  is reported. The same readout is applied to each target as a ground-truth
  reference.
- **Mode count:** hypercube local maxima with mass at least 1% of the peak.
- **Held-out MMD** on mixture samples (secondary diagnostic).

## Frozen tests

1. **Continuous-conditioning test:** dense-grid interpolation must have lower
   median bitstring TVD than oracle adjacent-angle lookup at every held-out
   angle. Oracle lookup evaluates both adjacent stored conditions against the
   midpoint target and retains the lower TVD.
2. **Center-count control test:** the median effective center count at each
   training angle must lie within \(\pm1\) of its class \(K\), and the
   medians must increase strictly with \(K\).

## Claim boundary

Success would establish interpolation along this one-dimensional log-\(K\)
angle path between planted Hamming-ball classes. It would not establish
extrapolation beyond \(K=8\) or \(\pi/4\), arbitrary per-qubit conditioning,
or control over structure outside the planted ball family.

The effective-\(K\) readout is a control diagnostic, not a distributional
test: the readout of a midpoint mixture equals its right class's readout (its
right-class centers each carry at least 5% of the fitted mass), so effective
\(K\) alone cannot distinguish a midpoint mixture from the stored right-class
condition. Distributional agreement is carried by bitstring TVD.

## Outcome (after the ten-seed run)

The frozen tests fail. The median effective center count collapses to one for
\(K=1,2,4\) and to zero for \(K=8\) (the fitted distribution is nearly
uniform); median training-angle bitstring TVD rises from \(0.47\) (\(K=1\)) to
\(0.83\) (\(K=8\)); and dense interpolation beats oracle adjacent-angle lookup
at only the lowest midpoint. The conditional MMD loss converges while the
within-shell structure that distinguishes ball counts is not learned.

Interpretation: the Hamming-RBF kernel is permutation-symmetric and the
multiscale budgets emphasize low-order characters, so the objective cannot
constrain the center structure. The study is kept as an internal negative
control and is not included in the manuscript; the manuscript's structured
multimodal focus remains the Hopfield-beta study.

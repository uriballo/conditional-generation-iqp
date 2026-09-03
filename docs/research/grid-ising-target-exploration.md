# Conditional grid-Ising target exploration

## Question

Could a local Ising model conditioned on temperature replace the unsuccessful
beta-binomial benchmark, and would this add evidence beyond the conditional
Curie--Weiss experiment?

## Target

For spins (s_i\in\{-1,+1\}) on a rectangular nearest-neighbor lattice, the
exploration uses

\[
p(s\mid\beta,h)\propto
\exp\left[
\beta\sum_{\langle i,j\rangle}s_i s_j+h\sum_i s_i
\right].
\]

Both conditions are dimensionless. The coupling is fixed to one. The first
condition controls local order; the second selects a magnetization branch. The
family obeys the exact cross-condition equivariance

\[
p(-s\mid\beta,h)=p(s\mid\beta,-h).
\]

At (h=0), this becomes the usual within-condition global spin-flip
invariance.

## Scientific assessment

A temperature-only version is valid but overlaps too strongly with
Curie--Weiss. Both would use inverse temperature as the only condition and
global spin flip as the main symmetry. It would also extend the fixed-
temperature (4\times4) Ising experiment in Section 6 and the experimental
section of Recio-Armengol, Ahmed, and Bowles rather than introduce a clearly
different conditional symmetry.

The two-condition ((\beta,h)) family is stronger. It differs from
Curie--Weiss through nearest-neighbor geometry, domain structure, and an exact
symmetry that maps (h) to (-h). The prior work trained one symmetry-enforced
Ising distribution at fixed temperature; it did not test a shared model across
temperature and field or cross-condition equivariance.

## Exact target behavior

Exact enumeration on the periodic (4\times4) lattice shows a large but smooth
finite-size response around the infinite-lattice critical inverse temperature
\(\beta_c\simeq0.4407\). At zero field, the expected absolute magnetization per
spin increases from 0.343 at \(\beta=0.2\) to 0.843 at \(\beta=0.44\), 0.954 at
\(\beta=0.55\), and 0.996 at \(\beta=0.8\). The last condition is almost
deterministic and should not dominate a pilot.

At \(\beta=0.44\), fields \(h=\pm0.05\) already yield mean magnetization
\(\mathbb E[M]/n\simeq\pm0.531\), while \(h=\pm0.1\) yields approximately
\(\pm0.795\). A useful development grid should therefore use fields no larger
than about 0.1.

## Isotropic-MMD audit

The Hamming-RBF kernel remains characteristic on the hypercube. It does not
encode lattice locality, so the relevant question is estimator efficiency.
The exact character audit gives the following result at \(\beta=0.44,h=0\):

| Lattice | Edges | Expected edge masks in bandwidth 1 | Bandwidth 2 | Bandwidth 3 |
|---|---:|---:|---:|---:|
| (2\times4) | 12 | 256.4 | 264.3 | 101.9 |
| (3\times4) | 24 | 215.5 | 136.9 | 13.7 |
| (4\times4) | 32 | 157.3 | 53.1 | 1.2 |

Each value assumes 2,000 nonidentity sampled masks. On the (4\times4)
lattice, adjacent pairs account for about 29.4% of the exact two-body MMD
signal. Their combinatorial share among all pairs is 26.7%. The remaining
two-body signal comes from longer-range correlations, so it is not noise. At
the lowest-body bandwidth, the exact MMD is dominated by two-body characters.
At the middle and high-body bandwidths, four- and higher-body characters carry
substantial signal. This is the intended broad correlation coverage of the
three-bandwidth protocol.

The audit therefore finds no immediate incompatibility with the isotropic
MMD. It does not show that optimization will succeed.

## Geometry limitation

The proposed size sequence (2\times4), (3\times4), and (4\times4) is not
a strict fixed-geometry scaling law. Deduplicating periodic edges gives the
(2\times4) lattice degree three and the larger lattices degree four. We
should either present size as a finite-lattice robustness check or restrict the
headline local-Ising analysis to (4\times4). We should not interpret shifts
across these sizes as pure qubit scaling.

## Recommended next gate

Keep this target for a small (n=8) development study, but use the full
\((\beta,h)\) condition rather than temperature alone. Before designing a
specialized encoder, run target-aware direct fits at a disordered point, a
near-transition point, and an ordered point. This separates model and loss
capacity from conditional-encoding failure. If the direct fits pass exact-TVD
thresholds, compare a generic two-coordinate product encoder with a
complement-equivariant encoder satisfying

\[
X^{\otimes n}|\phi(\beta,h)\rangle
=|\phi(\beta,-h)\rangle.
\]

The pilot should use zero and three ancillas, matched seeds, exact TVD, local
edge-correlation error, magnetization-profile TVD, and the equivariance defect.
No encoder or training campaign has been frozen at this stage.

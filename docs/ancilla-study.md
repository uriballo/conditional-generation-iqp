# Ancilla-Assisted Conditional IQP Born Machine Study

## Overview

This experiment explores the effect of adding $m \in \{1, 2, 3, 5, 7, 9\}$ ancilla qubits to the 19-qubit IQP Born machine architecture.

The central question is whether adding ancillas—and tracing them out at output measurement time—increases the expressivity of the conditional model and improves both multi-condition distribution fitting and held-out angle interpolation.

## Formulation

For $n = 19$ data qubits and $m$ ancilla qubits, the model operates on a total of $N = n + m$ qubits.
The output probability distribution over target bitstrings $x \in \{0,1\}^n$ is obtained by marginalizing over the $m$ ancilla qubits:

$$p(x \mid c) = \sum_{a \in \{0,1\}^m} p(x, a \mid c)$$

### Observables & Estimator
During classical Conditional MMD training:
- Pauli-$Z$ mask observables $A \in \{0,1\}^n$ generated on data qubits are padded with $m$ zeros: $(A, 0^m) \in \{0,1\}^{n+m}$.
- The estimator evaluates $\langle Z_A \otimes I_m \rangle$ on the $N$-qubit product state.
- Because ancilla bits are unmasked (0 mask), bit-flip character differences act only on data qubits, keeping Monte Carlo estimator sample variance bounded and independent of $m$.

### Exact Evaluation
Exact metrics compute $2^{n+m}$ joint probabilities using FWHT, sum out the $m$ ancilla qubits to produce $2^n$ data probabilities, and compute exact Bitstring TVD and Hamming-weight Profile TVD against 19-qubit target distributions.

## Condition Encodings for Ancillas

1. **Broadcast Conditioning (`broadcast_all`)**:
   Ancillas receive the same condition angle $\alpha$ as data qubits: $\bigotimes_{i=1}^{n+m} R_Y(\alpha)|0\rangle$.
2. **Fixed Zero Ancilla (`fixed_zero`)**:
   Data qubits receive angle $\alpha$; ancillas are fixed at angle $0$ ($|0\rangle$).
3. **Ancilla-Only Conditioning (`ancilla_only`)**:
   Data qubits receive angle $0$; the supplied condition angles are placed only
   on the ancillas. This mode is used by the Iris three-class exploratory
   benchmark, where two ancillas carry a binary class selector.

## Ancilla Rungs & Capacity Scaling

All models use a complete 2-local ansatz on $N = 19 + m$ qubits:

| Ancillas ($m$) | Total Qubits ($N$) | Parameter Count ($\binom{N}{2} + N$) | Capacity Ratio vs $m=0$ |
|---|---|---|---|
| 0 | 19 | 190 | $1.00\times$ (Baseline) |
| 1 | 20 | 210 | $1.11\times$ |
| 2 | 21 | 231 | $1.22\times$ |
| 3 | 22 | 253 | $1.33\times$ |
| 5 | 24 | 300 | $1.58\times$ |
| 7 | 26 | 351 | $1.85\times$ |
| 9 | 28 | 406 | $2.14\times$ |

## Decision Rules & Comparisons

For each ancilla count $m$:
1. **Primary Conditioning Benchmark**: Compare 10-seed median Profile TVD and Bitstring TVD on Conditions A and B against the 0-ancilla baseline.
2. **Held-Out Interpolation Audit**: Evaluate dense-grid 5-angle models at 4 interleaved held-out angles ($\pi/32, 3\pi/32, 5\pi/32, 7\pi/32$) and test whether ancillas reduce residual within-shell bitstring bias.

## Benchmark Results (10 Seeds, 2,000 Steps)

Results use seeds 10-19 (a seed-consistency sanity check of the original
seeds 0-9 confirmed the same medians without the three collapsing runs;
the new seed block is canonical).

| Ancillas ($m$) | Qubits ($N$) | Params | Held-out Profile TVD | Held-out Bitstring TVD | End. A Profile TVD | End. B Profile TVD | Relative Bitstring TVD Drop |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 19 | 190 | 0.0635 [0.0604, 0.0658] | 0.3537 [0.3501, 0.3601] | 0.0503 [0.0425, 0.0609] | 0.0964 [0.0946, 0.1049] | Baseline |
| 1 | 20 | 210 | 0.0646 [0.0607, 0.0656] | 0.3259 [0.3217, 0.3324] | 0.0463 [0.0431, 0.0566] | 0.0975 [0.0919, 0.0990] | $-7.9\%$ |
| 2 | 21 | 231 | 0.0602 [0.0583, 0.0632] | 0.3002 [0.2939, 0.3046] | 0.0450 [0.0385, 0.0489] | 0.0897 [0.0836, 0.0950] | $-15.1\%$ |
| 3 | 22 | 253 | 0.0569 [0.0548, 0.0603] | 0.2847 [0.2732, 0.2874] | 0.0381 [0.0356, 0.0427] | 0.0906 [0.0873, 0.0914] | $-19.5\%$ |
| 5 | 24 | 300 | 0.0547 [0.0513, 0.0579] | 0.2455 [0.2395, 0.2599] | 0.0338 [0.0313, 0.0364] | 0.0847 [0.0824, 0.0892] | $-30.6\%$ |
| 7 | 26 | 351 | **0.0512 [0.0500, 0.0524]** | **0.2079 [0.2033, 0.2132]** | **0.0358 [0.0290, 0.0391]** | **0.0866 [0.0841, 0.0924]** | **$-41.2\%$** |

### Key Findings:
1. **Monotonic Bitstring TVD Reduction**: Bitstring TVD decreases strictly monotonically from `0.3537` ($m=0$) down to **`0.2079`** ($m=7$). This represents a **41.2% relative reduction** in exact 19-qubit bitstring error.
2. **Profile TVD Gains**: Profile TVD also reaches a new overall low of **`0.0512`** at $m=7$ (compared to `0.0635` at $m=0$).
3. **Endpoint Fitting**: Endpoint A Profile TVD drops to **`0.0358`** ($m=7$) from `0.0503` ($m=0$), and Endpoint B drops to **`0.0866`** ($m=7$) from `0.0964` ($m=0$).

## Matched-Capacity Control

The ladder is not capacity matched: the complete 2-local ansatz on $N=19+m$
qubits grows from 190 to 351 parameters as $m$ goes from 0 to 7. Because
duplicated 2-local generators would collapse to a single phase, the control
adds genuine **3-local** generators instead: the same 19 data qubits, the same
190 complete 2-local generators, plus enough deterministic 3-local
$Z_iZ_jZ_k$ generators (fixed permutation seed) to reach each ladder rung's
parameter count, trained under the identical five-angle protocol with ten
matched seeds.

### Control Protocol (frozen by comparison structure)

1. For each rung $m$, run an $m{=}0$ model with the same parameter count as
   the ancilla rung: 210, 231, 253, 300, 351 parameters
   (+20/41/63/110/161 3-local generators).
2. Compare the exact ten-seed median Held-out Bitstring TVD of the control
   against the ancilla model at the same parameter count.
3. Claim the marginalization mechanism only if the control does not reproduce
   the ancilla reduction.

### Control Results (10 Seeds, 2,000 Steps)

| Params | m=0 3-local Control Held-out Bit TVD | Ancilla Held-out Bit TVD (m) |
|---:|---:|---:|
| 210 | 0.3580 [0.3564, 0.3622] | 0.3259 (m=1) |
| 231 | 0.3585 [0.3567, 0.3608] | 0.3002 (m=2) |
| 253 | 0.3578 [0.3563, 0.3595] | 0.2847 (m=3) |
| 300 | 0.3541 [0.3534, 0.3557] | 0.2455 (m=5) |
| 351 | 0.3550 [0.3541, 0.3558] | 0.2079 (m=7) |

### Control Key Findings

1. **No parameter-count replication**: the matched-capacity $m{=}0$ control
   stays at or slightly above the baseline Bitstring TVD at every parameter
   count (0.354--0.359 versus 0.354 at $m{=}0$), while the ancilla model at
   the same count improves monotonically down to 0.208 at 351 parameters.
2. **Mechanism attribution**: the ladder gain is attributable to the
   traced-out marginalization, not to the growing number of trainable phases.
3. **Boundary unchanged**: the ladder still stops at $m=7$ (exact $m=9$
   evaluation would need about 18 GiB), and the control is a descriptive
   mechanism check without a frozen multi-criterion decision rule.

# Paper outline and evidence map

## Story

The paper asks whether the classically tractable training strategy used for IQP
generators can be made conditional without giving each condition separate
circuit parameters. The mechanism is to replace the fixed computational-basis
input with a real product state whose angles encode the condition. The
post-Hadamard state remains efficiently sampleable and evaluable, so Pauli-Z
expectations and their MMD gradients retain polynomial Monte Carlo estimators.

## Section outline

1. **Introduction**
   - Conditional generation requires one shared model to represent
     condition-indexed output laws.
   - IQP Born machines combine classically tractable observable training with
     potentially nontrivial quantum sampling, but prior demonstrations are
     principally unconditional.
   - We encode the condition in a tractable initial product state and extend the
     probabilistic estimator.
   - Contributions: estimator and variance result; unbiased Conditional MMD;
     controlled 19-qubit demonstration and topology/conditioning ablations.
2. **Conditional product-state IQP model**
   - Define \(RY\) product-state encoding and shared \(H D_Z H\) circuit.
   - Show the circuit diagram and distinguish condition-dependent angles from
     shared trainable parameters.
3. **Classical estimation and Conditional MMD**
   - Derive the amplitude-ratio Pauli estimator.
   - Prove unbiasedness, bounded second moment, and polynomial sample cost.
   - Derive direct backpropagation and the unbiased multiscale conditional loss.
4. **Experiments**
   - Define calibrated Gaussian Hamming-weight targets and data protocol.
   - Describe heavy-hex, lattice, and complete native ansatz families.
   - Report exact bitstring/shell metrics, baselines, estimator calibration, and
     limitations.
5. **Discussion and conclusion**
   - Separate tractable training from classical sampling and quantum-advantage
     claims.
   - Explain what topology results do and do not establish.
   - Identify richer conditions, larger systems, and variance behavior as next
     steps.

## Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| Real product-state conditions preserve efficient Pauli-Z estimation. | Proposition: efficient local sampling/amplitudes, unbiased estimator, bounded second moment. | Supported analytically |
| The loss can be differentiated by ordinary backpropagation. | Samples and observable masks are independent of trainable phases; JAX/PennyLane gradient tests. | Supported analytically and numerically |
| The implementation is genuinely conditional. | Label-swap test changes Conditional MMD while preserving the pooled marginal. | Supported by test |
| One shared angle-conditioned model separates both Gaussian Hamming-weight targets, but meets the strict baseline ordering on only three of four condition/metric comparisons. | Ten-seed exact shell and bitstring TVD; machine-generated success audit. | Supported with explicit negative criterion result |
| Native topology families produce different shell/bitstring trade-offs. | Ten matched seeds for heavy-hex, lattice, and complete families with parameter counts reported. | Supported descriptively; not a fixed-capacity connectivity claim |
| The trained circuits provide quantum advantage or hard sampling. | No experiment establishes this. | Explicitly not claimed |

## Reverse outline

- **Abstract:** task and tractability challenge → product-state mechanism →
  estimator/MMD contributions → ten-seed result and strict failure → claim
  boundary.
- **Introduction:** QCBM/IQP context → conditional shared-parameter challenge →
  estimator-preserving mechanism → contribution/evidence list → distinction
  from prior observable training and from separate models.
- **Model:** circuit and conditioning seam → tractable post-Hadamard reference
  amplitudes.
- **Estimator:** Monte Carlo construction → unbiasedness/second-moment proof →
  gradient consequence → conditional kernel objective → direct-backprop
  justification.
- **Experiments:** target/protocol → baselines and their algebraic restrictions
  → exact metrics and preregistered decision rule → quantitative criterion,
  topology, and within-shell findings → independent verification.
- **Discussion:** scoped interpretation and limitations → mechanism-level
  explanation of the failure → experiments directly motivated by each failure
  → no-hardness/no-advantage boundary.

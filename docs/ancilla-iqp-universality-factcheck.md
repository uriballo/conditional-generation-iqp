# Ancillas and IQP universality: primary-source fact-check

## Verdict

The statement that an \(n\)-bit IQP generator becomes universal after adding \(n+1\) hidden qubits is real, but "universal" has a narrow meaning here. Kurkin et al. prove an **exact distributional representability** result for an unconditional IQP quantum circuit Born machine (IQP-QCBM): every distribution on \(n\) visible bits is the visible marginal of an IQP circuit with \(n+1\) hidden qubits. The result does not establish universal quantum computation.

Three qualifications are necessary.

- The \(n+1\) count is sufficient, not necessary. Whether it is optimal remains open for \(n>2\).
- The proof uses an unrestricted diagonal IQP block with independently tunable phases for all joint computational-basis strings. Equivalently, it permits Pauli-\(Z\) generators of arbitrary support and exponentially many parameters.
- The theorem therefore does not prove universality of the manuscript's complete two-local ansatz, its small ancilla ladder, or a condition-indexed family implemented by one shared unitary.

The unsafe sentence is therefore: "IQP circuits become universal with \(n+1\) ancillas." It conflates distributional and computational universality and makes a sufficient upper bound sound necessary.

## The \(n+1\) theorem

The most complete primary source is Andrii Kurkin, Kevin Shen, Susanne Pielawa, Hao Wang, and Vedran Dunjko, *Universality and kernel-adaptive training for classically trained, quantum-deployed generative models*, [arXiv:2510.08476v1](https://arxiv.org/abs/2510.08476), [arXiv DOI 10.48550/arXiv.2510.08476](https://doi.org/10.48550/arXiv.2510.08476). This expanded preprint supersedes the shorter note [arXiv:2504.05997v1](https://arxiv.org/abs/2504.05997), [arXiv DOI 10.48550/arXiv.2504.05997](https://doi.org/10.48550/arXiv.2504.05997).

Exact locations in arXiv:2510.08476v1:

- Definition 2, p. 3, defines \(U(\theta)=H^{\otimes n}D(\theta)H^{\otimes n}\), where the diagonal block may contain a \(Z\) word on any qubit subset.
- Definition 3, p. 3, defines universality as representation of arbitrary probability distributions, not arbitrary quantum circuits or unitaries.
- Definition 5 and Eq. (4), p. 4, define hidden qubits by tracing them out before reading the \(n\)-bit output distribution.
- Theorem 1, p. 5, is titled "Exact universality" and states that \(m=n+1\) hidden qubits suffice. Its operative phrase is "produces \(p\) exactly."
- Appendix B, pp. 15–18, gives the full proof. Lemma 7 and Eq. (5) write the visible reduced state as a uniform mixture of \(2^m\) states. Lemma 8 realizes any two-sparse visible distribution in one mixture component. Lemmas 9–10 show that every \(2^n\)-outcome distribution is a uniform mixture of at most \(2^{n+1}\) two-sparse distributions.
- The paragraph immediately after Theorem 1, p. 5, says that optimality is open for \(n>2\) and that the construction uses exponentially many parameters. Remark 1 then preserves efficient classical estimation of visible Pauli-\(Z\) moments, but it does not make the universal construction efficiently trainable.

The resource statement is

\[
m=n+1\quad\text{hidden qubits},\qquad N=n+m=2n+1\quad\text{total qubits}.
\]

This is an upper bound for exact representability. The paper does not prove that \(n+1\) hidden qubits are required. It even proves a sharper special case in Appendix C, Lemma 12: one hidden qubit suffices for every distribution on two visible bits.

## Why the theorem does not cover the manuscript's ansatz

The proof starts from an arbitrary joint phase table

\[
D(\theta)=\operatorname{diag}\!\left(e^{i\theta_{x,y}}\right),
\qquad x\in\{0,1\}^{m},\;y\in\{0,1\}^{n},
\]

so it can choose the visible phase pattern separately for every hidden string \(x\). Such a phase table generally requires arbitrary-body diagonal generators and exponentially many coefficients. By contrast, the manuscript uses only one- and two-body generators on a complete \(N\)-qubit graph. It has

\[
N+\binom{N}{2}=\frac{N(N+1)}{2}
\]

learned phase parameters, which is polynomial in \(N\). Adding ancillas to this ansatz adds one-body, ancilla–ancilla, and visible–ancilla generators and makes the visible output a marginal of a larger pure state. Those changes can enlarge the model family, but Kurkin et al.'s theorem does not characterize that enlargement.

The theorem is also unconditional. It constructs a separate phase table for one target distribution \(p\). It does not show that one fixed unitary, driven only by a condition-dependent input state, can represent an arbitrary family \(p(\cdot\mid c)\).

## Distinct meanings of universality

| Model or extension | Added resource | Proven statement | Relevance here |
| --- | --- | --- | --- |
| Hidden-unit IQP-QCBM, Kurkin et al. | Trace out \(n+1\) hidden qubits; arbitrary joint diagonal phases | Exact representation of every distribution on \(n\) visible bits | Direct source of the \(n+1\) number, but not applicable to the two-local ansatz |
| Ordinary IQP sampling | Commuting IQP circuit followed by measurement | Restricted sampling model with classical-hardness evidence under complexity assumptions | No claim of universal quantum computation |
| Post-IQP, Bremner–Jozsa–Shepherd | Postselect specified measurement outcomes | \(\mathrm{postIQP}=\mathrm{postBQP}=\mathrm{PP}\) | Computational-complexity result; postselection is essential |
| Ancilla-driven adaptive computation | Repeated ancilla coupling, measurement, and classical feed-forward | Universal quantum computation in a different adaptive model | Not ordinary IQP and not the source of the (n+1) bound |
| Clifford circuits with magic states | Non-stabilizer ancillary states and stabilizer operations | Universal quantum computation after magic-state injection/distillation | Different gate and state resource; absent from this construction |

Bremner, Jozsa, and Shepherd, *Classical simulation of commuting quantum computations implies collapse of the polynomial hierarchy*, [arXiv:1005.1407](https://arxiv.org/abs/1005.1407), [DOI 10.1098/rspa.2010.0301](https://doi.org/10.1098/rspa.2010.0301), prove the postselection statement in Section 3.1, Theorem 1, journal pp. 465–466, article PDF pp. 7–8. Figure 1 replaces each intermediate Hadamard with a gadget containing one fresh ancillary line and a postselected outcome. The number of ancillas therefore follows the number of intermediate Hadamards in the simulated circuit; it is not \(n+1\). Clean ancillas without postselection are not what gives this construction its computational power.

Representative primary sources for the other extensions are Anders et al., *Ancilla-driven universal quantum computation*, [DOI 10.1103/PhysRevA.82.020301](https://doi.org/10.1103/PhysRevA.82.020301), and Bravyi and Kitaev, *Universal quantum computation with ideal Clifford gates and noisy ancillas*, [arXiv:quant-ph/0403025](https://arxiv.org/abs/quant-ph/0403025), [DOI 10.1103/PhysRevA.71.022316](https://doi.org/10.1103/PhysRevA.71.022316). Neither supports the \(n+1\) IQP-QCBM claim.

## Evidence that marginalization can increase expressivity

The strongest direct evidence is the Kurkin et al. existence theorem itself. It should be paired with the ancilla-free counterexample in Recio-Armengol, Ahmed, and Bowles, *Train on classical, deploy on quantum: scaling generative quantum machine learning to a thousand qubits*, [arXiv:2503.02934v2](https://arxiv.org/abs/2503.02934), [arXiv DOI 10.48550/arXiv.2503.02934](https://doi.org/10.48550/arXiv.2503.02934). Section 5.3 and Appendix C show that an ancilla-free two-qubit IQP-QCBM cannot represent the distribution \((1/3,1/3,1/3,0)\). Kurkin et al., Appendix C, Lemma 12, then show that one hidden qubit suffices for every two-bit distribution. Together these results establish a strict expressivity increase in that small case.

A newer, distinct analysis is Lee and Kim, *Trainability and Mode Separation of Mixed IQP-QCBMs*, [arXiv:2607.27883v1](https://arxiv.org/abs/2607.27883), [arXiv DOI 10.48550/arXiv.2607.27883](https://doi.org/10.48550/arXiv.2607.27883). Section II.3, Eqs. (10)–(13), decomposes an unmeasured-ancilla construction into a mixture of IQP branches. Proposition 2 shows that a mixture can beat the best ancilla-free MMD only if its branches represent different distributions. This supports marginalization as an expressivity mechanism, but its compiled circuit uses higher-body ancilla-controlled generators. Its weighted version also uses additional state preparation. It is not a theorem about the manuscript's complete two-local circuit.

I found no primary-source theorem that proves a monotonic expressivity increase, much less universality, for the exact architecture used in the manuscript: a complete two-local IQP unitary, a condition-dependent joint product input, a small number of unmeasured ancillas, and one shared parameter vector across conditions. Any empirical improvement under that architecture should therefore be reported as an observed finite-benchmark result, not as a consequence of a universality theorem.

## Safe manuscript wording

Recommended wording:

> Marginalizing ancillary outputs can enlarge the family of visible distributions. For unrestricted parameterized IQP-QCBMs that permit diagonal generators of arbitrary support, \(n+1\) hidden qubits suffice to represent exactly any distribution on \(n\) visible bits. This is a distributional universality result, and the bound is not known to be minimal. Our construction retains only one- and two-body generators and uses fewer ancillas, so this theorem motivates ancillary marginalization without establishing universality of our model.

The implementation-specific mechanism can then be stated without invoking universality:

> In our construction, adding ancillas simultaneously enlarges the joint two-local circuit and extends the condition-dependent input across a larger register. Visible–ancilla interactions couple this richer encoding to the learned dynamics, and tracing out the ancillas yields the visible marginal. The ancilla sweep varies these effects together and does not identify which mechanism causes any observed improvement.

Avoid the following formulations:

- "IQP circuits are universal with ancillas."
- "An \(n\)-qubit IQP circuit needs \(n+1\) ancillas to become universal."
- "Our two-local ancilla-assisted model is universal."
- "The universality theorem explains the improvements in our experiments."

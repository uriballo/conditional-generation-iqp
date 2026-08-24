# Require half of the reprogrammable conditioning benefit

Fixed-circuit input conditioning need not outperform the more flexible
Condition-modulated-phase IQP baseline, but on held-out Ising Conditions it
must recover at least half of that baseline's Bitstring-TVD improvement over
No condition. The primary retention ratio is
\(R=(\mathrm{TVD}_{no}-\mathrm{TVD}_{fixed})/(\mathrm{TVD}_{no}-\mathrm{TVD}_{phase})\),
evaluated with per-instance guardrails and a required median \(R\geq0.5\).

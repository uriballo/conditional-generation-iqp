# Use normalized broadcast RY for the Ising Condition

The primary fixed-circuit Ising model maps the frozen interval affinely by
\(\alpha(\beta)=(\pi/4)(\beta-\beta_{min})/(\beta_{max}-\beta_{min})\) and
prepares \(R_Y(\alpha(\beta))^{\otimes16}|0^{16}\rangle\). The map preserves
well-behaved Reference amplitudes, introduces no learned or per-qubit features,
is target-family calibrated but shared across all coupling instances, and
cannot change after model results are observed.

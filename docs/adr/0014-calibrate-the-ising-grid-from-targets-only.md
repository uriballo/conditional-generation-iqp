# Calibrate the Ising grid from Targets only

The shared inverse-temperature interval and interleaved train/held-out grid
will be frozen using exact Target properties across every coupling instance,
including adjacent Target TVD, entropy, concentration, moments, and thermal
response diagnostics. The interval must include the PennyLane anchor
\(\beta=1/3\), remain materially distinguishable on every instance, and avoid
an extreme regime that is already unsuitable as a conditional test; it cannot
be moved after observing model performance.

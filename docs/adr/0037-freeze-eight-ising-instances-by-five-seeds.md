# Freeze eight Ising instances by five optimizer seeds

The full non-shell matrix uses eight independently frozen (4\times4)
coupling instances and five matched optimizer seeds. For each instance-seed
pair it trains fixed-circuit, parameter-matched phase-conditioned, and
No-condition shared models plus five Separate models, totaling 320 primary
runs and leaving 80 of the 400-run ceiling for feasibility and bounded tuning.

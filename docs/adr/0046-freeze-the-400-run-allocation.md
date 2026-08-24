# Freeze the 400-run allocation

The Ising campaign's 400 full training runs are allocated non-transferably:
20 to runtime feasibility and the Visible-only Ising representability gate;
36 to equal Training/Validation-only tuning and phase-feature screening; 320
to the frozen primary matrix; and 24 to the Bounded Ising ancilla extension,
which is released only after primary success. Target-only calibration and
estimator-only scaling use no full training runs. Unused runs after a failed
gate cannot be redirected to architecture search, threshold relaxation, or
other rescue experiments.

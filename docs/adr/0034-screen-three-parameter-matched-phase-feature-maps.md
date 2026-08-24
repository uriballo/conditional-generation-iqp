# Screen three parameter-matched phase-feature maps

To avoid a weak reprogrammable baseline, three fixed, instance-shared
Condition-to-phase assignments with 136 trainable coefficients will be
preregistered: a common normalized affine multiplier, deterministic signed
affine multipliers across generators, and a balanced constant/slope generator
bank. The matched Training/Validation tuning pilot selects one before any
Held-out evaluation; no further phase architecture search is permitted.

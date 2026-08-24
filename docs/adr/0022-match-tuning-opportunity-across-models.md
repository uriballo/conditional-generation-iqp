# Match bounded tuning opportunity across model families

Fixed-circuit, Condition-modulated-phase, and No-condition shared models may
use different frozen optimizer hyperparameters, because identical settings can
favor one landscape accidentally. Each family receives the same small,
preregistered tuning-run budget using Training and Validation Conditions only,
with equal data and maximum update counts; Held-out Conditions remain
inaccessible until every protocol is frozen.

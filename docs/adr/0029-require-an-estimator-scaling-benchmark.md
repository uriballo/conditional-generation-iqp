# Require estimator and gradient scaling evidence

The Quantum-ready package includes a dedicated estimator-only benchmark across
qubit count, generator count, Condition count, and Condition dimension. It
reports runtime, peak memory, estimator variance or RMSE where exact references
exist, and gradient-estimator bias, variance, and RMSE against exact gradients
where available; it uses
synthetic frozen parameters and does not consume the 400-run Ising training
budget or imply large-scale generative fidelity. Raw gradient magnitude and
landscape trainability are outside this benchmark's pass/fail scope.

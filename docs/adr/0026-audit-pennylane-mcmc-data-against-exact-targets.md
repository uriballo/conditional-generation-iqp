# Audit PennyLane MCMC data against exact Ising Targets

Training and Validation strings will come from independent chains using the
official PennyLane Ising generator's burn-in and thinning approach, while
exact enumeration supplies the Target truth. Before model training, every
instance and inverse temperature must pass frozen effective-sample-size,
chain-agreement, moment, correlation, and distributional checks; inadequate
mixing makes the Condition ineligible rather than becoming model error.

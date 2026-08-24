# Fix 5,000 Training and 2,000 Validation Ising strings

Each coupling-instance and Training-Condition pair supplies 5,000 PennyLane-
generator MCMC strings for Training and 2,000 independent strings for
Validation, with Condition-balanced minibatches of 256. A larger independent
chain pool may be used only for sampling-quality certification, and exact
enumerated probabilities rather than an empirical test set determine final
Bitstring TVD.

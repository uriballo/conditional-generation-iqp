# Separate sparse and Complete estimator scaling

Estimator and gradient scaling will use a bounded-degree IQP track with
\(O(n)\) generators to isolate qubit dependence and a Complete 2-local track
with \(O(n^2)\) generators to measure the practical cost of the experimental
ansatz. Both vary Condition count and dimension, use exact RMSE references at
small sizes, and switch to repeated-estimator variance and internal consistency
diagnostics when exact evaluation becomes unavailable. Gradient-estimator
accuracy is measured separately from raw gradient magnitude or landscape
trainability.

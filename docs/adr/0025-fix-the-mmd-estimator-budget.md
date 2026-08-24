# Fix the MMD estimator budget at 2,000 by 2,000

Every primary sampled Conditional MMD evaluation uses 2,000 independently
sampled Fourier operators per bandwidth and 2,000 product-reference samples
per operator and Condition, in both Training and Validation and for every model
family. A one-run runtime and memory benchmark is mandatory before releasing
the 400-run matrix; if the fixed budget exceeds available resources, the
protocol must be reopened explicitly rather than silently reduced.

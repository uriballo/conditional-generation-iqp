# Freeze the primary bootstrap and retention gate

Primary uncertainty uses 10,000 paired hierarchical bootstrap replicates with
a frozen random seed. Each replicate resamples the eight coupling instances
first and the five optimizer seeds within selected instances, preserving all
method, Condition, and evaluation-cell pairings. Report 95% percentile
intervals. The Fixed-circuit benefit-retention gate requires both median
\(R\ge 0.5\) and a lower interval bound strictly above zero. The existing
interpolation decision rules remain binding. Requiring the lower bound itself
to exceed one half would be disproportionate with only eight independent
coupling instances.

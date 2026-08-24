# Prioritize coupling instances over optimizer seeds

The Ising study treats independently frozen coupling matrices as the primary
scientific replicates and matched optimizer seeds as nested optimization
replicates. Given a fixed compute budget, the design favors more coupling
instances over ten seeds on one Hamiltonian, and paired hierarchical bootstrap
resampling will first sample instances and then matched seeds within instances
without breaking method or Condition pairing.

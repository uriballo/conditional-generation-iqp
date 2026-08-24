# Encode inverse temperature as the Ising Condition

The PennyLane-Ising experiment uses inverse temperature \(\beta=1/T\) as the
canonical Condition and maps it affinely into product-state angles. Thermal
probabilities are exponential-linear in \(\beta\), so training points,
held-out midpoints, and conditional-response analysis share one natural
coordinate; the published PennyLane point \(T=3\) remains the anchor
\(\beta=1/3\), and figures may report both units.

# Fix and disclose the Condition map before model evidence

Every conditional-response theorem conditions on a disclosed fixed map; no
Target outcome enters its derivation. Map-selection provenance must nevertheless
be stated separately. For the Ising campaign, ADR 0014 and ADR 0035 permit exact
Target-only diagnostics to select one common beta interval before model work.
The resulting normalization is therefore **target-family calibrated**, not
Target independent, but it is shared across all coupling instances and cannot
use Training, Validation, Held-out, checkpoint, or model-performance evidence.

Arbitrary or per-instance classical preprocessing could encode the desired
Target trajectory into product-state angles and make apparent quantum
expressivity vacuous. Richer maps may be studied only when their added spectral
bandwidth and selection evidence are credited explicitly to the encoder rather
than to the shared IQP circuit.

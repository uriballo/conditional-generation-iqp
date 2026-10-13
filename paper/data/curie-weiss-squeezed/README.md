# Curie–Weiss squeezed-encoder data

Final ten-repeat curves for the conditional (`shared_squeezed_parity`) and
temperature-specific (`independent_squeezed_parity`) models with the squeezed
parity encoder. `figure_data.json` is written by
`python -m paper_experiments.cw_squeezed_bundle CAMPAIGN_ROOT... --output DIR`
from completed `cw_encoder_hpo` campaigns, and stores the protocol hashes and
campaign receipts of each family.

Protocols: `paper_experiments/cw_squeezed_protocol.json` and
`paper_experiments/cw_squeezed_independent_protocol.json`. They inherit every
training, selection and reporting rule of the audited follow-up protocol. The
schedule constant `a` is searched on [0.3, 0.7] and `kappa` is fixed at 0.4.
Fit streams pair with the audited parity fits at every stage, repeat and start.

`cw_followup_reporting.render` merges this bundle with
`../curie-weiss-followup/figure_data.json`. Status 2026-10-13: the shared family
is final; the temperature-specific family is still running.

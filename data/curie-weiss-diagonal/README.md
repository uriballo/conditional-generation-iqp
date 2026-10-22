# Curie–Weiss, diagonal conditioning (D-IQP)

One circuit with all 136 one- and two-body phases acting on the parity-encoder
input at t = pi/2. The phases are a cubic Chebyshev polynomial in β,
theta(β) = sum_{k<4} T_k(β − 1) w_k, giving 544 parameters. Search, selection,
confirmation and final stages follow the parity protocol with the
per-temperature search space.

- `figure_data.json`: final curves and metrics for the ten final fits.
- `review-ready.json`: final status, winner and figure-data hash.
- `protocol.json`: protocol (`cw_diagonal_k4_protocol.json`).

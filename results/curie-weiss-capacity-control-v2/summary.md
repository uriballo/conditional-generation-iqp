# Matched-capacity Curie-Weiss control

Positive paired effects mean that the five-ancilla model has lower TVD.
Conditions are averaged within each optimizer seed before bootstrapping.

| Metric | Region | Ancilla | Visible 1/2/3-local | Effect [95% CI] |
|---|---:|---:|---:|---:|
| profile_tvd | training | 0.113 | 0.168 | +0.056 [+0.024, +0.085] |
| profile_tvd | held_out | 0.100 | 0.167 | +0.067 [+0.035, +0.095] |
| profile_tvd | critical | 0.078 | 0.196 | +0.118 [+0.082, +0.146] |
| profile_tvd | cold | 0.164 | 0.171 | +0.007 [-0.032, +0.048] |
| bitstring_tvd | training | 0.180 | 0.312 | +0.132 [+0.092, +0.175] |
| bitstring_tvd | held_out | 0.166 | 0.310 | +0.144 [+0.105, +0.185] |
| bitstring_tvd | critical | 0.146 | 0.331 | +0.184 [+0.145, +0.218] |
| bitstring_tvd | cold | 0.191 | 0.273 | +0.082 [+0.033, +0.127] |

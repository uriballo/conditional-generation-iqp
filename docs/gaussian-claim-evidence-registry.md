# Gaussian claim--evidence registry

This registry is generated only after all 1,020 declared atomic models and exact evaluations are complete. Positive values favor the product-state-conditioned circuit. There is no empirical pass/fail gate: every size--width cell is reported.

- Manifest digest: `386ce98be93501f078d45b29e7fa498db1600c44d0e22f4483c3c8d5c7ec0d8c`
- Complete report SHA-256: `3582aac23bff0850eca11ec2d308f4f259856adf931c06faa554cf70c03cae0b`
- Paired intervals: 10,000 resamples, seed `20260811`

| Claim | Supported | Mixed | Unsupported |
|---|---:|---:|---:|
| G-CONTROL-PROFILE | 12 | 0 | 0 |
| G-INTERPOLATION-LOOKUP-PROFILE | 0 | 2 | 10 |
| G-INTERPOLATION-MIXTURE-PROFILE | 2 | 1 | 9 |
| G-CONTROL-BITSTRING | 12 | 0 | 0 |
| G-INTERPOLATION-LOOKUP-BITSTRING | 2 | 1 | 9 |

## G-CONTROL-PROFILE

| Cell | Status | Paired effect [95% interval] | Wins |
|---|---|---:|---:|
| n8-w0.35 | supported | 0.2402 [0.2278, 0.2536] | 50/50 |
| n8-w0.50 | supported | 0.2170 [0.2015, 0.2418] | 50/50 |
| n8-w0.75 | supported | 0.1460 [0.1293, 0.1714] | 50/50 |
| n12-w0.35 | supported | 0.2419 [0.2352, 0.2505] | 48/50 |
| n12-w0.50 | supported | 0.2291 [0.2213, 0.2391] | 48/50 |
| n12-w0.75 | supported | 0.1608 [0.1515, 0.1722] | 50/50 |
| n16-w0.35 | supported | 0.2472 [0.2421, 0.2519] | 46/50 |
| n16-w0.50 | supported | 0.2440 [0.2354, 0.2537] | 48/50 |
| n16-w0.75 | supported | 0.1667 [0.1570, 0.1770] | 50/50 |
| n19-w0.35 | supported | 0.2620 [0.2594, 0.2645] | 46/50 |
| n19-w0.50 | supported | 0.2668 [0.2587, 0.2737] | 47/50 |
| n19-w0.75 | supported | 0.1807 [0.1648, 0.1999] | 49/50 |

## G-INTERPOLATION-LOOKUP-PROFILE

| Cell | Status | Paired effect [95% interval] | Wins |
|---|---|---:|---:|
| n8-w0.35 | unsupported | -0.0292 [-0.0468, -0.0108] | 23/80 |
| n8-w0.50 | unsupported | -0.0280 [-0.0336, -0.0225] | 8/80 |
| n8-w0.75 | unsupported | -0.0276 [-0.0339, -0.0214] | 19/80 |
| n12-w0.35 | unsupported | -0.0221 [-0.0255, -0.0190] | 30/80 |
| n12-w0.50 | unsupported | -0.0180 [-0.0213, -0.0146] | 18/80 |
| n12-w0.75 | unsupported | -0.0129 [-0.0151, -0.0107] | 20/80 |
| n16-w0.35 | unsupported | -0.0229 [-0.0279, -0.0174] | 11/80 |
| n16-w0.50 | mixed | -0.0040 [-0.0071, 0.0001] | 38/80 |
| n16-w0.75 | unsupported | -0.0192 [-0.0205, -0.0178] | 20/80 |
| n19-w0.35 | unsupported | -0.0217 [-0.0240, -0.0192] | 27/80 |
| n19-w0.50 | mixed | 0.0016 [-0.0012, 0.0043] | 40/80 |
| n19-w0.75 | unsupported | -0.0111 [-0.0165, -0.0049] | 28/80 |

## G-INTERPOLATION-MIXTURE-PROFILE

| Cell | Status | Paired effect [95% interval] | Wins |
|---|---|---:|---:|
| n8-w0.35 | unsupported | -0.0248 [-0.0418, -0.0064] | 28/80 |
| n8-w0.50 | unsupported | -0.0363 [-0.0432, -0.0292] | 10/80 |
| n8-w0.75 | unsupported | -0.0525 [-0.0584, -0.0461] | 6/80 |
| n12-w0.35 | mixed | -0.0016 [-0.0052, 0.0019] | 39/80 |
| n12-w0.50 | unsupported | -0.0251 [-0.0283, -0.0216] | 21/80 |
| n12-w0.75 | unsupported | -0.0548 [-0.0567, -0.0526] | 20/80 |
| n16-w0.35 | supported | 0.0117 [0.0067, 0.0172] | 50/80 |
| n16-w0.50 | unsupported | -0.0099 [-0.0133, -0.0057] | 30/80 |
| n16-w0.75 | unsupported | -0.0681 [-0.0696, -0.0666] | 20/80 |
| n19-w0.35 | supported | 0.0188 [0.0164, 0.0213] | 58/80 |
| n19-w0.50 | unsupported | -0.0029 [-0.0053, -0.0006] | 38/80 |
| n19-w0.75 | unsupported | -0.0619 [-0.0670, -0.0560] | 20/80 |

## G-CONTROL-BITSTRING

| Cell | Status | Paired effect [95% interval] | Wins |
|---|---|---:|---:|
| n8-w0.35 | supported | 0.2388 [0.2232, 0.2554] | 50/50 |
| n8-w0.50 | supported | 0.2055 [0.1855, 0.2306] | 50/50 |
| n8-w0.75 | supported | 0.1288 [0.1018, 0.1699] | 50/50 |
| n12-w0.35 | supported | 0.2578 [0.2494, 0.2684] | 50/50 |
| n12-w0.50 | supported | 0.2594 [0.2388, 0.2782] | 50/50 |
| n12-w0.75 | supported | 0.1687 [0.1409, 0.1999] | 50/50 |
| n16-w0.35 | supported | 0.2515 [0.2406, 0.2632] | 50/50 |
| n16-w0.50 | supported | 0.2575 [0.2402, 0.2765] | 50/50 |
| n16-w0.75 | supported | 0.2100 [0.1710, 0.2486] | 50/50 |
| n19-w0.35 | supported | 0.2394 [0.2256, 0.2523] | 50/50 |
| n19-w0.50 | supported | 0.2559 [0.2370, 0.2723] | 50/50 |
| n19-w0.75 | supported | 0.1940 [0.1476, 0.2405] | 50/50 |

## G-INTERPOLATION-LOOKUP-BITSTRING

| Cell | Status | Paired effect [95% interval] | Wins |
|---|---|---:|---:|
| n8-w0.35 | unsupported | -0.0243 [-0.0445, -0.0054] | 19/80 |
| n8-w0.50 | unsupported | -0.0380 [-0.0491, -0.0277] | 28/80 |
| n8-w0.75 | unsupported | -0.0099 [-0.0165, -0.0032] | 37/80 |
| n12-w0.35 | unsupported | -0.0371 [-0.0466, -0.0271] | 21/80 |
| n12-w0.50 | unsupported | -0.0356 [-0.0470, -0.0249] | 28/80 |
| n12-w0.75 | supported | 0.0156 [0.0083, 0.0233] | 42/80 |
| n16-w0.35 | unsupported | -0.0429 [-0.0538, -0.0319] | 25/80 |
| n16-w0.50 | unsupported | -0.0418 [-0.0510, -0.0318] | 30/80 |
| n16-w0.75 | supported | 0.0250 [0.0190, 0.0313] | 50/80 |
| n19-w0.35 | unsupported | -0.0610 [-0.0776, -0.0458] | 22/80 |
| n19-w0.50 | unsupported | -0.0606 [-0.0749, -0.0461] | 26/80 |
| n19-w0.75 | mixed | 0.0074 [-0.0032, 0.0165] | 39/80 |

## Prohibited claims

- extrapolation
- universal conditional expressivity
- quantum advantage
- efficient classical full-distribution sampling

# Ashkin-Teller sampled-training development comparison

Stages 1-2 result, 2026-09-07. All 11 conditions are development fitting/validation conditions. Confirmatory targets remain unopened.

## Result

| Encoder | Training loss | Mean TVD | Worst TVD | Mean C error | Worst C error | Mean encoder-only TVD gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cat | exact | 0.12405 | 0.18663 | 0.08757 | 0.22598 | 0.20698 |
| cat | sampled | 0.13556 | 0.18798 | 0.08678 | 0.23104 | 0.19547 |
| bond | exact | 0.10546 | 0.20659 | 0.09457 | 0.21948 | 0.15203 |
| bond | sampled | 0.10793 | 0.20956 | 0.09488 | 0.22853 | 0.14956 |

Means average three seed fits and 11 conditions; worst values range over every seed/condition. C error is the RMSE of the four connected interlayer bond correlations. These are exact output metrics, even when training uses sampled estimates.

## Paired engineering gate

Predeclared tolerance: median paired increase in mean TVD (sampled minus exact) ≤.03 for each encoder, with no unexplained estimator failures. This is an engineering tolerance, not a significance test.

- cat: paired gaps [-0.00108, +0.00922, +0.02639]; median +0.00922; PASS.
- bond: paired gaps [+0.00179, +0.00240, +0.00322]; median +0.00240; PASS.

## Every development condition

TVDs and gains below average the three seeds. Δ is sampled minus exact TVD; gain is encoder-only minus sampled TVD, so negative gain identifies a condition worsened by training.

| (a,b) | Cat exact | Cat sampled | Cat Δ | Cat gain | Bond exact | Bond sampled | Bond Δ | Bond gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (0.1,0.1) | 0.18543 | 0.18558 | 0.00015 | 0.29041 | 0.20658 | 0.20779 | 0.00121 | 0.01688 |
| (0.3,0.1) | 0.15718 | 0.16048 | 0.00330 | 0.28668 | 0.14578 | 0.14682 | 0.00104 | 0.08770 |
| (0.5,0.1) | 0.13307 | 0.14193 | 0.00887 | 0.22945 | 0.09938 | 0.10081 | 0.00143 | 0.14346 |
| (0.7,0.1) | 0.10818 | 0.13550 | 0.02732 | 0.16287 | 0.07064 | 0.07385 | 0.00321 | 0.17801 |
| (0.1,0.3) | 0.15292 | 0.16647 | 0.01354 | 0.28070 | 0.14579 | 0.14760 | 0.00181 | 0.08693 |
| (0.1,0.5) | 0.13180 | 0.14839 | 0.01659 | 0.22300 | 0.09938 | 0.10319 | 0.00381 | 0.14108 |
| (0.1,0.7) | 0.11764 | 0.12772 | 0.01008 | 0.17065 | 0.07065 | 0.07438 | 0.00373 | 0.17748 |
| (0.5,0.5) | 0.10015 | 0.11205 | 0.01190 | 0.15614 | 0.08280 | 0.08492 | 0.00212 | 0.21664 |
| (0.5,0.7) | 0.09267 | 0.10028 | 0.00762 | 0.13056 | 0.08067 | 0.08385 | 0.00318 | 0.20975 |
| (0.7,0.5) | 0.08822 | 0.10223 | 0.01400 | 0.12862 | 0.08068 | 0.08502 | 0.00434 | 0.20858 |
| (0.7,0.7) | 0.09731 | 0.11052 | 0.01321 | 0.09112 | 0.07772 | 0.07900 | 0.00127 | 0.17871 |

| (a,b) | Encoder-only C error | Cat exact C error | Cat sampled C error | Bond exact C error | Bond sampled C error |
| --- | ---: | ---: | ---: | ---: | ---: |
| (0.1,0.1) | 0.30740 | 0.22341 | 0.20875 | 0.21947 | 0.22022 |
| (0.3,0.1) | 0.27823 | 0.16626 | 0.14799 | 0.16238 | 0.16246 |
| (0.5,0.1) | 0.21601 | 0.10539 | 0.08526 | 0.09363 | 0.09321 |
| (0.7,0.1) | 0.13940 | 0.05189 | 0.04309 | 0.03281 | 0.03231 |
| (0.1,0.3) | 0.27823 | 0.16631 | 0.14749 | 0.16238 | 0.16275 |
| (0.1,0.5) | 0.21601 | 0.10459 | 0.08538 | 0.09363 | 0.09400 |
| (0.1,0.7) | 0.13940 | 0.04884 | 0.04121 | 0.03281 | 0.03358 |
| (0.5,0.5) | 0.12581 | 0.00435 | 0.03340 | 0.02777 | 0.02886 |
| (0.5,0.7) | 0.07311 | 0.02825 | 0.04955 | 0.06438 | 0.06445 |
| (0.7,0.5) | 0.07311 | 0.02575 | 0.05215 | 0.06438 | 0.06519 |
| (0.7,0.7) | 0.04085 | 0.03821 | 0.06034 | 0.08665 | 0.08670 |

## Interpretation

Both encoders pass at 2,048 reference samples; the 4,096-sample fallback is not needed. Sampled joint TVD improves on the encoder-only input in every seed/condition. The symmetry-only fits show more sensitivity to sampled optimization: their largest seed-average gap is .02639, compared with .00322 for bond-informed fits. Passing the median tolerance does not imply identical optimization trajectories.

Bond-informed input gives lower mean TVD, while symmetry-only input gives lower worst TVD and lower mean connected-correlation error. At (.7,.7), both sampled-trained models worsen connected-correlation error relative to their encoder-only inputs. The worst errors remain near .23. Thus this supports a controlled comparison of supplied physics and learned contributions, not a claim of uniformly superior bond encoding or uniformly accurate interlayer dependence.

These results support freezing stage 3 next. They are not themselves new generalization evidence; no confirmatory run has been launched.

## Protocol, verification and limits

All 12 fits use 500 Adam-cosine updates, 100 training samples per condition per update, 1,000 independently generated training/validation examples per condition, 512 masks per bandwidth/condition and 2,048 reference samples. The three seeds are 1701, 1702, 1703. No ancillas; both encoders use the same deduplicated 30-phase two-local bank.

Both training modes select the minimum fixed sampled development-validation MMD among the initial checkpoint and checks every 25 updates. Exact controls optimize population MMD; sampled fits additionally include finite training-data, mask-bank and reference noise. Thus the gap is not a pure reference-sampling-error estimate.

The supplied bond input has uncoupled-layer probability amplitudes with fixed phase eta=pi/8. Its intra-layer law already contains useful physics. The cat input supplies the layer partition and both spin-flip symmetries. Calling the latter completely physics-unaware would be misleading.

The final implementation passed all 218 tests. Every recorded checkpoint/data hash was checked; saved output probabilities were independently recomputed, normalized, and verified against both flips. All four variants share data-derived initialization, mask banks and batching for each seed. Per-seed and per-condition marginal/correlation metrics are retained in the JSON and raw checkpoint bundles.

A slow first implementation was stopped for runtime reasons. Partial fits remain in the sibling `-superseded-matrix-products` directory and are excluded here. Equivalent elementwise contractions and canceled ratio normalizers reduced runtime; deterministic input arrays were verified identical before restarting under the new source freeze.

Frozen manifest file SHA256: `c0013d4676fcf55cd046bc57a791697c2953e5b044b37f2d3e02861e348f8c3a`. Artifacts: `artifacts/ashkin-teller-development-2026-09-07`. The complete source snapshot is `source.tar.gz`; the JSON records its hash and full fit provenance.

The former worst-correlation ≤.10 development gate remains a recorded failure. Passing the new sampled-fidelity gate does not reverse it. No classical comparison or quantum-advantage claim is made here. This experiment is not a fresh generalization test and does not establish hardware preparation or scalable output sampling.

## Claim-to-evidence map

| Claim | Evidence | Limitation | Review status |
| --- | --- | --- | --- |
| Tractable bond reference access and phase gradients | [Method](bond-reference-method.md), reference/estimator/preparation tests | Floating-point size scope; preparation is a construction, not hardware execution | Automated checks passed; author review pending |
| Sampled training approximates paired exact development fits | Frozen 12-fit protocol and paired gaps above | Three seeds; data/mask/reference noise combined | Engineering gate reported; author review pending |
| Learned IQP phases improve on the supplied input | Encoder-only gains and connected-correlation errors above | C error worsens at (.7,.7); uneven correlation accuracy | Development evidence only |
| Generalization to fresh joint conditions | No evidence from this experiment | Stage 3 remains unrun | Open |

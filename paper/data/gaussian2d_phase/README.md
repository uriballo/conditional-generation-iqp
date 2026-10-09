# Phase-prior two-dimensional Gaussian diagnostics

This bundle contains the frozen sampled-training diagnostics for the phase-prior Gaussian experiment. The encoder uses a fixed shell-amplitude reference state, condition-dependent collective phases, and one shared learned rotation angle. The physical encoder measurement distribution is fixed; the conditions act through phases.

The campaign contains two protocols: separate variation and joint variation. Each protocol has five final fits at sixteen qubits. Checkpoint selection used exact population TVD on the declared development conditions; the fitting, development, and reporting condition sets were fixed before the campaign. The reporting conditions had been explored during encoder and hyperparameter development, so these results support a capability demonstration rather than an independent generalization claim.

The fixed rotation control uses `theta = 3*pi/8`, frozen before sampled training. It is retained as a diagnostic and does not establish necessity of the learned model or of the phase prior.

`DIAGNOSTICS.npz` is copied byte-for-byte from the frozen results directory. `SUMMARY.json` is a sanitized copy containing no absolute paths.

From the repository root, rebuild the figures with:

```sh
python paper/figure_sources/gaussian2d_phase.py
python paper/figure_sources/gaussian2d_phase.py --output paper/figures
```

The default command writes to the ignored `results/gaussian2d-phase-publication-replay/` directory. The explicit `paper/figures` command rebuilds the publication copies. The primary controls replay uses its own source and default output under `results/`; run it separately after its controls bundle is present.

`control_parameters.npz` contains only the initial, selected and terminal
angles, selected steps, seeds and reporting coordinates from the ten frozen
final fits. `control_provenance.json` records the originating checkpoint hashes.
`controls.json` and the two manuscript tables replay those parameters and
post-hoc target-slope controls through a 17-dimensional collective-X operator.
The product-prior control replaces only the initial amplitudes; its conditional
quadratic phase remains entangling. It has not undergone sampled training.

From the repository root, regenerate the controls without changing publication
artifacts:

```sh
.venv/bin/python paper/figure_sources/gaussian2d_phase_controls.py --tables results/gaussian2d-phase-publication-replay/tables
```

Explicit publication rebuild:

```sh
.venv/bin/python paper/figure_sources/gaussian2d_phase_controls.py --output paper/data/gaussian2d_phase/controls.json --tables paper/tables
```

The included data support exact figure/control replay. They are a compact
publication evidence package, not a standalone implementation of the complete
26-fit training campaign. The frozen campaign's source and fit artifacts remain
in the historical research records. `REVIEW.md` gives the claim-evidence map.

## Finite-target validation follow-up

The supplementary finite-validation bundle records twenty matched fits: five
paired fits for each of two training designs and two estimators. Its target
validation pools contain 128 strings per development condition. Checkpoint
selection uses empirical target MMD and exact model moments; it does not use
population target probabilities or TVD. The encoder and optimizer retain
previously selected settings, and reporting conditions were already examined.
The original heatmaps and response figures continue to represent the earlier
exact-development-TVD benchmark. The supplementary table reports the new
finite-validation fits separately.

`finite_validation_parameters.npz` retains compact reporting and fitting
coordinates, fitting-pool means and fitted angles. `finite_validation.json`
records per-fit results. `finite_validation_provenance.json` identifies the
source campaign and exported evidence. The standalone replay evaluates the
seventeen-dimensional model from these publication inputs without training.
Direct calibration controls estimate angles from finite fitting means only;
product-prior controls retain the entangling quadratic phase.

Replay the supplementary evidence and table to ignored outputs:

```sh
.venv/bin/python paper/figure_sources/gaussian2d_phase_finite_validation.py
```

The manuscript interprets the example as structured conditional generation
and calibration. Neither iterative optimization nor Gaussian-prior preparation
is established as necessary for its fidelity. Pair averaging controls individual
moment samples but gives no resolved accuracy benefit in the matched fits.

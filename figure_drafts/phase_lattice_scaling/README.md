# Phase-encoded lattice plots

These plots use the frozen September 30 phase-encoded scaling campaign. The paper assets are `../../paper/figures/curie_weiss_lattice_scaling_mmd.pdf` and `curie_weiss_ancilla_tvd.pdf`, each with a PNG preview. The adjacent `per_condition_mmd.pdf` has separate training and held-out pages; `train_held_out_mmd.pdf` compares the grid averages. These diagnostics are not manuscript figures.

`figure_data.json` contains the exact plotted per-seed arrays, input audit hashes and original plotting-source hash. `provenance.json` binds these arrays and delivered files to the sealed campaign. Full run directories and checkpoints are local results and are not part of this export.

Scaling uses the general product phase encoder RZ(0.75 beta) RY(pi/2) on every wire, no ancillas, and open row-major nearest-neighbor rectangular lattices. The sizes, grids and independently trained phase counts are:

| Qubits | Grid | Parameters |
| --- | --- | --- |
| 64 | 8 by 8 | 176 |
| 128 | 8 by 16 | 360 |
| 256 | 16 by 16 | 736 |
| 512 | 16 by 32 | 1488 |
| 1024 | 32 by 32 | 3008 |
| 2048 | 32 by 64 | 6048 |

Each of seeds 100, 101, 102 has three 5000-update restarts. The training protocol has 512 masks per bandwidth with stepwise refresh, 1024 reference draws and bandwidths corresponding to expected mask weights two and four. Validation MMD at training temperatures selects a restart and checkpoint. Reported trajectories follow that selected restart at updates 0, 500, ..., 5000, rather than an envelope across restarts. A selected checkpoint may lie between plotted points.

Independent reporting uses five evaluation banks per seed. The displayed line and band are the seed median and IQR after averaging banks within each seed. Training conditions are beta 0.6, 0.8, 1.2, 1.4. Held-out conditions are beta 0.95, 1, 1.05, 0.4, 1.6. Unbiased negative estimates are retained. The train and held-out curves average different temperature grids, so their difference is not a conventional generalization gap. Bandwidths vary with size, making absolute MMD values unsuitable for cross-size fidelity comparison. Convergence supports classical optimization feasibility; substantial high-beta residuals remain.

The ancilla figure has sixteen visible qubits on an open 4 by 4 lattice. The same phase encoder acts on all 16+r wires. Every visible qubit couples to every ancilla, with 40+16r parameters and no ancilla-only phases. Only the mixed coupling bank is shown, at r=0, 1, 3, 6, 10. Exact raw Hamming-profile TVD is extracted per condition from the audit rows, averaged over beta 0.4, 1, 1.6 within each seed 410, 411, 412, then summarized by seed median and IQR. The annotation "16 visible qubits" is placed inside the plot at the top right. The ancillas are marginalized. Increasing parameter count prevents interpreting this as a parameter-matched ancilla advantage. The previously plotted full visible-bitstring TVD is retained only as historical provenance.

To regenerate with NumPy, Matplotlib and optional LaTeX, run from the repository root:

```sh
python figure_drafts/phase_lattice_scaling/render.py --output-root /tmp/phase-lattice-plots
```

Only the ancillary and held-out scaling plots are written to the requested output root. Training and condition diagnostics stay in this source directory.

The standalone renderer needs neither the IQP library nor JAX nor raw experiment runs. It reproduces the exported curves and design from the recorded arrays. The committed PDFs are the reviewed publication assets; a later regeneration can differ in metadata and font rendering across software environments. Y axes autoscale to the curves and IQR bands; per-condition panels may have different y limits.

The optional aggregate training plot, `training_mmd.pdf`, stored in this source directory, uses the same design and seed median/IQR aggregation as the held-out plot. It averages the four training temperatures; the held-out plot averages its five different temperatures.

The ancilla and aggregate scaling plots have no panel titles.

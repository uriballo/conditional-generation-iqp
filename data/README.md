# Data

Result bundles behind the paper's figures and tables. `run_experiments.py --mode
figures` reads them and writes to `results/`.

| Bundle | Contents |
| --- | --- |
| `gaussian_lattice/` | Gaussian-shaped family, conditional IQP: five final fits per design, 40 lattice phases each |
| `gaussian2d-diagonal/` | Gaussian-shaped family, diagonal conditioning with an affine map (120 parameters) |
| `gaussian2d-diagonal-variants/` | Quadratic and all-to-all diagonal-conditioning variants |
| `curie-weiss-followup/` | Curie–Weiss C-IQP (P), per-temperature IQP and the remaining families of the parity protocol |
| `curie-weiss-twisted/` | Curie–Weiss C-IQP (TP), per-temperature IQP (TP) and the κ = 0 ablation |
| `curie-weiss-diagonal/` | Curie–Weiss D-IQP |
| `curie-weiss-classical/` | Classical RBM, Transformer and VAE baselines, small and large |
| `scaling/` | Lattice scaling at 64–2048 qubits, ⟨m²⟩ estimates, and the 64-qubit all-to-all comparison (`locality/`) |
| `hardware/` | IQM Emerald runs at 16 and 30 qubits with the calibration snapshot |

Curie–Weiss results use ten final repeats with three starts each and report the
mean and sample standard deviation over repeats.

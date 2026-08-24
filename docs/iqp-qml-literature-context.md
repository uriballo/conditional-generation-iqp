# IQP and quantum-generative-learning literature audit

Date: 2026-08-13  
Scope: contextualization of the Gaussian-only manuscript in `paper/main.tex`, beginning from the 18 records in `paper/references.bib`.

## Executive recommendation

The implemented manuscript uses a **36-source bibliography**: it retains the 30 scientifically relevant sources found during the audit, corrects three records, and adds six primary sources covering the original IQP definition, quantum feature maps, circuit learning, data re-uploading, and kernel concentration. All 36 sources are cited. The two remaining high-value recommendations below have titles centered on the benchmark family intentionally excluded from the Gaussian-only source archive, so they remain documented here rather than being inserted as uncited padding or weakening the archive boundary. Two further optional sources can bring a later revision to 37 or 38 if their associated claims remain in the text.

The most important scientific correction is that the paper must not imply that conditional QCBMs, or interpolation at an unseen condition, are new. Kiss et al. already trained a QCBM conditional on a continuous external parameter and held out an intermediate parameter value for interpolation [Kiss et al. (2022)](https://doi.org/10.1103/PhysRevA.106.022612). Zeng et al. subsequently proposed a conditional QCBM that encodes conditions using additional qubits and evaluated Gaussian-mixture and MNIST targets [Zeng et al. (2023)](https://doi.org/10.1016/j.physa.2023.128693). The present paper's defensible contribution is narrower:

> We introduce and systematically evaluate product-state conditioning for a fixed compiled IQP Born machine, showing that a family of conditional distributions can be learned without changing the circuit generators or sacrificing classical estimability of the selected MMD objective and its gradients.

Avoid “first conditional IQP model,” “first conditional Born machine,” and any sentence that treats conditional interpolation itself as unprecedented.

## Thematic map

### 1. IQP structure, simulation, and sampling complexity

The IQP model originates with Shepherd and Bremner's commuting, temporally unstructured computation [Shepherd and Bremner (2009)](https://doi.org/10.1098/rspa.2008.0443). Later work established hardness consequences for efficient classical simulation under specified approximation notions and complexity assumptions [Bremner, Jozsa, and Shepherd (2011)](https://doi.org/10.1098/rspa.2010.0301), [Bremner, Montanaro, and Shepherd (2016)](https://doi.org/10.1103/PhysRevLett.117.080501), and [Bremner, Montanaro, and Shepherd (2017)](https://doi.org/10.22331/q-2017-04-25-8). Those papers establish circuit-family complexity results; applying them to a particular learned instance or claiming a learning advantage would require additional argument not supplied by the present experiment.

The same circuit family also contains classically tractable observable-estimation regimes. Van den Nest gives probabilistic simulation methods for restricted circuits and observables [Van den Nest (2011)](https://arxiv.org/abs/0911.1624); Ni and Van den Nest explicitly contrast efficiently simulable and hard commuting-circuit regimes [Ni and Van den Nest (2012)](https://arxiv.org/abs/1204.4570). Fujii and Morimae connect commuting-circuit amplitudes to Ising partition functions and identify both tractable subclasses and hardness regimes [Fujii and Morimae (2017)](https://doi.org/10.1088/1367-2630/aa5fdb). Bravyi, Gosset, and Movassagh provide broader context for classical estimation of quantum mean values, under assumptions different from this manuscript's IQP estimator [Bravyi, Gosset, and Movassagh (2021)](https://doi.org/10.1038/s41567-020-01109-8).

**Positioning consequence:** use “classically estimable selected observables/loss/gradients,” not “classically simulable IQP circuit.” Reserve sampling-hardness language for the circuit-family motivation and state its assumptions.

### 2. Circuit Born machines and generative objectives

Liu and Wang introduced differentiable QCBM learning with MMD and analytic gradients [Liu and Wang (2018)](https://doi.org/10.1103/PhysRevA.98.062324). Benedetti et al. established a hardware-facing data-driven circuit-learning framework and generative benchmarks for shallow circuits [Benedetti et al. (2019)](https://doi.org/10.1038/s41534-019-0157-8). Coyle et al. then combined an IQP-related Ising Born machine with MMD, Stein, and Sinkhorn objectives while carefully separating training procedures from sampling-complexity claims [Coyle et al. (2020)](https://doi.org/10.1038/s41534-020-00288-9).

MMD is an integral probability metric defined through kernel mean embeddings and estimable from samples [Gretton et al. (2012)](https://www.jmlr.org/papers/v13/gretton12a.html). In QCBMs, loss choice and locality can materially affect optimization: Leadbeater et al. study adversarially estimated f-divergences and local generative costs [Leadbeater et al. (2021)](https://doi.org/10.3390/e23101281), while Rudolph et al. distinguish explicit/implicit models and losses and analyze trainability barriers for quantum generative models [Rudolph et al. (2024)](https://doi.org/10.1038/s41534-024-00902-0).

**Positioning consequence:** call the manuscript's kernel a **classical positive-definite kernel on bitstrings**. Do not conflate it with a “quantum kernel,” which conventionally means a kernel induced or evaluated by a quantum feature map.

### 3. Conditional quantum generation

The closest predecessor is Kiss et al.: their condition enters parameterized rotations throughout a QCBM, their target is conditioned on incoming particle energy, and they explicitly test a held-out intermediate energy [Kiss et al. (2022)](https://doi.org/10.1103/PhysRevA.106.022612). Zeng et al. encode conditional information using added qubits in a hybrid quantum-classical conditional QCBM and test Gaussian mixtures and MNIST [Zeng et al. (2023)](https://doi.org/10.1016/j.physa.2023.128693). These differ from the present model because the present condition acts only through the initial visible-qubit product state, while the compiled IQP generator list and learned phase vector are shared.

Kyriienko, Paine, and Elfving propose a distinct differentiable quantum generative model based on a phase feature map, latent-space training, and a fixed basis transformation for sampling [Kyriienko, Paine, and Elfving (2024)](https://doi.org/10.1103/PhysRevResearch.6.033291). It is useful broader context for input-dependent generative circuits, but it should not be described as the same fixed-product-input IQP construction.

**Positioning consequence:** the closest-work paragraph should compare **where the condition enters**, **whether the trained unitary is fixed across conditions**, and **what is classically estimable**. Those axes communicate novelty more precisely than saying only that the model is conditional.

### 4. Data encoding and Fourier response

Input-dependent rotations belong to the established quantum-circuit-learning and quantum-feature-map literature [Mitarai et al. (2018)](https://doi.org/10.1103/PhysRevA.98.032309), [Schuld and Killoran (2019)](https://doi.org/10.1103/PhysRevLett.122.040504), and [Havlíček et al. (2019)](https://doi.org/10.1038/s41586-019-0980-2). Repeated data injection, normally called data re-uploading, enlarges the functional family of a parameterized circuit [Pérez-Salinas et al. (2020)](https://doi.org/10.22331/q-2020-02-06-226) and provides the correct context for the manuscript's re-uploading ablation.

The Fourier-series view makes the effect of encoding quantitative: encoding spectra determine accessible frequencies [Schuld, Sweke, and Meyer (2021)](https://doi.org/10.1103/PhysRevA.103.032430), with input redundancy and multidimensional extensions developed by [Gil Vidal and Theis (2020)](https://doi.org/10.3389/fphy.2020.00297) and [Casas and Cervera-Lierta (2023)](https://doi.org/10.1103/PhysRevA.107.062612). The manuscript's propositions should therefore remain “specializations” or “consequences” of established circuit-Fourier theory.

**Positioning consequence:** describe the broadcast map as a deliberately low-complexity **condition encoding**, not as a novel feature-map theorem. “Symmetric re-uploading encoding” is clearer than “symmetric-reupload encoding.”

### 5. Classical estimability versus optimization trainability

IQPopt turns probabilistic IQP observable estimation into a JAX optimization tool for objectives expressible through Pauli-Z expectations [Armengol and Bowles (2025)](https://arxiv.org/abs/2501.04776). Recio-Armengol, Ahmed, and Bowles apply that interface to classically trained, quantum-deployed IQP generative models at large scale [Recio-Armengol, Ahmed, and Bowles (2025)](https://arxiv.org/abs/2503.02934).

Efficient unbiased or controlled-error estimation is not the same as a favorable loss landscape. Barren plateaus can arise in broad variational-circuit settings [McClean et al. (2018)](https://doi.org/10.1038/s41467-018-07090-4), depend on circuit depth and cost locality [Cerezo et al. (2021)](https://doi.org/10.1038/s41467-021-21728-w), and can also appear as exponential concentration in quantum kernel values [Thanasilp et al. (2024)](https://doi.org/10.1038/s41467-024-49287-w). Recent IQP-QCBM analyses make this caveat model-specific: gradient variance depends on generator structure, kernel spectrum, topology, and initialization [Shen et al. (2026)](https://arxiv.org/abs/2602.11042), while Lerch et al. compare data-dependent and data-agnostic initialization strategies in the same general IQP-MMD setting [Lerch et al. (2026)](https://arxiv.org/abs/2603.14576).

Recent extensions also limit broad expressivity claims. Kurkin et al. obtain universality only for a generalized classically trained, quantum-deployed model with hidden-qubit resources and study kernel adaptation [Kurkin et al. (2025)](https://arxiv.org/abs/2510.08476). Huang et al. generalize binary IQP Born machines through group Fourier analysis into spectral Born machines [Huang et al. (2026)](https://arxiv.org/abs/2607.06675).

**Positioning consequence:** define “classically trainable” operationally and locally: the selected observable-based MMD estimate and its derivatives are classically estimable at the stated error/budget. Do not let the term imply guaranteed convergence, efficient classical sampling, full probability access, or universal approximation.

### 6. Generalization terminology

Gili et al. study whether one QCBM trained on incomplete support generates unseen valid and high-quality bitstrings [Gili et al. (2023)](https://doi.org/10.1088/2058-9565/acd578). That is different from evaluating a conditional distribution at an unseen value of an explicit coordinate. The manuscript is right to use **held-out within-range condition interpolation** after defining its narrow meaning once. This distinction should be attributed directly to Gili et al. rather than presented as a generic terminology preference.

## Audited bibliography options: 36 implemented, up to 40 documented

### Keep: 30 current sources

All 30 current scientific sources are relevant, but three records need correction as noted below. The current set already includes the core IQP simulation/hardness line; IQPopt and train-on-classical/deploy-on-quantum papers; QCBM, conditional-QCBM, and differentiable-generative-model papers; Fourier-encoding papers; MMD; generalization/learnability papers; and recent IQP-specific trainability and universality preprints.

1. Van den Nest, “Simulating Quantum Computers with Probabilistic Methods” (2011), [arXiv:0911.1624](https://arxiv.org/abs/0911.1624).
2. Ni and Van den Nest, “Commuting Quantum Circuits: Efficient Classical Simulations versus Hardness Results” (2012), [arXiv:1204.4570](https://arxiv.org/abs/1204.4570).
3. Nakata and Murao, “Diagonal Quantum Circuits: Their Computational Power and Applications” (2014), [DOI](https://doi.org/10.1140/epjp/i2014-14152-9).
4. Bremner, Jozsa, and Shepherd, “Classical Simulation of Commuting Quantum Computations Implies Collapse of the Polynomial Hierarchy” (2011), [DOI](https://doi.org/10.1098/rspa.2010.0301).
5. Bremner, Montanaro, and Shepherd, “Average-Case Complexity versus Approximate Simulation of Commuting Quantum Computations” (2016), [DOI](https://doi.org/10.1103/PhysRevLett.117.080501).
6. Bremner, Montanaro, and Shepherd, “Achieving Quantum Supremacy with Sparse and Noisy Commuting Quantum Computations” (2017), [DOI](https://doi.org/10.22331/q-2017-04-25-8).
7. Armengol and Bowles, “IQPopt” (2025), [arXiv:2501.04776](https://arxiv.org/abs/2501.04776).
8. Recio-Armengol, Ahmed, and Bowles, “Train on Classical, Deploy on Quantum” (2025), [arXiv:2503.02934](https://arxiv.org/abs/2503.02934).
9. Gil Vidal and Theis, “Input Redundancy for Parameterized Quantum Circuits” (2020), [DOI](https://doi.org/10.3389/fphy.2020.00297).
10. Schuld, Sweke, and Meyer, “The Effect of Data Encoding on the Expressive Power of Variational Quantum-Machine-Learning Models” (2021), [DOI](https://doi.org/10.1103/PhysRevA.103.032430).
11. Casas and Cervera-Lierta, “Multidimensional Fourier Series with Quantum Circuits” (2023), [DOI](https://doi.org/10.1103/PhysRevA.107.062612).
12. Gretton et al., “A Kernel Two-Sample Test” (2012), [JMLR](https://www.jmlr.org/papers/v13/gretton12a.html).
13. Liu and Wang, “Differentiable Learning of Quantum Circuit Born Machine” (2018), [DOI](https://doi.org/10.1103/PhysRevA.98.062324).
14. Kiss et al., “Conditional Born Machine for Monte Carlo Event Generation” (2022), [DOI](https://doi.org/10.1103/PhysRevA.106.022612), [arXiv:2205.07674](https://arxiv.org/abs/2205.07674).
15. Gili et al., “Do Quantum Circuit Born Machines Generalize?” (2023), [DOI](https://doi.org/10.1088/2058-9565/acd578), [arXiv:2207.13645](https://arxiv.org/abs/2207.13645).
16. Cerezo et al., “Variational Quantum Algorithms” (2021), [DOI](https://doi.org/10.1038/s42254-021-00348-9).
17. Rudolph et al., “Trainability Barriers and Opportunities in Quantum Generative Modeling” (2024), [DOI](https://doi.org/10.1038/s41534-024-00902-0).
18. Benedetti et al., “A Generative Modeling Approach for Benchmarking and Training Shallow Quantum Circuits” (2019), [DOI](https://doi.org/10.1038/s41534-019-0157-8).
19. Gao, Zhang, and Duan, “A Quantum Machine Learning Algorithm Based on Generative Models” (2018), [DOI](https://doi.org/10.1126/sciadv.aat9004).
20. Sweke et al., “On the Quantum versus Classical Learnability of Discrete Distributions” (2021), [DOI](https://doi.org/10.22331/q-2021-03-23-417).
21. Leadbeater et al., “F-Divergences and Cost Function Locality in Generative Modelling with Quantum Circuits” (2021), [DOI](https://doi.org/10.3390/e23101281).
22. Du et al., “Power of Quantum Generative Learning” (2023), [DOI](https://doi.org/10.1103/PhysRevLett.131.140601).
23. Kyriienko, Paine, and Elfving, “Protocols for Trainable and Differentiable Quantum Generative Modeling” (2024), [DOI](https://doi.org/10.1103/PhysRevResearch.6.033291).
24. Zeng et al., “Conditional Quantum Circuit Born Machine Based on a Hybrid Quantum–Classical Framework” (2023), [DOI](https://doi.org/10.1016/j.physa.2023.128693).
25. Kurkin et al., “Universality and Kernel-Adaptive Training for Classically Trained, Quantum-Deployed Generative Models” (2025), [arXiv:2510.08476](https://arxiv.org/abs/2510.08476).
26. Shen et al., “Characterizing Trainability of Instantaneous Quantum Polynomial Circuit Born Machines” (2026), [arXiv:2602.11042](https://arxiv.org/abs/2602.11042).
27. Lerch et al., “IQP Born Machines under Data-dependent and Agnostic Initialization Strategies” (2026), [arXiv:2603.14576](https://arxiv.org/abs/2603.14576).
28. De Luca, “Trainability of IQP Quantum Circuit Born Machines under Gaussian Initialization” (2026), [arXiv:2606.10179](https://arxiv.org/abs/2606.10179).
29. Huang et al., “Spectral Born Machines: Classically Trainable Quantum Generative Models for Discrete Data” (2026), [arXiv:2607.06675](https://arxiv.org/abs/2607.06675).
30. McClean et al., “Barren Plateaus in Quantum Neural Network Training Landscapes” (2018), [DOI](https://doi.org/10.1038/s41467-018-07090-4).

### High-value sources considered

1. Shepherd and Bremner, “Temporally Unstructured Quantum Computation,” *Proceedings of the Royal Society A* 465, 1413–1439 (2009), [DOI](https://doi.org/10.1098/rspa.2008.0443), [arXiv:0809.0847](https://arxiv.org/abs/0809.0847).
2. Fujii and Morimae, “Commuting Quantum Circuits and Complexity of Ising Partition Functions,” *New Journal of Physics* 19, 033003 (2017), [DOI](https://doi.org/10.1088/1367-2630/aa5fdb), [arXiv:1311.2128](https://arxiv.org/abs/1311.2128). Documented but not added to the Gaussian-only source package.
3. Havlíček et al., “Supervised Learning with Quantum-Enhanced Feature Spaces,” *Nature* 567, 209–212 (2019), [DOI](https://doi.org/10.1038/s41586-019-0980-2), [arXiv:1804.11326](https://arxiv.org/abs/1804.11326).
4. Schuld and Killoran, “Quantum Machine Learning in Feature Hilbert Spaces,” *Physical Review Letters* 122, 040504 (2019), [DOI](https://doi.org/10.1103/PhysRevLett.122.040504), [arXiv:1803.07128](https://arxiv.org/abs/1803.07128).
5. Coyle et al., “The Born Supremacy: Quantum Advantage and Training of an Ising Born Machine,” *npj Quantum Information* 6, 60 (2020), [DOI](https://doi.org/10.1038/s41534-020-00288-9), [arXiv:1904.02214](https://arxiv.org/abs/1904.02214). Documented but not added to the Gaussian-only source package.
6. Mitarai et al., “Quantum Circuit Learning,” *Physical Review A* 98, 032309 (2018), [DOI](https://doi.org/10.1103/PhysRevA.98.032309), [arXiv:1803.00745](https://arxiv.org/abs/1803.00745).
7. Pérez-Salinas et al., “Data Re-uploading for a Universal Quantum Classifier,” *Quantum* 4, 226 (2020), [DOI](https://doi.org/10.22331/q-2020-02-06-226), [arXiv:1907.02085](https://arxiv.org/abs/1907.02085).
8. Thanasilp et al., “Exponential Concentration in Quantum Kernel Methods,” *Nature Communications* 15, 5200 (2024), [DOI](https://doi.org/10.1038/s41467-024-49287-w), [arXiv:2208.11060](https://arxiv.org/abs/2208.11060).

### Optional additions: choose zero, one, or two to land at 38–40

1. Sim, Johnson, and Aspuru-Guzik, “Expressibility and Entangling Capability of Parameterized Quantum Circuits for Hybrid Quantum-Classical Algorithms,” *Advanced Quantum Technologies* 2, 1900070 (2019), [DOI](https://doi.org/10.1002/qute.201900070), [arXiv:1905.10876](https://arxiv.org/abs/1905.10876). Add if the topology ablation is discussed in expressibility language.
2. Bravyi, Gosset, and Movassagh, “Classical Algorithms for Quantum Mean Values,” *Nature Physics* 17, 337–341 (2021), [DOI](https://doi.org/10.1038/s41567-020-01109-8), [arXiv:1909.11485](https://arxiv.org/abs/1909.11485). Add only for broader observable-estimation context; its assumptions do not establish this manuscript's estimator.

### Drop or replace

- **Drop no scientific source outright.** The current set is compact and relevant.
- **Replace the bibliographic record, not the source, for Gili et al.** It is a journal article in *Quantum Science and Technology* 8, 035021 (2023), DOI [10.1088/2058-9565/acd578](https://doi.org/10.1088/2058-9565/acd578), not merely an arXiv `@misc` record.
- **Correct IQPopt:** the current arXiv record lists **Erik Armengol** and an original 2025 date [arXiv:2501.04776](https://arxiv.org/abs/2501.04776).
- **Correct Train on Classical, Deploy on Quantum:** retain **Erik Recio-Armengol**, but use the original 2025 date unless the authors request citation of a later revision [arXiv:2503.02934](https://arxiv.org/abs/2503.02934).

## Claim-to-citation placements

Line references correspond to the manuscript state audited on 2026-08-13 and may drift after editing.

| Manuscript location | Recommended claim | Citations |
|---|---|---|
| Abstract, first two sentences | Do not add citations in the abstract. Replace “extend ... from learning one distribution per circuit” with language specific to the classically trainable IQP framework, because conditional QCBMs predate this work. | Substantiate later with [Recio-Armengol et al.](https://arxiv.org/abs/2503.02934), [Kiss et al.](https://doi.org/10.1103/PhysRevA.106.022612), [Zeng et al.](https://doi.org/10.1016/j.physa.2023.128693). |
| Introduction, opening paragraph (around lines 151–159) | “Prior conditional QCBMs inject external variables through trainable/data-encoding gates or additional qubits; here the condition changes only the initial visible-qubit product state.” | [Kiss et al.](https://doi.org/10.1103/PhysRevA.106.022612), [Zeng et al.](https://doi.org/10.1016/j.physa.2023.128693). |
| Introduction, IQP access paragraph (around lines 160–171) | Introduce IQP with the original paper; separate particular-observable simulation from approximate-sampling hardness. | [Shepherd and Bremner](https://doi.org/10.1098/rspa.2008.0443), [Van den Nest](https://arxiv.org/abs/0911.1624), [Ni and Van den Nest](https://arxiv.org/abs/1204.4570), existing Bremner hardness papers. |
| Introduction, construction paragraph (around lines 173–181) | Frame product-state rotations as condition encoding within the quantum feature-map/Fourier literature. | [Mitarai et al.](https://doi.org/10.1103/PhysRevA.98.032309), [Schuld and Killoran](https://doi.org/10.1103/PhysRevLett.122.040504), [Schuld et al.](https://doi.org/10.1103/PhysRevA.103.032430), [Casas and Cervera-Lierta](https://doi.org/10.1103/PhysRevA.107.062612). |
| Introduction, contribution paragraph (around lines 183–194) | Use the safe novelty sentence from the executive recommendation. Follow it with the empirical contribution: a systematic size–width–condition performance map and held-out within-range interpolation. | Closest-work contrast: [Kiss et al.](https://doi.org/10.1103/PhysRevA.106.022612), [Zeng et al.](https://doi.org/10.1016/j.physa.2023.128693), [Recio-Armengol et al.](https://arxiv.org/abs/2503.02934). |
| Related work, first paragraph (around lines 257–267) | Add the original IQP source, Ising-partition-function connection, and the broader mean-value-estimation context. Make clear that the manuscript uses the Van-den-Nest/IQPopt lineage for its actual estimator. | [Shepherd and Bremner](https://doi.org/10.1098/rspa.2008.0443), [Fujii and Morimae](https://doi.org/10.1088/1367-2630/aa5fdb), [Bravyi et al.](https://doi.org/10.1038/s41567-020-01109-8), [Armengol and Bowles](https://arxiv.org/abs/2501.04776). |
| Related work, encoding paragraph (around lines 269–278) | Distinguish quantum feature-map kernels from the classical Hamming kernel used inside MMD. Mention data re-uploading as the established comparator for repeated condition injection. | [Havlíček et al.](https://doi.org/10.1038/s41586-019-0980-2), [Schuld and Killoran](https://doi.org/10.1103/PhysRevLett.122.040504), [Pérez-Salinas et al.](https://doi.org/10.22331/q-2020-02-06-226), [Gretton et al.](https://www.jmlr.org/papers/v13/gretton12a.html). |
| Related work, QCBM paragraph (around lines 280–289) | Expand from two sources to a short historical arc: differentiable MMD QCBM; shallow hardware QCBM; IQP/Ising Born machine; conditional QCBMs. Explicitly state the fixed-product-input distinction. | [Liu and Wang](https://doi.org/10.1103/PhysRevA.98.062324), [Benedetti et al.](https://doi.org/10.1038/s41534-019-0157-8), [Coyle et al.](https://doi.org/10.1038/s41534-020-00288-9), [Kiss et al.](https://doi.org/10.1103/PhysRevA.106.022612), [Zeng et al.](https://doi.org/10.1016/j.physa.2023.128693). |
| Related work, generalization sentence (around lines 283–289) | Keep the distinction between unseen bitstrings and unseen condition coordinates; note that Kiss et al. also tested condition-coordinate interpolation, so novelty lies in the IQP/classical-training/fixed-unitary combination. | [Gili et al.](https://doi.org/10.1088/2058-9565/acd578), [Kiss et al.](https://doi.org/10.1103/PhysRevA.106.022612). |
| Related work, trainability paragraph (around lines 291–295) | State that classical estimability does not ensure non-vanishing or useful gradients; identify kernel spectrum, cost locality, topology, and initialization as relevant factors. | [McClean et al.](https://doi.org/10.1038/s41467-018-07090-4), [Cerezo et al.](https://doi.org/10.1038/s41467-021-21728-w), [Thanasilp et al.](https://doi.org/10.1038/s41467-024-49287-w), [Shen et al.](https://arxiv.org/abs/2602.11042), [Lerch et al.](https://arxiv.org/abs/2603.14576). |
| Classical training signals, after the MMD estimator (around lines 341–369) | Attribute QCBM MMD and explain that this paper's special step is a character-mask decomposition whose IQP observables are classically estimated. | [Gretton et al.](https://www.jmlr.org/papers/v13/gretton12a.html), [Liu and Wang](https://doi.org/10.1103/PhysRevA.98.062324), [Armengol and Bowles](https://arxiv.org/abs/2501.04776), [Recio-Armengol et al.](https://arxiv.org/abs/2503.02934). |
| Response propositions (around lines 371–405) | Preserve “direct specialization” wording. For the observable filter, cite the exact IQP active-generator source; for the frequency statement, cite circuit-Fourier sources. | Existing Fourier citations plus [Recio-Armengol et al.](https://arxiv.org/abs/2503.02934). |
| Ablations (around lines 567–590) | Describe re-uploading as an established expressivity mechanism; describe topology in terms of connectivity rather than generic expressive power; state that hidden-qubit universality is established elsewhere and the present ancilla panel is only an empirical finite-budget ablation. | [Pérez-Salinas et al.](https://doi.org/10.22331/q-2020-02-06-226), [Sim et al.](https://doi.org/10.1002/qute.201900070), [Kurkin et al.](https://arxiv.org/abs/2510.08476). |
| Limitations (around lines 592–616) | Add that fixed-kernel IQP optimization can suffer spectrum/topology/initialization-dependent gradient concentration; avoid implying that successful finite-size runs settle asymptotic trainability. | [Shen et al.](https://arxiv.org/abs/2602.11042), [Lerch et al.](https://arxiv.org/abs/2603.14576), [Rudolph et al.](https://doi.org/10.1038/s41534-024-00902-0). |

## Terminology and language recommendations

| Avoid or qualify | Prefer | Reason |
|---|---|---|
| “classically trainable circuit” without a local definition | “the selected MMD objective and its gradients are classically estimable for this IQP family” | Prevents inference of efficient sampling, full simulation, or guaranteed optimization. |
| “first conditional IQP/Born machine” | “product-state-conditioned fixed-IQP Born machine” | Conditional QCBMs already exist [Kiss et al.](https://doi.org/10.1103/PhysRevA.106.022612), [Zeng et al.](https://doi.org/10.1016/j.physa.2023.128693). |
| “generalization” throughout | Define once as “held-out within-range condition interpolation,” then use “interpolation” | Separates this test from unseen-bitstring generalization [Gili et al.](https://doi.org/10.1088/2058-9565/acd578). |
| “quantum kernel” for the MMD kernel | “classical Hamming radial-basis kernel on bitstrings” | Quantum kernels normally refer to kernels induced/evaluated by quantum feature maps [Havlíček et al.](https://doi.org/10.1038/s41586-019-0980-2). |
| “single-peak circuit/model” | “independently trained reference model at one condition” | States the training role and avoids confusion with target-profile peaks. |
| “bank” | “collection of independently trained reference models” on first use; “reference collection” thereafter | More natural paper language and less implementation-oriented. |
| “lookup baseline” | “neighbor-reference oracle” | Makes the target-aware selection and its generosity explicit. |
| “fixed circuit” alone | “fixed compiled IQP unitary and learned phase vector” | Clarifies what remains invariant while the prepared input changes. |
| “input encoding” when discussing the target law | “condition encoding” | Separates the scalar conditioning coordinate from bitstring data represented by the output distribution. |
| “it works rather well” | Report the direction and scope: “tracks the moving Hamming-weight profile and improves over a matched unconditioned circuit across all tested size–width cells” | Converts persuasion into a verifiable empirical statement. |
| “universal” for the visible-qubit model | “finite Fourier response family at fixed encoding width” | Universality results require generalized resources and should be attributed, e.g. [Kurkin et al.](https://arxiv.org/abs/2510.08476). |

## Suggested related-work narrative

The current related-work section reads as a list of ingredients. A stronger narrative is:

1. **QCBMs made circuit sampling into an implicit generative model**, with MMD enabling sample-based training [Liu and Wang (2018)](https://doi.org/10.1103/PhysRevA.98.062324) and early hardware demonstrations [Benedetti et al. (2019)](https://doi.org/10.1038/s41534-019-0157-8).
2. **IQP Born machines create a distinctive train/deploy split**: selected Pauli observables can be estimated classically even though generic sampling is conjectured hard [Van den Nest (2011)](https://arxiv.org/abs/0911.1624), [Armengol and Bowles (2025)](https://arxiv.org/abs/2501.04776), [Recio-Armengol et al. (2025)](https://arxiv.org/abs/2503.02934).
3. **Conditional QCBMs already show that external coordinates can control generated laws**, including held-out interpolation [Kiss et al. (2022)](https://doi.org/10.1103/PhysRevA.106.022612) and extra-qubit condition encoding [Zeng et al. (2023)](https://doi.org/10.1016/j.physa.2023.128693).
4. **This paper joins those lines at a narrower architectural seam**: it puts the condition entirely in an initial visible-qubit product state, leaving a single compiled IQP unitary and phase vector shared, and tests the resulting conditional-control/fidelity tradeoff across a systematic Gaussian Hamming-profile surface.

That four-step progression makes the paper's contribution legible without overclaiming either conditional generation or IQP theory.

## Verification notes

- Metadata for journal papers was checked against DOI/publisher records; arXiv-only metadata was checked against the arXiv record.
- The implemented `paper/references.bib` contains **36** entries, all used in the manuscript. This is within the requested 30–40 range. The two excluded-title recommendations and two optional sources remain documented for a later broader-scope revision.
- The recent 2025–2026 IQP papers are primary manuscripts but not all are peer reviewed. They should be labeled and cited as arXiv preprints rather than formatted as journal articles.
- This note recommends sources and wording; it intentionally does not alter the manuscript or bibliography.

## Candidate physically conditioned datasets (2026-08-13)

All three candidates below use a $4\times4$ binary lattice, so the full target law at every condition can be computed by direct summation over $2^{16}=65{,}536$ configurations. This makes finite-sample training compatible with exact post-training TVD, moment, and correlation diagnostics. The main selection issue is not enumerability; it is whether the current isotropic Hamming-RBF MMD sees the physical structure that changes with the condition.

### Candidate A — attractive lattice gas conditioned on chemical potential

**Physical interpretation.** Each bit $n_i\in\{0,1\}$ records whether lattice site $i$ is occupied. At fixed inverse temperature $\beta$ and attraction $\epsilon>0$, the condition is the chemical potential $\mu$, which controls adsorption/density. Lee and Yang proved the mathematical equivalence between the lattice gas and the Ising model in an external field and used it to analyze phase behavior [Lee and Yang (1952)](https://doi.org/10.1103/PhysRev.87.410).

**Target law.** On a periodic square lattice,
\[
p_\mu(n)=Z(\mu)^{-1}\exp\!\left[\beta\epsilon
\sum_{\langle i,j\rangle}n_i n_j+\beta\mu\sum_i n_i\right].
\]
Exact enumeration supplies $Z(\mu)$ and every bitstring probability. A condition grid should avoid a curve on which almost all mass jumps between nearly empty and nearly full states; first scan exact density, compressibility, and nearest-neighbor occupancy to choose a smooth but nontrivial interval.

**What interpolation tests.** Held-out $\mu$ values test whether one circuit learns an adsorption isotherm: continuously changing particle number together with interaction-driven clustering. Report density $\langle N\rangle/16$, density variance/compressibility, nearest-neighbor pair occupancy, and full TVD in addition to MMD.

**Isotropic-kernel risk.** Hamming distance directly measures particles added or removed, so the current kernel is aligned with the dominant density response. It is nevertheless blind to *where* occupied sites sit: configurations with the same particle number and Hamming separation can have very different contact energies and cluster morphology. A model may therefore obtain a favorable MMD and density curve while learning incorrect spatial correlations.

**Recommendation: strongest first extension.** It has an immediately interpretable condition, preserves the 16-bit/exact-evaluation workflow, and is better aligned with Hamming geometry than a spin glass. Require full-bitstring TVD and pair-correlation diagnostics; do not present Hamming-MMD alone as evidence of learning the interacting gas.

### Candidate B — random-bond Ising model conditioned on temperature

**Physical interpretation.** Each bit maps to a spin $s_i\in\{-1,+1\}$, and the condition is inverse temperature $\beta=(k_BT)^{-1}$. Use one preregistered periodic $4\times4$ coupling realization $J_{ij}\in\{\pm1\}$, or a small fixed panel of instances. The square-lattice ferromagnet is the canonical finite-temperature order/disorder model [Onsager (1944)](https://doi.org/10.1103/PhysRev.65.117). For the disordered variant, Saul and Kardar explicitly compute exact density-of-states/partition-function information for periodic $L\times L$ two-dimensional $\pm J$ systems beginning at $L=4$ [Saul and Kardar (1994)](https://doi.org/10.1016/0550-3213(94)90037-X).

**Target law.** For a frozen coupling instance,
\[
p_\beta(s)=Z(\beta)^{-1}\exp\!\left(
\beta\sum_{\langle i,j\rangle}J_{ij}s_i s_j\right).
\]
Direct enumeration gives exact Gibbs laws, energy, correlations, and TVD. A ferromagnet is easier to interpret; a random-bond instance avoids reducing the task mostly to magnetization but adds instance variability.

**What interpolation tests.** Held-out temperatures test whether one fixed conditional generator follows the redistribution from high-entropy configurations toward low-energy correlated states. Primary physical diagnostics should be mean energy and specific heat; add magnetization for the ferromagnet or bond satisfaction/selected correlations for a disordered instance.

**Isotropic-kernel risk.** This is the most serious mismatch of the three candidates. Hamming distance counts spin flips but ignores the interaction graph and bond signs. In zero field, a configuration and its global spin flip have equal Gibbs weight while lying at maximal Hamming distance 16. Conversely, two one-spin changes have equal Hamming length even when their energy changes differ greatly. The Hamming-RBF objective can therefore emphasize magnetization shells while under-resolving energy and local correlations.

**Recommendation: scientifically strongest, but not under an isotropic-only objective.** Retain it as the preferred physics study after adding a preregistered energy- or graph-aware characteristic kernel, or at minimum a combined Hamming plus bond-feature MMD. If the present kernel must remain unchanged, treat the Ising study as a metric-stress experiment and make exact TVD/energy the primary evidence rather than expecting a clean positive conditional-generation result.

### Candidate C — spanning-conditioned site percolation conditioned on occupancy

**Physical interpretation.** Each bit marks an occupied site in a porous medium; the condition $p$ is site-occupation probability. Percolation was introduced to study how a random medium controls fluid connectivity [Broadbent and Hammersley (1957)](https://doi.org/10.1017/S0305004100032680). Square-lattice site percolation exhibits a sharp spanning crossover with occupancy; Newman and Ziff quantified spanning probabilities across $p$ [Newman and Ziff (2000)](https://doi.org/10.1103/PhysRevLett.85.4104), and exact finite-lattice spanning probabilities can be computed by enumeration/transfer methods [Mertens (2022)](https://doi.org/10.1088/1751-8121/ac4195).

**Target law.** Unconditioned site percolation,
\[
p_p(x)=p^{|x|}(1-p)^{16-|x|},
\]
is a factorized product law and would be too close to a trivial positive control for product-state conditioning. The useful target is the law conditional on a top-to-bottom spanning cluster $A$,
\[
p_p(x\mid A)=\frac{p^{|x|}(1-p)^{16-|x|}\mathbf 1_A(x)}
{\sum_y p^{|y|}(1-p)^{16-|y|}\mathbf 1_A(y)}.
\]
For $4\times4$, both the normalizer and all target probabilities are exactly enumerable.

**What interpolation tests.** Held-out $p$ values test whether the generator changes the density and morphology of configurations while preserving a global connectivity constraint. Report full TVD, spanning validity, occupied-site count, largest-cluster size, and shortest spanning-path length.

**Isotropic-kernel risk.** The spanning event is topological, whereas the kernel only sees Hamming distance. Bitstrings at the same occupancy can differ completely in connectivity, and a single strategically located bit can create or destroy a span without looking exceptional to an isotropic kernel. Conditioning on $A$ also makes low-$p$ targets concentrate on rare, thin paths, which can create abrupt finite-size changes and high estimator variance.

**Recommendation: do not use as the next headline dataset with the current loss.** It is an excellent adversarial ablation for demonstrating why a topology-aware kernel or explicit validity term is needed. If included now, label unconditioned percolation as an encoding sanity check and spanning-conditioned percolation as a deliberately kernel-misaligned stress test, not as the main second application.

### Comparative decision

| Candidate | Physical condition | Hamming-RBF alignment | Main scientific value | Recommendation |
|---|---|---:|---|---|
| Attractive lattice gas | chemical potential $\mu$ | moderate | density control plus interacting spatial correlations | **Run first** with TVD and pair diagnostics |
| Random-bond Ising | inverse temperature $\beta$ | weak | canonical thermal conditional generation | **Run after kernel extension** |
| Spanning-conditioned percolation | occupation probability $p$ | weak for connectivity | exposes metric blindness to topology | **Use as stress test only** |

The lattice gas is therefore the best immediate companion to the Gaussian Hamming-profile study. It changes the dominant Hamming-weight statistic for a physical reason while adding spatial correlations that expose whether the generator learns more than the shell profile. The Ising model remains the stronger long-term physics benchmark, but its scientific value is undermined if training and selection use only a topology-blind isotropic kernel.

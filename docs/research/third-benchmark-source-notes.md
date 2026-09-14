# Candidate third benchmark: source notes

These notes record primary sources relevant to selecting a third generative
benchmark. They are evidence for dataset definitions and exact finite targets,
not a recommendation by themselves.

## Bars and Stripes (BAS)

* Benedetti et al., “A generative modeling approach for benchmarking and
  training shallow quantum circuits,” *npj Quantum Information* 5, 45 (2019),
  [publisher page](https://www.nature.com/articles/s41534-019-0157-8) and
  [arXiv record](https://arxiv.org/abs/1801.07686). The paper explicitly uses
  BAS as a sampling benchmark, reports that its experiments include BAS along
  with zero-temperature ferromagnet and random thermal distributions, and
  introduces the qBAS score. It describes BAS as a canonical pattern-sampling
  task and discusses the role of entanglement in representing the patterns.

* Hamilton, Dumitrescu, and Pooser, “Generative model benchmarks for
  superconducting qubits,” *Physical Review A* 99, 062323 (2019),
  [accepted manuscript](https://link.aps.org/accepted/10.1103/PhysRevA.99.062323)
  and [arXiv record](https://arxiv.org/abs/1811.09905). This is a direct hardware
  benchmark using the 2x2 BAS set on fewer than five qubits. It evaluates the
  learned distribution with KL divergence and two F1 adaptations, and compares
  shallow circuit constructions under hardware connectivity constraints.

* For the finite support count, the primary paper’s BAS construction gives
  (2^n+2^m-2) patterns for an (n\times m) binary image (the two uniform
  images are shared by bars and stripes). Thus 2x4 has 22 valid images and
  3x4 has 30. This count is also stated in the authors’ maintained
  [qBAS protocol](https://quantumbenchmarkzoo.org/content/application-level-benchmark/protocols/qbas),
  which defines the ideal distribution as uniform over the valid patterns.
  These counts describe standard noiseless BAS. A biased or noisy conditional
  extension defines a different target and should be named explicitly.

## Exact finite-lattice Ising targets

* Wu and Hu, “Exact partition functions of the Ising model on MxN planar
  lattices with periodic-aperiodic boundary conditions,” *Journal of Physics
  A* 35 (2002), [paper record](https://arxiv.org/abs/cond-mat/0204217) and
  [full text](https://in.ncu.edu.tw/mcwu/publications/full_text/M_C_Wu_JPA_35_5189%282002%29.pdf).
  The authors derive exact partition functions for finite MxN square,
  triangular, and honeycomb lattices under four periodic/antiperiodic boundary
  conditions, and use them to study finite-size and boundary effects. For a
  small lattice, enumerating each spin configuration with its Boltzmann weight
  therefore gives an exact normalized Gibbs distribution and exact observables.

* Galluccio, Loebl, and Vondrák, “New algorithm for the Ising problem:
  partition function for finite lattice graphs,” *Physical Review Letters* 84,
  5924 (2000), [PubMed/DOI record](https://pubmed.ncbi.nlm.nih.gov/10991090/).
  The paper presents an algorithm for the Ising partition function on finite
  lattice graphs, including toroidal lattices. It supports treating boundary
  condition and finite-graph choice as explicit benchmark parameters rather
  than silently importing an infinite-lattice approximation.

## Practical limitation

BAS is structurally sparse and has a small, exactly known support; it is useful
for validity, mode coverage, and global pattern structure, but it does not test
a smooth distribution over all bitstrings. A finite-lattice Ising Gibbs family
has full support at finite temperature and exposes local correlations,
temperature, field, and boundary-condition effects, but exact support tables
grow as (2^N). These are factual differences to weigh against the project’s
existing Curie--Weiss and grid-Ising pilots.

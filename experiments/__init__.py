"""Paper experiments built on the `iqp-cond` library.

Studies live in one directory each and share the infrastructure in
:mod:`experiments.common`: deterministic target datasets, encoder maps,
a run harness with run directories, metrics, and the planned tracking and
sweep adapters.  The boundary rule is that experiments import only the public
``iqp_cond`` API, never campaign internals.
"""

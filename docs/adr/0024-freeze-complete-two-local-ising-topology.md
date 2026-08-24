# Freeze the Complete 2-local Ising topology

Every primary Ising model is visible-only (\(m=0\)) and uses the Complete
2-local IQP topology, giving 136 trainable generators at 16 qubits before any
explicitly labeled reference capacity is added. This preserves continuity with
the current mechanism study, does not inject the Target's square-lattice graph
into the model, and prevents topology or ancilla search; neither a
target-aligned lattice diagnostic nor an ancilla extension can rescue a failed
primary gate.

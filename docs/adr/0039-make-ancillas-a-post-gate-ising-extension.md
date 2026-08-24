# Make ancillas a post-gate Ising extension

The full 8-instance by 5-seed primary Ising matrix, its representability gate,
and its conditional success rules use the Visible-only Ising primary
(\(m=0\)). A bounded ancilla experiment may run only after that primary passes.
It uses the canonical and one frozen secondary coupling instance, three
matched optimizer seeds, and \(m\in\{3,5\}\). Every ancilla rung is paired with
a visible-only Matched-capacity control, yielding 24 new training runs. It
cannot change a failed visible-only verdict into a successful one. This keeps
the paper's central evidence attributable to product-state conditioning in a
fixed visible circuit while preserving the preliminary observation that
ancillas can add useful modeling capacity.

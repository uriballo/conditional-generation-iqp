"""Ancilla-purified product inputs for one shared visible IQP diagonal.

The condition controls only an ancilla product state.  Ancilla-controlled
``RY`` rotations then generate a low-rank family of visible product inputs.
The ancillas never enter the diagonal IQP generator bank: every latent input
branch is processed by the same visible phase vector.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .exact import (
    enumerate_bitstrings,
    exact_iqp_probabilities_many_jax,
    generator_characters,
)


def latent_bitstrings(ancillas: int) -> jnp.ndarray:
    """Return all computational-basis strings for ``ancillas`` controls."""
    if ancillas < 1:
        raise ValueError("ancillas must be positive")
    return jnp.asarray(enumerate_bitstrings(ancillas), dtype=jnp.float64)


def bernoulli_branch_weights(
    condition: jnp.ndarray,
    ancillas: int,
    *,
    margin: float = 0.0,
) -> jnp.ndarray:
    """Return product-ancilla branch weights for normalized conditions.

    With ``margin=0``, each ancilla bit is Bernoulli(``condition``).  A positive
    margin applies the target-independent overlap map
    ``p(t) = margin + (1 - 2 margin) t``.
    """
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must lie in [0, 0.5)")
    bits = latent_bitstrings(ancillas)
    condition = jnp.asarray(condition, dtype=jnp.float64)
    if condition.ndim != 1:
        raise ValueError("condition must be one-dimensional")
    probabilities = margin + (1.0 - 2.0 * margin) * condition
    factors = jnp.where(
        bits[None, :, :] > 0.5,
        probabilities[:, None, None],
        1.0 - probabilities[:, None, None],
    )
    return jnp.prod(factors, axis=-1)


def latent_input_angles(
    base_angles: jnp.ndarray,
    input_couplings: jnp.ndarray,
) -> jnp.ndarray:
    r"""Return branch-visible angles ``alpha_i(t,z)``.

    The output has shape ``(2**r, n_conditions, n_visible)`` and implements

    .. math::
       \alpha_i(t,z)=\alpha_i^{(0)}(t)+\sum_k L_{ik}(-1)^{z_k}.

    Since all controlled rotations share the ``Y`` axis, this is realized by
    an ordinary visible ``RY`` layer followed by ancilla-controlled ``RY``
    rotations.
    """
    base_angles = jnp.asarray(base_angles, dtype=jnp.float64)
    input_couplings = jnp.asarray(input_couplings, dtype=jnp.float64)
    if base_angles.ndim != 2:
        raise ValueError("base_angles must have shape (n_conditions, n_visible)")
    if input_couplings.ndim != 2:
        raise ValueError("input_couplings must have shape (n_visible, ancillas)")
    if base_angles.shape[1] != input_couplings.shape[0]:
        raise ValueError("base angles and input couplings disagree on n_visible")
    signs = 1.0 - 2.0 * latent_bitstrings(input_couplings.shape[1])
    shifts = signs @ input_couplings.T
    return base_angles[None, :, :] + shifts[:, None, :]


def exact_latent_input_probabilities(
    phases: jnp.ndarray,
    generators: jnp.ndarray,
    base_angles: jnp.ndarray,
    condition: jnp.ndarray,
    input_couplings: jnp.ndarray,
    *,
    margin: float = 0.0,
    characters: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Evaluate the visible marginal of the ancilla-purified input circuit.

    ``phases`` is a single visible IQP phase vector.  It is reused unchanged
    for every ancilla string; only the product-input angles vary by branch.
    """
    phases = jnp.asarray(phases, dtype=jnp.float64)
    generators = jnp.asarray(generators, dtype=jnp.int8)
    input_couplings = jnp.asarray(input_couplings, dtype=jnp.float64)
    if generators.ndim != 2:
        raise ValueError("generators must have shape (n_generators, n_visible)")
    if phases.shape != (generators.shape[0],):
        raise ValueError("phases must contain one value per visible generator")
    if input_couplings.shape[0] != generators.shape[1]:
        raise ValueError("input couplings and generators disagree on n_visible")
    if characters is None:
        characters = jnp.asarray(generator_characters(np.asarray(generators)))
    angles = latent_input_angles(base_angles, input_couplings)

    def one_branch(branch_angles: jnp.ndarray) -> jnp.ndarray:
        return exact_iqp_probabilities_many_jax(
            phases,
            generators,
            branch_angles,
            characters=characters,
        )

    branch_probabilities = jax.vmap(one_branch)(angles)
    weights = bernoulli_branch_weights(
        condition,
        input_couplings.shape[1],
        margin=margin,
    )
    return jnp.einsum("cb,bcp->cp", weights, branch_probabilities)

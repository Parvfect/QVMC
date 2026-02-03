

import jax
import jax.numpy as jnp


def radial_and_pairwise_features(x, n_particles, dim):
    """
    x: (batch, n_particles * dim)
    returns:
        r:    (batch, n_particles, 1)
        rij:  (batch, n_pairs, 1)
    """
    batch = x.shape[0]

    # (batch, N, d)
    R = x.reshape(batch, n_particles, dim)

    # One-body radii |r_i|
    r = jnp.linalg.norm(R, axis=-1, keepdims=True)

    # Pairwise distances |r_i - r_j|
    diff = R[:, :, None, :] - R[:, None, :, :]   # (batch, N, N, d)
    dist = jnp.linalg.norm(diff, axis=-1)        # (batch, N, N)

    # Keep only i < j
    i, j = jnp.triu_indices(n_particles, k=1)
    rij = dist[:, i, j][:, :, None]               # (batch, n_pairs, 1)

    return r, rij

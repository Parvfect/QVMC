

import jax
import jax.numpy as jnp


def radial_and_pairwise_features(x, n_particles, dim):
    """
    x: (n_particles * dim,)
    returns:
        r:   (n_particles, 1)
        rij: (n_pairs, 1)
    """

    # (N, d)
    R = x.reshape(n_particles, dim)

    # -------------------------
    # One-body radii |r_i|
    # -------------------------
    r = jnp.linalg.norm(R, axis=-1, keepdims=True)   # (N, 1)

    # -------------------------
    # Pairwise distances |r_i - r_j|
    # -------------------------
    diff = R[:, None, :] - R[None, :, :]             # (N, N, d)
    dist = jnp.linalg.norm(diff, axis=-1)            # (N, N)

    # Keep only i < j
    i, j = jnp.triu_indices(n_particles, k=1)
    rij = dist[i, j][:, None]                         # (n_pairs, 1)

    eps = 1e-6
    
    #r = jnp.sqrt(jnp.sum(R**2, axis=-1, keepdims=True) + eps)
    #rij = jnp.sqrt(jnp.sum(diff**2, axis=-1) + eps)

    phi_r  = r / (1.0 + r)
    phi_ij = rij / (1.0 + rij)

    return phi_r, phi_ij


import jax
import jax.numpy as jnp
from jax import lax





def get_local_energy_fn(f, Z, nelectrons):
    
    def pe(x):
        """
        Full Coulomb potential energy for a single atom at the origin.

        x: (3 * nelectrons,)
        returns: scalar
        """

        # (N, 3)
        R = x.reshape(nelectrons, 3)

        # -------------------------
        # Electron–nucleus term
        # -------------------------
        r_i = jnp.linalg.norm(R, axis=-1)      # (N,)
        V_en = -Z * jnp.sum(1.0 / r_i)         # scalar

        # -------------------------
        # Electron–electron term
        # -------------------------
        diff = R[:, None, :] - R[None, :, :]   # (N, N, 3)
        dist = jnp.linalg.norm(diff, axis=-1)  # (N, N)

        # keep i < j only
        i, j = jnp.triu_indices(nelectrons, k=1)
        r_ij = dist[i, j]                      # (n_pairs,)

        V_ee = jnp.sum(1.0 / r_ij)             # scalar

        return V_en + V_ee
    
    def _lapl_over_f(params, data):
        n = data.shape[0]
        eye = jnp.eye(n)

        grad_f = jax.grad(f, argnums=1)
        
        def grad_f_closure(x):
            return grad_f(params, x)

        primal, dgrad_f = jax.linearize(grad_f_closure, data)

        hessian_diagonal = lambda i: dgrad_f(eye[i])[i]

        result = -0.5 * lax.fori_loop(
            0, n, lambda i, val: val + hessian_diagonal(i), 0.0)
        return result - 0.5 * jnp.sum(primal ** 2)

    def te(params, x):
        return pe(x) + _lapl_over_f(params, x)
    
    return jax.vmap(te, in_axes=(None, 0), out_axes=(0))

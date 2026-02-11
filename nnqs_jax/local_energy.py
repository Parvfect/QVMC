
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
    
    """
    def _lapl_over_f(params, data):
        n = data.shape[0]
        eye = jnp.eye(n)

        grad_f = jax.grad(f, argnums=1, holomorphic=True)
        
        def grad_f_closure(x):
            return grad_f(params, x)

        primal, dgrad_f = jax.linearize(grad_f_closure, data)

        hessian_diagonal = lambda i: dgrad_f(eye[i])[i]

        result = -0.5 * lax.fori_loop(
            0, n, lambda i, val: val + hessian_diagonal(i), 0.0)
        return result - 0.5 * jnp.sum(primal ** 2)
    """
    def kinetic(params, x):
        psi = f(params, x)
        lap = jnp.trace(jax.hessian(lambda y: f(params, y))(x))
        return -0.5 * lap / psi
    
    def kinetic_vjp(params, x):
        psi = f(params, x)

        # gradient of psi
        grad_psi = jax.grad(lambda y: f(params, y))(x)

        # divergence of grad_psi via VJP
        def grad_psi_fn(y):
            return jax.grad(lambda z: f(params, z))(y)

        _, vjp_fn = jax.vjp(grad_psi_fn, x)
        lap_psi = jnp.sum(vjp_fn(jnp.ones_like(x))[0])

        return -0.5 * lap_psi / psi
    
    def kinetic_logpsi_vjp(params, x):
        logpsi = f(params, x)

        grad_logpsi = jax.grad(lambda y: f(params, y))(x)

        def grad_fn(y):
            return jax.grad(lambda z: f(params, z))(y)

        _, vjp_fn = jax.vjp(grad_fn, x)
        lap_logpsi = jnp.sum(vjp_fn(jnp.ones_like(x))[0])

        return -0.5 * (lap_logpsi + jnp.sum(grad_logpsi**2))

    
    ## This needs to be redefined - log psis are creating problems for this
    def _lapl_over_f(params, data):
        """
        Computes (∇² ψ) / ψ using explicit real/imag decomposition.
        """

        n = data.shape[0]
        eye = jnp.eye(n)

        # --- Split psi into real and imaginary parts ---
        def psi_real(params, x):
            return jnp.real(jnp.log(f(params, x)))

        def psi_imag(params, x):
            return jnp.imag(jnp.log(f(params, x)))

        # --- Gradients ---
        grad_u = jax.grad(psi_real, argnums=1)
        grad_v = jax.grad(psi_imag, argnums=1)

        def grad_u_closure(x):
            return grad_u(params, x)

        def grad_v_closure(x):
            return grad_v(params, x)

        u, du = jax.linearize(grad_u_closure, data)
        v, dv = jax.linearize(grad_v_closure, data)

        # --- Hessian diagonals ---
        hess_u = lambda i: du(eye[i])[i]
        hess_v = lambda i: dv(eye[i])[i]

        lap_u = lax.fori_loop(0, n, lambda i, val: val + hess_u(i), 0.0)
        lap_v = lax.fori_loop(0, n, lambda i, val: val + hess_v(i), 0.0)

        lap = lax.fori_loop(
            0, n, lambda i, val: val + hess_u(i) + 1j * hess_v(i), 0.0)

        # --- Recombine ---
        result = -0.5 * lap
        result -= 0.5 * jnp.sum(u ** 2)
        result += 0.5 * jnp.sum(v ** 2)
        result -= 1j * jnp.sum(u * v)
        
        return result

    def te(params, x):
        return pe(x) + _lapl_over_f(params, x) 
    
    return jax.jit(jax.vmap(te, in_axes=(None, 0)))

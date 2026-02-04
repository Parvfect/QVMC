
import jax
import jax.numpy as jnp
from jax import lax





def get_local_energy_fn(f):
    
    def pe(x):
        r = jnp.linalg.norm(x)
        return -1/r
    
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
        return pe(x) + 0.5 * _lapl_over_f(params, x)
    
    return jax.vmap(te, in_axes=(None, 0), out_axes=(0))

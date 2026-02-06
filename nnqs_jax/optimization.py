
import jax
import jax.numpy as jnp


def get_energy_grads_fn(local_energy_fn, f_b):

    def energy_loss(params, data):
        eloc = local_energy_fn(params, data)   # (B,)
        return jnp.mean(eloc)


    def energy_grads(params, data):
        # ----- forward pass -----
        eloc = local_energy_fn(params, data)          # (B,)
        loss = jnp.mean(eloc)
        diff = eloc - loss                         # (B,)

        # ----- log ψ -----
        log_psi = lambda p: jnp.log(f_b(p, data))  # (B,)

        # ----- VJP -----
        _, vjp_fn = jax.vjp(log_psi, params)

        # Contract batch dimension first
        cotangent = diff / diff.shape[0]            # (B,)

        grads = vjp_fn(cotangent)[0]

        return loss, grads

    return energy_grads

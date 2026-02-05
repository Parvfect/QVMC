
import jax
from sampling import metropolis_step
import optax
from jax.flatten_util import ravel_pytree

# ------------------
# Optimizer
# ------------------
learning_rate = 2e-2
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# ------------------
# Burn-in
# ------------------
burn_in = 100
for i in range(burn_in):
    pos, pmove = metropolis_step(
        params, f_b, pos, key, mcmc_width=0.03
    )

# ------------------
# Optimization loop
# ------------------
nsteps = 100
for step in range(nsteps):

    # --- MCMC move ---
    pos, pmove = metropolis_step(
        params, f_b, pos, key, mcmc_width=0.03
    )

    # --- Energy + gradients ---
    loss, grads = energy_grads(params, pos)

    # --- Optax update ---
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    print(
        f"Step {step:04d} | "
        f"Energy {loss:.6f} | "
        f"Acceptance {pmove:.3f}"
    )

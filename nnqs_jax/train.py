
import jax
import optax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from sampling import metropolis_step
from local_energy import get_local_energy_fn
from optimization import get_energy_grads_fn
from networks import MLP, psi_nn

print("Same problems in a different font")

n_dim = 3
n_electrons = 2
Z = 2
batch_size = 4200

key = jax.random.key(seed=13)
pos = jax.random.normal(
    key, (n_dim * n_electrons))


model = MLP(input_dim=3, n_hidden_layers=2, hidden_dim=3, output_size=1)
variables = model.init(key, jax.random.normal(key, 3))
params = variables["params"]
f = lambda params, x: psi_nn(params, model, x, n_electrons, n_dim, Z)

local_energy = get_local_energy_fn(f, Z, n_electrons)

f_b = jax.vmap(f, in_axes=(None, 0))
local_energy_b = jax.vmap(local_energy, in_axes= (None, 0))
loss_grads_fn = get_energy_grads_fn(local_energy_b, f_b)  # Might want to make this batchless too


learning_rate = 1e-2
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

burn_in = 10
for i in range(burn_in):
    pos, pmove = metropolis_step(
        params, f_b, pos, key, mcmc_width=0.12
    )

nsteps = 100
for step in range(nsteps):

    pos, pmove = metropolis_step(
        params, f_b, pos, key, mcmc_width=0.12
    )

    loss, grads = loss_grads_fn(params, pos)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    print(
        f"Step {step:04d} | "
        f"Energy {loss:.6f} | "
        f"Acceptance {pmove:.3f}"
    )

    
    # Adjust width every k moves
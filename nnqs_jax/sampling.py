import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from jax import lax


def mh_accept(x1, x2, lp_1, lp_2, ratio, key, num_accepts):
  """Given state, proposal, and probabilities, execute MH accept/reject step."""
  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2, x1)
  lp_new = jnp.where(cond, lp_2, lp_1)
  num_accepts += jnp.sum(cond)
  return x_new, key, lp_new, num_accepts

def metropolis_step(
      params, f, data, key, mcmc_width=0.12,
      num_accepts=0, n_steps=10):

    def step_fn(i, carry):
      x1, key, num_accepts = carry
      key, subkey = jax.random.split(key)

      mcmc_width = 0.02
      x2 = x1 + mcmc_width * jax.random.normal(
        subkey, shape=x1.shape)

      lp_1 = 2.0 * f(x1)
      lp_2 = 2.0 * f(x2)
      ratio = lp_2 - lp_1
      
      x_new, key, lp_new, num_accepts = mh_accept(
        x1, x2, lp_1, lp_2, ratio, key, num_accepts)
      return x_new, key, num_accepts
      
    new_data, key, num_accepts = lax.fori_loop(
        0, n_steps, step_fn, (data, key, 0)
    )
    pmove = jnp.sum(num_accepts) / (
      n_steps * data.shape[0])

    return new_data, pmove


def update_mcmc_width(
    t: int,
    width: jnp.ndarray,
    adapt_frequency: int,
    pmove: jnp.ndarray,
    pmoves: np.ndarray,
    pmove_max: float = 0.55,
    pmove_min: float = 0.5,
) -> tuple[jnp.ndarray, np.ndarray]:
  """Updates the width in MCMC steps.

  Args:
    t: Current step.
    width: Current MCMC width.
    adapt_frequency: The number of iterations after which the update is applied.
    pmove: Acceptance ratio in the last step.
    pmoves: Acceptance ratio over the last N steps, where N is the number of
      steps between MCMC width updates.
    pmove_max: The upper threshold for the range of allowed pmove values
    pmove_min: The lower threshold for the range of allowed pmove values

  Returns:
    width: Updated MCMC width.
    pmoves: Updated `pmoves`.
  """

  t_since_mcmc_update = t % adapt_frequency
  # update `pmoves`; `pmove` should be the same across devices
  pmoves[t_since_mcmc_update] = pmove.reshape(-1)[0].item()
  if t > 0 and t_since_mcmc_update == 0:
    if np.mean(pmoves) > pmove_max:
      width *= 1.1
    elif np.mean(pmoves) < pmove_min:
      width /= 1.1
  return width, pmoves

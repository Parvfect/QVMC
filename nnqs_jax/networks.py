
import  jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence
from features import radial_and_pairwise_features


class MLP(nn.Module):
    input_dim: int
    n_hidden_layers: int
    hidden_dim: int
    output_size: int

    @nn.compact
    def __call__(self, x):
        # Input layer
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)

        # Hidden layers
        for _ in range(self.n_hidden_layers - 1):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.tanh(x)

        # Output layer
        x = nn.Dense(self.output_size)(x)
        return x
    
class ComplexMLP(nn.Module):
    n_hidden_layers: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        # Shared trunk
        h = x
        for _ in range(self.n_hidden_layers):
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.tanh(h)

        # Two real outputs
        log_amp = nn.Dense(1)(h)   # log |ψ|
        phase   = nn.Dense(1)(h)   # θ

        return log_amp, phase


def psi_nn(
    params,
    model,
    x,
    n_particles,
    dim
):
    """
    x: (batch, n_particles * dim)
    """

    r, rij = radial_and_pairwise_features(x, n_particles, dim)

    # Flatten features for NN
    features = jnp.concatenate(
        [
            r.reshape(r.shape[0], -1),
            rij.reshape(rij.shape[0], -1),
        ],
        axis=1,
    )

    nn_out = model.apply(params, features)

    # Hydrogenic term (example: Z=2)
    hydrogenic = -2.0 * jnp.sum(r, axis=1)

    # Jastrow term (example form)
    jastrow = jnp.sum(
        rij / (2.0 * (1.0 + 0.5 * rij)),
        axis=1,
    )

    return hydrogenic + jastrow + jnp.squeeze(nn_out)


def psi_nn_complex(
    params,
    model,
    x,
    n_particles,
    dim,
    return_log=False,
):
    """
    Returns:
        ψ(x) or log ψ(x)
    """

    r, rij = radial_and_pairwise_features(x, n_particles, dim)

    features = jnp.concatenate(
        [
            r.reshape(r.shape[0], -1),
            rij.reshape(rij.shape[0], -1),
        ],
        axis=1,
    )

    log_amp_nn, phase_nn = model.apply(params, features)

    # Physics terms (real)
    hydrogenic = -2.0 * jnp.sum(r, axis=1)
    jastrow = jnp.sum(
        rij / (2.0 * (1.0 + 0.5 * rij)),
        axis=1,
    )

    log_amp = hydrogenic + jastrow + jnp.squeeze(log_amp_nn)
    phase = jnp.squeeze(phase_nn)

    if return_log:
        return log_amp + 1j * phase

    return jnp.exp(log_amp + 1j * phase)

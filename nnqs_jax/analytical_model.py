import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from sampling import metropolis_step
from local_energy import get_local_energy_fn
import matplotlib.pyplot as plt
from local_energy import get_local_energy_fn

# Energy levels of H
E1 = -0.5
E2 = -0.125

# Sampling parameters
batch_size = 512
n_electrons = 1
n_dim = 3

key = jax.random.key(2)

key, subkey = jax.random.split(key)
pos = jax.random.uniform(subkey, (batch_size, n_electrons * n_dim))
params = jnp.array([0.0])

def get_energy_mean_variance(energies):
    me = jnp.mean(energies)
    var = jnp.mean((energies - me) ** 2)
    return me, var

def f(params, x):
    r = jnp.linalg.norm(x)
    psi = (
        (1 / jnp.sqrt(jnp.pi)) * jnp.exp(-r)
        + (1 / (4 * jnp.sqrt(2 * jnp.pi))) * (2 - r) * jnp.exp(-r / 2)
    ) / jnp.sqrt(2)
    return psi

def psi_t(params, x, t):
    r = jnp.linalg.norm(x)

    psi_1s = (1 / jnp.sqrt(jnp.pi)) * jnp.exp(-r)
    psi_2s = (1 / (4 * jnp.sqrt(2 * jnp.pi))) * (2 - r) * jnp.exp(-r / 2)

    return (
        psi_1s
        + psi_2s * jnp.exp(-1j * (E2-E1) * t)
    ) / jnp.sqrt(2)


f_b = jax.vmap(f, in_axes=(None, 0))
psi_b = jax.vmap(psi_t, in_axes=(None, 0, None))

mc_step = metropolis_step(params, f_b, pos, key)
mc_step = jax.jit(mc_step)


print("Running warmup steps")
## Warmup steps
for i in range(100):
    key, subkey = jax.random.split(key)
    pos, pmove = mc_step(params, pos, subkey, 0.5)
print(pmove)
print("Warmup completed")


loc_energy = get_local_energy_fn(f, 1, 1)

le = loc_energy(params, pos)

print("Initial Local energy, variance")
print(get_energy_mean_variance(le))


# For importance sampling
psi_ref = psi_t(params, pos, 0.0)
density_ref = jnp.abs(psi_ref) ** 2


t = jnp.linspace(0, 200.0, 40000)
r_expect = []

print("Starting simulation")
for ti in tqdm(t):

    #key, subkey = jax.random.split(key)
    #pos = jax.random.uniform(subkey, (200, 3))
    #key, subkey = jax.random.split(key)

    for i in range(10):
        key, subkey = jax.random.split(key)
        pos, pmove = mc_step(params, pos, subkey, 0.5)

    psi = psi_b(params, pos, ti)
    density = jnp.abs(psi) ** 2
    r = jnp.linalg.norm(pos, axis=1)

    #r_expect_t = jnp.sum(weights * r) / jnp.sum(weights)
    #r_expect.append(r_expect_t)

    r_expect_t = jnp.sum(density * r) / jnp.sum(density)
    r_expect.append(r_expect_t)
    
    #print(get_energy_mean_variance(loc_energy(params, pos)))

r_expect = jnp.array(r_expect)


plt.figure(figsize=(8, 4))
plt.plot(t, r_expect)
plt.xlabel("Time (a.u.)")
plt.ylabel(r"$\langle r \rangle(t)$")
plt.title("Radial Expectation Value Oscillation (1s–2s)")
plt.grid(True)
plt.tight_layout()
plt.show()


print("Getting peak frequency")
signal = r_expect - jnp.mean(r_expect)
dt = t[1] - t[0]

fft_vals = np.fft.rfft(signal)
freqs = np.fft.rfftfreq(len(signal), dt)

power = np.abs(fft_vals) ** 2

idx = np.argmax(power[1:]) + 1   # skip zero frequency
f_peak = freqs[idx]

omega_peak = 2 * np.pi * f_peak
print("Extracted ω:", omega_peak)


dt = t[1] - t[0]

signal = r_expect - np.mean(r_expect)
window = np.hanning(len(signal))

fft_vals = np.fft.rfft(signal * window)
freqs = np.fft.rfftfreq(len(signal), dt)

power = np.abs(fft_vals) ** 2
omega = 2 * np.pi * freqs

plt.figure(figsize=(8, 4))
plt.plot(omega, power)
plt.axvline(0.375, color="red", linestyle="--", label="ΔE (theory)")
plt.axvline(omega[np.argmax(power[1:]) + 1],
            color="green", linestyle=":",
            label=f"Peak = {omega[np.argmax(power[1:]) + 1]:.4f}")

plt.xlim(0, 1.0)
plt.xlabel(r"Angular frequency $\omega$")
plt.ylabel("Power")
plt.title("Frequency Spectrum of ⟨r⟩(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

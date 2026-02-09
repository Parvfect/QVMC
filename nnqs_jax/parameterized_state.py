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
pos = jax.random.uniform(key, (batch_size, n_electrons * n_dim))
params = jnp.array([0.0])

def get_energy_mean_variance(energies):
    me = jnp.mean(energies)
    var = jnp.mean((energies - me) ** 2)
    return me, var

def f(params, x):
    """
    Wavefunction representing the superposition of the ground and first excited state of Hydrogen.
    The time dependent parameter controls the relative phase evolution.
    """
    r = jnp.linalg.norm(x)
    c1 = 1/jnp.sqrt(jnp.pi)
    c2 = 1/(4 * jnp.sqrt(2 * jnp.pi))
    return c1 * jnp.exp(-r) + c2 * jnp.exp(-r/2) * (2 - r) * jnp.exp(1j * params.squeeze())

f_b = jax.vmap(f, in_axes=(None, 0))
f_b = jax.jit(f_b)
loc_energy = get_local_energy_fn(f, Z=1, nelectrons=1)

def get_update(params, pos):
    """Gets theta dot"""
    o_alpha = jax.jacfwd(lambda p: f_b(p, pos))(params)
    o_alpha_centered = o_alpha - jnp.mean(o_alpha)
    s = jnp.mean(jnp.conj(o_alpha) * o_alpha_centered)
    #s += 1e-8
    energy = loc_energy(params, pos)
    energy_centered = energy - jnp.mean(energy)
    F_sup = jnp.mean(jnp.conj(o_alpha) * energy_centered)
    F = -1j * F_sup
    theta_dot = jnp.real(F)/jnp.real(s)
    
    return theta_dot

mc_step = metropolis_step(params, f_b, pos, key)
theta_update = jax.jit(get_update)

print("Running warmup steps")
## Warmup steps
for i in range(10):
    pos, pmove = mc_step(params, pos, key, 0.5)
print(pmove)
print("Warmup completed")


le = loc_energy(params, pos)
print("Initial Local energy, variance")
print(get_energy_mean_variance(le))


t = jnp.linspace(0, 50.0, 200)
r_expect = []
dt = 0.01

print("Starting simulation")
key, subkey = jax.random.split(key)
pos = jax.random.uniform(subkey, (200, 3))
r_expect = []
for i in range(10000):

    # --- (1) Initialize / refresh walkers ---
    #key, subkey = jax.random.split(key)
    #pos = jax.random.uniform(subkey, (200, 3))

    #for j in range(10):
    #    key, subkey = jax.random.split(key)
    #    pos, pmove = mc_step(params, pos, subkey, 0.5)

    # --- (2) RK2 step ---
    # k1
    k1 = theta_update(params, pos)

    # midpoint parameters
    params_mid = params + 0.5 * dt * k1

    # k2 (same walkers!)
    k2 = theta_update(params_mid, pos)

    # full update
    params = params + dt * k2
    print(k2)

    # --- (3) Observables ---
    psi = f_b(params, pos)
    density = jnp.abs(psi) ** 2

    r = jnp.linalg.norm(pos, axis=1)
    r_expect_t = jnp.sum(density * r) / jnp.sum(density)
    r_expect.append(r_expect_t)

    Eloc = loc_energy(params, pos)

    t = i * dt

    if t % 5 == 0:
        print(f"Timestep {t}")
        print(
            f"Energy-variance {get_energy_mean_variance(Eloc)}")
        print(f"R exp {r_expect_t}")
        print(f"Theta {params}")
        print(f"Expected {0.375 * t}")


plt.figure(figsize=(8, 4))
plt.plot(r_expect)
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

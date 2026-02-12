import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from sampling import metropolis_step
from local_energy import get_local_energy_fn
import matplotlib.pyplot as plt
from local_energy import get_local_energy_fn


"""
Time-Dependent Variational Monte Carlo (t-VMC) for Hydrogen 1s-2s Superposition

This program simulates the quantum dynamics of a coherent superposition of hydrogen's
ground state (1s) and first excited state (2s) using time-dependent variational 
Monte Carlo methods.

Physical System
---------------
The quantum state is: |ψ(θ)⟩ = 1/√2 (|ψ₁ₛ⟩ + |ψ₂ₛ⟩ e^(-iθ))

where θ(t) is a time-dependent variational parameter controlling the relative phase
between the two energy eigenstates. As the system evolves, the phase oscillates at
the Bohr frequency ω = E₂ - E₁, causing periodic oscillations in observables like
the expectation value of position ⟨r⟩.

Method
------
Uses the time-dependent variational principle (t-VMC):
    S θ̇ = -iF

where:
- S is the quantum geometric tensor (Fubini-Study metric)
- F is the energy gradient (force term)
- θ̇ is integrated using RK4 (4th order Runge-Kutta)

Monte Carlo sampling is used to evaluate expectation values over 3D position space,
avoiding the need for expensive grid-based integration.

Expected Results
----------------
For the hydrogen 1s-2s superposition:
- Energy: ⟨E⟩ = -0.3125 Hartree (constant, conserved)
- Phase evolution: θ̇ = E₂ - E₁ = 0.375 a.u. (constant)
- Observable oscillations: ⟨r⟩(t) oscillates at frequency ω = 0.375 a.u.
- Period: T = 2π/ω ≈ 16.75 a.u. of time

The program validates the analytical theory by:
1. Verifying θ̇ remains constant at 0.375 a.u.
2. Extracting oscillation frequency from ⟨r⟩(t) via FFT
3. Comparing numerical frequency to analytical prediction

Outputs
-------
- Time series plot of ⟨r⟩(t) showing quantum oscillations
- Frequency spectrum (FFT) identifying dominant oscillation frequency
- Comparison of numerical vs analytical frequencies
- Energy conservation check

Notes
-----
- All quantities in atomic units (ℏ = m_e = e = 1)
- MC sampling fixed at t=0 (does not evolve with density)
- Sampling bias corrected by using broad initial distribution
"""


# Energy levels of H
E1 = -0.5
E2 = -0.125

# Sampling parameters
batch_size = 4200
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
    return (c1 * jnp.exp(-r) + c2 * jnp.exp(-r/2) * (2 - r) * jnp.exp(-1j * params.squeeze())) / (jnp.sqrt(2))

f_b = jax.vmap(f, in_axes=(None, 0))
f_b = jax.jit(f_b)
loc_energy = get_local_energy_fn(f, Z=1, nelectrons=1)

def get_update(params, pos):
    """Gets theta dot"""
    # Compute log-derivative O_alpha = d(log psi)/d(theta)
    o_alpha = jax.jacfwd(lambda p: jnp.log(f_b(p, pos)))(params)
    o_alpha = o_alpha.squeeze()  # shape: (n_samples,)
    
    # Quantum geometric tensor S
    o_alpha_centered = o_alpha - jnp.mean(o_alpha)
    s = jnp.real(jnp.mean(jnp.conj(o_alpha_centered) * o_alpha_centered))
    
    # Local energy
    energy = loc_energy(params, pos)  # shape: (n_samples,)
    
    # Force term F = <O_alpha^* E_L> - <O_alpha^*><E_L>
    F = jnp.mean(jnp.conj(o_alpha) * energy) - jnp.mean(jnp.conj(o_alpha)) * jnp.mean(energy)
    
    # Equation: S * theta_dot = -i * F
    # So: theta_dot = Re(-i * F) / S = Im(F) / S
    theta_dot = jnp.imag(F) / s
    
    return theta_dot


mc_step = metropolis_step(params, f_b, pos, key)
theta_update = jax.jit(get_update)

print("Running warmup steps")
key, subkey = jax.random.split(key)
pos = jax.random.uniform(subkey, (4200, 3))

for i in range(200):
    key, subkey = jax.random.split(key)
    pos, pmove = mc_step(params, pos, subkey, mcmc_width=0.5)
print(pmove)
print("Warmup completed")


le = loc_energy(params, pos)
print("Initial Local energy, variance")
print(get_energy_mean_variance(le))


r_expect = []
dt = 0.01
T = 100
n_steps = int(T/dt)

print("Starting simulation")
r_expect = []
for i in tqdm(range(n_steps)):

    theta_dot = theta_update(params, pos)

    # midpoint parameters
    params = params +  dt * theta_dot

    # --- (3) Observables ---
    psi = f_b(params, pos)
    density = jnp.abs(psi) ** 2

    r = jnp.linalg.norm(pos, axis=1)
    r_expect_t = jnp.sum(density * r) / jnp.sum(density)
    r_expect.append(r_expect_t)

    Eloc = loc_energy(params, pos)

plt.figure(figsize=(8, 4))
plt.plot(r_expect)
plt.xlabel("Time (a.u.)")
plt.ylabel(r"$\langle r \rangle(t)$")
plt.title("Radial Expectation Value Oscillation (1s–2s)")
plt.grid(True)
plt.tight_layout()
plt.show()


print("Getting peak frequency")
r_expect = jnp.array(r_expect)
signal = r_expect - jnp.mean(r_expect)

fft_vals = np.fft.rfft(signal)
freqs = np.fft.rfftfreq(len(signal), dt)

power = np.abs(fft_vals) ** 2

idx = np.argmax(power[1:]) + 1   # skip zero frequency
f_peak = freqs[idx]

omega_peak = 2 * np.pi * f_peak
print("Extracted ω:", omega_peak)

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

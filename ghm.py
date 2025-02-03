import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.fftpack as fft
import scipy.signal as signal
import pywt  # Wavelet Analysis
from scipy.integrate import odeint
from scipy.stats import linregress

# --- Constants ---
LAMBDA = 0.5   # Damping coefficient
ETA = 1.0      # Renormalization constant
T0 = 1.0       # Initial time offset
N = 150        # Number of terms in series expansions
NU = 0.1       # Viscosity parameter for Navier-Stokes

# ------------------------------
# 1. Generalized Harmonia Mechanics Energy Evolution
# ------------------------------
def ghm_dynamics(E, t, lambda_, eta, t0):
    weight = np.exp(-eta / np.log(t + t0))
    oscillatory_sum = sum([np.cos(2 * np.pi * (n + 1) * t) / (n + 1) for n in range(N)])
    dE_dt = -lambda_ * E * weight + oscillatory_sum
    return dE_dt

t_range = np.linspace(1, 100, 500)
E0 = 1.0
E_solution = odeint(ghm_dynamics, E0, t_range, args=(LAMBDA, ETA, T0))

plt.figure(figsize=(8, 5))
plt.plot(t_range, E_solution, label="GHM Energy Evolution")
plt.xlabel("Time")
plt.ylabel("Energy E(t)")
plt.title("GHM Energy Evolution under Damping & Oscillatory Influence")
plt.legend()
plt.grid()
plt.show()

# ------------------------------
# 2. Navier-Stokes Extended Analysis (Turbulence)
# ------------------------------
def navier_stokes_turbulence(u0, t, nu):
    du_dt = -nu * u0 + np.cos(2 * np.pi * t)
    return du_dt

t_ns = np.linspace(0, 50, 500)
u_solution = odeint(navier_stokes_turbulence, E0, t_ns, args=(NU,))

plt.figure(figsize=(8, 5))
plt.plot(t_ns, u_solution, label="Navier-Stokes Stability")
plt.xlabel("Time")
plt.ylabel("Velocity Field Energy")
plt.title("Navier-Stokes Stability with Turbulence Correction")
plt.legend()
plt.grid()
plt.show()

# Compute Kolmogorov Spectrum (Fourier Transform)
freqs = fft.fftfreq(len(t_ns), d=t_ns[1] - t_ns[0])
spectrum = np.abs(fft.fft(u_solution.flatten()))

plt.figure(figsize=(8, 5))
plt.loglog(freqs[freqs > 0], spectrum[freqs > 0], label="Kolmogorov Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Energy Spectrum")
plt.title("Kolmogorov Turbulence Spectrum (Navier-Stokes)")
plt.legend()
plt.grid()
plt.show()

# ------------------------------
# 3. Prime Gaps - Autocorrelation & Fourier Analysis
# ------------------------------
def compute_prime_gaps(n):
    primes = [2]
    num = 3
    while len(primes) < n:
        if all(num % p != 0 for p in primes):
            primes.append(num)
        num += 2
    return [primes[i+1] - primes[i] for i in range(len(primes)-1)]

prime_gaps = compute_prime_gaps(N)
prime_autocorr = np.correlate(prime_gaps, prime_gaps, mode="full")

plt.figure(figsize=(8, 5))
plt.plot(prime_autocorr[len(prime_autocorr)//2:], label="Prime Gap Autocorrelation")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.title("Autocorrelation of Prime Gaps")
plt.legend()
plt.grid()
plt.show()

# Fourier Transform of Prime Gaps
prime_spectrum = np.abs(fft.fft(prime_gaps))
plt.figure(figsize=(8, 5))
plt.plot(prime_spectrum, label="Prime Gap Fourier Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Fourier Spectrum of Prime Gaps")
plt.legend()
plt.grid()
plt.show()

# ------------------------------
# 4. Wavelet Analysis - GHM Energy Oscillations
# ------------------------------
wavelet_coeffs, freqs = pywt.cwt(E_solution.flatten(), scales=np.arange(1, 100), wavelet='cmor')

plt.figure(figsize=(8, 5))
plt.imshow(np.abs(wavelet_coeffs), aspect='auto', cmap='inferno', extent=[t_range.min(), t_range.max(), 1, 100])
plt.colorbar(label="Magnitude")
plt.xlabel("Time")
plt.ylabel("Scale (Frequency Inverse)")
plt.title("Wavelet Transform of GHM Energy Oscillations")
plt.show()

# ------------------------------
# 5. Hubble Tension Extended Cosmology
# ------------------------------
def hubble_modified(a, eta=0.17, p=1.0):
    rho_m0, rho_L0 = 0.3, 0.7
    return np.sqrt((rho_m0 * a**-3 + rho_L0) * np.exp(eta / a**p))

a_values = np.linspace(0.1, 2.0, 100)
H_modified = [hubble_modified(a) for a in a_values]

plt.figure(figsize=(8, 5))
plt.plot(a_values, H_modified, label="Hubble Parameter w/ GHM Correction")
plt.xlabel("Scale Factor a")
plt.ylabel("Hubble Parameter H(a)")
plt.title("Hubble Parameter Evolution with GHM")
plt.legend()
plt.grid()
plt.show()

# ------------------------------
# 6. Monte Carlo Stability Analysis
# ------------------------------
num_trials = 1000
stability_results = []

for _ in range(num_trials):
    perturbed_X = prime_gaps + np.random.normal(0, 0.1, len(prime_gaps))
    stability_results.append(np.var(perturbed_X))

plt.figure(figsize=(8, 5))
plt.hist(stability_results, bins=50, alpha=0.7, color='blue', label="Monte Carlo Stability")
plt.axvline(np.mean(stability_results), color='red', linestyle="--", label="Mean Variance")
plt.xlabel("Stability Functional Variance")
plt.ylabel("Frequency")
plt.title("Monte Carlo Simulation of Stability Functional")
plt.legend()
plt.grid()
plt.show()

# ------------------------------
# Final Summary
# ------------------------------
print("\n=== Summary of Computations ===")
print(f"GHM Stability Functional: {np.var(E_solution):.4f}")
print(f"Navier-Stokes Turbulence Variance: {np.var(u_solution):.4f}")
print(f"Prime Gap Variance (Autocorr): {np.var(prime_autocorr):.4f}")
print(f"Hubble Parameter at a=1 (GHM Corrected): {hubble_modified(1):.4f}")


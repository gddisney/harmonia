# Re-import required libraries since execution state was reset
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Simulated photon trajectory data (radial distance over time)
# Assuming the photon moves in a stable oscillatory orbit rather than collapsing.
# Replace this with actual simulation data if available.

T = 10  # Total time in seconds
dt = 0.01  # Time step
t = np.arange(0, T, dt)  # Time array

# Example: Harmonic oscillation in radial motion (replace with real trajectory data)
r_photon = 10 + 2 * np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * (1.618) * t)

# Compute FFT of the radial motion
N = len(t)
fft_values = fft(r_photon)
freqs = fftfreq(N, dt)

# Only consider positive frequencies for analysis
positive_freqs = freqs[:N//2]
positive_fft_values = np.abs(fft_values[:N//2])

# Plot the FFT result
plt.figure(figsize=(10, 5))
plt.plot(positive_freqs, positive_fft_values, label="FFT Magnitude", color="blue")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Fourier Transform of Photon's Radial Motion")
plt.xlim(0, 5)  # Focus on lower frequencies for harmonic structure
plt.grid()
plt.legend()
plt.show()

# Identify dominant frequencies
dominant_freqs = positive_freqs[np.argsort(positive_fft_values)[-5:]]  # Top 5 frequencies

# Return the detected dominant frequencies
dominant_freqs


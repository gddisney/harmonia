import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Load the radial motion data (simulated time-series data)
T = 10  # Total time (s)
dt = 0.01  # Time step (s)
t = np.arange(0, T, dt)

# Simulated photon's radial motion (example: an oscillatory function influenced by gravity)
# This function models a photon trapped in a quasi-stable orbit influenced by a Harmonia resonance.
r_motion = np.exp(-0.1 * t) * np.cos(2 * np.pi * (0.618 * t))  # Golden ratio based

# Compute the FFT of the radial motion
fft_vals = fft(r_motion)
freqs = fftfreq(len(r_motion), dt)

# Extract positive frequencies
positive_freqs = freqs[:len(freqs)//2]
positive_fft_vals = np.abs(fft_vals[:len(fft_vals)//2])

# Find the dominant frequencies
dominant_freqs = positive_freqs[np.argsort(positive_fft_vals)[-5:]]  # Top 5 frequencies

# Plot the FFT spectrum
plt.figure(figsize=(10, 5))
plt.plot(positive_freqs, positive_fft_vals, color='blue', label="FFT Magnitude")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Fourier Transform of Photon's Radial Motion")
plt.legend()
plt.grid()

# Display dominant frequencies
for freq in dominant_freqs:
    plt.axvline(freq, linestyle='dashed', color='red', alpha=0.6)
    plt.text(freq, max(positive_fft_vals) * 0.7, f"{freq:.3f} Hz", color='red')

plt.xlim(0, 5)  # Focus on low frequencies
plt.show()

# Check for golden ratio scaling
golden_ratio = (1 + np.sqrt(5)) / 2  # ~1.618
golden_ratio_matches = [f for f in dominant_freqs if np.isclose(f / dominant_freqs[0], golden_ratio, atol=0.05)]

# Output results
golden_ratio_matches, dominant_freqs[:5]


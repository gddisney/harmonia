import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
M = 1.989e30     # Mass of central object (e.g., Sun in kg)
c = 3.0e8        # Speed of light (m/s)
num_frames = 3000  # Number of frames in animation
dt = 0.000000001        # Time step for simulation

# Initial Conditions (Photon starts at r=5, ?=0 with tangential velocity)
r_init = 5.0  # Initial radius (arbitrary units)
theta_init = 0.0
v_init = 0.9 * c  # Initial velocity (as fraction of c)

# Arrays for photon trajectory
r_vals = np.zeros(num_frames)
theta_vals = np.zeros(num_frames)
x_vals = np.zeros(num_frames)
y_vals = np.zeros(num_frames)

# Set initial values
r_vals[0] = r_init
theta_vals[0] = theta_init

# Compute trajectory with gravitational lensing
for i in range(1, num_frames):
    # Harmonia Mechanics correction factor (oscillatory stabilization)
    harmonia_factor = 1 + 0.05 * np.sin(2 * np.pi * i / num_frames)

    # Gravitational acceleration (approximated)
    a_gravity = (G * M / r_vals[i-1]**2) / c**2  # Normalized

    # Update radial distance (with Harmonia correction)
    r_vals[i] = r_vals[i-1] - dt * a_gravity * harmonia_factor

    # Update angular position (Keplerian motion)
    theta_vals[i] = theta_vals[i-1] + (v_init / r_vals[i-1]) * dt

    # Convert to Cartesian coordinates
    x_vals[i] = r_vals[i] * np.cos(theta_vals[i])
    y_vals[i] = r_vals[i] * np.sin(theta_vals[i])

# Plot setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-r_init, r_init)
ax.set_ylim(-r_init, r_init)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Photon Bending Near Massive Object")

# Background object (e.g., black hole or star)
ax.plot(0, 0, 'yo', markersize=10, label="Massive Object")

# Initialize photon
photon_dot, = ax.plot([], [], 'ro', markersize=6)

# Fix: Function must return a **tuple of lists** not single values
def update(frame):
    photon_dot.set_data([x_vals[frame]], [y_vals[frame]])  # Wrapping values in lists
    return photon_dot,

# Create animation (blit=False fixes some backend issues)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=30, blit=False)

plt.legend()
plt.show()


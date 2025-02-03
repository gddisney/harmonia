import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# -----------------------------
# Define Parametric Wormhole Geometry
# -----------------------------

u = np.linspace(0, 1, 50)
v = np.linspace(0, 2 * np.pi, 50)
U, V = np.meshgrid(u, v)

def entry_region(U, V):
    """ Flat entry plane (disk). """
    x = U * np.cos(V)
    y = U * np.sin(V)
    z = np.zeros_like(U)
    return x, y, z

def throat_region(U, V):
    """ Cylindrical wormhole throat. """
    x = np.cos(V)
    y = np.sin(V)
    z = 2 * U - 1  # Maps U from [0,1] to z in [-1,1]
    return x, y, z

def exit_region(U, V):
    """ Bent toroidal exit region. """
    R0 = 1.0  # Major (centerline) radius (controls bending)
    r  = 0.3  # Minor (tube) radius
    theta = np.pi * U  # bending angle from 0 to c (half torus)
    x = (R0 + r * np.cos(V)) * np.cos(theta)
    y = (R0 + r * np.cos(V)) * np.sin(theta)
    z = r * np.sin(V)
    return x, y, z

def healing_region(U, V, t_current):
    """ Healing dimple that fades over time with gravitational relaxation. """
    d = 0.3 * np.exp(-0.5 * t_current)  # Exponential decay mimicking energy dissipation
    x = U * np.cos(V)
    y = U * np.sin(V)
    z = -d * (1 - U**2)  # Max depression at center, flattens at the boundary
    return x, y, z

# -----------------------------
# Define a Spacetime Field with Gravitational Waves
# -----------------------------

def spacetime_field(X, Y, t_current):
    """ Oscillatory energy density field with propagating gravitational waves. """
    wave_speed = 0.8  # Wave propagation speed
    wavelength = 1.5
    frequency = 1.0
    phase_shift = np.pi / 4

    # Gaussian wave pulse spreading outward after closure
    ripple = np.exp(-0.5 * (X**2 + Y**2)) * np.cos(2 * np.pi * (t_current * frequency - np.sqrt(X**2 + Y**2) / wavelength) + phase_shift)
    
    return 0.5 * np.cos(2 * np.pi * 0.5 * t_current) * np.exp(-0.5 * (X**2 + Y**2)) + ripple

X_space, Y_space = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))

# -----------------------------
# Simulating Particle Trajectories Through the Wormhole
# -----------------------------

num_particles = 10
particle_positions = np.random.uniform(-1, 1, (num_particles, 3))
particle_velocities = np.random.uniform(-0.1, 0.1, (num_particles, 3))

def update_particles():
    """ Updates particle positions as they move through the wormhole. """
    global particle_positions
    particle_positions += particle_velocities
    # Time dilation effect: slow down near the throat
    particle_positions[:, 2] *= 0.98

# -----------------------------
# Set Up Simulation
# -----------------------------

T = 10
dt = 0.01
t = np.arange(0, T, dt)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    global dt
    p = (frame % 400) / 100.0  # Normalized morphing phase [0,4)

    x_entry, y_entry, z_entry = entry_region(U, V)
    x_throat, y_throat, z_throat = throat_region(U, V)
    x_exit, y_exit, z_exit = exit_region(U, V)
    t_current = frame * dt

    if p <= 1:
        # Entry morphing into Throat
        lam = p
        x_w = (1 - lam) * x_entry + lam * x_throat
        y_w = (1 - lam) * y_entry + lam * y_throat
        z_w = (1 - lam) * z_entry + lam * z_throat
        morph_label = "Entry ? Throat"

    elif p <= 2:
        # Throat morphing into Exit
        mu = p - 1
        x_w = (1 - mu) * x_throat + mu * x_exit
        y_w = (1 - mu) * y_throat + mu * y_exit
        z_w = (1 - mu) * z_throat + mu * z_exit
        morph_label = "Throat ? Exit"

    elif p <= 3:
        # Exit morphing into Healing (Dimple)
        nu = p - 2
        x_heal, y_heal, z_heal = healing_region(U, V, t_current)
        x_w = (1 - nu) * x_exit + nu * x_heal
        y_w = (1 - nu) * y_exit + nu * y_heal
        z_w = (1 - nu) * z_exit + nu * z_heal
        morph_label = "Exit ? Healing"

    else:
        # Healing Phase Finalizing (Fading Dimple)
        gamma = p - 3
        x_heal, y_heal, z_heal = healing_region(U, V, t_current)
        x_w = (1 - gamma) * x_heal + gamma * x_entry
        y_w = (1 - gamma) * y_heal + gamma * y_entry
        z_w = (1 - gamma) * z_heal + gamma * z_entry
        morph_label = "Healing ? Flat"

    # Clear frame
    ax.cla()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Apply Time Dilation Effect
    time_dilation_factor = 1.0 / (1.0 + np.exp(-t_current + 5))
    dt_modified = dt * time_dilation_factor
    t_current = frame * dt_modified

    # Spacetime Field with Gravitational Waves
    Z_space = spacetime_field(X_space, Y_space, t_current)
    ax.plot_surface(X_space, Y_space, Z_space, cmap='plasma', alpha=0.4, edgecolor='none')

    # Wormhole Surface
    ax.plot_surface(x_w, y_w, z_w, cmap='viridis', edgecolor='none')

    # Update and Plot Particle Trajectories
    update_particles()
    ax.scatter(particle_positions[:, 0], particle_positions[:, 1], particle_positions[:, 2], color='red', s=10)

    # Title update
    ax.set_title(f"Wormhole Evolution: {morph_label} (t = {t_current:.2f}s)")

    return ax

# Create Animation
ani = animation.FuncAnimation(fig, update, frames=400, interval=50, blit=False)

plt.show()

